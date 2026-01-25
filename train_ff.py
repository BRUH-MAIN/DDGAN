"""Training script for Deepfake Detection on FaceForensics++ dataset.

This script provides training with:
- FaceForensics++ image dataset support
- Multiple data imbalance handling strategies
- AAML (Additive Angular Margin Loss) support
- Weighted sampling and focal loss

Usage:
    python train_ff.py --data_dir ./images_dataset --epochs 50
    python train_ff.py --help
"""

import argparse
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import get_default_config
from src.data.ff_dataset import FaceForensicsDataset
from src.data.transforms import get_gpu_transform, cpu_transform
from src.training.ff_trainer import FFDeepfakeGANModule
from src.training.imbalance_losses import compute_class_weights


class FFDataModule(L.LightningDataModule):
    """Lightning DataModule for FaceForensics++ dataset.
    
    Handles data loading with:
    - Weighted sampling for class imbalance
    - GPU transforms for efficiency
    - Proper train/val splitting by video ID
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        use_weighted_sampling: bool = True,
        max_samples_per_category: int = None,
        categories: list = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampling = use_weighted_sampling
        self.max_samples_per_category = max_samples_per_category
        self.categories = categories
        self.seed = seed
        
        self.gpu_transform = None
        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None
    
    def setup(self, stage: str = None):
        """Set up datasets."""
        # Create training dataset
        self.train_dataset = FaceForensicsDataset(
            root_dir=self.data_dir,
            categories=self.categories,
            max_samples_per_category=self.max_samples_per_category,
            split='train',
            seed=self.seed,
        )
        
        # Create validation dataset
        self.val_dataset = FaceForensicsDataset(
            root_dir=self.data_dir,
            categories=self.categories,
            max_samples_per_category=self.max_samples_per_category,
            split='val',
            seed=self.seed,
        )
        
        # Compute class weights from training data
        train_stats = self.train_dataset.get_statistics()
        self.class_weights = compute_class_weights(
            n_fake=train_stats['n_fake'],
            n_real=train_stats['n_real'],
            method='effective'  # Best for extreme imbalance
        )
        
        # Print dataset statistics
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Training samples: {train_stats['total_samples']}")
        print(f"  - Real: {train_stats['n_real']}")
        print(f"  - Fake: {train_stats['n_fake']}")
        print(f"  - Imbalance ratio: {train_stats['imbalance_ratio']:.2f}:1")
        print(f"  - Class weights: fake={self.class_weights[0]:.3f}, real={self.class_weights[1]:.3f}")
        print(f"\nValidation samples: {self.val_dataset.get_statistics()['total_samples']}")
        print(f"Categories: {list(train_stats['categories'].keys())}")
        print("=" * 60 + "\n")
        
        # Create GPU transform
        self.gpu_transform = get_gpu_transform()
    
    def get_class_weights(self) -> torch.Tensor:
        """Get computed class weights."""
        return self.class_weights
    
    def collate_fn(self, batch):
        """Custom collate function."""
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor(
            [item['label'] for item in batch], 
            dtype=torch.float32
        )
        return images, labels
    
    def train_dataloader(self):
        sampler = None
        shuffle = True
        
        if self.use_weighted_sampling:
            sampler = self.train_dataset.get_weighted_sampler()
            shuffle = False
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply GPU transforms."""
        images, labels = batch
        if self.gpu_transform is not None:
            if not hasattr(self, '_transform_device') or self._transform_device != images.device:
                self.gpu_transform = self.gpu_transform.to(images.device)
                self._transform_device = images.device
            images = self.gpu_transform(images)
        return images, labels


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Deepfake Detection on FaceForensics++"
    )
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./images_dataset",
                       help="Path to FaceForensics++ images dataset")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples per category (for debugging)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--d_lr", type=float, default=2e-4,
                       help="Discriminator learning rate")
    parser.add_argument("--g_lr", type=float, default=2e-4,
                       help="Generator learning rate")
    parser.add_argument("--precision", type=str, default="16-mixed",
                       choices=["32", "16-mixed", "bf16-mixed"],
                       help="Training precision")
    
    # Model arguments
    parser.add_argument("--no_pretrained", action="store_true",
                       help="Don't use pretrained weights")
    parser.add_argument("--epsilon", type=float, default=0.03,
                       help="Perturbation strength")
    
    # Imbalance handling arguments
    parser.add_argument("--no_weighted_sampling", action="store_true",
                       help="Disable weighted random sampling")
    parser.add_argument("--loss_type", type=str, default="focal",
                       choices=["bce", "focal", "weighted_bce", "aaml", "combined"],
                       help="Loss function type for handling imbalance")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Gamma for focal loss")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                       help="Alpha for focal loss")
    parser.add_argument("--aaml_margin", type=float, default=0.5,
                       help="Margin for AAML loss")
    parser.add_argument("--aaml_scale", type=float, default=30.0,
                       help="Scale for AAML loss")
    
    # Category filtering
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                       help="Categories to include (default: all)")
    
    # Path arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory for logs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Run a fast development test")
    parser.add_argument("--refresh_rate", type=int, default=0,
                       help="Progress bar refresh rate")
    
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Set seed
    L.seed_everything(args.seed, workers=True)
    
    # Create data module
    data_module = FFDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampling=not args.no_weighted_sampling,
        max_samples_per_category=args.max_samples,
        categories=args.categories,
        seed=args.seed,
    )
    
    # Setup to get class weights
    data_module.setup()
    class_weights = data_module.get_class_weights()
    
    # Create model
    model = FFDeepfakeGANModule(
        pretrained=not args.no_pretrained,
        epsilon=args.epsilon,
        d_lr=args.d_lr,
        g_lr=args.g_lr,
        total_epochs=args.epochs,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        aaml_margin=args.aaml_margin,
        aaml_scale=args.aaml_scale,
        class_weights=class_weights,
    )
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="best-{epoch:02d}-{val/f1:.4f}",
            monitor="val/f1",  # F1 is better for imbalanced data
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="checkpoint-{epoch:02d}",
            every_n_epochs=5,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=args.refresh_rate),
        EarlyStopping(
            monitor="val/f1",
            patience=10,
            mode="max",
            verbose=True,
        ),
    ]
    
    # Logger
    logger = CSVLogger(
        save_dir=args.log_dir,
        name="ff_deepfake_gan",
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        log_every_n_steps=10,
        val_check_interval=1.0,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
    )
    
    # Print configuration
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION GAN - FaceForensics++ Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Loss type: {args.loss_type}")
    print(f"Weighted sampling: {not args.no_weighted_sampling}")
    print(f"Class weights: {class_weights.tolist()}")
    print(f"Precision: {args.precision}")
    print("=" * 60 + "\n")
    
    # Train
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model: {args.checkpoint_dir}")
    print(f"Logs: {args.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
