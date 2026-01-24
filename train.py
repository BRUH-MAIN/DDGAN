"""Training script for Deepfake Detection GAN using PyTorch Lightning.

This script handles the complete training pipeline including data loading,
model initialization, and training with automatic checkpointing and logging.

Usage:
    python train.py --dataset_name "your-dataset/name" --epochs 50
    python train.py --help
"""

import argparse
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from config import get_default_config
from src.data.dataset import load_deepfake_dataset
from src.data.transforms import collate_fn, get_gpu_transform
from src.training.trainer import DeepfakeGANModule


class GPUTransformDataModule(L.LightningDataModule):
    """Lightning DataModule that applies GPU transforms.
    
    Wraps the dataset and applies GPU transforms during batch transfer.
    """
    
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
        cache_dir: str = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.gpu_transform = None
        self.dataset = None
    
    def prepare_data(self):
        """Download data if needed."""
        # This will download and cache the dataset
        load_deepfake_dataset(self.dataset_name, cache_dir=self.cache_dir)
    
    def setup(self, stage: str = None):
        """Set up datasets for each stage."""
        self.dataset = load_deepfake_dataset(
            self.dataset_name, 
            cache_dir=self.cache_dir
        )
        self.dataset.set_format(columns=['image', 'label'])
        
        # Create GPU transform
        self.gpu_transform = get_gpu_transform()
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset['val'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply GPU transforms after batch is transferred to GPU."""
        images, labels = batch
        if self.gpu_transform is not None:
            # Move transform to same device as images
            if not hasattr(self, '_transform_device') or self._transform_device != images.device:
                self.gpu_transform = self.gpu_transform.to(images.device)
                self._transform_device = images.device
            images = self.gpu_transform(images)
        return images, labels


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Deepfake Detection GAN")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="HuggingFace dataset name")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for datasets")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
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
    
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Load default config and override with args
    config = get_default_config()
    
    if args.dataset_name:
        config.dataset.dataset_name = args.dataset_name
    if args.cache_dir:
        config.dataset.cache_dir = args.cache_dir
    config.dataset.num_workers = args.num_workers
    
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.d_lr = args.d_lr
    config.training.g_lr = args.g_lr
    config.training.precision = args.precision
    
    config.model.pretrained = not args.no_pretrained
    config.model.epsilon = args.epsilon
    
    config.paths.checkpoint_dir = args.checkpoint_dir
    config.paths.log_dir = args.log_dir
    
    config.seed = args.seed
    
    # Set seed for reproducibility
    L.seed_everything(config.seed, workers=True)
    
    # Create data module
    data_module = GPUTransformDataModule(
        dataset_name=config.dataset.dataset_name,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        cache_dir=config.dataset.cache_dir,
    )
    
    # Create model
    model = DeepfakeGANModule(
        pretrained=config.model.pretrained,
        epsilon=config.model.epsilon,
        d_lr=config.training.d_lr,
        g_lr=config.training.g_lr,
        max_grad_norm=config.training.max_grad_norm,
        total_epochs=config.training.epochs,
    )
    
    # Callbacks
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            dirpath=config.paths.checkpoint_dir,
            filename="best-{epoch:02d}-{val/accuracy:.4f}",
            monitor="val/accuracy",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        # Save checkpoints periodically
        ModelCheckpoint(
            dirpath=config.paths.checkpoint_dir,
            filename="checkpoint-{epoch:02d}",
            every_n_epochs=5,
            save_top_k=-1,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
        # Progress bar
        RichProgressBar(),
        # Early stopping (optional)
        EarlyStopping(
            monitor="val/accuracy",
            patience=10,
            mode="max",
            verbose=True,
        ),
    ]
    
    # Logger
    logger = CSVLogger(
        save_dir=config.paths.log_dir,
        name="deepfake_gan",
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.training.epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.training.precision,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=config.logging.log_every_n_steps,
        val_check_interval=config.logging.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )
    
    # Print config
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION GAN - Training")
    print("=" * 60)
    print(f"Dataset: {config.dataset.dataset_name}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Precision: {config.training.precision}")
    print(f"Accelerator: {config.accelerator}")
    print(f"Checkpoint dir: {config.paths.checkpoint_dir}")
    print("=" * 60 + "\n")
    
    # Train
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved to: {config.paths.checkpoint_dir}")
    print(f"Logs saved to: {config.paths.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
