"""
Main training script for Deepfake Detection GAN
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

from config import default_config
from data import DeepfakeDataModule
from lightning_module import DeepfakeGAN


def main(args):
    """Main training function"""
    
    # Set random seed for reproducibility
    pl.seed_everything(default_config.seed)
    
    # Print configuration
    print("=" * 60)
    print("DEEPFAKE DETECTION GAN - TRAINING")
    print("=" * 60)
    print(f"Experiment: {default_config.experiment_name}")
    
    # Print dataset source
    if default_config.data.hf_dataset_id:
        print(f"Dataset: {default_config.data.hf_dataset_id} (HuggingFace)")
    else:
        print(f"Dataset: {default_config.data.dataset_root} (Local)")
    
    print(f"Batch size: {default_config.data.batch_size}")
    print(f"Max epochs: {default_config.training.max_epochs}")
    print(f"Learning rate: {default_config.training.learning_rate}")
    print(f"Devices: {default_config.training.devices} {default_config.training.accelerator}")
    print(f"Precision: {default_config.training.precision}")
    print("=" * 60 + "\n")
    
    # Initialize DataModule
    print("Initializing DataModule...")
    datamodule = DeepfakeDataModule(
        data_dir=str(default_config.data.dataset_root) if not default_config.data.hf_dataset_id else None,
        hf_dataset_id=default_config.data.hf_dataset_id,
        batch_size=default_config.data.batch_size,
        num_workers=default_config.data.num_workers,
        pin_memory=default_config.data.pin_memory,
        persistent_workers=default_config.data.persistent_workers,
        mean=default_config.data.mean,
        std=default_config.data.std
    )
    
    # Initialize model
    print("\nInitializing Model...")
    model = DeepfakeGAN(
        d_backbone=default_config.model.d_backbone,
        d_pretrained=default_config.model.d_pretrained,
        g_base_channels=default_config.model.g_base_channels,
        epsilon=default_config.model.epsilon,
        lr=default_config.training.learning_rate,
        betas=default_config.training.betas,
        weight_decay=default_config.training.weight_decay,
        adv_weight=default_config.training.adv_weight,
        perturb_weight=default_config.training.perturb_weight,
        scheduler_type=default_config.training.scheduler_type,
        max_epochs=default_config.training.max_epochs
    )
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    
    # Progress bar callback
    progress_bar = RichProgressBar(refresh_rate=default_config.training.get('refresh_rate', 1))
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=default_config.training.checkpoint_dir,
        filename='{epoch:02d}-{val/accuracy:.4f}',
        monitor='val/accuracy',
        mode='max',
        save_top_k=default_config.training.save_top_k,
        save_last=True,
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Early stopping (optional)
    early_stopping = EarlyStopping(
        monitor='val/accuracy',
        patience=10,
        mode='max',
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=default_config.experiment_name
    )
    
    # Initialize trainer
    print("\nInitializing Trainer...")
    trainer = pl.Trainer(
        max_epochs=default_config.training.max_epochs,
        accelerator=default_config.training.accelerator,
        devices=default_config.training.devices,
        strategy=default_config.training.strategy if default_config.training.devices > 1 else 'auto',
        precision=default_config.training.precision,
        callbacks=[progress_bar, checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,
        log_every_n_steps=default_config.training.log_every_n_steps,
        val_check_interval=default_config.training.val_check_interval,
        gradient_clip_val=default_config.training.gradient_clip_val,
        deterministic=False,  # Set to False to allow benchmark optimization
        benchmark=True,  # Enable cudnn benchmarking for faster training
        enable_progress_bar=default_config.training.get('refresh_rate', 1) > 0,
        # Progress bar refresh rate controlled by RichProgressBar callback
    )
    
    # Print trainer info
    print(f"\nTrainer configuration:")
    print(f"  Max epochs: {trainer.max_epochs}")
    print(f"  Precision: {trainer.precision}")
    print(f"  Accelerator: {trainer.accelerator}")
    print(f"  Devices: {trainer.num_devices}")
    print(f"  Strategy: {trainer.strategy.__class__.__name__}")
    print(f"  Gradient clipping: {trainer.gradient_clip_val}")
    
    # Start training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    
    try:
        trainer.fit(model, datamodule=datamodule)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Print best model info
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 60 + "\n")
    
    # Save final model
    final_model_path = default_config.training.checkpoint_dir / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Deepfake Detection GAN')
    parser.add_argument('--data-dir', type=str, default='celebdfv2_images',
                        help='Path to local dataset directory')
    parser.add_argument('--hf-dataset', type=str, default=None,
                        help='HuggingFace dataset ID (e.g., RohanRamesh/celebdfv2_224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--devices', type=int, default=2,
                        help='Number of GPUs to use')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained weights for discriminator')
    parser.add_argument('--refresh-rate', type=int, default=1,
                        help='Progress bar refresh rate (updates per second). Set to 0 to disable.')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.hf_dataset:
        default_config.data.hf_dataset_id = args.hf_dataset
        default_config.data.dataset_root = None
    elif args.data_dir:
        default_config.data.dataset_root = args.data_dir
        default_config.data.hf_dataset_id = None
    if args.batch_size:
        default_config.data.batch_size = args.batch_size
    if args.epochs:
        default_config.training.max_epochs = args.epochs
    if args.lr:
        default_config.training.learning_rate = args.lr
    if args.devices:
        default_config.training.devices = args.devices
    if args.no_pretrained:
        default_config.model.d_pretrained = False
    if hasattr(args, 'refresh_rate'):
        default_config.training.refresh_rate = args.refresh_rate
    
    # Run training
    main(args)
