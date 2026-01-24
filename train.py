"""Training script for Deepfake Detection GAN.

This script handles the complete training pipeline including data loading,
model initialization, training loop, validation, and checkpointing.

Usage:
    python train.py --dataset_name "your-dataset/name" --epochs 50
    python train.py --help
"""

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, get_default_config
from src.data.dataset import load_deepfake_dataset
from src.data.transforms import collate_fn, get_gpu_transform
from src.models.discriminator import DCTDiscriminator
from src.models.generator import UNetGenerator
from src.training.trainer import DeepfakeGANTrainer
from src.utils.metrics import MetricsTracker
from src.utils.visualization import plot_training_curves, visualize_adversarial_examples


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
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
    parser.add_argument("--no_amp", action="store_true",
                       help="Disable automatic mixed precision")
    
    # Model arguments
    parser.add_argument("--no_pretrained", action="store_true",
                       help="Don't use pretrained weights")
    parser.add_argument("--epsilon", type=float, default=0.03,
                       help="Perturbation strength")
    
    # Logging arguments
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="deepfake-gan",
                       help="W&B project name")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log every N batches")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--val_interval", type=int, default=1,
                       help="Validate every N epochs")
    
    # Path arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory for outputs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
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
    config.training.use_amp = not args.no_amp
    
    config.model.pretrained = not args.no_pretrained
    config.model.epsilon = args.epsilon
    
    config.logging.use_wandb = args.wandb
    config.logging.wandb_project = args.wandb_project
    config.logging.log_interval = args.log_interval
    config.logging.save_interval = args.save_interval
    config.logging.val_interval = args.val_interval
    
    config.paths.checkpoint_dir = args.checkpoint_dir
    config.paths.output_dir = args.output_dir
    
    config.seed = args.seed
    
    # Set random seeds
    set_seed(config.seed)
    
    # Enable cuDNN benchmark for faster training
    cudnn.benchmark = True
    
    # Setup device
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Initialize W&B if enabled
    if config.logging.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                config=config.to_dict()
            )
        except ImportError:
            print("Warning: wandb not installed. Disabling W&B logging.")
            config.logging.use_wandb = False
    
    # Load dataset
    print(f"Loading dataset: {config.dataset.dataset_name}")
    dataset = load_deepfake_dataset(
        config.dataset.dataset_name,
        cache_dir=config.dataset.cache_dir
    )
    
    # Set format to keep PIL images - transforms applied on-the-fly in collate_fn
    # This avoids caching transformed tensors to disk (which causes 90GB+ storage)
    print("Setting up dataset (transforms applied on-the-fly)...")
    dataset.set_format(columns=['image', 'label'])
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset['train'],
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=config.dataset.num_workers > 0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset['val'],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=config.dataset.num_workers > 0
    )
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples: {len(dataset['val'])}")
    
    # Initialize models
    print("Initializing models...")
    discriminator = DCTDiscriminator(pretrained=config.model.pretrained)
    generator = UNetGenerator()
    
    # Create GPU transform and compile it
    gpu_transform = get_gpu_transform().to(device)
    gpu_transform = torch.compile(gpu_transform)
    
    # Initialize trainer
    trainer = DeepfakeGANTrainer(
        discriminator=discriminator,
        generator=generator,
        device=device,
        d_lr=config.training.d_lr,
        g_lr=config.training.g_lr,
        use_amp=config.training.use_amp,
        max_grad_norm=config.training.max_grad_norm
    )
    
    # Setup schedulers
    trainer.setup_schedulers(config.training.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, metrics = trainer.load_checkpoint(args.resume)
        best_val_acc = metrics.get('val_accuracy', 0.0)
        start_epoch += 1  # Start from next epoch
    
    # Metrics tracking
    metrics_tracker = MetricsTracker()
    history: Dict[str, List[float]] = {
        'd_loss': [], 'g_loss': [],
        'd_acc_real': [], 'd_acc_fake': [],
        'val_accuracy': [], 'val_f1': []
    }
    
    # Training loop
    print(f"\nStarting training for {config.training.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.training.epochs):
        trainer.discriminator.train()
        trainer.generator.train()
        metrics_tracker.reset()
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Apply GPU transforms
            images = gpu_transform(images)
            
            # Training step
            metrics = trainer.train_step(images, labels)
            
            if metrics is not None:
                metrics_tracker.update(metrics)
                
                # Update progress bar
                avg_metrics = metrics_tracker.get_average()
                pbar.set_postfix({
                    'd_loss': f"{avg_metrics['d_loss']:.4f}",
                    'g_loss': f"{avg_metrics['g_loss']:.4f}",
                    'd_acc': f"{avg_metrics['d_acc_real']:.2%}"
                })
                
                # Log to W&B
                if config.logging.use_wandb and batch_idx % config.logging.log_interval == 0:
                    import wandb
                    wandb.log({
                        'train/' + k: v for k, v in metrics.items()
                    }, step=epoch * len(train_loader) + batch_idx)
        
        # Epoch metrics
        epoch_metrics = metrics_tracker.get_average()
        history['d_loss'].append(epoch_metrics['d_loss'])
        history['g_loss'].append(epoch_metrics['g_loss'])
        history['d_acc_real'].append(epoch_metrics['d_acc_real'])
        history['d_acc_fake'].append(epoch_metrics['d_acc_fake'])
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(metrics_tracker.pretty_print())
        
        # Validation
        if (epoch + 1) % config.logging.val_interval == 0:
            print("\nRunning validation...")
            val_metrics = trainer.validate(val_loader, gpu_transform)
            
            history['val_accuracy'].append(val_metrics['val_accuracy'])
            history['val_f1'].append(val_metrics['val_f1'])
            
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['val_accuracy']:.2%}")
            print(f"  Val F1: {val_metrics['val_f1']:.4f}")
            print(f"  Val Precision: {val_metrics['val_precision']:.4f}")
            print(f"  Val Recall: {val_metrics['val_recall']:.4f}")
            
            if config.logging.use_wandb:
                import wandb
                wandb.log({
                    'val/' + k: v for k, v in val_metrics.items()
                }, step=(epoch + 1) * len(train_loader))
            
            # Save best model
            if val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']
                best_path = os.path.join(config.paths.checkpoint_dir, "best_model.pth")
                trainer.save_checkpoint(best_path, epoch, val_metrics)
                print(f"  New best model saved! Accuracy: {best_val_acc:.2%}")
        
        # Save checkpoint
        if (epoch + 1) % config.logging.save_interval == 0:
            checkpoint_path = os.path.join(
                config.paths.checkpoint_dir, 
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            trainer.save_checkpoint(checkpoint_path, epoch, epoch_metrics)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Step learning rate schedulers
        trainer.step_schedulers()
        
        print("-" * 60)
    
    # Save final model
    final_path = os.path.join(config.paths.checkpoint_dir, "final_model.pth")
    trainer.save_checkpoint(final_path, config.training.epochs - 1, epoch_metrics)
    print(f"\nFinal model saved: {final_path}")
    
    # Plot and save training curves
    curves_path = os.path.join(config.paths.output_dir, "training_curves.png")
    plot_training_curves(history, curves_path)
    print(f"Training curves saved: {curves_path}")
    
    # Generate sample adversarial examples
    print("\nGenerating sample adversarial examples...")
    trainer.discriminator.eval()
    trainer.generator.eval()
    
    sample_images, _ = next(iter(val_loader))
    sample_images = sample_images[:4].to(device)
    sample_images = gpu_transform(sample_images)
    
    with torch.no_grad():
        adv_images, perturbation = trainer.generator(sample_images, config.model.epsilon)
    
    adv_viz_path = os.path.join(config.paths.output_dir, "adversarial_examples.png")
    visualize_adversarial_examples(sample_images, adv_images, perturbation, adv_viz_path)
    print(f"Adversarial examples saved: {adv_viz_path}")
    
    # Finish W&B run
    if config.logging.use_wandb:
        import wandb
        wandb.finish()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
