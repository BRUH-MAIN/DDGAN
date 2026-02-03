"""
Main training script for Deepfake Detection GAN
"""
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from pathlib import Path

from config import default_config
from data import DeepfakeDataModule
from lightning_module import DeepfakeGAN


class CleanProgressBar(TQDMProgressBar):
    """Custom progress bar that shows one bar per epoch with live metrics"""
    
    def __init__(self, refresh_rate: int = 1):
        super().__init__(refresh_rate=refresh_rate)
    
    def init_train_tqdm(self):
        """Override to configure the main training progress bar"""
        from tqdm import tqdm
        bar = tqdm(
            desc="Training",
            position=0,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar
    
    def init_validation_tqdm(self):
        """Override to configure validation progress bar"""
        from tqdm import tqdm
        bar = tqdm(
            desc="Validating",
            position=0,
            disable=self.is_disabled,
            leave=False,  # Don't leave validation bar after completion
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Reset and configure bar at epoch start"""
        super().on_train_epoch_start(trainer, pl_module)
        # Update description with epoch number
        if self.train_progress_bar is not None:
            self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}/{trainer.max_epochs-1}")
    
    def get_metrics(self, trainer, pl_module):
        """Get metrics to display in progress bar - shortened for Kaggle display"""
        items = super().get_metrics(trainer, pl_module)
        # Remove 'v_num' as it clutters the display
        items.pop("v_num", None)
        
        # Shorten metric names for better display in Kaggle notebooks
        shortened = {}
        for key, value in items.items():
            # Remove prefixes and suffixes to shorten names
            short_key = key.replace("train/", "").replace("val/", "v_")
            short_key = short_key.replace("_step", "").replace("_epoch", "")
            short_key = short_key.replace("_loss", "L").replace("loss", "L")
            shortened[short_key] = value
        
        return shortened


class MultiCriteriaEarlyStop(Callback):
    """Stop when any stopping rule is satisfied."""

    def __init__(
        self,
        auc_patience: int = 2,
        gap_threshold: float = 0.01,
        margin_violation_threshold: float = 0.05,
        min_epochs: int = 1
    ):
        super().__init__()
        self.auc_patience = auc_patience
        self.gap_threshold = gap_threshold
        self.margin_violation_threshold = margin_violation_threshold
        self.min_epochs = min_epochs
        self._best_auc = None
        self._epochs_since_improve = 0

    def _to_float(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return float(value)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.min_epochs:
            return

        metrics = trainer.callback_metrics
        auc_adv = self._to_float(metrics.get('val/auc_adv'))
        robustness_gap = self._to_float(metrics.get('val/robustness_gap'))
        margin_violation_rate = self._to_float(metrics.get('train/margin_violation_rate'))

        stop_reasons = []

        if auc_adv is not None:
            if self._best_auc is None or auc_adv > self._best_auc:
                self._best_auc = auc_adv
                self._epochs_since_improve = 0
            else:
                self._epochs_since_improve += 1

            if self._epochs_since_improve >= self.auc_patience:
                stop_reasons.append(
                    f"val/auc_adv not improved for {self.auc_patience} epochs"
                )

        if robustness_gap is not None and robustness_gap > self.gap_threshold:
            stop_reasons.append(
                f"val/robustness_gap {robustness_gap:.4f} > {self.gap_threshold}"
            )

        if margin_violation_rate is not None and margin_violation_rate < self.margin_violation_threshold:
            stop_reasons.append(
                f"train/margin_violation_rate {margin_violation_rate:.4f} < {self.margin_violation_threshold}"
            )

        if stop_reasons:
            trainer.should_stop = True
            if trainer.is_global_zero:
                print("[EarlyStop] " + " | ".join(stop_reasons))


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
    print(f"Num workers: {default_config.data.num_workers}")
    print(f"Image size: {default_config.data.image_size}")
    print(f"Max epochs: {default_config.training.max_epochs}")
    print(f"Learning rate: {default_config.training.learning_rate}")
    print(f"D LR mult: {default_config.training.d_lr_mult}")
    print(f"G LR mult: {default_config.training.g_lr_mult}")
    print(f"Weight decay: {default_config.training.weight_decay}")
    print(f"Consistency weight: {default_config.training.consistency_weight}")
    print(f"Margin max: {default_config.training.margin_max}")
    print(f"Perturb weight: {default_config.training.perturb_weight}")
    print(f"Epsilon: {default_config.model.epsilon}")
    print(f"Discriminator type: {default_config.model.d_type}")
    print(f"Fusion type: {default_config.model.d_fusion_type}")
    print(f"Devices: {default_config.training.devices} {default_config.training.accelerator}")
    print(f"Precision: {default_config.training.precision}")
    print(f"Seed: {default_config.seed}")
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
        d_type=default_config.model.d_type,
        d_fusion_type=default_config.model.d_fusion_type,
        g_base_channels=default_config.model.g_base_channels,
        epsilon=default_config.model.epsilon,
        lr=default_config.training.learning_rate,
        d_lr_mult=default_config.training.d_lr_mult,
        g_lr_mult=default_config.training.g_lr_mult,
        betas=default_config.training.betas,
        weight_decay=default_config.training.weight_decay,
        consistency_weight=default_config.training.consistency_weight,
        margin_max=default_config.training.margin_max,
        perturb_weight=default_config.training.perturb_weight,
        scheduler_type=default_config.training.scheduler_type,
        max_epochs=default_config.training.max_epochs
    )
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    
    # Progress bar callback - single bar per epoch with live metric updates
    progress_bar = CleanProgressBar(refresh_rate=default_config.training.refresh_rate)
    
    # Model checkpoint callback (monitor adversarial AUC for robustness)
    checkpoint_callback = ModelCheckpoint(
        dirpath=default_config.training.checkpoint_dir,
        filename='{epoch:02d}-{val/auc_adv:.4f}',
        monitor='val/auc_adv',
        mode='max',
        save_top_k=default_config.training.save_top_k,
        save_last=True,
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Multi-criteria early stopping (formalized rules)
    multi_criteria_stop = MultiCriteriaEarlyStop(
        auc_patience=2,
        gap_threshold=0.01,
        margin_violation_threshold=0.05,
        min_epochs=1
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=default_config.experiment_name
    )
    
    # Initialize trainer
    print("\nInitializing Trainer...")
    # Note: gradient_clip_val removed - incompatible with manual optimization
    # Gradient clipping is handled in lightning_module.py via self.clip_gradients()
    trainer = pl.Trainer(
        max_epochs=default_config.training.max_epochs,
        accelerator=default_config.training.accelerator,
        devices=default_config.training.devices,
        strategy=default_config.training.strategy if default_config.training.devices > 1 else 'auto',
        precision=default_config.training.precision,
        callbacks=[progress_bar, checkpoint_callback, lr_monitor, multi_criteria_stop],
        logger=logger,
        log_every_n_steps=default_config.training.log_every_n_steps,
        val_check_interval=default_config.training.val_check_interval,
        deterministic=False,  # Set to False to allow benchmark optimization
        benchmark=True,  # Enable cudnn benchmarking for faster training
        enable_progress_bar=default_config.training.refresh_rate > 0,
        # Progress bar refresh rate controlled by RichProgressBar callback
    )
    
    # Print trainer info
    print(f"\nTrainer configuration:")
    print(f"  Max epochs: {trainer.max_epochs}")
    print(f"  Precision: {trainer.precision}")
    print(f"  Accelerator: {trainer.accelerator}")
    print(f"  Devices: {trainer.num_devices}")
    print(f"  Strategy: {trainer.strategy.__class__.__name__}")
    print(f"  Gradient clipping: 1.0 (manual, in lightning_module)")
    
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
    print(f"Best val/auc_adv: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 60 + "\n")
    
    # Save final model
    checkpoint_dir = Path(default_config.training.checkpoint_dir)
    final_model_path = checkpoint_dir / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Deepfake Detection GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ==================== Data Configuration ====================
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-dir', type=str, default='celebdfv2_images',
                            help='Path to local dataset directory')
    data_group.add_argument('--hf-dataset', type=str, default=None,
                            help='HuggingFace dataset ID (e.g., RohanRamesh/celebdfv2_224)')
    data_group.add_argument('--batch-size', type=int, default=32,
                            help='Batch size for training')
    data_group.add_argument('--num-workers', type=int, default=4,
                            help='Number of data loader workers')
    data_group.add_argument('--no-pin-memory', action='store_true',
                            help='Disable pinned memory for data loading')
    data_group.add_argument('--no-persistent-workers', action='store_true',
                            help='Disable persistent workers for data loading')
    data_group.add_argument('--image-size', type=int, default=224,
                            help='Input image size')
    
    # ==================== Model Configuration ====================
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--d-backbone', type=str, default='convnext_tiny',
                             choices=['convnext_tiny', 'convnext_small', 'convnext_base'],
                             help='Discriminator backbone architecture')
    model_group.add_argument('--no-pretrained', action='store_true',
                             help='Do not use pretrained weights for discriminator')
    model_group.add_argument('--g-base-channels', type=int, default=64,
                             help='Generator base channels')
    model_group.add_argument('--epsilon', type=float, default=0.03,
                             help='Maximum perturbation magnitude for generator')
    
    # ==================== Training Configuration ====================
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_group.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    train_group.add_argument('--beta1', type=float, default=0.5,
                             help='Adam beta1 parameter')
    train_group.add_argument('--beta2', type=float, default=0.999,
                             help='Adam beta2 parameter')
    train_group.add_argument('--weight-decay', type=float, default=0.01,
                             help='Weight decay for AdamW optimizer')
    
    # ==================== Loss Configuration ====================
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument('--consistency-weight', type=float, default=1.0,
                            help='Weight for consistency loss in discriminator')
    loss_group.add_argument('--margin-max', type=float, default=0.3,
                            help='Max margin for generator loss (warmup to this value)')
    loss_group.add_argument('--perturb-weight', type=float, default=0.005,
                            help='Weight for perturbation regularization')
    
    # ==================== Hardware Configuration ====================
    hw_group = parser.add_argument_group('Hardware Configuration')
    hw_group.add_argument('--devices', type=int, default=2,
                          help='Number of GPUs to use')
    hw_group.add_argument('--accelerator', type=str, default='gpu',
                          choices=['gpu', 'cpu', 'auto'],
                          help='Accelerator type')
    hw_group.add_argument('--strategy', type=str, default='ddp',
                          choices=['ddp', 'ddp_spawn', 'auto'],
                          help='Distributed training strategy')
    hw_group.add_argument('--precision', type=str, default='16-mixed',
                          choices=['32', '16-mixed', 'bf16-mixed'],
                          help='Training precision')
    
    # ==================== Scheduler Configuration ====================
    sched_group = parser.add_argument_group('Scheduler Configuration')
    sched_group.add_argument('--scheduler', type=str, default='cosine',
                             choices=['cosine', 'step', 'none'],
                             help='Learning rate scheduler type')
    
    # ==================== Logging & Checkpointing ====================
    log_group = parser.add_argument_group('Logging & Checkpointing')
    log_group.add_argument('--log-every-n-steps', type=int, default=50,
                           help='Log metrics every N training steps')
    log_group.add_argument('--val-check-interval', type=float, default=1.0,
                           help='Validation check interval (1.0 = every epoch)')
    log_group.add_argument('--save-top-k', type=int, default=3,
                           help='Number of best checkpoints to save')
    log_group.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                           help='Directory to save checkpoints')
    log_group.add_argument('--refresh-rate', type=int, default=1,
                           help='Progress bar refresh rate (updates per second). Set to 0 to disable.')
    log_group.add_argument('--experiment-name', type=str, default='deepfake_gan',
                           help='Experiment name for logging')
    
    # ==================== Misc Configuration ====================
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
    misc_group.add_argument('--gradient-clip-val', type=float, default=1.0,
                            help='Gradient clipping value')
    
    args = parser.parse_args()
    
    # ==================== Update config with command line arguments ====================
    
    # Data config
    if args.hf_dataset:
        default_config.data.hf_dataset_id = args.hf_dataset
        default_config.data.dataset_root = None
    else:
        default_config.data.dataset_root = args.data_dir
        default_config.data.hf_dataset_id = None
    default_config.data.batch_size = args.batch_size
    default_config.data.num_workers = args.num_workers
    default_config.data.pin_memory = not args.no_pin_memory
    default_config.data.persistent_workers = not args.no_persistent_workers
    default_config.data.image_size = args.image_size
    
    # Model config
    default_config.model.d_backbone = args.d_backbone
    default_config.model.d_pretrained = not args.no_pretrained
    default_config.model.g_base_channels = args.g_base_channels
    default_config.model.epsilon = args.epsilon
    
    # Training config
    default_config.training.max_epochs = args.epochs
    default_config.training.learning_rate = args.lr
    default_config.training.betas = (args.beta1, args.beta2)
    default_config.training.weight_decay = args.weight_decay
    default_config.training.consistency_weight = args.consistency_weight
    default_config.training.margin_max = args.margin_max
    default_config.training.perturb_weight = args.perturb_weight
    default_config.training.gradient_clip_val = args.gradient_clip_val
    default_config.training.precision = args.precision
    default_config.training.scheduler_type = args.scheduler
    default_config.training.log_every_n_steps = args.log_every_n_steps
    default_config.training.val_check_interval = args.val_check_interval
    default_config.training.save_top_k = args.save_top_k
    default_config.training.checkpoint_dir = args.checkpoint_dir
    default_config.training.refresh_rate = args.refresh_rate
    default_config.training.accelerator = args.accelerator
    default_config.training.devices = args.devices
    default_config.training.strategy = args.strategy
    
    # Experiment config
    default_config.experiment_name = args.experiment_name
    default_config.seed = args.seed
    
    # Run training
    main(args)
