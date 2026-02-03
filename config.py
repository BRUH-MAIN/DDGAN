"""
Configuration file for Deepfake Detection GAN
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration"""
    # Data source (use either dataset_root OR hf_dataset_id, not both)
    dataset_root: Path = Path('celebdfv2_images')  # Local dataset path
    hf_dataset_id: str = 'RohanRamesh/celebdfv2_224'  # HuggingFace dataset ID
    
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    image_size: int = 224
    
    # Normalization stats (ImageNet)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # DCT settings
    dct_size: int = 224
    
    # Discriminator settings
    d_backbone: str = 'convnext_tiny'
    d_pretrained: bool = True
    d_num_classes: int = 1
    d_type: str = 'dual_stream'  # 'single_stream' (legacy DCT-only) or 'dual_stream' (RGB + DCT)
    d_fusion_type: str = 'concat'  # For dual_stream: 'concat' or 'add'
    
    # Generator settings
    g_input_channels: int = 3
    g_base_channels: int = 64
    epsilon: float = 0.03  # Maximum perturbation magnitude


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    max_epochs: int = 30
    learning_rate: float = 1e-4  # Base LR (scaled by D/G multipliers)
    d_lr_mult: float = 0.2  # Discriminator LR multiplier
    g_lr_mult: float = 0.5  # Generator LR multiplier
    betas: tuple = (0.5, 0.999)
    weight_decay: float = 0.01
    
    # Loss weights (adversarial robustness training)
    consistency_weight: float = 1.0  # Weight for consistency loss in discriminator
    margin_max: float = 0.3  # Max margin for generator loss (warmup to this value)
    perturb_weight: float = 0.02  # Weight for perturbation regularization (balanced to prevent feature destruction)
    
    # Optimization
    gradient_clip_val: float = 1.0
    precision: str = '16-mixed'  # Re-enabled after fixing normalization issues
    
    # Scheduler
    scheduler_type: str = 'cosine'
    
    # Logging and checkpointing
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    save_top_k: int = 3
    checkpoint_dir: Path = Path('checkpoints')
    refresh_rate: int = 1  # Progress bar refresh rate (updates per second)
    
    # Hardware
    accelerator: str = 'gpu'
    devices: int = 2  # Number of GPUs
    strategy: str = 'ddp'  # Distributed Data Parallel


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment settings
    experiment_name: str = 'deepfake_gan'
    seed: int = 42


# Create default config
default_config = Config()
