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
    hf_dataset_id: str = None  # HuggingFace dataset ID (e.g., 'RohanRamesh/celebdfv2_224')
    
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
    
    # Generator settings
    g_input_channels: int = 3
    g_base_channels: int = 64
    epsilon: float = 0.03  # Maximum perturbation magnitude


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    max_epochs: int = 50
    learning_rate: float = 1e-4  # Reduced from 2e-4 for stability
    betas: tuple = (0.5, 0.999)
    weight_decay: float = 0.01
    
    # Loss weights
    adv_weight: float = 0.5  # Weight for adversarial loss in discriminator
    perturb_weight: float = 0.1  # Weight for perturbation regularization
    
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
    # Use ddp_find_unused_parameters_true for GAN training where D and G
    # are trained alternately, so not all params are used in each backward pass
    strategy: str = 'ddp_find_unused_parameters_true'


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
