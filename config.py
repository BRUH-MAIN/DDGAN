"""Configuration settings for deepfake detection GAN.

This module provides configuration dataclasses for all training,
model, and dataset settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DatasetConfig:
    """Dataset configuration settings."""
    dataset_name: str = "your-dataset/name"  # HuggingFace dataset name
    cache_dir: Optional[str] = None
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration settings."""
    pretrained: bool = True  # Use pretrained ConvNeXt weights
    epsilon: float = 0.03    # Perturbation strength for generator


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    batch_size: int = 32
    epochs: int = 50
    d_lr: float = 2e-4       # Discriminator learning rate
    g_lr: float = 2e-4       # Generator learning rate
    precision: str = "16-mixed"  # Lightning precision: "32", "16-mixed", "bf16-mixed"
    max_grad_norm: float = 1.0   # Gradient clipping
    accumulate_grad_batches: int = 1  # Gradient accumulation


@dataclass
class PathConfig:
    """Path configuration settings."""
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0  # Validate every epoch (1.0 = every epoch)


@dataclass
class Config:
    """Main configuration container."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Device configuration
    accelerator: str = field(default_factory=lambda: "gpu" if torch.cuda.is_available() else "cpu")
    devices: int = 1
    seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values.
            
        Returns:
            Config instance.
        """
        dataset = DatasetConfig(**config_dict.get('dataset', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        paths = PathConfig(**config_dict.get('paths', {}))
        logging = LoggingConfig(**config_dict.get('logging', {}))
        
        return cls(
            dataset=dataset,
            model=model,
            training=training,
            paths=paths,
            logging=logging,
            accelerator=config_dict.get('accelerator', "gpu" if torch.cuda.is_available() else "cpu"),
            devices=config_dict.get('devices', 1),
            seed=config_dict.get('seed', 42)
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config.
        """
        return {
            'dataset': {
                'dataset_name': self.dataset.dataset_name,
                'cache_dir': self.dataset.cache_dir,
                'num_workers': self.dataset.num_workers,
            },
            'model': {
                'pretrained': self.model.pretrained,
                'epsilon': self.model.epsilon,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'd_lr': self.training.d_lr,
                'g_lr': self.training.g_lr,
                'precision': self.training.precision,
                'max_grad_norm': self.training.max_grad_norm,
                'accumulate_grad_batches': self.training.accumulate_grad_batches,
            },
            'paths': {
                'checkpoint_dir': self.paths.checkpoint_dir,
                'log_dir': self.paths.log_dir,
                'output_dir': self.paths.output_dir,
            },
            'logging': {
                'log_every_n_steps': self.logging.log_every_n_steps,
                'val_check_interval': self.logging.val_check_interval,
            },
            'accelerator': self.accelerator,
            'devices': self.devices,
            'seed': self.seed,
        }


def get_default_config() -> Config:
    """Get default configuration.
    
    Returns:
        Default Config instance.
    """
    return Config()
