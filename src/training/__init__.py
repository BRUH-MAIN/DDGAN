"""Training package for deepfake detection GAN."""

from .trainer import DeepfakeGANTrainer
from .losses import PerceptualLoss, AdversarialLoss, CombinedGeneratorLoss

__all__ = [
    'DeepfakeGANTrainer',
    'PerceptualLoss',
    'AdversarialLoss',
    'CombinedGeneratorLoss',
]
