"""Training package for deepfake detection GAN."""

from .trainer import DeepfakeGANModule
from .losses import PerceptualLoss, AdversarialLoss, CombinedGeneratorLoss

__all__ = [
    'DeepfakeGANModule',
    'PerceptualLoss',
    'AdversarialLoss',
    'CombinedGeneratorLoss',
]
