"""Training package for deepfake detection GAN."""

from .trainer import DeepfakeGANModule
from .losses import PerceptualLoss, AdversarialLoss, CombinedGeneratorLoss
from .ff_trainer import FFDeepfakeGANModule
from .imbalance_losses import (
    FocalLoss,
    WeightedBCELoss,
    AAMLoss,
    AsymmetricFocalLoss,
    CombinedImbalanceLoss,
    compute_class_weights,
)

__all__ = [
    'DeepfakeGANModule',
    'FFDeepfakeGANModule',
    'PerceptualLoss',
    'AdversarialLoss',
    'CombinedGeneratorLoss',
    'FocalLoss',
    'WeightedBCELoss',
    'AAMLoss',
    'AsymmetricFocalLoss',
    'CombinedImbalanceLoss',
    'compute_class_weights',
]
