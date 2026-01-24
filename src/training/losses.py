"""Loss functions for deepfake detection GAN.

This module provides custom loss functions including perceptual loss
and adversarial loss wrappers.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features.
    
    Computes L1 distance between feature maps extracted from
    VGG16 at specified layers.
    
    Args:
        layers: List of layer indices to extract features from.
               Default uses relu1_2, relu2_2, relu3_3.
    """
    
    def __init__(self, layers: List[int] = None) -> None:
        super().__init__()
        
        # Default VGG16 layers: relu1_2 (idx 4), relu2_2 (idx 9), relu3_3 (idx 16)
        self.layers = layers or [4, 9, 16]
        
        # Load pretrained VGG16
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # Extract feature layers up to the last layer we need
        max_layer = max(self.layers) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])
        
        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG16."""
        return (x - self.mean) / self.std
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features at specified layers."""
        features = []
        x = self._normalize(x)
        
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layers:
                features.append(x)
        
        return features
    
    def forward(
        self, 
        input_img: torch.Tensor, 
        target_img: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss between input and target.
        
        Args:
            input_img: Input image tensor [B, 3, H, W].
            target_img: Target image tensor [B, 3, H, W].
            
        Returns:
            Perceptual loss value (L1 distance of features).
        """
        input_features = self._extract_features(input_img)
        target_features = self._extract_features(target_img)
        
        loss = 0.0
        for inp_feat, tgt_feat in zip(input_features, target_features):
            loss += F.l1_loss(inp_feat, tgt_feat)
        
        return loss / len(self.layers)


class AdversarialLoss(nn.Module):
    """Wrapper for adversarial training loss.
    
    Provides convenient methods for computing discriminator and
    generator losses in GAN training.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def discriminator_loss(
        self, 
        real_pred: torch.Tensor, 
        fake_pred: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss.
        
        Args:
            real_pred: Predictions for real images [B, 1].
            fake_pred: Predictions for fake images [B, 1].
            
        Returns:
            Dictionary with total loss and individual components.
        """
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        
        real_loss = self.criterion(real_pred, real_labels)
        fake_loss = self.criterion(fake_pred, fake_labels)
        total_loss = real_loss + fake_loss
        
        return {
            'total': total_loss,
            'real': real_loss,
            'fake': fake_loss
        }
    
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute generator loss (tries to fool discriminator).
        
        Args:
            fake_pred: Discriminator predictions on generated images [B, 1].
            
        Returns:
            Generator adversarial loss.
        """
        # Generator wants discriminator to predict fake as real
        real_labels = torch.ones_like(fake_pred)
        return self.criterion(fake_pred, real_labels)


class CombinedGeneratorLoss(nn.Module):
    """Combined loss for generator with adversarial and perceptual components.
    
    Args:
        perceptual_weight: Weight for perceptual loss (default: 0.1).
        l1_weight: Weight for L1 perturbation regularization (default: 0.1).
    """
    
    def __init__(
        self, 
        perceptual_weight: float = 0.1, 
        l1_weight: float = 0.1
    ) -> None:
        super().__init__()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss()
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
    
    def forward(
        self,
        fake_pred: torch.Tensor,
        original_img: torch.Tensor,
        adversarial_img: torch.Tensor,
        perturbation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined generator loss.
        
        Args:
            fake_pred: Discriminator predictions on adversarial images.
            original_img: Original input images.
            adversarial_img: Generated adversarial images.
            perturbation: Generated perturbation.
            
        Returns:
            Dictionary with total loss and components.
        """
        # Adversarial loss (fool discriminator)
        adv_loss = self.adversarial_loss.generator_loss(fake_pred)
        
        # Perceptual loss (maintain visual similarity)
        perc_loss = self.perceptual_loss(adversarial_img, original_img)
        
        # L1 regularization on perturbation
        l1_loss = torch.mean(torch.abs(perturbation))
        
        # Combined loss
        total_loss = (
            adv_loss + 
            self.perceptual_weight * perc_loss + 
            self.l1_weight * l1_loss
        )
        
        return {
            'total': total_loss,
            'adversarial': adv_loss,
            'perceptual': perc_loss,
            'l1': l1_loss
        }
