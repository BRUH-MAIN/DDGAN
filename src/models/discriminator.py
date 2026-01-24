"""Discriminator model for deepfake detection.

This module implements a DCT-based discriminator using ConvNeXt-Tiny
as the backbone for extracting features from DCT representations.
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from .dct_extractor import DCTFeatureExtractor


class DCTDiscriminator(nn.Module):
    """DCT-based discriminator for deepfake detection.
    
    Uses DCT feature extraction followed by a ConvNeXt-Tiny backbone
    to classify images as real or fake.
    
    Args:
        pretrained: Whether to use pretrained ConvNeXt weights (default: True).
    """
    
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        
        # DCT feature extractor for grayscale frequency features
        self.dct_extractor = DCTFeatureExtractor(size=224)
        
        # Load ConvNeXt-Tiny backbone
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=weights)
        
        # Modify first conv layer to accept 1 input channel instead of 3
        original_conv = convnext.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize new conv by averaging original RGB weights if pretrained
        if pretrained:
            with torch.no_grad():
                # Average across input channel dimension: [out, 3, k, k] -> [out, 1, k, k]
                new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                if original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data.clone()
        
        # Replace the first conv layer
        convnext.features[0][0] = new_conv
        
        # Store backbone components
        self.backbone = convnext.features  # Feature extractor
        self.avgpool = convnext.avgpool    # Global average pooling
        
        # Custom classifier for binary classification
        # ConvNeXt-Tiny has 768 output channels
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1)
        )
        
        # Store intermediate features for visualization
        self._features = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification.
        
        Args:
            x: Input RGB tensor of shape [B, 3, 224, 224].
            
        Returns:
            Logits tensor of shape [B, 1] (no sigmoid applied).
        """
        # Extract DCT features: [B, 3, 224, 224] -> [B, 1, 224, 224]
        dct_features = self.dct_extractor(x)
        
        # Pass through ConvNeXt backbone: [B, 1, 224, 224] -> [B, 768, 7, 7]
        features = self.backbone(dct_features)
        self._features = features  # Store for visualization
        
        # Global average pooling: [B, 768, 7, 7] -> [B, 768, 1, 1]
        pooled = self.avgpool(features)
        
        # Flatten: [B, 768, 1, 1] -> [B, 768]
        flattened = pooled.flatten(start_dim=1)
        
        # Classify: [B, 768] -> [B, 1]
        logits = self.classifier(flattened)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features for visualization.
        
        Args:
            x: Input RGB tensor of shape [B, 3, 224, 224].
            
        Returns:
            Feature maps of shape [B, 768, 7, 7].
        """
        # Run forward pass to populate features
        _ = self.forward(x)
        return self._features
    
    def get_dct_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DCT features for visualization.
        
        Args:
            x: Input RGB tensor of shape [B, 3, 224, 224].
            
        Returns:
            DCT features of shape [B, 1, 224, 224].
        """
        return self.dct_extractor(x)
