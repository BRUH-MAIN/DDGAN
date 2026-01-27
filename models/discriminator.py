"""
Discriminator for Deepfake Detection
Uses DCT features + ConvNeXt backbone for classification
"""
import torch
import torch.nn as nn
import timm
from .dct_extractor import DCT2D


class Discriminator(nn.Module):
    """
    Discriminator for deepfake detection
    
    Architecture:
    1. RGB Image → DCT Feature Extractor → DCT features [B, 1, 224, 224]
    2. ConvNeXt-Tiny backbone (pretrained on ImageNet)
    3. Global Average Pooling
    4. Classification head
    
    Key features:
    - Uses frequency-domain (DCT) features instead of raw pixels
    - Transfer learning from ImageNet (ConvNeXt pretrained weights)
    - Modified input layer to accept 1-channel DCT features
    """
    
    def __init__(
        self,
        backbone='convnext_tiny',
        pretrained=True,
        num_classes=1,
        image_size=224
    ):
        """
        Initialize Discriminator
        
        Args:
            backbone: ConvNeXt variant ('convnext_tiny', 'convnext_small', etc.)
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes (1 for binary classification)
            image_size: Input image size
        """
        super(Discriminator, self).__init__()
        
        self.image_size = image_size
        
        # DCT feature extractor
        self.dct_extractor = DCT2D(size=image_size)
        
        # Load ConvNeXt backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling (we'll add our own)
        )
        
        # Modify first convolution to accept 1-channel input
        # Original: Conv2d(3, 96, kernel_size=4, stride=4)
        # Modified: Conv2d(1, 96, kernel_size=4, stride=4)
        if pretrained:
            # Average the pretrained RGB weights across channels
            original_conv = self.backbone.stem[0]
            new_conv = nn.Conv2d(
                1, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Average RGB weights to initialize single-channel conv
            with torch.no_grad():
                new_conv.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
                if original_conv.bias is not None:
                    new_conv.bias = nn.Parameter(original_conv.bias)
            
            self.backbone.stem[0] = new_conv
        else:
            # Create new conv layer without pretrained weights
            original_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, image_size, image_size)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )
        
        print(f"✓ Discriminator initialized:")
        print(f"  - Backbone: {backbone}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: RGB image [B, 3, H, W]
            
        Returns:
            logits: Classification logits [B, 1]
        """
        # Extract DCT features
        dct_features = self.dct_extractor(x)  # [B, 1, H, W]
        
        # Pass through backbone
        features = self.backbone(dct_features)  # [B, C, H', W']
        
        # Global average pooling
        features = self.global_pool(features)  # [B, C, 1, 1]
        features = features.flatten(1)  # [B, C]
        
        # Classification
        logits = self.head(features)  # [B, 1]
        
        return logits
    
    def get_features(self, x):
        """
        Extract features without classification
        
        Args:
            x: RGB image [B, 3, H, W]
            
        Returns:
            features: Feature vector [B, feature_dim]
        """
        dct_features = self.dct_extractor(x)
        features = self.backbone(dct_features)
        features = self.global_pool(features)
        features = features.flatten(1)
        return features


if __name__ == "__main__":
    # Test Discriminator
    print("Testing Discriminator...")
    
    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Initialize discriminator
    discriminator = Discriminator(
        backbone='convnext_tiny',
        pretrained=False,  # Faster for testing
        num_classes=1
    )
    
    # Forward pass
    with torch.no_grad():
        logits = discriminator(x)
        features = discriminator.get_features(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Feature vector shape: {features.shape}")
    print("✓ Discriminator test passed!")
