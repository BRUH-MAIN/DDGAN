"""
Discriminator for Deepfake Detection
Supports single-stream (DCT-only) and dual-stream (RGB + DCT fusion) architectures
"""
import torch
import torch.nn as nn
import timm
from .dct_extractor import DCT2D


class Discriminator(nn.Module):
    """
    Single-stream Discriminator for deepfake detection (Legacy)
    
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
        
        # DCT feature extractor (legacy grayscale mode)
        self.dct_extractor = DCT2D(size=image_size, per_channel=False)
        
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


class DualStreamDiscriminator(nn.Module):
    """
    Dual-Stream Discriminator for deepfake detection
    
    Architecture:
    1. Two parallel stems:
       - RGB stream: Original ConvNeXt stem (3→96 channels) with pretrained weights
       - DCT stream: Per-channel DCT with learnable filters (3→96 channels)
    2. Early fusion: Concatenate stem outputs (192 channels) → 1x1 conv → 96 channels
    3. Shared ConvNeXt stages 1-3
    4. Global Average Pooling
    5. Classification head
    
    Key advantages over single-stream:
    - Preserves RGB spatial information (color inconsistencies, blending seams)
    - Per-channel DCT captures frequency signatures across R, G, B independently
    - Learnable frequency filters adapt to dataset-specific artifacts
    - Retains full pretrained ImageNet weights for RGB path
    """
    
    def __init__(
        self,
        backbone='convnext_tiny',
        pretrained=True,
        num_classes=1,
        image_size=224,
        fusion_type='concat'  # 'concat' or 'add'
    ):
        """
        Initialize DualStreamDiscriminator
        
        Args:
            backbone: ConvNeXt variant ('convnext_tiny', 'convnext_small', etc.)
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes (1 for binary classification)
            image_size: Input image size
            fusion_type: How to fuse RGB and DCT features ('concat' or 'add')
        """
        super(DualStreamDiscriminator, self).__init__()
        
        self.image_size = image_size
        self.fusion_type = fusion_type
        
        # ========== DCT FEATURE EXTRACTOR ==========
        # Per-channel DCT with learnable frequency filters
        self.dct_extractor = DCT2D(
            size=image_size, 
            per_channel=True, 
            learnable_filters=True
        )
        
        # ========== LOAD BASE CONVNEXT ==========
        base_model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # ========== RGB STEM (Pretrained) ==========
        # Keep original stem: Conv2d(3, 96, kernel_size=4, stride=4) + LayerNorm
        self.rgb_stem = base_model.stem
        
        # ========== DCT STEM (Fresh) ==========
        # Create new stem for 3-channel DCT input
        # Initialize from scratch since DCT statistics differ from RGB
        original_conv = base_model.stem[0]
        self.dct_stem = nn.Sequential(
            nn.Conv2d(
                3,  # 3-channel DCT input
                original_conv.out_channels,  # 96
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            ),
            nn.LayerNorm(
                [original_conv.out_channels, image_size // 4, image_size // 4],
                eps=1e-6
            )
        )
        
        # Initialize DCT stem with small random weights
        nn.init.kaiming_normal_(self.dct_stem[0].weight, mode='fan_out', nonlinearity='relu')
        if self.dct_stem[0].bias is not None:
            nn.init.zeros_(self.dct_stem[0].bias)
        
        # ========== FUSION LAYER ==========
        stem_channels = original_conv.out_channels  # 96
        if fusion_type == 'concat':
            # Concatenate and project back to 96 channels
            self.fusion = nn.Sequential(
                nn.Conv2d(stem_channels * 2, stem_channels, kernel_size=1, bias=False),
                nn.LayerNorm([stem_channels, image_size // 4, image_size // 4], eps=1e-6),
                nn.GELU()
            )
        else:  # 'add'
            # Simple addition (no learnable params)
            self.fusion = None
        
        # ========== SHARED CONVNEXT STAGES ==========
        # Stages 0-3 (after stem)
        self.stages = base_model.stages
        
        # ========== CLASSIFICATION HEAD ==========
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            # Simulate forward through stems and stages
            rgb_feat = self.rgb_stem(dummy_input)
            dct_dummy = torch.randn(1, 3, image_size, image_size)
            dct_feat = self.dct_stem(dct_dummy)
            if fusion_type == 'concat':
                fused = self.fusion(torch.cat([rgb_feat, dct_feat], dim=1))
            else:
                fused = rgb_feat + dct_feat
            for stage in self.stages:
                fused = stage(fused)
            feature_dim = fused.shape[1]
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"✓ DualStreamDiscriminator initialized:")
        print(f"  - Backbone: {backbone}")
        print(f"  - Pretrained RGB stem: {pretrained}")
        print(f"  - Fusion type: {fusion_type}")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: RGB image [B, 3, H, W]
            
        Returns:
            logits: Classification logits [B, 1]
        """
        # ========== PARALLEL STEM PROCESSING ==========
        # RGB path (keeps spatial/color information)
        rgb_features = self.rgb_stem(x)  # [B, 96, H/4, W/4]
        
        # DCT path (frequency information per channel)
        dct_coeffs = self.dct_extractor(x)  # [B, 3, H, W]
        dct_features = self.dct_stem(dct_coeffs)  # [B, 96, H/4, W/4]
        
        # ========== FUSION ==========
        if self.fusion_type == 'concat':
            fused = torch.cat([rgb_features, dct_features], dim=1)  # [B, 192, H/4, W/4]
            fused = self.fusion(fused)  # [B, 96, H/4, W/4]
        else:
            fused = rgb_features + dct_features  # [B, 96, H/4, W/4]
        
        # ========== SHARED STAGES ==========
        features = fused
        for stage in self.stages:
            features = stage(features)
        
        # ========== CLASSIFICATION ==========
        features = self.global_pool(features)  # [B, C, 1, 1]
        features = features.flatten(1)  # [B, C]
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
        # RGB path
        rgb_features = self.rgb_stem(x)
        
        # DCT path
        dct_coeffs = self.dct_extractor(x)
        dct_features = self.dct_stem(dct_coeffs)
        
        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([rgb_features, dct_features], dim=1)
            fused = self.fusion(fused)
        else:
            fused = rgb_features + dct_features
        
        # Stages
        features = fused
        for stage in self.stages:
            features = stage(features)
        
        features = self.global_pool(features)
        features = features.flatten(1)
        return features
    
    def get_stream_features(self, x):
        """
        Get separate RGB and DCT stream features (for analysis/debugging)
        
        Args:
            x: RGB image [B, 3, H, W]
            
        Returns:
            dict with 'rgb_features', 'dct_features', 'dct_coeffs'
        """
        rgb_features = self.rgb_stem(x)
        dct_coeffs = self.dct_extractor(x)
        dct_features = self.dct_stem(dct_coeffs)
        
        return {
            'rgb_features': rgb_features,
            'dct_features': dct_features,
            'dct_coeffs': dct_coeffs
        }


def create_discriminator(
    discriminator_type='dual_stream',
    backbone='convnext_tiny',
    pretrained=True,
    num_classes=1,
    image_size=224,
    **kwargs
):
    """
    Factory function to create discriminator
    
    Args:
        discriminator_type: 'single_stream' (legacy DCT-only) or 'dual_stream' (RGB + DCT)
        backbone: ConvNeXt variant
        pretrained: Use pretrained weights
        num_classes: Number of output classes
        image_size: Input image size
        **kwargs: Additional arguments for specific discriminator types
        
    Returns:
        Discriminator instance
    """
    if discriminator_type == 'single_stream':
        return Discriminator(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size
        )
    elif discriminator_type == 'dual_stream':
        return DualStreamDiscriminator(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size,
            fusion_type=kwargs.get('fusion_type', 'concat')
        )
    else:
        raise ValueError(f"Unknown discriminator_type: {discriminator_type}. "
                        f"Choose 'single_stream' or 'dual_stream'.")


if __name__ == "__main__":
    # Test Discriminators
    print("=" * 60)
    print("Testing Discriminators...")
    print("=" * 60)
    
    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # ===== Test Single-Stream (Legacy) Discriminator =====
    print("\n--- Single-Stream Discriminator (Legacy) ---")
    discriminator_single = Discriminator(
        backbone='convnext_tiny',
        pretrained=False,  # Faster for testing
        num_classes=1
    )
    
    with torch.no_grad():
        logits_single = discriminator_single(x)
        features_single = discriminator_single.get_features(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits_single.shape}")
    print(f"Feature vector shape: {features_single.shape}")
    
    # ===== Test Dual-Stream Discriminator =====
    print("\n--- Dual-Stream Discriminator (RGB + DCT) ---")
    discriminator_dual = DualStreamDiscriminator(
        backbone='convnext_tiny',
        pretrained=False,
        num_classes=1,
        fusion_type='concat'
    )
    
    with torch.no_grad():
        logits_dual = discriminator_dual(x)
        features_dual = discriminator_dual.get_features(x)
        stream_features = discriminator_dual.get_stream_features(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits_dual.shape}")
    print(f"Feature vector shape: {features_dual.shape}")
    print(f"RGB stem output shape: {stream_features['rgb_features'].shape}")
    print(f"DCT stem output shape: {stream_features['dct_features'].shape}")
    print(f"DCT coefficients shape: {stream_features['dct_coeffs'].shape}")
    
    # ===== Test Factory Function =====
    print("\n--- Factory Function Test ---")
    disc_factory_single = create_discriminator(
        discriminator_type='single_stream',
        pretrained=False
    )
    disc_factory_dual = create_discriminator(
        discriminator_type='dual_stream',
        pretrained=False,
        fusion_type='concat'
    )
    print(f"Single-stream type: {type(disc_factory_single).__name__}")
    print(f"Dual-stream type: {type(disc_factory_dual).__name__}")
    
    print("\n" + "=" * 60)
    print("✓ All discriminator tests passed!")
    print("=" * 60)
