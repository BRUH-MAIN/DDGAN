"""
Generator for Adversarial Perturbations
U-Net architecture with Frequency-Aware Bottleneck
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and GELU"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.block(x)


class FrequencyAwareBottleneck(nn.Module):
    """
    Frequency-aware bottleneck with attention mechanism
    
    Uses:
    - Large kernels (7x7) to capture frequency-like patterns
    - Depthwise separable convolutions for efficiency
    - Channel attention gate to emphasize important frequency bands
    - Residual connection for gradient flow
    """
    
    def __init__(self, channels):
        super(FrequencyAwareBottleneck, self).__init__()
        
        # Depthwise convolution (7x7 for frequency patterns)
        self.depthwise = nn.Conv2d(
            channels, channels, 
            kernel_size=7, padding=3, 
            groups=channels
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.norm1 = nn.BatchNorm2d(channels)
        self.act1 = nn.GELU()
        
        # Frequency gate (channel attention)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.GELU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        # Second convolution block
        self.depthwise2 = nn.Conv2d(
            channels, channels,
            kernel_size=7, padding=3,
            groups=channels
        )
        self.pointwise2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act2 = nn.GELU()
    
    def forward(self, x):
        identity = x
        
        # First block
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.norm1(out)
        out = self.act1(out)
        
        # Apply frequency gate
        gate = self.gate(out)
        out = out * gate
        
        # Second block
        out = self.depthwise2(out)
        out = self.pointwise2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        # Residual connection
        return out + identity


class Generator(nn.Module):
    """
    U-Net Generator with Frequency-Aware Bottleneck
    
    Generates adversarial perturbations to make discriminator more robust
    
    Architecture:
    - Encoder: 4 levels of downsampling (64→128→256→512 channels)
    - Bottleneck: Frequency-aware processing
    - Decoder: 4 levels of upsampling with skip connections
    - Output: Perturbation map bounded by tanh
    """
    
    def __init__(self, input_channels=3, base_channels=64, epsilon=0.03):
        """
        Initialize Generator
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels (doubled at each level)
            epsilon: Maximum perturbation magnitude
        """
        super(Generator, self).__init__()
        
        self.epsilon = epsilon
        
        # Encoder (downsampling path)
        self.enc1 = ConvBlock(input_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with frequency-aware processing
        bottleneck_channels = base_channels * 8
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
            FrequencyAwareBottleneck(bottleneck_channels),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU()
        )
        
        # Decoder (upsampling path with skip connections)
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 4)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels)
        
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # Output layer: generate perturbation map
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, input_channels, 1),
            nn.Tanh()  # Bound output to [-1, 1]
        )
        
        print(f"✓ Generator initialized:")
        print(f"  - Base channels: {base_channels}")
        print(f"  - Epsilon: {epsilon}")
        print(f"  - Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input RGB image [B, 3, H, W]
            
        Returns:
            adversarial_image: Perturbed image [B, 3, H, W]
            perturbation: Perturbation map [B, 3, H, W]
        """
        # Encoder
        e1 = self.enc1(x)  # [B, 64, 224, 224]
        p1 = self.pool1(e1)  # [B, 64, 112, 112]
        
        e2 = self.enc2(p1)  # [B, 128, 112, 112]
        p2 = self.pool2(e2)  # [B, 128, 56, 56]
        
        e3 = self.enc3(p2)  # [B, 256, 56, 56]
        p3 = self.pool3(e3)  # [B, 256, 28, 28]
        
        e4 = self.enc4(p3)  # [B, 512, 28, 28]
        p4 = self.pool4(e4)  # [B, 512, 14, 14]
        
        # Bottleneck with frequency-aware processing
        b = self.bottleneck(p4)  # [B, 512, 14, 14]
        
        # Decoder with skip connections
        d4 = self.up4(b)  # [B, 512, 28, 28]
        d4 = torch.cat([d4, e4], dim=1)  # [B, 1024, 28, 28]
        d4 = self.dec4(d4)  # [B, 256, 28, 28]
        
        d3 = self.up3(d4)  # [B, 256, 56, 56]
        d3 = torch.cat([d3, e3], dim=1)  # [B, 512, 56, 56]
        d3 = self.dec3(d3)  # [B, 128, 56, 56]
        
        d2 = self.up2(d3)  # [B, 128, 112, 112]
        d2 = torch.cat([d2, e2], dim=1)  # [B, 256, 112, 112]
        d2 = self.dec2(d2)  # [B, 64, 112, 112]
        
        d1 = self.up1(d2)  # [B, 64, 224, 224]
        d1 = torch.cat([d1, e1], dim=1)  # [B, 128, 224, 224]
        d1 = self.dec1(d1)  # [B, 64, 224, 224]
        
        # Generate perturbation (bounded to [-1, 1])
        perturbation = self.output(d1)  # [B, 3, 224, 224]
        
        # Scale by epsilon and add to input
        scaled_perturbation = self.epsilon * perturbation
        adversarial_image = x + scaled_perturbation
        
        # ===== DIAGNOSTIC: Check if input was normalized (ImageNet) =====
        # If input is ImageNet-normalized, range is ~[-2.1, 2.6], not [0, 1]
        # Clamping to [0, 1] would destroy the image!
        input_min, input_max = x.min().item(), x.max().item()
        if input_min < -0.5 or input_max > 1.5:
            # Input appears to be normalized - use appropriate clamp range
            # ImageNet normalized range approximately: [-2.12, 2.64]
            adversarial_image = torch.clamp(adversarial_image, -3.0, 3.0)
        else:
            # Input is in [0, 1] range - use standard clamp
            adversarial_image = torch.clamp(adversarial_image, 0, 1)
        
        return adversarial_image, perturbation


if __name__ == "__main__":
    # Test Generator
    print("Testing Generator...")
    
    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    x = torch.clamp(x, 0, 1)  # Ensure valid pixel range
    
    # Initialize generator
    generator = Generator(
        input_channels=3,
        base_channels=64,
        epsilon=0.03
    )
    
    # Forward pass
    with torch.no_grad():
        adv_images, perturbations = generator(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Adversarial image shape: {adv_images.shape}")
    print(f"Perturbation shape: {perturbations.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Adversarial range: [{adv_images.min():.3f}, {adv_images.max():.3f}]")
    print(f"Perturbation range: [{perturbations.min():.3f}, {perturbations.max():.3f}]")
    print("✓ Generator test passed!")
