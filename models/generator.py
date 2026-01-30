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
    Frequency-aware bottleneck with EXPLICIT DCT/IDCT transforms
    
    Architecture:
    1. DCT Transform: Convert spatial features to frequency domain
    2. Frequency Processing: Learnable manipulation of frequency coefficients
    3. Channel Attention: Emphasize important frequency bands
    4. IDCT Transform: Convert back to spatial domain
    5. Residual Connection: Gradient flow preservation
    
    This allows the generator to directly manipulate frequency components
    when crafting adversarial perturbations.
    """
    
    def __init__(self, channels, spatial_size=14):
        """
        Args:
            channels: Number of input/output channels
            spatial_size: Spatial dimension at bottleneck (default 14 for 224 input with 4 poolings)
        """
        super(FrequencyAwareBottleneck, self).__init__()
        
        self.channels = channels
        self.spatial_size = spatial_size
        
        # Precompute DCT basis matrix (non-trainable)
        dct_matrix = self._create_dct_matrix(spatial_size)
        self.register_buffer('dct_matrix', dct_matrix)
        
        # Frequency domain processing (learnable)
        # Process each channel's frequency coefficients
        self.freq_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.freq_norm1 = nn.BatchNorm2d(channels)
        self.freq_act1 = nn.GELU()
        
        self.freq_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.freq_norm2 = nn.BatchNorm2d(channels)
        self.freq_act2 = nn.GELU()
        
        # Frequency band attention (channel-wise gating in frequency domain)
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.GELU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        # Learnable frequency mask (emphasize/suppress specific frequency bands)
        # Initialized to ones (no initial bias)
        self.freq_mask = nn.Parameter(torch.ones(1, channels, spatial_size, spatial_size))
        
        # Post-IDCT refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
    
    def _create_dct_matrix(self, size):
        """
        Create DCT-II transformation matrix
        
        Args:
            size: Matrix dimension
            
        Returns:
            DCT transformation matrix [size, size]
        """
        import numpy as np
        matrix = np.zeros((size, size))
        
        for k in range(size):
            for n in range(size):
                if k == 0:
                    # DC component normalization
                    matrix[k, n] = np.sqrt(1 / size)
                else:
                    # AC components normalization
                    matrix[k, n] = np.sqrt(2 / size) * np.cos(
                        np.pi * k * (2 * n + 1) / (2 * size)
                    )
        
        return torch.FloatTensor(matrix)
    
    def dct_2d(self, x):
        """
        Apply 2D DCT to each channel independently
        
        Args:
            x: [B, C, H, W] spatial domain features
            
        Returns:
            [B, C, H, W] frequency domain coefficients
        """
        B, C, H, W = x.shape
        
        # Reshape for batch matrix multiplication: [B*C, H, W]
        x_flat = x.view(B * C, H, W)
        
        # DCT along rows (right multiply with transpose)
        dct_rows = torch.matmul(x_flat, self.dct_matrix.t())
        
        # DCT along columns (left multiply)
        dct_2d = torch.matmul(self.dct_matrix, dct_rows)
        
        # Reshape back: [B, C, H, W]
        return dct_2d.view(B, C, H, W)
    
    def idct_2d(self, x):
        """
        Apply 2D Inverse DCT (IDCT) to each channel independently
        
        For orthonormal DCT-II, IDCT = DCT^T (transpose)
        
        Args:
            x: [B, C, H, W] frequency domain coefficients
            
        Returns:
            [B, C, H, W] spatial domain features
        """
        B, C, H, W = x.shape
        
        # Reshape for batch matrix multiplication: [B*C, H, W]
        x_flat = x.view(B * C, H, W)
        
        # IDCT along columns (left multiply with transpose)
        idct_cols = torch.matmul(self.dct_matrix.t(), x_flat)
        
        # IDCT along rows (right multiply)
        idct_2d = torch.matmul(idct_cols, self.dct_matrix)
        
        # Reshape back: [B, C, H, W]
        return idct_2d.view(B, C, H, W)
    
    def forward(self, x):
        """
        Forward pass: Spatial → DCT → Process → IDCT → Spatial
        
        Args:
            x: [B, C, H, W] input features
            
        Returns:
            [B, C, H, W] processed features with frequency-aware perturbations
        """
        identity = x
        
        # ===== 1. Transform to frequency domain =====
        freq = self.dct_2d(x)  # [B, C, H, W] frequency coefficients
        
        # ===== 2. Process in frequency domain =====
        # Learnable frequency manipulation
        freq = self.freq_conv1(freq)
        freq = self.freq_norm1(freq)
        freq = self.freq_act1(freq)
        
        freq = self.freq_conv2(freq)
        freq = self.freq_norm2(freq)
        freq = self.freq_act2(freq)
        
        # ===== 3. Apply frequency-aware attention =====
        # Channel attention in frequency domain
        gate = self.freq_gate(freq)
        freq = freq * gate
        
        # Apply learnable frequency mask (band selection)
        freq = freq * self.freq_mask
        
        # ===== 4. Transform back to spatial domain =====
        out = self.idct_2d(freq)  # [B, C, H, W] spatial features
        
        # ===== 5. Post-IDCT refinement =====
        out = self.refine(out)
        
        # ===== 6. Residual connection =====
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
        
        # Bottleneck with frequency-aware processing (explicit DCT/IDCT)
        bottleneck_channels = base_channels * 8
        # Spatial size at bottleneck: 224 / (2^4) = 14
        bottleneck_spatial_size = 14
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
            FrequencyAwareBottleneck(bottleneck_channels, spatial_size=bottleneck_spatial_size),
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
