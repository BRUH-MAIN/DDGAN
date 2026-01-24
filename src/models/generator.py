"""Generator model for adversarial perturbation.

This module implements a U-Net based generator with frequency-aware
bottleneck for creating adversarial perturbations to improve
discriminator robustness.
"""

import torch
import torch.nn as nn
from typing import Tuple


class FrequencyAwareBottleneck(nn.Module):
    """Frequency-aware bottleneck module with channel attention.
    
    Uses depthwise separable convolutions and a frequency gate
    mechanism for channel-wise attention in the bottleneck.
    
    Args:
        channels: Number of input/output channels (default: 512).
    """
    
    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        
        # First frequency convolution block (depthwise + pointwise)
        self.freq_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),  # Depthwise
            nn.Conv2d(channels, channels, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # Second frequency convolution block
        self.freq_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),  # Depthwise
            nn.Conv2d(channels, channels, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # Frequency gate for channel attention
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context: [B, C, H, W] -> [B, C, 1, 1]
            nn.Conv2d(channels, channels // 16, kernel_size=1),  # Reduce channels
            nn.GELU(),
            nn.Conv2d(channels // 16, channels, kernel_size=1),  # Expand channels
            nn.Sigmoid()  # Gate values in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frequency-aware processing.
        
        Args:
            x: Input tensor of shape [B, C, H, W].
            
        Returns:
            Output tensor of shape [B, C, H, W] with residual connection.
        """
        residual = x
        
        # First convolution block
        out = self.freq_conv1(x)
        
        # Apply frequency gate (channel attention)
        gate = self.freq_gate(out)
        out = out * gate
        
        # Second convolution block
        out = self.freq_conv2(out)
        
        # Residual connection
        out = out + residual
        
        return out


class UNetGenerator(nn.Module):
    """U-Net based generator for adversarial perturbation.
    
    Generates bounded adversarial perturbations using a U-Net architecture
    with skip connections and a frequency-aware bottleneck.
    
    Args:
        input_channels: Number of input channels (default: 3 for RGB).
        output_channels: Number of output channels (default: 3 for RGB).
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3) -> None:
        super().__init__()
        
        # Encoder blocks
        self.enc1 = self._encoder_block(input_channels, 64)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        
        # Bottleneck with frequency-aware processing
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            FrequencyAwareBottleneck(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        
        # Decoder blocks (input channels doubled due to skip connections)
        self.dec4 = self._decoder_block(1024, 256)  # 512 + 512 from skip
        self.dec3 = self._decoder_block(512, 128)   # 256 + 256 from skip
        self.dec2 = self._decoder_block(256, 64)    # 128 + 128 from skip
        self.dec1 = self._decoder_block(128, 64)    # 64 + 64 from skip
        
        # Output layer: generates perturbation in [-1, 1]
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Pooling and upsampling layers
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _encoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create an encoder block.
        
        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            
        Returns:
            Sequential encoder block.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    
    def _decoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create a decoder block.
        
        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            
        Returns:
            Sequential decoder block.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        epsilon: float = 0.03
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate adversarial perturbation and adversarial image.
        
        Args:
            x: Input RGB tensor of shape [B, 3, 224, 224].
            epsilon: Maximum perturbation strength (default: 0.03).
            
        Returns:
            Tuple of:
                - adversarial: Perturbed image clamped to [0, 1], shape [B, 3, 224, 224]
                - perturbation: Raw perturbation in [-1, 1], shape [B, 3, 224, 224]
        """
        # Encoder path
        e1 = self.enc1(x)                    # [B, 64, 224, 224]
        e2 = self.enc2(self.pool(e1))        # [B, 128, 112, 112]
        e3 = self.enc3(self.pool(e2))        # [B, 256, 56, 56]
        e4 = self.enc4(self.pool(e3))        # [B, 512, 28, 28]
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))   # [B, 512, 14, 14]
        
        # Decoder path with skip connections
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))   # [B, 256, 28, 28]
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))  # [B, 128, 56, 56]
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))  # [B, 64, 112, 112]
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))  # [B, 64, 224, 224]
        
        # Generate perturbation in [-1, 1]
        perturbation = self.output(d1)  # [B, 3, 224, 224]
        
        # Create adversarial image: x + epsilon * perturbation
        adversarial = x + epsilon * perturbation
        
        # Clamp to valid image range [0, 1]
        adversarial = torch.clamp(adversarial, 0.0, 1.0)
        
        return adversarial, perturbation
