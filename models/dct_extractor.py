"""
DCT Feature Extractor for Deepfake Detection
Implements 2D Discrete Cosine Transform with optional per-channel processing
and learnable frequency filters for adaptive frequency weighting.
"""
import torch
import torch.nn as nn
import numpy as np


class DCT2D(nn.Module):
    """
    2D Discrete Cosine Transform (DCT-II) layer
    
    Supports two modes:
    - Grayscale mode (legacy): RGB → grayscale → DCT [B, 1, H, W]
    - Per-channel mode: Apply DCT to each RGB channel separately [B, 3, H, W]
      with learnable frequency filters for adaptive band selection
    
    Mathematical Foundation:
    - DCT-II Transform Matrix: dct_matrix[k,n] = cos(π × k × (2n + 1) / (2 × N))
    - Normalization: First row sqrt(1/N), other rows sqrt(2/N)
    - Separable 2D DCT: apply 1D DCT along rows, then columns
    """
    
    def __init__(self, size=224, per_channel=False, learnable_filters=True):
        """
        Initialize DCT2D layer
        
        Args:
            size: Image size (assumes square images)
            per_channel: If True, apply DCT to each RGB channel separately
                        producing 3-channel output with learnable frequency filters.
                        If False (legacy), convert to grayscale first.
            learnable_filters: If True and per_channel=True, add learnable
                              frequency filter masks for each channel
        """
        super(DCT2D, self).__init__()
        self.size = size
        self.per_channel = per_channel
        self.learnable_filters = learnable_filters and per_channel
        
        # Precompute DCT basis matrix
        dct_matrix = self._create_dct_matrix(size)
        
        # Register as buffer (non-trainable, moves to GPU automatically)
        self.register_buffer('dct_matrix', dct_matrix)
        
        # Learnable frequency filters (per-channel mode only)
        if self.learnable_filters:
            # Initialize with slight preference for low-mid frequencies
            # Shape: [1, 3, H, W] - one filter per RGB channel
            init_filter = self._create_initial_frequency_filter(size)
            self.freq_filter = nn.Parameter(init_filter)
    
    def _create_initial_frequency_filter(self, size):
        """
        Create initial frequency filter with slight low-frequency bias
        
        Returns:
            Initial filter tensor [1, 3, H, W]
        """
        # Create radial frequency map (distance from DC component)
        y_coords = torch.arange(size).float()
        x_coords = torch.arange(size).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalized distance from (0,0) - DC component
        freq_dist = torch.sqrt(yy**2 + xx**2) / (size * np.sqrt(2))
        
        # Slight low-mid frequency preference (Gaussian-like falloff)
        # Values range from ~1.0 (low freq) to ~0.5 (high freq)
        init_weights = 0.5 + 0.5 * torch.exp(-2 * freq_dist**2)
        
        # Expand to [1, 3, H, W] - same init for all channels
        init_filter = init_weights.unsqueeze(0).unsqueeze(0).expand(1, 3, size, size).clone()
        
        return init_filter
        
    def _create_dct_matrix(self, size):
        """
        Create DCT-II transformation matrix
        
        Args:
            size: Matrix dimension
            
        Returns:
            DCT transformation matrix [size, size]
        """
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
    
    def rgb_to_grayscale(self, rgb_image):
        """
        Convert RGB image to grayscale using standard weights
        
        Args:
            rgb_image: [B, 3, H, W] RGB image
            
        Returns:
            grayscale: [B, 1, H, W] grayscale image
        """
        # Standard grayscale conversion weights (ITU-R BT.601)
        # Y = 0.299R + 0.587G + 0.114B
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_image.device)
        weights = weights.view(1, 3, 1, 1)
        
        grayscale = torch.sum(rgb_image * weights, dim=1, keepdim=True)
        return grayscale
    
    def _apply_dct_single_channel(self, x):
        """
        Apply 2D DCT to a single channel
        
        Args:
            x: Single channel tensor [B, H, W]
            
        Returns:
            DCT coefficients [B, H, W]
        """
        # Apply separable 2D DCT
        # Step 1: DCT along rows (right multiply)
        dct_rows = torch.matmul(x, self.dct_matrix.t())  # [B, H, W]
        
        # Step 2: DCT along columns (left multiply)
        dct_2d = torch.matmul(self.dct_matrix, dct_rows)  # [B, H, W]
        
        return dct_2d
    
    def forward(self, x):
        """
        Apply 2D DCT transform
        
        Pipeline (per_channel=False, legacy mode):
        1. RGB [B, 3, H, W] → Grayscale [B, 1, H, W]
        2. 2D DCT transform
        3. Log scaling for better feature distribution
        
        Pipeline (per_channel=True):
        1. Apply 2D DCT to each RGB channel separately
        2. Apply learnable frequency filters (if enabled)
        3. Log scaling
        4. Output [B, 3, H, W]
        
        Args:
            x: RGB image tensor [B, 3, H, W]
            
        Returns:
            DCT features [B, 1, H, W] (legacy) or [B, 3, H, W] (per_channel)
        """
        if self.per_channel:
            # Per-channel DCT mode
            batch_size = x.size(0)
            dct_channels = []
            
            for c in range(3):
                # Extract single channel [B, H, W]
                channel = x[:, c, :, :]
                
                # Apply 2D DCT
                dct_c = self._apply_dct_single_channel(channel)
                
                dct_channels.append(dct_c)
            
            # Stack channels [B, 3, H, W]
            dct_2d = torch.stack(dct_channels, dim=1)
            
            # Apply learnable frequency filters
            if self.learnable_filters:
                dct_2d = dct_2d * self.freq_filter
            
            # Clamp extreme values to prevent overflow in fp16
            dct_2d = torch.clamp(dct_2d, -1e6, 1e6)
            
            # Apply log scaling per channel
            dct_2d = torch.log(torch.abs(dct_2d) + 1e-6)
            
            return dct_2d
        else:
            # Legacy grayscale mode
            # Convert to grayscale
            gray = self.rgb_to_grayscale(x)  # [B, 1, H, W]
            
            # Remove channel dimension for matrix operations
            gray = gray.squeeze(1)  # [B, H, W]
            
            # Apply 2D DCT
            dct_2d = self._apply_dct_single_channel(gray)
            
            # ===== DIAGNOSTIC: Check for numerical issues before log =====
            if torch.isnan(dct_2d).any() or torch.isinf(dct_2d).any():
                print(f"[DCT WARNING] Pre-log: has_nan={torch.isnan(dct_2d).any()}, has_inf={torch.isinf(dct_2d).any()}")
                print(f"  gray input: min={gray.min():.3f}, max={gray.max():.3f}, mean={gray.mean():.3f}")
                print(f"  dct_2d: min={dct_2d.min():.3f}, max={dct_2d.max():.3f}, abs_max={dct_2d.abs().max():.3f}")
            
            # Clamp extreme values to prevent overflow in fp16
            dct_2d = torch.clamp(dct_2d, -1e6, 1e6)
            
            # Apply log scaling to compress dynamic range
            # DCT coefficients have huge range (DC >> AC components)
            # Log scaling makes features more learnable
            dct_2d = torch.log(torch.abs(dct_2d) + 1e-6)
            
            # Add channel dimension back
            dct_2d = dct_2d.unsqueeze(1)  # [B, 1, H, W]
            
            return dct_2d


if __name__ == "__main__":
    # Test DCT2D layer
    print("Testing DCT2D layer...")
    
    # Create dummy RGB image
    batch_size = 2
    image_size = 224
    x = torch.randn(batch_size, 3, image_size, image_size)
    
    # Test legacy grayscale mode
    print("\n--- Legacy Grayscale Mode ---")
    dct_legacy = DCT2D(size=image_size, per_channel=False)
    dct_features_legacy = dct_legacy(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {dct_features_legacy.shape}")
    print(f"Output range: [{dct_features_legacy.min():.3f}, {dct_features_legacy.max():.3f}]")
    
    # Test per-channel mode with learnable filters
    print("\n--- Per-Channel Mode (Learnable Filters) ---")
    dct_perchannel = DCT2D(size=image_size, per_channel=True, learnable_filters=True)
    dct_features_perchannel = dct_perchannel(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {dct_features_perchannel.shape}")
    print(f"Output range: [{dct_features_perchannel.min():.3f}, {dct_features_perchannel.max():.3f}]")
    print(f"Frequency filter shape: {dct_perchannel.freq_filter.shape}")
    print(f"Frequency filter range: [{dct_perchannel.freq_filter.min():.3f}, {dct_perchannel.freq_filter.max():.3f}]")
    
    # Test per-channel mode without learnable filters
    print("\n--- Per-Channel Mode (Fixed) ---")
    dct_perchannel_fixed = DCT2D(size=image_size, per_channel=True, learnable_filters=False)
    dct_features_fixed = dct_perchannel_fixed(x)
    print(f"Output shape: {dct_features_fixed.shape}")
    print(f"Has freq_filter: {hasattr(dct_perchannel_fixed, 'freq_filter')}")
    
    print("\n✓ DCT2D layer test passed!")
