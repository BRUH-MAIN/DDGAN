"""DCT (Discrete Cosine Transform) feature extraction module.

This module implements 2D DCT transformation and feature extraction
for deepfake detection using frequency-domain features.
"""

import torch
import torch.nn as nn
import math


class DCT2D(nn.Module):
    """2D Discrete Cosine Transform (DCT-II) module.
    
    Implements separable 2D DCT using precomputed transformation matrices.
    The DCT is computed via matrix multiplication for efficiency.
    
    Args:
        size: The spatial size of input images (default: 224).
              Assumes square images of shape [B, C, size, size].
    """
    
    def __init__(self, size: int = 224) -> None:
        super().__init__()
        self.size = size
        
        # Precompute the DCT-II transformation matrix
        # Formula: dct_matrix[k, n] = cos(π * k * (2n + 1) / (2 * size))
        # First row scaled by sqrt(1/size), remaining rows by sqrt(2/size)
        dct_matrix = self._create_dct_matrix(size)
        
        # Register as buffer (non-trainable, moves with model to device)
        self.register_buffer('dct_matrix', dct_matrix)
    
    def _create_dct_matrix(self, size: int) -> torch.Tensor:
        """Create the DCT-II transformation matrix.
        
        Args:
            size: Size of the transformation matrix.
            
        Returns:
            DCT transformation matrix of shape [size, size].
        """
        # Create indices
        k = torch.arange(size).float().unsqueeze(1)  # [size, 1]
        n = torch.arange(size).float().unsqueeze(0)  # [1, size]
        
        # Compute DCT matrix: cos(π * k * (2n + 1) / (2 * size))
        dct_matrix = torch.cos(math.pi * k * (2 * n + 1) / (2 * size))
        
        # Scale factors: sqrt(1/size) for first row, sqrt(2/size) for rest
        scale = torch.ones(size) * math.sqrt(2 / size)
        scale[0] = math.sqrt(1 / size)
        
        # Apply scaling to rows
        dct_matrix = dct_matrix * scale.unsqueeze(1)
        
        return dct_matrix
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D DCT transform using separable computation.
        
        The 2D DCT is computed as: DCT_2D = D @ X @ D^T
        where D is the 1D DCT matrix and X is the input.
        
        Args:
            x: Input tensor of shape [B, C, H, W].
            
        Returns:
            DCT-transformed tensor of shape [B, C, H, W].
        """
        # Validate input shape
        assert x.dim() == 4, f"Expected 4D input [B, C, H, W], got {x.dim()}D"
        
        # Separable 2D DCT: apply along rows, then along columns
        # DCT along rows: D @ X (for each row)
        x = torch.matmul(self.dct_matrix, x)
        
        # DCT along columns: X @ D^T (equivalent to transposing, applying D, transposing back)
        x = torch.matmul(x, self.dct_matrix.t())
        
        return x


class DCTFeatureExtractor(nn.Module):
    """DCT-based feature extractor for deepfake detection.
    
    Converts RGB images to grayscale, applies 2D DCT, and performs
    log scaling for better feature representation.
    
    Args:
        size: The spatial size of input images (default: 224).
    """
    
    def __init__(self, size: int = 224) -> None:
        super().__init__()
        self.dct = DCT2D(size=size)
        
        # RGB to grayscale conversion weights (ITU-R BT.601 standard)
        # Register as buffer for proper device handling
        grayscale_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.register_buffer('grayscale_weights', grayscale_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DCT features from RGB input.
        
        Args:
            x: Input RGB tensor of shape [B, 3, H, W].
            
        Returns:
            Log-scaled DCT features of shape [B, 1, H, W].
        """
        # Validate input
        assert x.dim() == 4, f"Expected 4D input [B, 3, H, W], got {x.dim()}D"
        assert x.size(1) == 3, f"Expected 3 channels (RGB), got {x.size(1)}"
        
        # Convert RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
        grayscale = (x * self.grayscale_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply 2D DCT transform
        dct_features = self.dct(grayscale)  # [B, 1, H, W]
        
        # Apply log scaling for better feature representation
        # log(|DCT| + epsilon) to handle zeros and negative values
        dct_features = torch.log(torch.abs(dct_features) + 1e-8)
        
        return dct_features
