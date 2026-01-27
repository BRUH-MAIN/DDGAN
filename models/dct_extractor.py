"""
DCT Feature Extractor for Deepfake Detection
Implements 2D Discrete Cosine Transform on grayscale images
"""
import torch
import torch.nn as nn
import numpy as np


class DCT2D(nn.Module):
    """
    2D Discrete Cosine Transform (DCT-II) layer
    
    Converts RGB images to grayscale and applies 2D DCT transform
    to extract frequency-domain features useful for deepfake detection.
    
    Mathematical Foundation:
    - DCT-II Transform Matrix: dct_matrix[k,n] = cos(π × k × (2n + 1) / (2 × N))
    - Normalization: First row sqrt(1/N), other rows sqrt(2/N)
    - Separable 2D DCT: apply 1D DCT along rows, then columns
    """
    
    def __init__(self, size=224):
        """
        Initialize DCT2D layer
        
        Args:
            size: Image size (assumes square images)
        """
        super(DCT2D, self).__init__()
        self.size = size
        
        # Precompute DCT basis matrix
        dct_matrix = self._create_dct_matrix(size)
        
        # Register as buffer (non-trainable, moves to GPU automatically)
        self.register_buffer('dct_matrix', dct_matrix)
        
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
    
    def forward(self, x):
        """
        Apply 2D DCT transform
        
        Pipeline:
        1. RGB [B, 3, H, W] → Grayscale [B, 1, H, W]
        2. 2D DCT transform
        3. Log scaling for better feature distribution
        
        Args:
            x: RGB image tensor [B, 3, H, W]
            
        Returns:
            DCT features [B, 1, H, W]
        """
        # Convert to grayscale
        gray = self.rgb_to_grayscale(x)  # [B, 1, H, W]
        
        # Remove channel dimension for matrix operations
        gray = gray.squeeze(1)  # [B, H, W]
        
        # Apply separable 2D DCT
        # Step 1: DCT along rows (right multiply)
        dct_rows = torch.matmul(gray, self.dct_matrix.t())  # [B, H, W]
        
        # Step 2: DCT along columns (left multiply)
        dct_2d = torch.matmul(self.dct_matrix, dct_rows)  # [B, H, W]
        
        # Apply log scaling to compress dynamic range
        # DCT coefficients have huge range (DC >> AC components)
        # Log scaling makes features more learnable
        dct_2d = torch.log(torch.abs(dct_2d) + 1e-8)
        
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
    
    # Initialize DCT layer
    dct = DCT2D(size=image_size)
    
    # Forward pass
    dct_features = dct(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {dct_features.shape}")
    print(f"Output range: [{dct_features.min():.3f}, {dct_features.max():.3f}]")
    print("✓ DCT2D layer test passed!")
