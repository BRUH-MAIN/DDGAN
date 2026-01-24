"""Image transforms for deepfake detection pipeline.

This module provides CPU and GPU transforms for preprocessing images
before feeding them to the model.
"""

from typing import Dict, List, Any

import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms


# CPU transforms for initial data loading (applied in DataLoader)
cpu_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),  # Ensure RGB
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Scale to [0, 1]
])


def get_gpu_transform() -> nn.Module:
    """Create GPU-accelerated transforms for model input.
    
    Returns:
        A compiled nn.Module that performs resize and normalization.
    """
    return nn.Sequential(
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )


def preprocess_function(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a batch of images using CPU transforms.
    
    This function is designed to be used with HuggingFace datasets.map().
    
    Args:
        batch: Dictionary containing 'image' key with PIL images.
        
    Returns:
        Modified batch with transformed images.
    """
    batch["image"] = [cpu_transform(img) for img in batch["image"]]
    return batch


def collate_fn(batch: List[Dict[str, Any]]) -> tuple:
    """Custom collate function for DataLoader.
    
    Stacks images into a batch tensor and extracts labels.
    Applies CPU transforms on-the-fly to avoid caching transformed data.
    
    Args:
        batch: List of dictionaries, each containing 'image' and 'label'.
        
    Returns:
        Tuple of (images, labels) tensors.
    """
    # Apply transforms on-the-fly (no caching = no disk bloat)
    images = torch.stack([cpu_transform(item['image']) for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    return images, labels
