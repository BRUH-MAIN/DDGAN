"""HuggingFace Dataset loader for FaceForensics++ deepfake detection.

This module provides dataset classes for loading the FaceForensics++ dataset
from Hugging Face Hub with support for handling class imbalance.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from .transforms import cpu_transform

# Try to import datasets library
try:
    from datasets import load_dataset, DatasetDict
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFaceFFDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace FaceForensics++ dataset.
    
    This dataset loads images from a HuggingFace Hub dataset that contains
    the FaceForensics++ deepfake detection images.
    
    Args:
        hf_dataset: HuggingFace dataset split (e.g., dataset['train'])
        transform: Optional transform to apply to images
        max_samples_per_category: Maximum samples per category (for balancing)
        seed: Random seed for reproducibility
    """
    
    # Map category names to binary labels
    LABEL_MAP = {
        'original': 1,  # REAL
        'Deepfakes': 0,  # FAKE
        'Face2Face': 0,  # FAKE
        'FaceSwap': 0,  # FAKE
        'FaceShifter': 0,  # FAKE
        'NeuralTextures': 0,  # FAKE
        'DeepFakeDetection': 0,  # FAKE
    }
    
    def __init__(
        self,
        hf_dataset,
        transform=None,
        max_samples_per_category: Optional[int] = None,
        seed: int = 42,
    ):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.seed = seed
        
        # Get indices for each category
        self.indices = list(range(len(hf_dataset)))
        
        # Balance dataset if max_samples_per_category is specified
        if max_samples_per_category is not None:
            self.indices = self._balance_indices(max_samples_per_category)
        
        # Extract labels for the selected indices
        self.labels = np.array([
            hf_dataset[i]['label'] for i in self.indices
        ])
        
        # Compute class weights for handling imbalance
        self._compute_class_weights()
    
    def _balance_indices(self, max_samples: int) -> List[int]:
        """Balance dataset by limiting samples per category."""
        np.random.seed(self.seed)
        
        # Group indices by category
        category_indices = {}
        for i in range(len(self.hf_dataset)):
            cat = self.hf_dataset[i]['category']
            if cat not in category_indices:
                category_indices[cat] = []
            category_indices[cat].append(i)
        
        # Sample from each category
        balanced_indices = []
        for cat, indices in category_indices.items():
            if len(indices) > max_samples:
                indices = np.random.choice(indices, max_samples, replace=False).tolist()
            balanced_indices.extend(indices)
        
        np.random.shuffle(balanced_indices)
        return balanced_indices
    
    def _compute_class_weights(self):
        """Compute class weights for handling imbalance."""
        class_counts = np.bincount(self.labels.astype(int), minlength=2)
        total = len(self.labels)
        
        # Inverse frequency weighting
        self.class_weights = total / (len(class_counts) * np.maximum(class_counts, 1))
        
        # Per-sample weights for weighted sampling
        self.sample_weights = self.class_weights[self.labels.astype(int)]
        
        # Store class distribution info
        self.n_fake = int(class_counts[0]) if len(class_counts) > 0 else 0
        self.n_real = int(class_counts[1]) if len(class_counts) > 1 else 0
        self.imbalance_ratio = self.n_fake / max(self.n_real, 1)
    
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Get a weighted random sampler for balanced training.
        
        Returns:
            WeightedRandomSampler that oversamples the minority class.
        """
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )
    
    def get_class_weights_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Get class weights as a tensor for weighted loss functions.
        
        Args:
            device: Device to place the tensor on.
            
        Returns:
            Tensor of shape [2] with [fake_weight, real_weight].
        """
        weights = torch.tensor(self.class_weights, dtype=torch.float32)
        if device is not None:
            weights = weights.to(device)
        return weights
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get the actual index in the HF dataset
        actual_idx = self.indices[idx]
        item = self.hf_dataset[actual_idx]
        
        # Get image (HF datasets returns PIL Image directly)
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = cpu_transform(image)
        
        # Validate image tensor - check for NaN/Inf and clamp
        if torch.is_tensor(image):
            if torch.isnan(image).any() or torch.isinf(image).any():
                # Replace corrupted image with zeros (will be skipped in training)
                image = torch.zeros_like(image)
            else:
                # Clamp to valid range [0, 1]
                image = torch.clamp(image, 0.0, 1.0)
        
        # Get label (already binary in HF dataset)
        label = item['label']
        
        return {
            'image': image,
            'label': label,
            'category': item['category'],
            'video_id': item['video_id'],
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        # Count categories
        category_counts = {}
        for i in self.indices:
            cat = self.hf_dataset[i]['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        stats = {
            'total_samples': len(self),
            'n_real': self.n_real,
            'n_fake': self.n_fake,
            'imbalance_ratio': self.imbalance_ratio,
            'class_weights': self.class_weights.tolist(),
            'categories': category_counts,
        }
        return stats


def load_hf_ff_dataset(
    repo_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> DatasetDict:
    """Load FaceForensics++ dataset from Hugging Face Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/ff-images-dataset')
        cache_dir: Optional directory for caching downloaded data
        token: Optional HuggingFace token for private datasets
        
    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits
        
    Raises:
        ImportError: If the datasets library is not installed
        ValueError: If the dataset doesn't have required splits
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "The 'datasets' library is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        )
    
    # Prepare kwargs
    kwargs = {}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    if token:
        kwargs['token'] = token
    
    print(f"Loading dataset from Hugging Face: {repo_id}")
    dataset = load_dataset(repo_id, **kwargs)
    
    # Validate splits
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected DatasetDict, got {type(dataset)}")
    
    required_splits = ['train']
    for split in required_splits:
        if split not in dataset:
            raise ValueError(f"Dataset {repo_id} must have a '{split}' split")
    
    # Handle validation split naming
    if 'validation' not in dataset and 'val' in dataset:
        dataset['validation'] = dataset['val']
        del dataset['val']
    
    print(f"Dataset loaded with splits: {list(dataset.keys())}")
    return dataset


def create_hf_ff_dataloaders(
    repo_id: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    max_samples_per_category: Optional[int] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Create train and validation dataloaders from HuggingFace dataset.
    
    Args:
        repo_id: HuggingFace repository ID
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        use_weighted_sampling: Whether to use weighted sampling for imbalance
        max_samples_per_category: Max samples per category (None = no limit)
        cache_dir: Cache directory for downloaded data
        token: HuggingFace token for private datasets
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, dataset_info)
    """
    # Load dataset from HuggingFace
    hf_dataset = load_hf_ff_dataset(repo_id, cache_dir=cache_dir, token=token)
    
    # Determine validation split name
    val_split = 'validation' if 'validation' in hf_dataset else 'val'
    if val_split not in hf_dataset:
        val_split = 'test'  # Fallback to test if no validation
    
    # Create PyTorch datasets
    train_dataset = HuggingFaceFFDataset(
        hf_dataset=hf_dataset['train'],
        max_samples_per_category=max_samples_per_category,
        seed=seed,
    )
    
    val_dataset = HuggingFaceFFDataset(
        hf_dataset=hf_dataset[val_split],
        max_samples_per_category=max_samples_per_category,
        seed=seed,
    )
    
    # Create sampler for training (handles imbalance)
    train_sampler = None
    shuffle = True
    if use_weighted_sampling:
        train_sampler = train_dataset.get_weighted_sampler()
        shuffle = False  # Sampler handles shuffling
    
    # Collate function
    def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor(
            [item['label'] for item in batch], 
            dtype=torch.float32
        )
        return images, labels
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Dataset info
    dataset_info = {
        'train_stats': train_dataset.get_statistics(),
        'val_stats': val_dataset.get_statistics(),
        'class_weights': train_dataset.get_class_weights_tensor(),
    }
    
    return train_loader, val_loader, dataset_info
