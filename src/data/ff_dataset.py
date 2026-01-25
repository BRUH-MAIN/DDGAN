"""FaceForensics++ Dataset loader for deepfake detection.

This module provides dataset classes and utilities for loading the
FaceForensics++ image dataset with support for handling class imbalance.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np

from .transforms import cpu_transform


class FaceForensicsDataset(Dataset):
    """PyTorch Dataset for FaceForensics++ image dataset.
    
    This dataset loads images from the FaceForensics++ dataset structure
    where images are organized by manipulation type (original, Deepfakes,
    Face2Face, FaceSwap, FaceShifter, NeuralTextures, DeepFakeDetection).
    
    Args:
        root_dir: Root directory containing the image_dataset_metadata.csv
        csv_path: Path to the metadata CSV file (optional, auto-detected if None)
        categories: List of categories to include (None = all)
        transform: Optional transform to apply to images
        max_samples_per_category: Maximum samples per category (for balancing)
        split: Data split ('train', 'val', 'test')
        train_ratio: Ratio of data for training (default: 0.8)
        val_ratio: Ratio of data for validation (default: 0.1)
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
        root_dir: str,
        csv_path: Optional[str] = None,
        categories: Optional[List[str]] = None,
        transform=None,
        max_samples_per_category: Optional[int] = None,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        # Load metadata
        if csv_path is None:
            csv_path = self.root_dir / 'image_dataset_metadata.csv'
        
        self.metadata = pd.read_csv(csv_path)
        
        # Filter by categories if specified
        if categories is not None:
            self.metadata = self.metadata[
                self.metadata['category'].isin(categories)
            ].reset_index(drop=True)
        
        # Balance dataset if max_samples_per_category is specified
        if max_samples_per_category is not None:
            balanced_dfs = []
            for category in self.metadata['category'].unique():
                cat_df = self.metadata[self.metadata['category'] == category]
                if len(cat_df) > max_samples_per_category:
                    cat_df = cat_df.sample(
                        n=max_samples_per_category, 
                        random_state=seed
                    )
                balanced_dfs.append(cat_df)
            self.metadata = pd.concat(balanced_dfs).reset_index(drop=True)
        
        # Split data
        self.metadata = self._split_data(train_ratio, val_ratio, seed)
        
        # Create label mapping
        self.labels = self.metadata['category'].map(self.LABEL_MAP).values
        
        # Compute class weights for imbalanced sampling
        self._compute_class_weights()
        
    def _split_data(
        self, 
        train_ratio: float, 
        val_ratio: float, 
        seed: int
    ) -> pd.DataFrame:
        """Split data by video_id to prevent data leakage."""
        np.random.seed(seed)
        
        # Get unique video IDs
        video_ids = self.metadata['video_id'].unique()
        np.random.shuffle(video_ids)
        
        n_train = int(len(video_ids) * train_ratio)
        n_val = int(len(video_ids) * val_ratio)
        
        if self.split == 'train':
            selected_ids = video_ids[:n_train]
        elif self.split == 'val':
            selected_ids = video_ids[n_train:n_train + n_val]
        else:  # test
            selected_ids = video_ids[n_train + n_val:]
        
        return self.metadata[
            self.metadata['video_id'].isin(selected_ids)
        ].reset_index(drop=True)
    
    def _compute_class_weights(self):
        """Compute class weights for handling imbalance."""
        class_counts = np.bincount(self.labels.astype(int))
        total = len(self.labels)
        
        # Inverse frequency weighting
        self.class_weights = total / (len(class_counts) * class_counts)
        
        # Per-sample weights for weighted sampling
        self.sample_weights = self.class_weights[self.labels.astype(int)]
        
        # Store class distribution info
        self.n_real = int(class_counts[1]) if len(class_counts) > 1 else 0
        self.n_fake = int(class_counts[0]) if len(class_counts) > 0 else 0
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
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = self.root_dir / row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = cpu_transform(image)
        
        # Get label
        label = self.LABEL_MAP[row['category']]
        
        return {
            'image': image,
            'label': label,
            'category': row['category'],
            'video_id': row['video_id'],
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self),
            'n_real': self.n_real,
            'n_fake': self.n_fake,
            'imbalance_ratio': self.imbalance_ratio,
            'class_weights': self.class_weights.tolist(),
            'categories': self.metadata['category'].value_counts().to_dict(),
        }
        return stats


def create_ff_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    max_samples_per_category: Optional[int] = None,
    categories: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Create train and validation dataloaders for FaceForensics++.
    
    Args:
        root_dir: Root directory of the dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        use_weighted_sampling: Whether to use weighted sampling for imbalance
        max_samples_per_category: Max samples per category (None = no limit)
        categories: Categories to include (None = all)
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, dataset_info)
    """
    # Create datasets
    train_dataset = FaceForensicsDataset(
        root_dir=root_dir,
        categories=categories,
        max_samples_per_category=max_samples_per_category,
        split='train',
        seed=seed,
    )
    
    val_dataset = FaceForensicsDataset(
        root_dir=root_dir,
        categories=categories,
        max_samples_per_category=max_samples_per_category,
        split='val',
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
