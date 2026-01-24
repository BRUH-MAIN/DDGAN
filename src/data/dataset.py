"""Dataset loading utilities for deepfake detection.

This module provides functions for loading deepfake detection datasets
from HuggingFace Hub.
"""

from typing import Optional
from datasets import load_dataset, DatasetDict


def load_deepfake_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """Load a deepfake detection dataset from HuggingFace.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub.
        cache_dir: Optional directory for caching downloaded data.
        
    Returns:
        DatasetDict containing 'train' and 'val' splits.
        
    Raises:
        ValueError: If the dataset doesn't have required splits.
    """
    # Load dataset from HuggingFace Hub
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    dataset = load_dataset(dataset_name, **kwargs)
    
    # Ensure we have the required splits
    if isinstance(dataset, DatasetDict):
        # Handle different split naming conventions
        if 'train' not in dataset:
            raise ValueError(f"Dataset {dataset_name} must have a 'train' split")
        
        # Check for validation split (might be named 'val', 'validation', or 'test')
        val_key = None
        for key in ['val', 'validation', 'test']:
            if key in dataset:
                val_key = key
                break
        
        if val_key is None:
            # If no validation split, create one from train (10%)
            train_test = dataset['train'].train_test_split(test_size=0.1, seed=42)
            return DatasetDict({
                'train': train_test['train'],
                'val': train_test['test']
            })
        
        return DatasetDict({
            'train': dataset['train'],
            'val': dataset[val_key]
        })
    
    raise ValueError(f"Unexpected dataset format from {dataset_name}")
