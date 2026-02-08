"""
Dataset and DataModule for Deepfake Detection
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace datasets not available. Install with: pip install datasets")


class DeepfakeDataset(Dataset):
    """
    Deepfake Detection Dataset
    
    Loads images from directory structure:
    root/
        train/
            real/
            fake/
        test/
            real/
            fake/
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory of dataset
            split: 'train' or 'test'
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Collect all image paths with labels
        # Convention: real=1 (positive), fake=0 (negative)
        self.samples = []
        label_map = {'real': 1, 'fake': 0}
        for label in ['real', 'fake']:
            label_dir = self.root_dir / split / label
            if not label_dir.exists():
                print(f"Warning: {label_dir} not found!")
                continue
            label_idx = label_map[label]
            
            # Check if directory has subdirectories (part_0, part_1, etc.)
            subdirs = sorted(label_dir.glob('part_*'))
            if subdirs:
                # Has subdirectories - iterate through them
                for subdir in subdirs:
                    for img_path in subdir.glob('*.jpg'):
                        self.samples.append((str(img_path), label_idx))
            else:
                # No subdirectories - directly get images
                for img_path in label_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), label_idx))
        
        print(f"Loaded {len(self.samples)} images from {split} set")
        
        # Calculate class distribution
        real_count = sum(1 for _, label in self.samples if label == 1)
        fake_count = sum(1 for _, label in self.samples if label == 0)
        print(f"  Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


class HuggingFaceDeepfakeDataset(Dataset):
    """
    Deepfake Detection Dataset from HuggingFace Hub
    
    Loads images from HuggingFace dataset
    """
    
    def __init__(self, hf_dataset, transform=None, print_stats=True, label_map=None):
        """
        Initialize dataset from HuggingFace
        
        Args:
            hf_dataset: HuggingFace dataset split
            transform: Image transformations
            print_stats: Whether to print dataset statistics
        """
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.label_map = label_map or self._infer_label_map()
        
        if print_stats:
            print(f"Loaded {len(self.hf_dataset)} images from HuggingFace")
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Get image and label
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        
        label = item['label']
        if isinstance(label, str):
            if label not in self.label_map:
                raise ValueError(f"Unknown label string '{label}'")
            label = self.label_map[label]
        else:
            label = int(label)
            if label not in self.label_map:
                raise ValueError(f"Unknown label id '{label}'")
            label = self.label_map[label]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def _infer_label_map(self):
        label_feature = self.hf_dataset.features.get("label") if hasattr(self.hf_dataset, "features") else None
        if label_feature is not None and hasattr(label_feature, "names"):
            names = [n.lower() for n in label_feature.names]
            if "real" in names and "fake" in names:
                real_idx = names.index("real")
                fake_idx = names.index("fake")
                return {
                    real_idx: 1,
                    fake_idx: 0,
                    "real": 1,
                    "fake": 0,
                }

        print("Warning: Could not infer label names; defaulting to 0=real, 1=fake.")
        return {
            0: 1,
            1: 0,
            "real": 1,
            "fake": 0,
        }


class DeepfakeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Deepfake Detection
    
    Handles data loading, transformations, and dataloaders
    """
    
    def __init__(
        self,
        data_dir=None,
        hf_dataset_id=None,
        hf_label_map=None,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        """
        Initialize DataModule
        
        Args:
            data_dir: Root directory of local dataset (for local loading)
            hf_dataset_id: HuggingFace dataset ID (for HF loading)
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            mean: Normalization mean
            std: Normalization std
        """
        super().__init__()
        
        # Validate input
        if data_dir is None and hf_dataset_id is None:
            raise ValueError("Either data_dir or hf_dataset_id must be provided")
        if data_dir is not None and hf_dataset_id is not None:
            raise ValueError("Only one of data_dir or hf_dataset_id should be provided")
        
        self.data_dir = Path(data_dir) if data_dir else None
        self.hf_dataset_id = hf_dataset_id
        self.use_hf = hf_dataset_id is not None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.mean = mean
        self.std = std
        self.hf_label_map = hf_label_map
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def setup(self, stage=None):
        """Setup datasets for training and validation"""
        if stage == 'fit' or stage is None:
            if self.use_hf:
                # Load from HuggingFace
                if not HF_AVAILABLE:
                    raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
                
                print(f"Loading dataset from HuggingFace: {self.hf_dataset_id}")
                
                # Load train and validation/test splits separately for faster loading
                print("  Loading train split...")
                train_data = load_dataset(
                    self.hf_dataset_id, 
                    split='train'
                )
                
                # Shuffle train data to ensure balanced batches
                # (dataset is ordered: all real first, then all fake)
                train_data = train_data.shuffle(seed=42)
                print(f"  Shuffled train data for balanced batches")
                
                # Try to find validation split
                val_split_name = 'test'  # Default to 'test'
                print(f"  Loading {val_split_name} split...")
                val_data = load_dataset(
                    self.hf_dataset_id,
                    split=val_split_name
                )
                
                print(f"  Train samples: {len(train_data)}")
                print(f"  Val samples: {len(val_data)}")
                
                self.train_dataset = HuggingFaceDeepfakeDataset(
                    train_data,
                    transform=self.train_transform,
                    print_stats=False,
                    label_map=self.hf_label_map,
                )
                
                self.val_dataset = HuggingFaceDeepfakeDataset(
                    val_data,
                    transform=self.val_transform,
                    print_stats=False,
                    label_map=self.hf_label_map,
                )
            else:
                # Load from local directory
                self.train_dataset = DeepfakeDataset(
                    self.data_dir,
                    split='train',
                    transform=self.train_transform
                )
                
                self.val_dataset = DeepfakeDataset(
                    self.data_dir,
                    split='test',
                    transform=self.val_transform
                )
    
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True  # Drop incomplete batches for stable training
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )


if __name__ == "__main__":
    # Test DataModule
    print("Testing DeepfakeDataModule...")
    
    dm = DeepfakeDataModule(
        data_dir='celebdfv2_images',
        batch_size=32,
        num_workers=4
    )
    
    # Setup
    dm.setup('fit')
    
    # Get dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label distribution: Real={torch.sum(labels == 0).item()}, Fake={torch.sum(labels == 1).item()}")
    
    print("\n✓ DataModule test passed!")
