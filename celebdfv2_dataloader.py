import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class CelebDFImageDataset(Dataset):
    """CelebDF-v2 Image Dataset for PyTorch."""

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Collect all image paths
        self.samples = []
        for label_idx, label in enumerate(['real', 'fake']):
            label_dir = self.root_dir / split / label
            for img_path in label_dir.glob('*.jpg'):
                self.samples.append((str(img_path), label_idx))

        print(f"Loaded {len(self.samples)} images from {split} set")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage:
if __name__ == "__main__":
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CelebDFImageDataset('celebdfv2_images', split='train', transform=train_transform)
    test_dataset = CelebDFImageDataset('celebdfv2_images', split='test', transform=test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
