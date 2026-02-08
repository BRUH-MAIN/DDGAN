"""
Visualize DCT features for real and fake images from CelebDFv2 dataset
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from scipy.fft import dctn

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224

# Image transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def load_celebdfv2_small():
    """Load CelebDFv2 small dataset from Hugging Face"""
    print("Loading CelebDFv2 small dataset from Hugging Face...")
    dataset = load_dataset('celebdf2_small', split='train', streaming=False)
    return dataset

def extract_dct_features(images):
    """
    Extract DCT features for a batch of images using SciPy (repository-independent).

    Args:
        images: Tensor of shape [B, 3, H, W]

    Returns:
        dct_features: Tensor of shape [B, 3, H, W] with DCT coefficients
    """
    images_np = images.detach().cpu().numpy()
    dct_list = []
    for img in images_np:
        # DCT on each channel (H, W)
        channels = []
        for c in range(img.shape[0]):
            channels.append(dctn(img[c], type=2, norm="ortho"))
        dct_list.append(np.stack(channels, axis=0))
    dct_features = torch.from_numpy(np.stack(dct_list, axis=0)).to(images.device)
    return dct_features

def plot_dct_comparison(real_images, fake_images, num_samples=3):
    """
    Plot original images and their DCT representations side-by-side
    
    Args:
        real_images: Tensor of real images [N, 3, H, W]
        fake_images: Tensor of fake images [N, 3, H, W]
        num_samples: Number of samples to display
    """
    with torch.no_grad():
        real_dct = extract_dct_features(real_images[:num_samples])
        fake_dct = extract_dct_features(fake_images[:num_samples])
    
    # Denormalize for visualization
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    real_viz = torch.clamp(denorm(real_images[:num_samples]), 0, 1)
    fake_viz = torch.clamp(denorm(fake_images[:num_samples]), 0, 1)
    
    # Log scale for DCT visualization (better contrast)
    real_dct_log = torch.log(torch.abs(real_dct) + 1e-6)
    fake_dct_log = torch.log(torch.abs(fake_dct) + 1e-6)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        # Real image
        ax = axes[i, 0] if num_samples > 1 else axes[0]
        ax.imshow(real_viz[i].permute(1, 2, 0).cpu().numpy())
        ax.set_title('Real Image', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Real DCT (log scale)
        ax = axes[i, 1] if num_samples > 1 else axes[1]
        dct_mean = real_dct_log[i].mean(dim=0).cpu().numpy()
        im = ax.imshow(dct_mean, cmap='viridis')
        ax.set_title('Real DCT (log)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        # Fake image
        ax = axes[i, 2] if num_samples > 1 else axes[2]
        ax.imshow(fake_viz[i].permute(1, 2, 0).cpu().numpy())
        ax.set_title('Fake Image', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Fake DCT (log scale)
        ax = axes[i, 3] if num_samples > 1 else axes[3]
        dct_mean = fake_dct_log[i].mean(dim=0).cpu().numpy()
        im = ax.imshow(dct_mean, cmap='viridis')
        ax.set_title('Fake DCT (log)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('dct_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'dct_comparison.png'")
    plt.show()

def plot_dct_spectrum(real_images, fake_images):
    """
    Plot average DCT spectrum (frequency content) for real vs fake
    """
    with torch.no_grad():
        real_dct = extract_dct_features(real_images).mean(dim=(0, 1))  # [H, W]
        fake_dct = extract_dct_features(fake_images).mean(dim=(0, 1))  # [H, W]
    
    # Compute magnitude spectrum
    real_spectrum = torch.log(torch.abs(real_dct) + 1e-6)
    fake_spectrum = torch.log(torch.abs(fake_dct) + 1e-6)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Real spectrum
    im = axes[0].imshow(real_spectrum.cpu().numpy(), cmap='viridis')
    axes[0].set_title('Real - DCT Spectrum', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Horizontal Frequency')
    axes[0].set_ylabel('Vertical Frequency')
    plt.colorbar(im, ax=axes[0], label='log(|DCT|)')
    
    # Fake spectrum
    im = axes[1].imshow(fake_spectrum.cpu().numpy(), cmap='viridis')
    axes[1].set_title('Fake - DCT Spectrum', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Horizontal Frequency')
    axes[1].set_ylabel('Vertical Frequency')
    plt.colorbar(im, ax=axes[1], label='log(|DCT|)')
    
    # Difference
    diff = (fake_spectrum - real_spectrum).cpu().numpy()
    im = axes[2].imshow(diff, cmap='RdBu', vmin=-diff.std(), vmax=diff.std())
    axes[2].set_title('Difference (Fake - Real)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Horizontal Frequency')
    axes[2].set_ylabel('Vertical Frequency')
    plt.colorbar(im, ax=axes[2], label='DCT Difference')
    
    plt.tight_layout()
    plt.savefig('dct_spectrum.png', dpi=150, bbox_inches='tight')
    print("Saved spectrum visualization to 'dct_spectrum.png'")
    plt.show()

def main():
    # Load dataset
    dataset = load_celebdfv2_small()
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    dataset = dataset.shuffle(seed=42)

    label_feature = dataset.features.get("label") if hasattr(dataset, "features") else None
    label_names = getattr(label_feature, "names", None)

    def is_real(label):
        if isinstance(label, str):
            return label.lower() == "real"
        if label_names:
            return label_names[label].lower() == "real"
        return label == 0

    def is_fake(label):
        if isinstance(label, str):
            return label.lower() == "fake"
        if label_names:
            return label_names[label].lower() == "fake"
        return label == 1
    
    # Separate real and fake images
    real_batch = []
    fake_batch = []
    
    max_scan = 50000
    for i, sample in enumerate(dataset):
        if i >= max_scan:
            break

        image = sample["image"]
        label = sample["label"]  # 0 for real, 1 for fake

        # Transform image
        img_tensor = transform(image)

        if is_real(label):
            real_batch.append(img_tensor)
        elif is_fake(label):
            fake_batch.append(img_tensor)

        if len(real_batch) >= 3 and len(fake_batch) >= 3:
            break

    if len(real_batch) < 3 or len(fake_batch) < 3:
        raise RuntimeError(
            "Not enough real/fake samples found within the scan window. "
            "Increase max_scan or verify dataset labels."
        )
    
    # Stack into batches
    real_images = torch.stack(real_batch).to(DEVICE)
    fake_images = torch.stack(fake_batch).to(DEVICE)
    
    print(f"Real images shape: {real_images.shape}")
    print(f"Fake images shape: {fake_images.shape}")
    
    # Generate visualizations
    print("\nGenerating DCT comparison plot...")
    plot_dct_comparison(real_images, fake_images, num_samples=min(3, len(real_batch), len(fake_batch)))
    
    print("Generating DCT spectrum plot...")
    plot_dct_spectrum(real_images, fake_images)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
