"""
Push CelebDF-v2 processed dataset to HuggingFace Hub
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder
from tqdm import tqdm


def push_dataset_to_hf(
    dataset_dir='celebdfv2_images',
    repo_name='celebdfv2_224',
    username='RohanRamesh',
    private=False
):
    """
    Push dataset to HuggingFace Hub
    
    Args:
        dataset_dir: Local directory containing the dataset
        repo_name: Name of the HuggingFace repository
        username: HuggingFace username
        private: Whether to make the repo private
    """
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file. Please add it.")
    
    print("=" * 60)
    print("PUSHING DATASET TO HUGGINGFACE HUB")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Repository: {username}/{repo_name}")
    print(f"Private: {private}")
    print("=" * 60 + "\n")
    
    # Check if dataset directory exists
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Count files
    print("Scanning dataset...")
    train_real = len(list((dataset_path / 'train' / 'real').glob('*.jpg')))
    train_fake = len(list((dataset_path / 'train' / 'fake').glob('*.jpg')))
    test_real = len(list((dataset_path / 'test' / 'real').glob('*.jpg')))
    test_fake = len(list((dataset_path / 'test' / 'fake').glob('*.jpg')))
    
    total_files = train_real + train_fake + test_real + test_fake
    
    print(f"\nDataset statistics:")
    print(f"  Train:")
    print(f"    Real: {train_real:,} images")
    print(f"    Fake: {train_fake:,} images")
    print(f"    Total: {train_real + train_fake:,} images")
    print(f"  Test:")
    print(f"    Real: {test_real:,} images")
    print(f"    Fake: {test_fake:,} images")
    print(f"    Total: {test_real + test_fake:,} images")
    print(f"  Total: {total_files:,} images\n")
    
    # Initialize HuggingFace API
    api = HfApi(token=hf_token)
    
    # Create repository
    repo_id = f"{username}/{repo_name}"
    print(f"Creating repository: {repo_id}...")
    
    try:
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=private,
            repo_type="dataset",
            exist_ok=True
        )
        print(f"✓ Repository created/found: https://huggingface.co/datasets/{repo_id}\n")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Create README
    print("Creating README.md...")
    readme_content = f"""---
license: cc-by-nc-4.0
task_categories:
- image-classification
- zero-shot-image-classification
tags:
- deepfake-detection
- face
- synthetic
size_categories:
- 100K<n<1M
---

# CelebDF-v2 224x224 Processed Dataset

This dataset contains preprocessed images from the CelebDF-v2 dataset, resized and face-cropped to 224×224 pixels.

## Dataset Description

- **Task**: Deepfake detection
- **Format**: RGB images, 224×224 pixels
- **Classes**: Real (0) and Fake (1)
- **Total Images**: {total_files:,}

## Dataset Structure

```
celebdfv2_224/
├── train/
│   ├── real/  ({train_real:,} images)
│   └── fake/  ({train_fake:,} images)
└── test/
    ├── real/  ({test_real:,} images)
    └── fake/  ({test_fake:,} images)
```

## Statistics

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | {train_real:,} | {train_fake:,} | {train_real + train_fake:,} |
| Test  | {test_real:,} | {test_fake:,} | {test_real + test_fake:,} |
| **Total** | **{train_real + test_real:,}** | **{train_fake + test_fake:,}** | **{total_files:,}** |

## Preprocessing

The original CelebDF-v2 videos were processed using:
1. MTCNN face detection
2. Face cropping with 30% margin
3. Resizing to 224×224 pixels
4. Frame extraction (30 frames per video)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access splits
train_dataset = dataset['train']
test_dataset = dataset['test']
```

## Source

Original dataset: [CelebDF-v2](https://github.com/yuezunli/celeb-deepfakeforensics)

## License

This dataset follows the same license as the original CelebDF-v2 dataset (CC BY-NC 4.0).

## Citation

If you use this dataset, please cite the original CelebDF-v2 paper:

```bibtex
@inproceedings{{li2020celeb,
  title={{Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics}},
  author={{Li, Yuezun and Yang, Xin and Sun, Pu and Qi, Honggang and Lyu, Siwei}},
  booktitle={{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
  year={{2020}}
}}
```

## Processed by

Dataset preprocessed by RohanRamesh for deepfake detection research.

---

**Note**: This dataset is for research purposes only. Use responsibly and ethically.
"""
    
    readme_path = dataset_path / 'README.md'
    readme_path.write_text(readme_content)
    print(f"✓ README.md created\n")
    
    # Upload dataset
    print(f"Uploading dataset to {repo_id}...")
    print("This may take a while depending on your internet connection...")
    print("Using large folder upload for better reliability...\n")
    
    try:
        api.upload_large_folder(
            folder_path=dataset_dir,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.txt", "*.md"],
            ignore_patterns=[".git/*", "__pycache__/*", "*.pyc"],
            num_workers=4
        )
        print(f"\n✓ Dataset uploaded successfully!")
        print(f"\n{'=' * 60}")
        print(f"Dataset available at:")
        print(f"https://huggingface.co/datasets/{repo_id}")
        print(f"{'=' * 60}\n")
        
    except Exception as e:
        print(f"\n✗ Error uploading dataset: {e}")
        print("\nYou can try uploading manually using:")
        print(f"  huggingface-cli upload {repo_id} {dataset_dir} --repo-type dataset")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Push CelebDF-v2 dataset to HuggingFace')
    parser.add_argument('--dataset-dir', type=str, default='celebdfv2_images',
                        help='Path to dataset directory')
    parser.add_argument('--repo-name', type=str, default='celebdfv2_224',
                        help='Name of HuggingFace repository')
    parser.add_argument('--username', type=str, default='RohanRamesh',
                        help='HuggingFace username')
    parser.add_argument('--private', action='store_true',
                        help='Make repository private')
    
    args = parser.parse_args()
    
    push_dataset_to_hf(
        dataset_dir=args.dataset_dir,
        repo_name=args.repo_name,
        username=args.username,
        private=args.private
    )
