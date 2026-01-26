#!/usr/bin/env python3
"""Push images_dataset to Hugging Face Hub.

This script uploads the FaceForensics++ image dataset to Hugging Face Hub
with proper structure for easy loading via the datasets library.

The script reads the HF_TOKEN from .env file for authentication.

Usage:
    python push_to_huggingface.py --repo_id your-username/ff-images-dataset
    
    # Or with custom settings
    python push_to_huggingface.py \
        --repo_id your-username/ff-images-dataset \
        --data_dir ./images_dataset \
        --private
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import HfApi, create_repo, login
from tqdm import tqdm


def create_hf_dataset(data_dir: str, split_seed: int = 42) -> DatasetDict:
    """Create a HuggingFace DatasetDict from the local image dataset.
    
    Args:
        data_dir: Path to the images_dataset directory
        split_seed: Random seed for train/val/test split
        
    Returns:
        DatasetDict with train, validation, and test splits
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / "image_dataset_metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)
    
    print(f"Total images: {len(metadata)}")
    print(f"Categories: {metadata['category'].unique().tolist()}")
    
    # Split data by video_id to prevent data leakage
    video_ids = metadata['video_id'].unique()
    import numpy as np
    np.random.seed(split_seed)
    np.random.shuffle(video_ids)
    
    n_train = int(len(video_ids) * 0.8)
    n_val = int(len(video_ids) * 0.1)
    
    train_ids = video_ids[:n_train]
    val_ids = video_ids[n_train:n_train + n_val]
    test_ids = video_ids[n_train + n_val:]
    
    train_df = metadata[metadata['video_id'].isin(train_ids)]
    val_df = metadata[metadata['video_id'].isin(val_ids)]
    test_df = metadata[metadata['video_id'].isin(test_ids)]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} images from {len(train_ids)} videos")
    print(f"  Val:   {len(val_df)} images from {len(val_ids)} videos")
    print(f"  Test:  {len(test_df)} images from {len(test_ids)} videos")
    
    def df_to_dataset(df: pd.DataFrame, split_name: str) -> Dataset:
        """Convert DataFrame to HuggingFace Dataset with images."""
        records = []
        
        print(f"\nProcessing {split_name} split...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            img_path = data_dir / row['image_path']
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
                
            # Map label to binary (1 = REAL, 0 = FAKE)
            binary_label = 1 if row['label'] == 'REAL' else 0
            
            records.append({
                'image': str(img_path),
                'label': binary_label,
                'category': row['category'],
                'video_id': str(row['video_id']),
                'frame_number': int(row['frame_number']),
                'label_text': row['label'],
            })
        
        # Create dataset with proper features
        features = Features({
            'image': HFImage(),
            'label': Value('int64'),
            'category': Value('string'),
            'video_id': Value('string'),
            'frame_number': Value('int64'),
            'label_text': Value('string'),
        })
        
        return Dataset.from_dict(
            {k: [r[k] for r in records] for k in records[0].keys()},
            features=features
        )
    
    # Create datasets for each split
    train_dataset = df_to_dataset(train_df, 'train')
    val_dataset = df_to_dataset(val_df, 'validation')
    test_dataset = df_to_dataset(test_df, 'test')
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset,
    })


def push_to_hub(
    dataset: DatasetDict,
    repo_id: str,
    token: str,
    private: bool = False,
    commit_message: str = "Upload FaceForensics++ image dataset"
) -> None:
    """Push dataset to Hugging Face Hub.
    
    Args:
        dataset: DatasetDict to push
        repo_id: Repository ID (username/repo-name)
        token: HuggingFace API token
        private: Whether to make the repo private
        commit_message: Commit message for the push
    """
    print(f"\nPushing dataset to {repo_id}...")
    
    # Create repo if it doesn't exist
    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Push dataset
    dataset.push_to_hub(
        repo_id,
        private=private,
        commit_message=commit_message,
        token=token,
    )
    
    print(f"\n✅ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")


def create_dataset_card(repo_id: str, data_dir: str) -> str:
    """Create a README.md (dataset card) content for the dataset."""
    data_dir = Path(data_dir)
    metadata = pd.read_csv(data_dir / "image_dataset_metadata.csv")
    
    categories = metadata['category'].value_counts().to_dict()
    n_real = len(metadata[metadata['label'] == 'REAL'])
    n_fake = len(metadata[metadata['label'] == 'FAKE'])
    
    card = f"""---
license: cc-by-nc-4.0
task_categories:
- image-classification
tags:
- deepfake-detection
- faceforensics
- computer-vision
- binary-classification
size_categories:
- 100K<n<1M
---

# FaceForensics++ Image Dataset

This dataset contains preprocessed images from the FaceForensics++ benchmark for deepfake detection.

## Dataset Description

- **Total Images:** {len(metadata):,}
- **Real Images:** {n_real:,}
- **Fake Images:** {n_fake:,}
- **Imbalance Ratio:** {n_fake/n_real:.2f}:1 (fake:real)

### Categories

| Category | Count |
|----------|-------|
"""
    for cat, count in categories.items():
        card += f"| {cat} | {count:,} |\n"
    
    card += f"""
## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Example: iterate over training data
for sample in train_data:
    image = sample['image']  # PIL Image
    label = sample['label']  # 0 = FAKE, 1 = REAL
    category = sample['category']  # e.g., 'original', 'Deepfakes', etc.
```

## Dataset Structure

Each sample contains:
- `image`: The face image (PIL Image)
- `label`: Binary label (0 = FAKE, 1 = REAL)
- `category`: Original category (original, Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures, DeepFakeDetection)
- `video_id`: Source video identifier
- `frame_number`: Frame number within the video
- `label_text`: Text label ("REAL" or "FAKE")

## Splits

The dataset is split by video ID to prevent data leakage:
- **Train:** 80% of videos
- **Validation:** 10% of videos  
- **Test:** 10% of videos

## Citation

If you use this dataset, please cite the original FaceForensics++ paper:

```bibtex
@inproceedings{{roessler2019faceforensicspp,
  author = {{Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Niessner, Matthias}},
  title = {{FaceForensics++: Learning to Detect Manipulated Facial Images}},
  booktitle = {{International Conference on Computer Vision (ICCV)}},
  year = {{2019}}
}}
```
"""
    return card


def upload_dataset_card(repo_id: str, card_content: str, token: str) -> None:
    """Upload the dataset card to the repository."""
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    print("📄 Dataset card uploaded")


def main():
    parser = argparse.ArgumentParser(
        description="Push images_dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./images_dataset",
        help="Path to the images_dataset directory"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split"
    )
    parser.add_argument(
        "--skip_card",
        action="store_true",
        help="Skip uploading dataset card"
    )
    
    args = parser.parse_args()
    
    # Load environment variables from .env
    load_dotenv()
    
    # Get HuggingFace token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in environment. "
            "Please add HF_TOKEN=your_token to your .env file"
        )
    
    # Login to HuggingFace
    print("Logging in to Hugging Face...")
    login(token=hf_token)
    print("✅ Logged in successfully")
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    print("\n" + "=" * 60)
    print("Pushing FaceForensics++ Dataset to Hugging Face Hub")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Repository: {args.repo_id}")
    print(f"Private: {args.private}")
    print("=" * 60)
    
    # Create dataset
    dataset = create_hf_dataset(args.data_dir, split_seed=args.seed)
    
    # Push to hub
    push_to_hub(dataset, args.repo_id, token=hf_token, private=args.private)
    
    # Upload dataset card
    if not args.skip_card:
        card_content = create_dataset_card(args.repo_id, args.data_dir)
        upload_dataset_card(args.repo_id, card_content, token=hf_token)
    
    print("\n" + "=" * 60)
    print("✅ Upload complete!")
    print(f"View your dataset: https://huggingface.co/datasets/{args.repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
