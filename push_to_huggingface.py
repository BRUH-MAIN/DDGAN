"""
Push CelebDF-v2 processed dataset to HuggingFace Hub

Properly handles:
- part_* storage shards (collapsed, not treated as splits)
- Explicit ClassLabel schema
- Parquet-backed, streamable dataset
- Production-grade sharding
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, Image, ClassLabel
from huggingface_hub import HfApi
from tqdm import tqdm


# Dataset configuration
LABEL_NAMES = ["real", "fake"]
LABEL_MAP = {name: i for i, name in enumerate(LABEL_NAMES)}
LABEL_FEATURE = ClassLabel(names=LABEL_NAMES)


def build_split(root_dir: Path, split: str) -> Dataset:
    """
    Build a dataset split, correctly handling part_* subdirectories.
    
    Args:
        root_dir: Root dataset directory
        split: 'train' or 'test'
    
    Returns:
        HuggingFace Dataset with proper schema
    """
    images = []
    labels = []
    
    split_dir = root_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    
    for cls in LABEL_NAMES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        
        # Check if there are part_* subdirectories or direct images
        part_dirs = sorted([d for d in cls_dir.iterdir() if d.is_dir() and d.name.startswith('part_')])
        
        if part_dirs:
            # Has part_* subdirectories - iterate through them
            for part_dir in tqdm(part_dirs, desc=f"  {split}/{cls}"):
                for img in sorted(part_dir.glob("*.jpg")):
                    images.append(str(img))
                    labels.append(LABEL_MAP[cls])
        else:
            # No part_* subdirectories - get images directly
            for img in tqdm(sorted(cls_dir.glob("*.jpg")), desc=f"  {split}/{cls}"):
                images.append(str(img))
                labels.append(LABEL_MAP[cls])
    
    # Create dataset with proper schema
    dataset = Dataset.from_dict({
        "image": images,
        "label": labels,
    }).cast_column("image", Image()).cast_column("label", LABEL_FEATURE)
    
    return dataset


def push_dataset_to_hf(
    dataset_dir: str = 'celebdfv2_images_reorganized',
    repo_name: str = 'celebdfv2_224',
    username: str = 'RohanRamesh',
    private: bool = False,
    train_shards: int = 20,
    test_shards: int = 5,
    verify: bool = True
):
    """
    Build and push dataset to HuggingFace Hub with proper schema.
    
    Args:
        dataset_dir: Local directory containing the dataset
        repo_name: Name of the HuggingFace repository
        username: HuggingFace username
        private: Whether to make the repo private
        train_shards: Number of Parquet shards for train split
        test_shards: Number of Parquet shards for test split
        verify: Whether to verify the upload by streaming
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
    print(f"Train shards: {train_shards}")
    print(f"Test shards: {test_shards}")
    print("=" * 60 + "\n")
    
    # Check if dataset directory exists
    root_dir = Path(dataset_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Build dataset splits
    print("Building dataset splits...")
    print("(This correctly handles part_* subdirectories)\n")
    
    print("Building train split...")
    train_dataset = build_split(root_dir, "train")
    print(f"  ✓ Train: {len(train_dataset):,} images\n")
    
    print("Building test split...")
    test_dataset = build_split(root_dir, "test")
    print(f"  ✓ Test: {len(test_dataset):,} images\n")
    
    # Create DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })
    
    # Print dataset info
    print("Dataset schema:")
    print(f"  {dataset}")
    print(f"  Features: {dataset['train'].features}\n")
    
    # Print class distribution
    train_labels = train_dataset['label']
    test_labels = test_dataset['label']
    
    train_real = sum(1 for l in train_labels if l == 0)
    train_fake = sum(1 for l in train_labels if l == 1)
    test_real = sum(1 for l in test_labels if l == 0)
    test_fake = sum(1 for l in test_labels if l == 1)
    
    print("Class distribution:")
    print(f"  Train: real={train_real:,}, fake={train_fake:,}")
    print(f"  Test:  real={test_real:,}, fake={test_fake:,}")
    print(f"  Total: {len(train_dataset) + len(test_dataset):,} images\n")
    
    # Push to Hub
    repo_id = f"{username}/{repo_name}"
    print(f"Pushing to HuggingFace Hub: {repo_id}")
    print("(This creates proper Parquet shards for streaming)\n")
    
    try:
        dataset.push_to_hub(
            repo_id,
            token=hf_token,
            private=private,
            num_shards={"train": train_shards, "test": test_shards}
        )
        print(f"\n✓ Dataset pushed successfully!")
        print(f"  URL: https://huggingface.co/datasets/{repo_id}\n")
        
    except Exception as e:
        print(f"\n✗ Error pushing dataset: {e}")
        raise
    
    # Verification step
    if verify:
        print("=" * 60)
        print("VERIFICATION (streaming mode)")
        print("=" * 60)
        
        try:
            from datasets import load_dataset as load_ds
            
            print(f"Loading {repo_id} in streaming mode...")
            ds = load_ds(repo_id, streaming=True)
            
            # Get first sample from each split
            train_sample = next(iter(ds["train"]))
            test_sample = next(iter(ds["test"]))
            
            print(f"\n✓ Streaming verification passed!")
            print(f"  Train sample: image={type(train_sample['image'])}, label={train_sample['label']}")
            print(f"  Test sample: image={type(test_sample['image'])}, label={test_sample['label']}")
            print(f"\nDataset is production-ready and streamable!")
            
        except Exception as e:
            print(f"\n⚠ Verification failed: {e}")
            print("Dataset was pushed but may not be streamable.")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Push CelebDF-v2 dataset to HuggingFace (proper Parquet format)')
    parser.add_argument('--dataset-dir', type=str, default='celebdfv2_images_reorganized',
                        help='Path to dataset directory')
    parser.add_argument('--repo-name', type=str, default='celebdfv2_224',
                        help='Name of HuggingFace repository')
    parser.add_argument('--username', type=str, default='RohanRamesh',
                        help='HuggingFace username')
    parser.add_argument('--private', action='store_true',
                        help='Make repository private')
    parser.add_argument('--train-shards', type=int, default=20,
                        help='Number of Parquet shards for train split')
    parser.add_argument('--test-shards', type=int, default=5,
                        help='Number of Parquet shards for test split')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification step')
    
    args = parser.parse_args()
    
    push_dataset_to_hf(
        dataset_dir=args.dataset_dir,
        repo_name=args.repo_name,
        username=args.username,
        private=args.private,
        train_shards=args.train_shards,
        test_shards=args.test_shards,
        verify=not args.no_verify
    )
