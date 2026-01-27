"""
Reorganize dataset to avoid HuggingFace 10k files per directory limit
"""
import shutil
from pathlib import Path
from tqdm import tqdm
import math


def reorganize_directory(source_dir, max_files_per_dir=9000):
    """
    Reorganize a directory with too many files into subdirectories
    
    Args:
        source_dir: Directory with too many files
        max_files_per_dir: Maximum files per subdirectory (default: 9000 to be safe)
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Directory not found: {source_dir}")
        return
    
    # Get all image files
    image_files = sorted(source_path.glob('*.jpg'))
    total_files = len(image_files)
    
    print(f"\nReorganizing: {source_dir}")
    print(f"Total files: {total_files:,}")
    
    if total_files <= max_files_per_dir:
        print(f"✓ No reorganization needed (under {max_files_per_dir:,} files)")
        return
    
    # Calculate number of subdirectories needed
    num_subdirs = math.ceil(total_files / max_files_per_dir)
    print(f"Creating {num_subdirs} subdirectories...")
    
    # Create subdirectories and move files
    for i in range(num_subdirs):
        subdir = source_path / f"part_{i}"
        subdir.mkdir(exist_ok=True)
        
        # Calculate file range for this subdirectory
        start_idx = i * max_files_per_dir
        end_idx = min((i + 1) * max_files_per_dir, total_files)
        
        files_to_move = image_files[start_idx:end_idx]
        
        print(f"\nMoving {len(files_to_move):,} files to {subdir.name}/")
        for file_path in tqdm(files_to_move, desc=f"  Subdir {i}"):
            dest_path = subdir / file_path.name
            if not dest_path.exists():
                shutil.move(str(file_path), str(dest_path))
    
    print(f"✓ Reorganization complete for {source_dir}")


def reorganize_dataset(dataset_dir='celebdfv2_images', output_dir='celebdfv2_images_reorganized'):
    """
    Reorganize entire dataset for HuggingFace upload
    
    Args:
        dataset_dir: Original dataset directory
        output_dir: Output directory for reorganized dataset
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    print("=" * 60)
    print("REORGANIZING DATASET FOR HUGGINGFACE")
    print("=" * 60)
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Create output directory structure
    output_path.mkdir(exist_ok=True)
    
    # Directories to check
    directories_to_check = [
        dataset_path / 'train' / 'real',
        dataset_path / 'train' / 'fake',
        dataset_path / 'test' / 'real',
        dataset_path / 'test' / 'fake'
    ]
    
    for dir_path in directories_to_check:
        if not dir_path.exists():
            continue
        
        # Get relative path from dataset root
        rel_path = dir_path.relative_to(dataset_path)
        output_subdir = output_path / rel_path
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Count files in this directory
        files = list(dir_path.glob('*.jpg'))
        num_files = len(files)
        
        print(f"\n{rel_path}: {num_files:,} files")
        
        if num_files > 9000:
            # Need to split into subdirectories
            num_subdirs = math.ceil(num_files / 9000)
            print(f"  Splitting into {num_subdirs} subdirectories...")
            
            for i in range(num_subdirs):
                subdir = output_subdir / f"part_{i}"
                subdir.mkdir(exist_ok=True)
                
                start_idx = i * 9000
                end_idx = min((i + 1) * 9000, num_files)
                files_to_copy = files[start_idx:end_idx]
                
                print(f"  Copying {len(files_to_copy):,} files to {rel_path}/part_{i}/")
                for file_path in tqdm(files_to_copy, desc=f"    Part {i}"):
                    dest_path = subdir / file_path.name
                    if not dest_path.exists():
                        shutil.copy2(str(file_path), str(dest_path))
        else:
            # Copy directly
            print(f"  Copying files directly (under limit)...")
            for file_path in tqdm(files, desc=f"  {rel_path}"):
                dest_path = output_subdir / file_path.name
                if not dest_path.exists():
                    shutil.copy2(str(file_path), str(dest_path))
    
    # Copy README if exists
    readme_src = dataset_path / 'README.md'
    if readme_src.exists():
        shutil.copy2(str(readme_src), str(output_path / 'README.md'))
    
    print("\n" + "=" * 60)
    print("REORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"Reorganized dataset saved to: {output_dir}")
    print(f"\nNew structure:")
    
    # Print new structure
    for split in ['train', 'test']:
        for label in ['real', 'fake']:
            label_dir = output_path / split / label
            if label_dir.exists():
                subdirs = sorted(label_dir.glob('part_*'))
                if subdirs:
                    print(f"  {split}/{label}/")
                    for subdir in subdirs:
                        count = len(list(subdir.glob('*.jpg')))
                        print(f"    {subdir.name}/  ({count:,} files)")
                else:
                    count = len(list(label_dir.glob('*.jpg')))
                    print(f"  {split}/{label}/  ({count:,} files)")
    
    print("\n✓ Ready for HuggingFace upload!")
    print(f"\nUpload with:")
    print(f"  python push_to_huggingface.py --dataset-dir {output_dir} --repo-name celebdfv2_224 --username RohanRamesh")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize dataset for HuggingFace')
    parser.add_argument('--dataset-dir', type=str, default='celebdfv2_images',
                        help='Original dataset directory')
    parser.add_argument('--output-dir', type=str, default='celebdfv2_images_reorganized',
                        help='Output directory for reorganized dataset')
    
    args = parser.parse_args()
    
    reorganize_dataset(args.dataset_dir, args.output_dir)
