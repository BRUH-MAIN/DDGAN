"""
FaceForensics++ Video to Image Dataset Converter

This script extracts frames from videos, detects faces, crops them, and creates
an organized image dataset suitable for deepfake detection training.

Methods implemented:
1. Uniform frame sampling (configurable N frames per video)
2. Face detection using MTCNN
3. Face cropping and resizing
4. Metadata CSV generation
"""

import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not available. Install with: pip install facenet-pytorch")


class VideoToImageConverter:
    """Convert FaceForensics++ videos to image dataset"""
    
    def __init__(self, 
                 dataset_root,
                 output_root,
                 frames_per_video=32,
                 output_size=(224, 224),
                 use_face_detection=True,
                 min_face_size=20,
                 save_format='jpg',
                 quality=95):
        """
        Initialize converter
        
        Args:
            dataset_root: Path to FaceForensics++ dataset root
            output_root: Path to save extracted images
            frames_per_video: Number of frames to extract per video
            output_size: Tuple (width, height) for output images
            use_face_detection: Whether to detect and crop faces
            min_face_size: Minimum face size for detection
            save_format: Image format ('jpg' or 'png')
            quality: JPEG quality (1-100)
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.frames_per_video = frames_per_video
        self.output_size = output_size
        self.use_face_detection = use_face_detection
        self.min_face_size = min_face_size
        self.save_format = save_format.lower()
        self.quality = quality
        
        # Initialize face detector
        self.face_detector = None
        if self.use_face_detection and MTCNN_AVAILABLE:
            try:
                self.face_detector = MTCNN(
                    keep_all=False,
                    min_face_size=min_face_size,
                    device='cuda' if self._check_cuda() else 'cpu'
                )
                print(f"✓ Face detector initialized (device: {'cuda' if self._check_cuda() else 'cpu'})")
            except Exception as e:
                print(f"Warning: Could not initialize MTCNN: {e}")
                self.face_detector = None
        
        # Categories in FaceForensics++
        self.categories = [
            'original',
            'Deepfakes',
            'Face2Face',
            'FaceSwap',
            'NeuralTextures',
            'FaceShifter',
            'DeepFakeDetection'
        ]
        
        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        
    def _check_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def extract_frames_uniform(self, video_path, num_frames):
        """
        Extract frames uniformly distributed across video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frames (numpy arrays)
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Calculate frame indices to extract
        if total_frames <= num_frames:
            # If video has fewer frames than requested, extract all
            frame_indices = list(range(total_frames))
        else:
            # Uniformly sample frames
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((idx, frame))
        
        cap.release()
        return frames
    
    def detect_and_crop_face(self, frame):
        """
        Detect face in frame and return cropped face
        
        Args:
            frame: Input frame (RGB numpy array)
            
        Returns:
            Cropped face image or None if no face detected
        """
        if self.face_detector is None:
            # No face detection, return resized frame
            img = Image.fromarray(frame)
            img = img.resize(self.output_size, Image.LANCZOS)
            return np.array(img)
        
        try:
            # Detect face
            img = Image.fromarray(frame)
            boxes, probs = self.face_detector.detect(img)
            
            if boxes is not None and len(boxes) > 0:
                # Get the face with highest confidence
                best_idx = np.argmax(probs)
                box = boxes[best_idx]
                
                # Expand box slightly (10% padding)
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                padding = 0.1
                
                x1 = max(0, int(x1 - width * padding))
                y1 = max(0, int(y1 - height * padding))
                x2 = min(frame.shape[1], int(x2 + width * padding))
                y2 = min(frame.shape[0], int(y2 + height * padding))
                
                # Crop and resize
                face = frame[y1:y2, x1:x2]
                face_img = Image.fromarray(face)
                face_img = face_img.resize(self.output_size, Image.LANCZOS)
                return np.array(face_img)
            else:
                # No face detected, return resized frame
                img = img.resize(self.output_size, Image.LANCZOS)
                return np.array(img)
                
        except Exception as e:
            # On error, return resized frame
            img = Image.fromarray(frame)
            img = img.resize(self.output_size, Image.LANCZOS)
            return np.array(img)
    
    def process_video(self, video_path, category, video_id):
        """
        Process a single video and extract images
        
        Args:
            video_path: Path to video file
            category: Category name (e.g., 'Deepfakes', 'original')
            video_id: Video identifier
            
        Returns:
            List of metadata dictionaries for saved images
        """
        if not video_path.exists():
            return []
        
        # Extract frames
        frames = self.extract_frames_uniform(video_path, self.frames_per_video)
        
        if len(frames) == 0:
            return []
        
        # Create output directory for this category
        category_dir = self.output_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for i, (frame_idx, frame) in enumerate(frames):
            # Detect and crop face
            if self.use_face_detection:
                processed_frame = self.detect_and_crop_face(frame)
            else:
                # Just resize
                img = Image.fromarray(frame)
                img = img.resize(self.output_size, Image.LANCZOS)
                processed_frame = np.array(img)
            
            # Generate output filename
            output_filename = f"{video_id}_frame{i:04d}.{self.save_format}"
            output_path = category_dir / output_filename
            
            # Save image
            img = Image.fromarray(processed_frame)
            if self.save_format == 'jpg':
                img.save(output_path, 'JPEG', quality=self.quality)
            else:
                img.save(output_path, 'PNG')
            
            # Store metadata
            label = 'REAL' if category == 'original' else 'FAKE'
            metadata.append({
                'image_path': str(output_path.relative_to(self.output_root)),
                'video_path': str(video_path.relative_to(self.dataset_root)),
                'category': category,
                'label': label,
                'video_id': video_id,
                'frame_number': i,
                'original_frame_idx': frame_idx,
                'width': self.output_size[0],
                'height': self.output_size[1]
            })
        
        return metadata
    
    def process_category(self, category):
        """
        Process all videos in a category
        
        Args:
            category: Category name
            
        Returns:
            List of metadata dictionaries
        """
        category_path = self.dataset_root / category
        
        if not category_path.exists():
            print(f"Warning: Category '{category}' not found at {category_path}")
            return []
        
        # Get all video files
        video_files = sorted(category_path.glob('*.mp4'))
        
        if len(video_files) == 0:
            print(f"Warning: No videos found in category '{category}'")
            return []
        
        print(f"\nProcessing category: {category} ({len(video_files)} videos)")
        
        all_metadata = []
        
        for video_path in tqdm(video_files, desc=f"  {category}"):
            video_id = video_path.stem
            metadata = self.process_video(video_path, category, video_id)
            all_metadata.extend(metadata)
        
        return all_metadata
    
    def convert_dataset(self, categories=None):
        """
        Convert entire dataset to images
        
        Args:
            categories: List of categories to process (None for all)
            
        Returns:
            DataFrame with metadata
        """
        if categories is None:
            categories = self.categories
        
        print("=" * 70)
        print("FaceForensics++ Video to Image Conversion")
        print("=" * 70)
        print(f"Dataset root: {self.dataset_root}")
        print(f"Output root: {self.output_root}")
        print(f"Frames per video: {self.frames_per_video}")
        print(f"Output size: {self.output_size}")
        print(f"Face detection: {self.use_face_detection}")
        print(f"Format: {self.save_format.upper()}")
        print("=" * 70)
        
        all_metadata = []
        
        for category in categories:
            metadata = self.process_category(category)
            all_metadata.extend(metadata)
        
        # Create DataFrame
        df = pd.DataFrame(all_metadata)
        
        # Save metadata CSV
        csv_path = self.output_root / 'image_dataset_metadata.csv'
        df.to_csv(csv_path, index=False)
        
        print("\n" + "=" * 70)
        print("Conversion Complete!")
        print("=" * 70)
        print(f"Total images extracted: {len(df)}")
        print(f"REAL images: {len(df[df['label'] == 'REAL'])}")
        print(f"FAKE images: {len(df[df['label'] == 'FAKE'])}")
        print(f"\nMetadata saved to: {csv_path}")
        print(f"Images saved to: {self.output_root}")
        print("=" * 70)
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Convert FaceForensics++ videos to image dataset'
    )
    
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/run/media/rohan/New Volume/FaceForensics++_C23',
        help='Path to FaceForensics++ dataset root'
    )
    
    parser.add_argument(
        '--output_root',
        type=str,
        default='/run/media/rohan/New Volume/FaceForensics++_C23/images_dataset',
        help='Path to save extracted images'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=32,
        help='Number of frames to extract per video (default: 32)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Output image size (square) (default: 224)'
    )
    
    parser.add_argument(
        '--no_face_detection',
        action='store_true',
        help='Disable face detection (extract full frames)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['jpg', 'png'],
        default='jpg',
        help='Output image format (default: jpg)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality 1-100 (default: 95)'
    )
    
    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='Specific categories to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = VideoToImageConverter(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        frames_per_video=args.frames,
        output_size=(args.size, args.size),
        use_face_detection=not args.no_face_detection,
        save_format=args.format,
        quality=args.quality
    )
    
    # Convert dataset
    df = converter.convert_dataset(categories=args.categories)
    
    print(f"\n✓ Dataset conversion successful!")
    print(f"  {len(df)} images created")


if __name__ == '__main__':
    main()
