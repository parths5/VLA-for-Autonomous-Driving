"""
Bench2Drive dataset loader for VLAD project.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bench2drive_utils import (
    load_annotation,
    get_ego_transform,
    get_ego_speed,
    get_navigation_command,
    get_target_point,
    compute_waypoints_from_trajectory,
    map_command_to_simlingo,
)
from dataloaders.transforms import ImagePreprocessor


class Bench2DriveDataset(Dataset):
    """
    PyTorch Dataset for Bench2Drive data.
    Outputs data in SimLingo-compatible format.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_waypoints: int = 11,
        waypoint_spacing: float = 0.2,
        fps: float = 10.0,
        skip_first_n_frames: int = 10,
        skip_last_n_frames: int = 30,
        img_augmentation: bool = True,
        img_augmentation_prob: float = 0.5,
        cut_bottom_quarter: bool = True,
    ):
        """
        Initialize Bench2Drive dataset.
        
        Args:
            data_root: Root directory containing extracted Bench2Drive clips
            split: Dataset split ('train' or 'val')
            num_waypoints: Number of future waypoints to predict
            waypoint_spacing: Time spacing between waypoints (seconds)
            fps: Frames per second of dataset
            skip_first_n_frames: Skip first N frames of each clip
            skip_last_n_frames: Skip last N frames (to ensure we have future waypoints)
            img_augmentation: Whether to apply image augmentation
            img_augmentation_prob: Probability of applying augmentations
            cut_bottom_quarter: Whether to cut bottom of image (car hood)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_waypoints = num_waypoints
        self.waypoint_spacing = waypoint_spacing
        self.fps = fps
        self.skip_first_n_frames = skip_first_n_frames
        self.skip_last_n_frames = skip_last_n_frames
        
        # Image preprocessor
        self.img_preprocessor = ImagePreprocessor(
            cut_bottom_quarter=cut_bottom_quarter,
            augment=img_augmentation,
            augment_prob=img_augmentation_prob,
        )
        
        # Find all clips
        self.clips = self._find_clips()
        
        # Build index of all valid frames
        self.samples = self._build_sample_index()
        
        print(f"[{split}] Loaded {len(self.samples)} samples from {len(self.clips)} clips")
    
    def _find_clips(self) -> List[Path]:
        """Find all clip directories in data_root."""
        clips = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        # For now, we don't have a train/val split in Bench2Drive Mini
        # All 10 clips are used for both train and val
        # In a real scenario, you'd split by clip or by town
        
        if self.split == "val":
            # Use first 2 clips for validation
            clips = clips[:2]
        else:
            # Use remaining clips for training
            clips = clips[2:]
        
        return clips
    
    def _build_sample_index(self) -> List[Dict]:
        """
        Build an index of all valid samples.
        Each sample is a dictionary with clip path and frame index.
        """
        samples = []
        
        for clip_path in tqdm(self.clips, desc=f"Indexing {self.split} clips"):
            # Check if clip has required directories
            anno_dir = clip_path / "anno"
            rgb_dir = clip_path / "camera" / "rgb_front"
            
            if not anno_dir.exists() or not rgb_dir.exists():
                print(f"Warning: Skipping {clip_path.name}, missing required directories")
                continue
            
            # Count frames
            anno_files = sorted(anno_dir.glob("*.json.gz"))
            num_frames = len(anno_files)
            
            # Calculate valid frame range
            start_frame = self.skip_first_n_frames
            # Need enough future frames for waypoints
            frames_needed = int(self.num_waypoints * self.waypoint_spacing * self.fps)
            end_frame = num_frames - max(self.skip_last_n_frames, frames_needed)
            
            if end_frame <= start_frame:
                print(f"Warning: Skipping {clip_path.name}, not enough frames")
                continue
            
            # Add all valid frames from this clip
            for frame_idx in range(start_frame, end_frame):
                samples.append({
                    'clip_path': clip_path,
                    'frame_idx': frame_idx,
                    'num_frames': num_frames,
                })
        
        return samples
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - rgb: RGB image (C, H, W) uint8
                - rgb_org_size: Original size RGB image (C, H, W) uint8
                - waypoints: Future waypoints (num_waypoints, 2) float32
                - speed: Current speed (float)
                - command: Navigation command (int)
                - target_point: Target waypoint (2,) float32
                - clip_name: Name of the clip (str)
                - frame_idx: Frame index (int)
        """
        # Disable threading for OpenCV (important for DataLoader)
        cv2.setNumThreads(0)
        
        sample_info = self.samples[idx]
        clip_path = sample_info['clip_path']
        frame_idx = sample_info['frame_idx']
        
        # Load current frame annotation
        anno_path = clip_path / "anno" / f"{frame_idx:05d}.json.gz"
        current_anno = load_annotation(str(anno_path))
        
        # Load all annotations for waypoint computation
        # We need current + future frames
        frames_needed = int(self.num_waypoints * self.waypoint_spacing * self.fps) + 1
        annotations = []
        for i in range(frames_needed):
            anno_path_i = clip_path / "anno" / f"{frame_idx + i:05d}.json.gz"
            annotations.append(load_annotation(str(anno_path_i)))
        
        # Load RGB image
        rgb_path = clip_path / "camera" / "rgb_front" / f"{frame_idx:05d}.jpg"
        rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Keep original size copy
        rgb_org_size = rgb_image.copy()
        
        # Preprocess image
        rgb_image = self.img_preprocessor(rgb_image, apply_augment=(self.split == 'train'))
        
        # Compute waypoints
        waypoints = compute_waypoints_from_trajectory(
            annotations=annotations,
            current_idx=0,  # Current frame is at index 0 in our loaded annotations
            num_waypoints=self.num_waypoints,
            waypoint_spacing=self.waypoint_spacing,
            fps=self.fps,
        )
        
        # Get other data
        speed = get_ego_speed(current_anno)
        command_near, command_far = get_navigation_command(current_anno)
        command = map_command_to_simlingo(command_near)
        
        # Get target point (already in ego-centric coordinates in Bench2Drive)
        x_target, y_target = get_target_point(current_anno)
        target_point = np.array([x_target, y_target], dtype=np.float32)
        
        # Convert images to (C, H, W) format
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        rgb_org_size = np.transpose(rgb_org_size, (2, 0, 1))
        
        return {
            'rgb': rgb_image,
            'rgb_org_size': rgb_org_size,
            'waypoints': waypoints,
            'speed': np.float32(speed),
            'command': command,
            'target_point': target_point,
            'clip_name': clip_path.name,
            'frame_idx': frame_idx,
        }


if __name__ == "__main__":
    
    data_root = "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_mini/extracted"
    
    print("Testing Bench2Drive Dataset...")
    print("=" * 50)
    
    # Create dataset
    dataset = Bench2DriveDataset(
        data_root=data_root,
        split="train",
        img_augmentation=False,  # Disable for testing
    )
    
    print(f"\nDataset size: {len(dataset)} samples")
    
    # Test loading a few samples
    print("\nTesting sample loading...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Clip: {sample['clip_name']}")
        print(f"  Frame: {sample['frame_idx']}")
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  Waypoints shape: {sample['waypoints'].shape}")
        print(f"  Speed: {sample['speed']:.2f} m/s")
        print(f"  Command: {sample['command']}")
        print(f"  Target point: {sample['target_point']}")
    
    print("\n" + "=" * 50)
    print("Dataset test completed successfully!")

