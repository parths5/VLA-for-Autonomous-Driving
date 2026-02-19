"""
Bench2Drive dataset loader for VLAD project.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Get logger (inherits configuration from parent)
logger = logging.getLogger(__name__)

from utils.bench2drive_utils import (
    load_annotation,
    get_ego_transform,
    get_ego_speed,
    compute_waypoints_from_trajectory,
    world_to_ego
)


def bench2drive_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching Bench2Drive history samples.

    Args:
        batch: List of sample dictionaries from __getitem__

    Returns:
        Batched dictionary with:
            - history_state: (B, history_len, 6) torch.Tensor float32
            - history_ts: (B, history_len) torch.Tensor float32
            - history_image_embeddings: (B, history_len, ...) torch.Tensor float32
            - waypoints: (B, num_waypoints, 2) torch.Tensor float32
    """
    # Stack numpy arrays and convert to tensors
    batched = {
        'history_state': torch.stack([i['history_state'] for i in batch]),
        'history_ts': torch.stack([i['history_ts'] for i in batch]).unsqueeze(-1),
        'history_image_embeddings': torch.stack([i['history_image_embeddings'] for i in batch]),
        'waypoints': torch.stack([i['waypoints'] for i in batch]),
    }
    return batched


class Bench2DriveHistoryDataset(Dataset):
    """
    PyTorch Dataset for Bench2Drive data.
    Outputs data in SimLingo-compatible format.
    """
    
    def __init__(
        self,
        data_root: str,
        embeddings_root: str,
        embeddings_suffix: str,
        history_len: int,
        clips: List[str],
        num_waypoints: int = 10,
        waypoint_spacing: float = 0.2,
        fps: float = 10.0,
        skip_first_n_frames: int = 10,
        skip_last_n_frames: int = 30,
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
        self.embeddings_root = Path(embeddings_root)
        self.embeddings_suffix = embeddings_suffix
        self.history_len = history_len
        self.clips = clips
        self.num_waypoints = num_waypoints
        self.waypoint_spacing = waypoint_spacing
        self.fps = fps
        self.skip_first_n_frames = skip_first_n_frames
        self.skip_last_n_frames = skip_last_n_frames
        
        # Build index of all valid frames and preload tensors
        self.samples = self._build_and_preload_samples()
        
    
    def _validate_annotation(self, anno: Dict, frame_idx: int, current_frame_idx: int) -> bool:
        """
        Validate annotation data for NaN/None values.
        
        Args:
            anno: Annotation dictionary
            frame_idx: Frame index being validated
            current_frame_idx: Current sample frame index (for relative validation)
            
        Returns:
            True if annotation is valid, False otherwise
        """
        # Check critical fields based on frame type
        if frame_idx == current_frame_idx:
            # Current frame needs x, y, theta, speed
            x_val = anno.get('x')
            y_val = anno.get('y')
            theta_val = anno.get('theta')
            speed_val = anno.get('speed')
            
            if x_val is None or y_val is None or theta_val is None or speed_val is None:
                logger.debug(f"Invalid current frame {frame_idx}: None in critical fields")
                return False
            if (np.isnan(x_val) or np.isnan(y_val) or
                np.isnan(theta_val) or np.isnan(speed_val)):
                logger.debug(f"Invalid current frame {frame_idx}: NaN in critical fields (x={x_val}, y={y_val}, theta={theta_val}, speed={speed_val})")
                return False
        else:
            # History and future frames need x, y, speed (theta not needed)
            x_val = anno.get('x')
            y_val = anno.get('y')
            speed_val = anno.get('speed')
            
            if x_val is None or y_val is None or speed_val is None:
                logger.debug(f"Invalid frame {frame_idx}: None in x, y, or speed")
                return False
            if np.isnan(x_val) or np.isnan(y_val) or np.isnan(speed_val):
                logger.debug(f"Invalid frame {frame_idx}: NaN in x, y, or speed (x={x_val}, y={y_val}, speed={speed_val})")
                return False
        
        return True
    
    def _build_and_preload_samples(self) -> List[Dict]:
        """
        Build an index of all valid samples and preload all tensors into memory in a single pass.
        Each sample is a dictionary with clip path and frame index.
        """
        samples = []

        # Initialize cache dictionaries
        self.embeddings_cache = {}
        self.annotations_cache = {}
        
        for clip_name in tqdm(self.clips, desc=f"Processing clips"):
            clip_path = self.data_root / clip_name
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
            start_frame = self.skip_first_n_frames + self.history_len
            # Need enough future frames for waypoints
            frames_needed = int(self.num_waypoints * self.waypoint_spacing * self.fps)
            end_frame = num_frames - max(self.skip_last_n_frames, frames_needed)
            
            if end_frame <= start_frame:
                print(f"Warning: Skipping {clip_path.name}, not enough frames")
                continue
            
            # Add all valid frames from this clip
            for frame_idx in range(start_frame, end_frame):
                # Collect all frames needed for this sample
                frames_to_load = set()
                
                # Current frame
                frames_to_load.add(frame_idx)
                
                # History frames (going backwards from current frame)
                for i in range(self.history_len):
                    frames_to_load.add(frame_idx - i)

                # Future frames for waypoint computation
                max_future = int(self.num_waypoints * self.waypoint_spacing * self.fps)
                for i in range(max_future + 1):
                    if frame_idx + i < num_frames:
                        frames_to_load.add(frame_idx + i)
                
                # Load and validate all required frames
                valid_sample = True
                
                for load_frame_idx in sorted(frames_to_load):
                    key = (str(clip_path), load_frame_idx)
                    
                    # Load embedding if not already cached
                    if key not in self.embeddings_cache:
                        embedding_clip_path = self.embeddings_root / clip_path.relative_to(self.data_root)
                        embedding_path = embedding_clip_path / "camera" / "rgb_front" / f"{load_frame_idx:05d}{self.embeddings_suffix}.pt"
                        
                        if not embedding_path.exists():
                            logger.debug(f"Missing embedding for frame {load_frame_idx} in {clip_path.name}")
                            valid_sample = False
                            break
                        
                        try:
                            self.embeddings_cache[key] = torch.load(embedding_path)
                        except Exception as e:
                            logger.warning(f"Failed to load embedding {embedding_path}: {e}")
                            valid_sample = False
                            break
                    
                    # Load annotation if not already cached, or if this is the current frame (need to revalidate)
                    if key not in self.annotations_cache or load_frame_idx == frame_idx:
                        anno_path = clip_path / "anno" / f"{load_frame_idx:05d}.json.gz"
                        
                        if not anno_path.exists():
                            logger.debug(f"Missing annotation for frame {load_frame_idx} in {clip_path.name}")
                            valid_sample = False
                            break
                        
                        try:
                            anno = load_annotation(str(anno_path))
                            
                            # Validate annotation data
                            if not self._validate_annotation(anno, load_frame_idx, frame_idx):
                                logger.debug(f"Invalid annotation for frame {load_frame_idx} in {clip_path.name}")
                                valid_sample = False
                                break
                            
                            self.annotations_cache[key] = anno
                        except Exception as e:
                            logger.warning(f"Failed to load annotation {anno_path}: {e}")
                            valid_sample = False
                            break
                
                if valid_sample:
                    samples.append({
                        'clip_path': clip_path,
                        'frame_idx': frame_idx,
                        'num_frames': num_frames
                    })

        logger.info(f"Dataset initialized with {len(samples)} samples from {len(self.clips)} clips")
        logger.info(f"Preloaded {len(self.embeddings_cache)} embeddings and {len(self.annotations_cache)} annotations")
        
        return samples

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - history_state: Historical states (history_len, 6) float32 [x_ego, y_ego, speed, throttle, steer, brake]
                - history_ts: Historical timestamps (history_len,) float32 [relative time in seconds]
                - history_image_embeddings: Historical image embeddings (history_len, ...) float32
                - waypoints: Future waypoints (num_waypoints, 2) float32
        """        
        sample_info = self.samples[idx]
        clip_path = Path(sample_info['clip_path'])
        frame_idx = sample_info['frame_idx']
        num_frames = sample_info['num_frames']

        # Load current frame annotation from cache
        key = (str(clip_path), frame_idx)
        current_anno = self.annotations_cache[key]
        current_x_ego, current_y_ego, current_ego_theta = get_ego_transform(current_anno)

        # Calculate how many future frames are actually available
        # How many frames available from current frame onwards (including current frame)
        max_future_frames = num_frames - frame_idx

        # Calculate how many frames we need for the desired waypoints (including current frame)
        future_frames_needed = int(self.num_waypoints * self.waypoint_spacing * self.fps) + 1

        # Use the minimum of what we need and what's available
        future_frames_to_load = min(future_frames_needed, max_future_frames)

        # Load all available future annotations from cache (including current frame as first element)
        future_annotations = []
        for i in range(future_frames_to_load):
            key = (str(clip_path), frame_idx + i)
            if key in self.annotations_cache:
                future_annotations.append(self.annotations_cache[key])
            else:
                # Stop if annotation doesn't exist
                break

        history_state = []
        history_ts = []
        history_embeddings = []
        for i in reversed(range(self.history_len)):
            # Load history annotation from cache
            key = (str(clip_path), frame_idx - i)
            history_anno = self.annotations_cache[key]

            history_x_world, history_y_world, _ = get_ego_transform(history_anno)
            history_x_ego, history_y_ego = world_to_ego(history_x_world, history_y_world, current_x_ego, current_y_ego, current_ego_theta)
            history_speed = get_ego_speed(history_anno)
            history_t = - i / self.fps
            state = [history_x_ego, history_y_ego, history_speed, history_anno['throttle'], history_anno['steer'], history_anno['brake']]

            # Check for NaNs in individual state components
            state_array = np.asarray(state)
            if np.isnan(state_array).any():
                state_names = ['x_ego', 'y_ego', 'speed', 'throttle', 'steer', 'brake']
                nan_mask = np.isnan(state_array)
                nan_fields = [state_names[j] for j in range(len(state_names)) if nan_mask[j]]
                logger.warning(f"NaN in state for clip {clip_path.name}, frame {frame_idx - i}, fields: {nan_fields}, values: {state}")

            history_state.append(state_array)
            history_ts.append(history_t)

            # Load embedding from cache
            embedding = self.embeddings_cache[key]

            # Check for NaNs in embeddings
            if torch.isnan(embedding).any():
                nan_count = torch.isnan(embedding).sum().item()
                logger.warning(f"NaN in embedding for clip {clip_path.name}, frame {frame_idx - i}: "
                             f"shape={embedding.shape}, nan_count={nan_count}")

            history_embeddings.append(embedding)
                
        # Calculate actual number of waypoints we can extract from available frames
        actual_num_waypoints = min(
            self.num_waypoints,
            len(future_annotations) if len(future_annotations) > 0 else 0
        )

        future_waypoints = compute_waypoints_from_trajectory(
            annotations=future_annotations,
            current_idx=0,
            num_waypoints=actual_num_waypoints,
            waypoint_spacing=self.waypoint_spacing,
            fps=self.fps,
        )

        # Check for NaNs in waypoints
        if np.isnan(future_waypoints).any():
            nan_count = np.isnan(future_waypoints).sum()
            logger.warning(f"NaN in waypoints for clip {clip_path.name}, frame {frame_idx}: "
                         f"shape={future_waypoints.shape}, nan_count={nan_count}")
        
        # Create return dict
        result = {
            'history_state': torch.tensor(np.asarray(history_state), dtype=torch.float32),
            'history_ts': torch.tensor(history_ts, dtype=torch.float32),
            'history_image_embeddings': torch.stack(history_embeddings).type(torch.float32),
            'waypoints': torch.tensor(future_waypoints).type(torch.float32),
        }

        # Check for NaNs in the data
        for key, value in result.items():
            if torch.isnan(value).any():
                nan_count = torch.isnan(value).sum().item()
                logger.warning(f"NaN detected in '{key}' for clip {clip_path.name}, frame {frame_idx}: "
                             f"shape={value.shape}, nan_count={nan_count}")

        return result

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset with proper batching.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch

        Returns:
            DataLoader configured with custom collate function
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=bench2drive_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


if __name__ == "__main__":
    # Import the unoptimized dataset for comparison
    from bench2drive_history_dataset import Bench2DriveHistoryDataset as UnoptimizedDataset

    test_clips = [
        "AccidentTwoWays_Town12_Route1109_Weather9",  # Had NaN theta and waypoints
    ]

    print("\n" + "="*80)
    print("TESTING OPTIMIZED DATASET (with preloading)")
    print("="*80)

    # Create optimized dataset (current class)
    optimized_dataset = Bench2DriveHistoryDataset(
        "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_base/extracted",
        "/ocean/projects/cis250252p/shared/afadia/VLAD/data/bench2drive_base/extracted",
        "_Qwen3-VL-2B-Instruct_features",
        8,
        test_clips,
    )

    print("\n" + "="*80)
    print("TESTING UNOPTIMIZED DATASET (on-demand loading)")
    print("="*80)

    # Create unoptimized dataset for comparison
    unoptimized_dataset = UnoptimizedDataset(
        "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_base/extracted",
        "/ocean/projects/cis250252p/shared/afadia/VLAD/data/bench2drive_base/extracted",
        "_Qwen3-VL-2B-Instruct_features",
        8,
        test_clips,
    )

    print("\n" + "="*80)
    print("VERIFYING OUTPUTS ARE IDENTICAL")
    print("="*80)

    # Check that both datasets have the same length
    assert len(optimized_dataset) == len(unoptimized_dataset), \
        f"Dataset lengths differ: optimized={len(optimized_dataset)}, unoptimized={len(unoptimized_dataset)}"
    print(f"✓ Both datasets have {len(optimized_dataset)} samples")

    # Compare outputs element by element
    mismatches = 0
    for idx in tqdm(range(len(optimized_dataset)), desc="Comparing samples"):
        opt_sample = optimized_dataset[idx]
        unopt_sample = unoptimized_dataset[idx]

        # Compare each key
        for key in opt_sample.keys():
            if key not in unopt_sample:
                print(f"✗ Key '{key}' missing in unoptimized dataset at idx {idx}")
                mismatches += 1
                continue

            opt_val = opt_sample[key]
            unopt_val = unopt_sample[key]

            # Check shapes match
            if opt_val.shape != unopt_val.shape:
                print(f"✗ Shape mismatch for '{key}' at idx {idx}: "
                      f"optimized={opt_val.shape}, unoptimized={unopt_val.shape}")
                mismatches += 1
                continue

            # Check values match (allowing for small floating point differences)
            if not torch.allclose(opt_val, unopt_val, rtol=1e-5, atol=1e-7, equal_nan=True):
                max_diff = torch.abs(opt_val - unopt_val).max().item()
                print(f"✗ Value mismatch for '{key}' at idx {idx}: max_diff={max_diff}")
                mismatches += 1

    print("\n" + "="*80)
    if mismatches == 0:
        print("✓ SUCCESS: All outputs are identical!")
    else:
        print(f"✗ FAILURE: Found {mismatches} mismatches")
    print("="*80)
