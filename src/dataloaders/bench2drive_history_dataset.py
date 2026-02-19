"""
Bench2Drive dataset loader for VLAD project.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
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
    get_navigation_command,
    get_target_point,
    compute_waypoints_from_trajectory,
    map_command_to_simlingo,
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
        
        # Build index of all valid frames
        self.samples = self._build_sample_index()
        
    
    def _build_sample_index(self) -> List[Dict]:
        """
        Build an index of all valid samples.
        Each sample is a dictionary with clip path and frame index.
        """
        samples = []
        
        for clip_name in tqdm(self.clips, desc=f"Indexing clips"):
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
                # Check if embedding exists
                embedding_clip_path = self.embeddings_root / clip_path.relative_to(self.data_root)
                embedding_path = embedding_clip_path / "camera" / "rgb_front" / f"{frame_idx:05d}{self.embeddings_suffix}.pt"
                if not embedding_path.exists():
                    continue

                # Load current frame annotation to check for NaN/None values
                anno_path = clip_path / "anno" / f"{frame_idx:05d}.json.gz"
                try:
                    current_anno = load_annotation(str(anno_path))

                    # Check critical fields for None or NaN
                    x_val = current_anno.get('x')
                    y_val = current_anno.get('y')
                    theta_val = current_anno.get('theta')
                    speed_val = current_anno.get('speed')

                    # Check for None values
                    if x_val is None or y_val is None or theta_val is None or speed_val is None:
                        logger.debug(f"Skipping frame {frame_idx} in {clip_path.name}: None in critical fields")
                        continue

                    # Check for NaN values
                    if (np.isnan(x_val) or np.isnan(y_val) or
                        np.isnan(theta_val) or np.isnan(speed_val)):
                        logger.debug(f"Skipping frame {frame_idx} in {clip_path.name}: NaN in critical fields (x={x_val}, y={y_val}, theta={theta_val}, speed={speed_val})")
                        continue

                    # Also check history frames for NaN/None values
                    # Note: theta is NOT needed for history frames, only x, y, speed
                    skip_sample = False
                    for i in range(self.history_len):
                        hist_frame_idx = frame_idx - i
                        hist_anno_path = clip_path / "anno" / f"{hist_frame_idx:05d}.json.gz"
                        if not hist_anno_path.exists():
                            skip_sample = True
                            break

                        hist_anno = load_annotation(str(hist_anno_path))
                        hist_x = hist_anno.get('x')
                        hist_y = hist_anno.get('y')
                        hist_speed = hist_anno.get('speed')

                        # Check for None (only x, y, speed needed for history frames)
                        if hist_x is None or hist_y is None or hist_speed is None:
                            logger.debug(f"Skipping frame {frame_idx} in {clip_path.name}: None in history frame {hist_frame_idx}")
                            skip_sample = True
                            break

                        # Check for NaN (only x, y, speed needed for history frames)
                        if np.isnan(hist_x) or np.isnan(hist_y) or np.isnan(hist_speed):
                            logger.debug(f"Skipping frame {frame_idx} in {clip_path.name}: NaN in history frame {hist_frame_idx}")
                            skip_sample = True
                            break

                    if skip_sample:
                        continue

                    # Also check future frames for waypoint computation
                    # We need to check frames that will be used for waypoints
                    max_future_frame = frame_idx + int(self.num_waypoints * self.waypoint_spacing * self.fps)
                    if frame_idx == 71:  # Debug frame 71 specifically
                        logger.warning(f"DEBUG: Validating frame 71, will check future frames {frame_idx + 1} to {min(max_future_frame, num_frames - 1)}")
                    for future_idx in range(frame_idx + 1, min(max_future_frame + 1, num_frames)):
                        future_anno_path = clip_path / "anno" / f"{future_idx:05d}.json.gz"
                        if not future_anno_path.exists():
                            logger.warning(f"Skipping frame {frame_idx} in {clip_path.name}: future frame {future_idx} missing")
                            skip_sample = True
                            break

                        future_anno = load_annotation(str(future_anno_path))
                        future_x = future_anno.get('x')
                        future_y = future_anno.get('y')

                        # Check for None (x, y needed for waypoints, theta not needed)
                        if future_x is None or future_y is None:
                            logger.warning(f"Skipping frame {frame_idx} in {clip_path.name}: None in future frame {future_idx} (x={future_x}, y={future_y})")
                            skip_sample = True
                            break

                        # Check for NaN
                        if np.isnan(future_x) or np.isnan(future_y):
                            logger.warning(f"Skipping frame {frame_idx} in {clip_path.name}: NaN in future frame {future_idx} (x={future_x}, y={future_y})")
                            skip_sample = True
                            break

                    if skip_sample:
                        if frame_idx == 71:
                            logger.warning(f"DEBUG: Frame 71 SKIPPED due to future frame issues")
                        continue

                    if frame_idx == 71:
                        logger.warning(f"DEBUG: Frame 71 PASSED validation, will be added to dataset")

                except Exception as e:
                    logger.warning(f"Failed to load annotation for {clip_path.name} frame {frame_idx}: {e}")
                    continue

                samples.append({
                    'clip_path': clip_path,
                    'frame_idx': frame_idx,
                    'num_frames': num_frames
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
                - history_state: Historical states (history_len, 6) float32 [x_ego, y_ego, speed, throttle, steer, brake]
                - history_ts: Historical timestamps (history_len,) float32 [relative time in seconds]
                - history_image_embeddings: Historical image embeddings (history_len, ...) float32
                - waypoints: Future waypoints (num_waypoints, 2) float32
        """        
        sample_info = self.samples[idx]
        clip_path = Path(sample_info['clip_path'])
        frame_idx = sample_info['frame_idx']
        num_frames = sample_info['num_frames']

        # Load current frame annotation
        anno_path = clip_path / "anno" / f"{frame_idx:05d}.json.gz"
        current_anno = load_annotation(str(anno_path))
        current_x_ego, current_y_ego, current_ego_theta = get_ego_transform(current_anno)

        # Calculate how many future frames are actually available
        # How many frames available from current frame onwards (including current frame)
        max_future_frames = num_frames - frame_idx

        # Calculate how many frames we need for the desired waypoints (including current frame)
        future_frames_needed = int(self.num_waypoints * self.waypoint_spacing * self.fps) + 1

        # Use the minimum of what we need and what's available
        future_frames_to_load = min(future_frames_needed, max_future_frames)

        # Load all available future annotations (including current frame as first element)
        future_annotations = []
        for i in range(future_frames_to_load):
            anno_path_i = clip_path / "anno" / f"{frame_idx + i:05d}.json.gz"
            if anno_path_i.exists():
                future_annotations.append(load_annotation(str(anno_path_i)))
            else:
                # Stop if annotation doesn't exist
                break

        history_state = []
        history_ts = []
        history_embeddings = []
        for i in reversed(range(self.history_len)):
            anno_path_i = clip_path / "anno" / f"{frame_idx - i:05d}.json.gz"
            if not anno_path_i.exists():
                raise FileNotFoundError(f"Missing annotation file that should exist: {anno_path_i}")

            history_anno = load_annotation(str(anno_path_i))
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

            embedding_clip_path = self.embeddings_root / clip_path.relative_to(self.data_root)
            embedding_path = embedding_clip_path / "camera" / "rgb_front" / f"{frame_idx - i:05d}{self.embeddings_suffix}.pt"
            if not embedding_path.exists():
                raise FileNotFoundError(f"Missing embedding file that should exist: {embedding_path}")
            embedding = torch.load(embedding_path)

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
    test_clips = [
        "AccidentTwoWays_Town12_Route1109_Weather9",  # Had NaN theta and waypoints
    ]
    train_dataset = Bench2DriveHistoryDataset(
        "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_base/extracted",
        "/ocean/projects/cis250252p/shared/afadia/VLAD/data/bench2drive_base/extracted",
        "_Qwen3-VL-2B-Instruct_features",
        8,
        test_clips,
    )
    for elem in tqdm(train_dataset):
        isnan = torch.isnan(elem['waypoints'])
        # print(isnan)