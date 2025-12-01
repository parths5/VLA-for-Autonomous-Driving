"""
Visualization utilities for autonomous driving data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Tuple
import cv2


def plot_image_with_waypoints(
    image: np.ndarray,
    waypoints: np.ndarray,
    speed: float,
    command: int,
    target_point: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot an image with waypoints overlaid.
    
    Args:
        image: RGB image (H, W, 3) or (3, H, W)
        waypoints: Waypoints array (N, 2)
        speed: Current speed in m/s
        command: Navigation command
        target_point: Target point (2,)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    # Convert image format if needed
    if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Command names
    command_names = {
        1: "Turn Left",
        2: "Turn Right",
        3: "Go Straight",
        4: "Follow Lane",
        5: "Lane Change Left",
        6: "Lane Change Right",
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot image
    ax1.imshow(image)
    ax1.set_title(f"RGB Image\nSpeed: {speed:.2f} m/s | Command: {command_names.get(command, 'Unknown')}")
    ax1.axis('off')
    
    # Plot bird's eye view with waypoints
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title("Bird's Eye View - Waypoints")
    
    # Plot ego vehicle at origin
    ego_circle = Circle((0, 0), 1.0, color='blue', fill=True, alpha=0.7, label='Ego Vehicle')
    ax2.add_patch(ego_circle)
    ax2.arrow(0, 0, 2, 0, head_width=0.5, head_length=0.5, fc='blue', ec='blue')
    
    # Plot waypoints
    if len(waypoints) > 0:
        ax2.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', linewidth=2, markersize=8, label='Waypoints')
        
        # Annotate waypoint indices
        for i, (x, y) in enumerate(waypoints):
            ax2.text(x, y, f'{i}', fontsize=8, ha='center', va='bottom')
    
    # Plot target point if provided
    if target_point is not None:
        ax2.plot(target_point[0], target_point[1], 'g*', markersize=15, label='Target Point')
    
    # Set axis limits
    if len(waypoints) > 0:
        max_range = max(np.abs(waypoints).max(), 10)
        ax2.set_xlim(-5, max_range + 5)
        ax2.set_ylim(-5, max_range + 5)
    else:
        ax2.set_xlim(-10, 30)
        ax2.set_ylim(-10, 30)
    
    ax2.legend(loc='upper right')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_video_from_samples(
    samples: list,
    output_path: str,
    fps: int = 10,
):
    """
    Create a video from a list of samples.
    
    Args:
        samples: List of sample dictionaries from dataset
        output_path: Path to save video
        fps: Frames per second
    """
    if len(samples) == 0:
        print("No samples to create video")
        return
    
    # Get image dimensions from first sample
    first_img = samples[0]['rgb']
    if first_img.shape[0] == 3:  # (C, H, W)
        height, width = first_img.shape[1], first_img.shape[2]
    else:  # (H, W, C)
        height, width = first_img.shape[0], first_img.shape[1]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for sample in samples:
        img = sample['rgb']
        
        # Convert format if needed
        if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        out.write(img_bgr)
    
    out.release()
    print(f"Video saved to {output_path}")


def print_dataset_statistics(dataset):
    """
    Print statistics about the dataset.
    
    Args:
        dataset: Bench2DriveDataset instance
    """
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Number of clips: {len(dataset.clips)}")
    
    # Sample a few examples to get statistics
    num_samples = min(100, len(dataset))
    speeds = []
    commands = []
    
    print(f"\nSampling {num_samples} examples for statistics...")
    for i in range(num_samples):
        sample = dataset[i]
        speeds.append(sample['speed'])
        commands.append(sample['command'])
    
    speeds = np.array(speeds)
    commands = np.array(commands)
    
    print(f"\nSpeed statistics:")
    print(f"  Mean: {speeds.mean():.2f} m/s")
    print(f"  Std: {speeds.std():.2f} m/s")
    print(f"  Min: {speeds.min():.2f} m/s")
    print(f"  Max: {speeds.max():.2f} m/s")
    
    print(f"\nCommand distribution:")
    command_names = {
        1: "Turn Left",
        2: "Turn Right",
        3: "Go Straight",
        4: "Follow Lane",
        5: "Lane Change Left",
        6: "Lane Change Right",
    }
    unique_cmds, counts = np.unique(commands, return_counts=True)
    for cmd, count in zip(unique_cmds, counts):
        print(f"  {command_names.get(cmd, f'Unknown ({cmd})')}: {count} ({count/len(commands)*100:.1f}%)")
    
    print("\n" + "=" * 60)

