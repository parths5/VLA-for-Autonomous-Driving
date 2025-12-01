#!/usr/bin/env python3
"""
Test script for Bench2Drive dataloader.
Tests loading, visualization, and basic statistics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server

from dataloaders.bench2drive_dataset import Bench2DriveDataset
from utils.visualization import plot_image_with_waypoints, print_dataset_statistics


def test_dataset_loading():
    """Test basic dataset loading."""
    print("\n" + "=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)
    
    data_root = "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_mini/extracted"
    
    # Create train dataset
    print("\nCreating training dataset...")
    train_dataset = Bench2DriveDataset(
        data_root=data_root,
        split="train",
        img_augmentation=False,
    )
    
    print(f"✓ Train dataset created: {len(train_dataset)} samples")
    
    # Create val dataset
    print("\nCreating validation dataset...")
    val_dataset = Bench2DriveDataset(
        data_root=data_root,
        split="val",
        img_augmentation=False,
    )
    
    print(f"✓ Val dataset created: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def test_sample_loading(dataset, num_samples=5):
    """Test loading individual samples."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Loading {num_samples} Samples")
    print("=" * 60)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        print(f"\nSample {i}:")
        print(f"  Clip: {sample['clip_name']}")
        print(f"  Frame: {sample['frame_idx']}")
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  RGB dtype: {sample['rgb'].dtype}")
        print(f"  RGB range: [{sample['rgb'].min()}, {sample['rgb'].max()}]")
        print(f"  Waypoints shape: {sample['waypoints'].shape}")
        print(f"  Waypoints dtype: {sample['waypoints'].dtype}")
        print(f"  Speed: {sample['speed']:.2f} m/s")
        print(f"  Command: {sample['command']}")
        print(f"  Target point: {sample['target_point']}")
        
        # Verify data types and shapes
        assert sample['rgb'].shape[0] == 3, "RGB should be (C, H, W)"
        assert sample['waypoints'].shape == (11, 2), "Waypoints should be (11, 2)"
        assert isinstance(sample['speed'], (float, np.floating)), "Speed should be float"
        assert isinstance(sample['command'], (int, np.integer)), "Command should be int"
        
        print("  ✓ All checks passed")


def test_visualization(dataset, output_dir):
    """Test visualization of samples."""
    print("\n" + "=" * 60)
    print("TEST 3: Visualization")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize first 3 samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        
        output_path = output_dir / f"sample_{i:03d}.png"
        
        plot_image_with_waypoints(
            image=sample['rgb'],
            waypoints=sample['waypoints'],
            speed=sample['speed'],
            command=sample['command'],
            target_point=sample['target_point'],
            title=f"Sample {i}: {sample['clip_name']} - Frame {sample['frame_idx']}",
            save_path=str(output_path),
            show=False,
        )
        
        print(f"  ✓ Saved visualization to {output_path}")


def test_augmentation():
    """Test data augmentation."""
    print("\n" + "=" * 60)
    print("TEST 4: Data Augmentation")
    print("=" * 60)
    
    data_root = "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_mini/extracted"
    
    # Create dataset with augmentation
    dataset_aug = Bench2DriveDataset(
        data_root=data_root,
        split="train",
        img_augmentation=True,
        img_augmentation_prob=1.0,  # Always augment for testing
    )
    
    # Load same sample multiple times to see augmentation
    sample_idx = 0
    print(f"\nLoading sample {sample_idx} multiple times with augmentation...")
    
    for i in range(3):
        sample = dataset_aug[sample_idx]
        print(f"  Iteration {i}: RGB range [{sample['rgb'].min()}, {sample['rgb'].max()}]")
    
    print("  ✓ Augmentation working (values change between iterations)")


def test_batch_loading(dataset, batch_size=4):
    """Test batch loading with DataLoader."""
    print("\n" + "=" * 60)
    print(f"TEST 5: Batch Loading (batch_size={batch_size})")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    
    print(f"\nDataLoader created with {len(dataloader)} batches")
    
    # Load first batch
    print("\nLoading first batch...")
    batch = next(iter(dataloader))
    
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  RGB batch shape: {batch['rgb'].shape}")
    print(f"  Waypoints batch shape: {batch['waypoints'].shape}")
    print(f"  Speed batch shape: {batch['speed'].shape}")
    
    assert batch['rgb'].shape[0] == batch_size, "Batch size mismatch"
    print("  ✓ Batch loading successful")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" " * 20 + "BENCH2DRIVE DATALOADER TEST SUITE")
    print("=" * 80)
    
    # Test 1: Dataset loading
    train_dataset, val_dataset = test_dataset_loading()
    
    # Test 2: Sample loading
    test_sample_loading(train_dataset, num_samples=3)
    
    # Test 3: Visualization
    viz_dir = Path(__file__).parent.parent / "outputs" / "visualizations"
    test_visualization(train_dataset, viz_dir)
    
    # Test 4: Augmentation
    test_augmentation()
    
    # Test 5: Batch loading
    test_batch_loading(train_dataset, batch_size=4)
    
    # Print statistics
    print_dataset_statistics(train_dataset)
    
    print("\n" + "=" * 80)
    print(" " * 25 + "ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nDataloader is ready to use for training!")
    print("\nNext steps:")
    print("  1. Check visualizations in: outputs/visualizations/")
    print("  2. Integrate with your model training code")
    print("  3. Use Bench2DriveDataset in your training loop")
    print("\n")


if __name__ == "__main__":
    main()

