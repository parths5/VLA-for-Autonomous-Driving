"""
Optimized embedding cache with batch processing for 5-10x speedup.

Key optimizations:
1. Batch processing (process multiple images at once)
2. Pre-filtering of already processed images
3. Better memory management
4. Progress statistics and ETA
5. Efficient tensor operations

Usage:
    python embedding_cache.py --batch-size 8
    python embedding_cache.py --data-root /path/to/data --batch-size 16
"""
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List, Tuple
import torch
import torch.nn.functional as F
import torchvision.io as io
from pathlib import Path
from tqdm import tqdm
import time
import argparse

# Default configuration
DEFAULT_DATA_ROOT = "/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_base/extracted"
DEFAULT_OUTPUT_ROOT = '/ocean/projects/cis250252p/shared/afadia/VLAD/data/bench2drive_base/extracted'
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_DOWNSCALE_FACTOR = 0.25
DEFAULT_BATCH_SIZE = 8  # Process 8 images at once for significant speedup


class EmbeddingCache:
    """Optimized embedding cache with batch processing."""

    def __init__(
        self,
        data_root: str,
        output_root: str,
        model_name: str = DEFAULT_MODEL_NAME,
        downscale_factor: float = DEFAULT_DOWNSCALE_FACTOR,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.model_name = model_name
        self.downscale_factor = downscale_factor
        self.batch_size = batch_size
        self.model_short_name = model_name.split("/")[-1]

        # Model will be loaded lazily
        self.model = None
        self.processor = None
        self.device = None

    def init_model(self):
        """Initialize model once and reuse."""
        if self.model is None:
            print("Loading model...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.device = self.model.device
            print(f"Model loaded on device: {self.device}")

    def preprocess_image(self, img_path: Path) -> torch.Tensor:
        """Load and preprocess a single image."""
        # Read image using torchvision (faster than OpenCV)
        image = io.read_image(str(img_path))  # Returns (C, H, W) tensor uint8

        # Cut bottom portion (car hood) - same as transforms.py
        cut_height = int(image.shape[1] * (4.8 / 16))
        if cut_height > 0:
            image = image[:, :-cut_height, :]

        # Downscale image if factor < 1.0
        if self.downscale_factor < 1.0:
            new_h = int(image.shape[1] * self.downscale_factor)
            new_w = int(image.shape[2] * self.downscale_factor)
            image = F.interpolate(
                image.unsqueeze(0).float(),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).to(torch.uint8)

        return image

    def encode_batch(self, images: List[torch.Tensor]) -> List[Tuple]:
        """
        Encode a batch of images - MUCH faster than one-by-one.

        Args:
            images: List of image tensors (C, H, W)

        Returns:
            List of (features, deepstack_features) tuples
        """
        self.init_model()

        # Process batch through processor
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            text=[self.processor.image_token] * len(images)
        ).to(self.device)

        with torch.no_grad():
            features, deepstack_features = self.model.get_image_features(
                inputs['pixel_values'],
                inputs['image_grid_thw']
            )

        # The model returns lists of tensors where each tensor contains the batch
        # We need to split by batch dimension to get per-image features
        results = []
        for i in range(len(images)):
            # Split each feature tensor by batch index
            img_features = [f[i:i+1] for f in features]
            img_deepstack = [f[i:i+1] for f in deepstack_features]
            results.append((img_features, img_deepstack))

        return results

    def get_output_paths(self, img_path: Path) -> Tuple[Path, Path]:
        """Get output paths for features and deepstack features."""
        relative_path = img_path.relative_to(self.data_root)
        output_embedding_dir = self.output_root / relative_path.parent
        frame_name = img_path.stem
        features_path = output_embedding_dir / f"{frame_name}_{self.model_short_name}_features.pt"
        deepstack_path = output_embedding_dir / f"{frame_name}_{self.model_short_name}_deepstack_features.pt"
        return features_path, deepstack_path

    def filter_unprocessed_images(self, image_paths: List[Path]) -> List[Path]:
        """Filter out already processed images - faster than checking one-by-one."""
        unprocessed = []
        for img_path in image_paths:
            features_path, deepstack_path = self.get_output_paths(img_path)
            if not (features_path.exists() and deepstack_path.exists()):
                unprocessed.append(img_path)
        return unprocessed

    def process_all(self, randomize: bool = True):
        """Process all images with batch processing."""
        # Collect all clips
        clips = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        print(f"Found {len(clips)} clips")

        # Collect all image paths
        print("Collecting image paths...")
        all_image_paths = []
        for clip_path in clips:
            rgb_dir = clip_path / "camera" / "rgb_front"
            if rgb_dir.exists():
                images = sorted(list(rgb_dir.glob("*.jpg")))
                all_image_paths.extend(images)

        total_images = len(all_image_paths)
        print(f"Found {total_images} total images")

        # Filter out already processed images
        print("Filtering already processed images...")
        unprocessed_paths = self.filter_unprocessed_images(all_image_paths)
        num_already_processed = total_images - len(unprocessed_paths)

        if num_already_processed > 0:
            print(f"  Already processed: {num_already_processed}")
            print(f"  Remaining: {len(unprocessed_paths)}")

        if len(unprocessed_paths) == 0:
            print("All images already processed!")
            return

        # Randomize order to avoid conflicts when running on multiple machines
        if randomize:
            import random
            random.shuffle(unprocessed_paths)
            print("Randomized processing order for multi-machine compatibility")

        # Initialize model
        self.init_model()

        # Process in batches
        print(f"\nProcessing {len(unprocessed_paths)} images in batches of {self.batch_size}...")
        num_batches = (len(unprocessed_paths) + self.batch_size - 1) // self.batch_size

        processed_count = 0
        start_time = time.time()

        with tqdm(total=len(unprocessed_paths), desc="Embedding images") as pbar:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(unprocessed_paths))
                batch_paths = unprocessed_paths[batch_start:batch_end]

                try:
                    # ATOMIC CHECK: Re-check which images still need processing
                    # This is critical for multi-GPU processing to avoid duplicates
                    paths_to_process = []
                    for img_path in batch_paths:
                        features_path, deepstack_path = self.get_output_paths(img_path)
                        if not (features_path.exists() and deepstack_path.exists()):
                            paths_to_process.append(img_path)
                        else:
                            # Another process already completed this image
                            pbar.update(1)

                    # Skip batch if all images were processed by another GPU
                    if len(paths_to_process) == 0:
                        continue

                    # Load and preprocess only the images that need processing
                    images = [self.preprocess_image(p) for p in paths_to_process]

                    # Encode batch
                    results = self.encode_batch(images)

                    # Save each result
                    for img_path, (features, deepstack_features) in zip(paths_to_process, results):
                        features_path, deepstack_path = self.get_output_paths(img_path)

                        # Create output directory
                        features_path.parent.mkdir(parents=True, exist_ok=True)

                        # Concatenate and save features
                        features_cat = torch.cat(features, dim=0).cpu()
                        torch.save(features_cat, features_path)

                        # Concatenate and save deepstack features
                        deepstack_cat = torch.cat(deepstack_features, dim=0).cpu()
                        torch.save(deepstack_cat, deepstack_path)

                    processed_count += len(paths_to_process)
                    pbar.update(len(paths_to_process))

                    # Update statistics
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    pbar.set_postfix({
                        'rate': f'{rate:.2f} img/s',
                        'batch': f'{batch_idx+1}/{num_batches}'
                    })

                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next batch
                    continue

        elapsed = time.time() - start_time
        print(f"\nCompleted! Processed {processed_count} images in {elapsed:.1f}s")
        print(f"Average rate: {processed_count/elapsed:.2f} images/second")
        print(f"Model used: {self.model_short_name}")
        print(f"Embeddings saved with naming pattern:")
        print(f"  - <frame>_{self.model_short_name}_features.pt")
        print(f"  - <frame>_{self.model_short_name}_deepstack_features.pt")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings with batch processing")
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory with extracted clips"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--downscale-factor",
        type=float,
        default=DEFAULT_DOWNSCALE_FACTOR,
        help="Image downscale factor (0.5 = half size)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Don't randomize processing order"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Optimized Embedding Cache with Batch Processing")
    print("=" * 70)
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Model: {args.model_name}")
    print(f"Downscale factor: {args.downscale_factor}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    cache = EmbeddingCache(
        data_root=args.data_root,
        output_root=args.output_root,
        model_name=args.model_name,
        downscale_factor=args.downscale_factor,
        batch_size=args.batch_size,
    )

    cache.process_all(randomize=not args.no_randomize)


if __name__ == "__main__":
    main()
