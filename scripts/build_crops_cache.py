"""Build preprocessed crops cache from raw video data."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.impact_detector.config import load_config, merge_configs
from src.impact_detector.data_utils import (
    build_sample_metadata,
    create_video_index,
    discover_data_files,
    extract_crop,
    extract_temporal_crop,
    load_video_labels,
)


def save_crop(crop: np.ndarray, output_path: Path) -> bool:
    """Save crop to disk as JPEG.

    Args:
        crop: Crop array (RGB)
        output_path: Output file path

    Returns:
        True if successful
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(output_path), crop_bgr)
    return success


def build_crops(config, tiny_mode: bool = False):
    """Build preprocessed crops from videos.

    Args:
        config: Configuration object
        tiny_mode: Whether to use tiny mode for fast iteration
    """
    print("=" * 60)
    print("Building Crops Cache")
    print("=" * 60)

    data_dir = Path(config.data.raw_dir)
    crops_dir = Path(config.data.crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Discover data files
    print("\n1. Discovering data files...")
    discovered = discover_data_files(data_dir)

    if not discovered["labels"]:
        print("✗ No label files found!")
        return False

    if not discovered["videos"]:
        print("✗ No video files found!")
        return False

    print(f"  Found {len(discovered['labels'])} label files")
    print(f"  Found {len(discovered['videos'])} video files")

    # Load labels
    print("\n2. Loading and filtering labels...")
    all_labels = []
    for labels_path in discovered["labels"]:
        labels_df = load_video_labels(labels_path, config.data.impact_filter)
        all_labels.append(labels_df)

    labels_df = pd.concat(all_labels, ignore_index=True)
    print(f"  Total labels: {len(labels_df)}")

    # Tiny mode filtering
    if tiny_mode or config.data.tiny_mode:
        print(f"\n  TINY MODE: Limiting to {config.data.tiny_videos} videos")

        # Get unique videos
        video_col = None
        for col in ["video", "videoName", "video_name"]:
            if col in labels_df.columns:
                video_col = col
                break

        if video_col:
            unique_videos = labels_df[video_col].unique()[:config.data.tiny_videos]
            labels_df = labels_df[labels_df[video_col].isin(unique_videos)]
            print(f"  Filtered to {len(labels_df)} labels from {len(unique_videos)} videos")

    # Create video index
    print("\n3. Creating video index...")
    video_index = create_video_index(data_dir)
    print(f"  Indexed {len(video_index)} video paths")

    # Build sample metadata
    print("\n4. Building sample metadata...")
    max_samples = config.data.tiny_max_samples if (tiny_mode or config.data.tiny_mode) else None
    metadata_df = build_sample_metadata(labels_df, video_index, max_samples)

    if metadata_df.empty:
        print("✗ No valid samples found!")
        print("  Check that video files match the labels")
        return False

    print(f"  Built metadata for {len(metadata_df)} samples")

    # Extract and save crops
    print("\n5. Extracting crops...")
    use_temporal = config.data.temporal_frames > 1

    successful = 0
    failed = 0

    crop_paths = []

    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting crops"):
        video_path = Path(row["video_path"])
        frame = row["frame"]
        bbox = (row["x1"], row["y1"], row["x2"], row["y2"])

        # Generate crop filename
        video_id = video_path.stem
        crop_name = f"{video_id}_frame{frame}_{idx}.jpg"
        crop_path = crops_dir / video_id / crop_name

        # Extract crop
        if use_temporal:
            crop = extract_temporal_crop(
                video_path,
                frame,
                bbox,
                config.data.temporal_window,
                config.data.crop_size,
            )
            # For temporal, save as numpy array
            if crop is not None:
                crop_path = crop_path.with_suffix(".npy")
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(crop_path, crop)
                successful += 1
                crop_paths.append(str(crop_path))
            else:
                failed += 1
                crop_paths.append(None)
        else:
            crop = extract_crop(video_path, frame, bbox, config.data.crop_size)

            if crop is not None and save_crop(crop, crop_path):
                successful += 1
                crop_paths.append(str(crop_path))
            else:
                failed += 1
                crop_paths.append(None)

    print(f"\n  Extracted {successful} crops successfully")
    if failed > 0:
        print(f"  Failed: {failed} crops")

    # Add crop paths to metadata
    metadata_df["crop_path"] = crop_paths

    # Remove failed samples
    metadata_df = metadata_df[metadata_df["crop_path"].notna()]

    print(f"  Final dataset: {len(metadata_df)} samples")

    # Save metadata
    print("\n6. Saving metadata...")
    metadata_path = Path(config.data.metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_df.to_parquet(metadata_path, index=False)
    print(f"  Saved metadata to {metadata_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("CACHE BUILD COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(metadata_df)}")
    print(f"Crops directory: {crops_dir}")
    print(f"Metadata file: {metadata_path}")

    # Print sample distribution
    if "game_key" in metadata_df.columns and "play_id" in metadata_df.columns:
        unique_plays = metadata_df.groupby(["game_key", "play_id"]).size()
        print(f"\nUnique plays: {len(unique_plays)}")
        print(f"Avg samples per play: {unique_plays.mean():.1f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build preprocessed crops cache")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny mode for fast iteration",
    )
    parser.add_argument(
        "--override-config",
        help="Optional override config (e.g., configs/tiny.yaml)",
    )

    args = parser.parse_args()

    # Load config
    if args.override_config:
        config = merge_configs(args.config, args.override_config)
    else:
        config = load_config(args.config)

    # Build crops
    success = build_crops(config, tiny_mode=args.tiny)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
