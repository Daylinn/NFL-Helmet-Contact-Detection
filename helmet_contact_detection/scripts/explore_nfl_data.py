#!/usr/bin/env python3
"""
Explore the NFL Impact Detection dataset.

Shows dataset statistics and sample images.
"""
import pandas as pd
import os
from pathlib import Path

def explore_dataset(data_dir='data/raw'):
    """Explore the NFL Impact Detection dataset."""

    print("="*60)
    print("NFL Impact Detection Dataset Explorer")
    print("="*60 + "\n")

    # Check if data exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("\nPlease download the dataset first:")
        print("  kaggle competitions download -c nfl-impact-detection")
        return

    # Load train labels
    labels_file = data_path / 'train_labels.csv'
    if labels_file.exists():
        print("📊 Loading train labels...")
        df = pd.read_csv(labels_file)

        print(f"\n✓ Loaded {len(df):,} rows")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Contact statistics
        if 'contact' in df.columns:
            contact_counts = df['contact'].value_counts()
            print(f"\n📈 Contact Statistics:")
            print(contact_counts)
            print(f"\nContact percentage: {df['contact'].mean()*100:.2f}%")

        # Unique videos/games
        if 'video' in df.columns:
            print(f"\n🎥 Unique videos: {df['video'].nunique()}")

    else:
        print(f"❌ Labels file not found: {labels_file}")

    # Check image directories
    train_dir = data_path / 'train'
    if train_dir.exists():
        # Count images
        image_files = list(train_dir.glob('**/*.jpg')) + list(train_dir.glob('**/*.png'))
        print(f"\n🖼️  Training images: {len(image_files):,}")

        if image_files:
            print(f"\nSample image: {image_files[0]}")

    print("\n" + "="*60)
    print("Dataset exploration complete!")
    print("="*60)


if __name__ == '__main__':
    explore_dataset()
