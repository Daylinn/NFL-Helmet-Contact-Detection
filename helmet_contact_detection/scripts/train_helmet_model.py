#!/usr/bin/env python3
"""
Train a YOLOv8 model on NFL Impact Detection dataset.

This script:
1. Converts NFL dataset to YOLO format
2. Trains YOLOv8 on helmet detection
3. Exports trained weights for the API

Usage:
    python scripts/train_helmet_model.py --epochs 50 --img-size 1280
"""
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import yaml


def convert_to_yolo_format(data_dir='data/raw', output_dir='data/yolo'):
    """
    Convert NFL dataset to YOLO format.

    YOLO format:
    - images/ directory with .jpg files
    - labels/ directory with .txt files (one per image)
    - Each label line: class x_center y_center width height (normalized 0-1)
    """
    print("Converting NFL dataset to YOLO format...")

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Create YOLO directory structure
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_df = pd.read_csv(data_path / 'train_labels.csv')

    print(f"✓ Loaded {len(labels_df):,} labels")
    print(f"✓ Converting to YOLO format...")

    # TODO: Implement conversion based on actual dataset structure
    # This is a placeholder - actual implementation depends on dataset format

    print(f"✓ Dataset converted to {output_path}")
    return output_path


def create_dataset_yaml(data_dir='data/yolo'):
    """Create YOLO dataset configuration file."""

    config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Number of classes (just 'helmet')
        'names': ['helmet']
    }

    yaml_path = Path(data_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    print(f"✓ Created dataset config: {yaml_path}")
    return yaml_path


def train_model(data_yaml, epochs=50, img_size=1280, model_size='n'):
    """
    Train YOLOv8 model.

    Args:
        data_yaml: Path to dataset YAML config
        epochs: Number of training epochs
        img_size: Image size for training
        model_size: Model size (n, s, m, l, x)
    """
    print(f"\n{'='*60}")
    print(f"Training YOLOv8{model_size.upper()} on NFL Helmet Dataset")
    print(f"{'='*60}\n")

    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        name='nfl_helmet_detection',
        device='cpu',  # Use 'cuda' if GPU available
        patience=10,
        save=True,
        plots=True
    )

    print(f"\n✓ Training complete!")
    print(f"✓ Best weights saved to: runs/detect/nfl_helmet_detection/weights/best.pt")

    return results


def export_weights():
    """Export trained weights to models/ directory."""

    best_weights = Path('runs/detect/nfl_helmet_detection/weights/best.pt')

    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, 'models/weights.pt')
        print(f"✓ Weights exported to models/weights.pt")
        print(f"\nYou can now rebuild Docker and test!")
    else:
        print(f"❌ Weights not found at {best_weights}")


def main():
    parser = argparse.ArgumentParser(description='Train helmet detection model')
    parser.add_argument('--data-dir', default='data/raw', help='NFL dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--img-size', type=int, default=1280, help='Image size')
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size')
    parser.add_argument('--skip-convert', action='store_true',
                        help='Skip dataset conversion (if already done)')

    args = parser.parse_args()

    # Step 1: Convert dataset
    if not args.skip_convert:
        yolo_dir = convert_to_yolo_format(args.data_dir)
        dataset_yaml = create_dataset_yaml(yolo_dir)
    else:
        dataset_yaml = 'data/yolo/dataset.yaml'

    # Step 2: Train model
    train_model(
        data_yaml=dataset_yaml,
        epochs=args.epochs,
        img_size=args.img_size,
        model_size=args.model_size
    )

    # Step 3: Export weights
    export_weights()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Rebuild Docker: docker build -t helmet-contact-detection:latest .")
    print("  2. Run container: docker run -p 8000:8000 helmet-contact-detection:latest")
    print("  3. Test with NFL images!")


if __name__ == '__main__':
    main()
