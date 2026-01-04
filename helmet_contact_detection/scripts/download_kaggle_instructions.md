# NFL Helmet Contact Detection Dataset

## Dataset Information

The NFL Helmet Contact Detection dataset is available on Kaggle:

**Competition**: [NFL 1st and Future - Impact Detection](https://www.kaggle.com/c/nfl-impact-detection)

This dataset contains:
- NFL game footage video frames
- Helmet bounding box annotations
- Contact/impact event labels
- Player tracking data

## Prerequisites

1. Kaggle account (free registration at https://www.kaggle.com)
2. Kaggle API credentials configured

## Setup Kaggle API

### 1. Get API Credentials

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json` with your credentials

### 2. Install Kaggle API

```bash
pip install kaggle
```

### 3. Configure Credentials

**Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

## Download Dataset

### Download Competition Data

```bash
# Accept competition rules on Kaggle website first
# Visit: https://www.kaggle.com/c/nfl-impact-detection/rules

# Download dataset (large - several GB)
kaggle competitions download -c nfl-impact-detection

# Unzip
unzip nfl-impact-detection.zip -d data/raw/
```

### Dataset Structure

After extraction:
```
data/raw/
├── train_labels.csv           # Training labels with contact annotations
├── train_video_metadata.csv   # Video metadata
├── test_video_metadata.csv    # Test metadata
├── sample_submission.csv      # Submission format
└── train/                     # Training videos and frames
    ├── video_1.mp4
    ├── video_2.mp4
    └── ...
```

## Training a YOLO Model

### 1. Prepare Data in YOLO Format

Convert CSV labels to YOLO format:

```python
import pandas as pd
from pathlib import Path

# Read labels
labels_df = pd.read_csv('data/raw/train_labels.csv')

# Convert to YOLO format: class x_center y_center width height
# Each row represents a helmet bounding box
# Save as .txt files in YOLO format
```

### 2. Train YOLO Model

```bash
# Install ultralytics
pip install ultralytics

# Train YOLOv8 model
yolo detect train \
  data=data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=1280 \
  batch=16
```

### 3. Export Trained Weights

After training, copy the best weights to the models directory:

```bash
cp runs/detect/train/weights/best.pt helmet_contact_detection/models/weights.pt
```

## Alternative: Pre-trained Weights

If you have access to pre-trained weights:

1. Download or obtain the trained YOLO weights file (`.pt` format)
2. Place in `helmet_contact_detection/models/weights.pt`
3. Ensure the model was trained on helmet detection task

## Data Licensing

The NFL dataset is subject to Kaggle competition rules and NFL data usage policies.
Please review the competition terms before using the data.

## Contact Detection Enhancement

For improved contact prediction beyond geometric heuristics:

1. **Temporal Features**: Use player tracking data for velocity/acceleration
2. **Fine-tune Model**: Add contact classification head to YOLO
3. **Multi-frame Context**: Use video sequences instead of single frames
4. **Impact Metrics**: Incorporate G-force and sensor data if available

## References

- Kaggle Competition: https://www.kaggle.com/c/nfl-impact-detection
- Ultralytics YOLO: https://docs.ultralytics.com/
- NFL Helmet Tracking: https://github.com/nfl-football-ops
