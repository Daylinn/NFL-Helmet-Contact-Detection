# Model Weights Directory

## Overview

This directory should contain the trained YOLO model weights for helmet detection.

**Required file:** `weights.pt`

## Getting Model Weights

### Option 1: Train Your Own Model

Follow the instructions in `scripts/download_kaggle_instructions.md` to:

1. Download the NFL Helmet Contact Detection dataset
2. Train a YOLOv8 model on helmet detection
3. Export the trained weights as `weights.pt`
4. Place the file in this directory

### Option 2: Use Pre-trained Weights

If you have access to pre-trained YOLO weights:

1. Obtain the `.pt` file (must be trained on helmet detection)
2. Rename it to `weights.pt`
3. Place it in this directory

### Option 3: Use YOLOv8 Pretrained Base

For testing/demo purposes, you can use a YOLOv8 pretrained model:

```python
from ultralytics import YOLO

# Download pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Save to this directory
model.save('models/weights.pt')
```

**Note:** This will NOT work well for helmet detection without fine-tuning on NFL data.

## File Specification

- **Format:** PyTorch (.pt) weights file
- **Model:** YOLOv8 (nano, small, medium, large, or xlarge)
- **Training:** Should be trained on helmet detection task
- **Classes:** Expects "helmet" as the primary detection class

## Expected Model Performance

A well-trained model should achieve:

- **Precision:** > 0.85 for helmet detection
- **Recall:** > 0.80 for helmet detection
- **Inference Time:** 50-200ms per frame (CPU), 10-30ms (GPU)

**Note:** These are target metrics, not guarantees. Actual performance depends on:
- Training data quality and quantity
- Model size (yolov8n vs yolov8x)
- Hardware (CPU vs GPU)
- Image resolution

## Placeholder Behavior

If `weights.pt` is not present:

- The API will start successfully
- Health check will show `model_loaded: false`
- Prediction endpoints will return HTTP 503 errors
- Logs will show "Model weights not found" warning

## File Size

Expected file sizes by model variant:

- YOLOv8n: ~6 MB
- YOLOv8s: ~22 MB
- YOLOv8m: ~52 MB
- YOLOv8l: ~87 MB
- YOLOv8x: ~136 MB

## Security Note

Do NOT commit actual model weights to version control if:

- They contain proprietary training data
- The dataset has licensing restrictions
- The model is considered intellectual property

Add `weights.pt` to `.gitignore` for protection.

## Testing Your Model

Once you have placed `weights.pt` in this directory:

```bash
# Rebuild Docker image
docker build -t helmet-contact-detection:latest .

# Run container
docker run -p 8000:8000 helmet-contact-detection:latest

# Test health endpoint
curl http://localhost:8000/health

# Should return:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "version": "1.0.0"
# }
```

## Troubleshooting

### "Model not loaded" Error

**Causes:**
- File `weights.pt` does not exist
- File is corrupted
- File is not a valid YOLO model
- Permissions issue

**Solutions:**
- Verify file exists: `ls -lh models/weights.pt`
- Check file size (should be several MB)
- Ensure file is readable
- Try re-downloading/re-training the model

### "Failed to load model" Error

**Causes:**
- Incompatible model format
- Wrong PyTorch version
- Model architecture mismatch

**Solutions:**
- Ensure model was saved with compatible ultralytics version
- Check model was exported correctly
- Verify model architecture matches expected format

## References

- Ultralytics YOLO: https://docs.ultralytics.com/
- NFL Dataset: https://www.kaggle.com/c/nfl-impact-detection
- Training Guide: See `scripts/download_kaggle_instructions.md`
