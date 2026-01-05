# NFL Helmet Impact Detection

A production-grade ML system for automatically detecting helmet-to-helmet impacts in NFL game footage. Upload a video clip and get back JSON predictions with timestamps, bounding boxes, and confidence scores, plus an optional annotated video.

## Features

- **Automatic helmet detection** using YOLOv8
- **Impact classification** with ResNet-based deep learning
- **Video processing pipeline** with configurable sampling
- **REST API** (FastAPI) for programmatic access
- **Web UI** (Streamlit) for interactive use
- **Docker deployment** ready for Render, Fly.io, or any cloud platform
- **Tiny dev mode** for fast iteration on CPU
- **Comprehensive tests** and CI/CD with GitHub Actions

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -e .

# Or using make
make install
```

### 2. Download Dataset

Set up Kaggle credentials:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Or create `~/.kaggle/kaggle.json`:
```json
{"username":"your_username","key":"your_api_key"}
```

Download the data:
```bash
make download-data
```

### 3. Build Preprocessed Cache

For full dataset:
```bash
make build-cache
```

For quick development (tiny mode - 3 videos, 200 samples):
```bash
make build-cache-tiny
```

### 4. Train Model

Full training:
```bash
make train
```

Tiny mode (5 epochs, fast iteration):
```bash
make train-tiny
```

### 5. Run Inference

**Option A: Streamlit Web UI**
```bash
make streamlit
# Navigate to http://localhost:8501
```

**Option B: FastAPI Server**
```bash
make api
# API docs at http://localhost:8000/docs
```

## Project Structure

```
nfl-impact-detector/
├── src/impact_detector/      # Main package
│   ├── config.py             # Configuration management
│   ├── data_utils.py         # Data loading and parsing
│   ├── dataset.py            # PyTorch datasets
│   ├── model.py              # Model architecture
│   ├── inference.py          # Inference pipeline
│   └── api.py                # FastAPI service
├── scripts/                  # Standalone scripts
│   ├── download_data.py      # Kaggle data download
│   ├── build_crops_cache.py  # Preprocessing
│   ├── train.py              # Training
│   ├── evaluate.py           # Evaluation
│   └── export_onnx.py        # ONNX export
├── configs/                  # YAML configurations
│   ├── base.yaml            # Base config
│   └── tiny.yaml            # Tiny mode overrides
├── tests/                    # pytest tests
├── app.py                    # Streamlit UI
├── Dockerfile                # Docker configuration
├── Makefile                  # Common commands
└── README.md                 # This file
```

## Configuration

All settings are in `configs/base.yaml`. Key parameters:

**Data:**
- `tiny_mode`: Enable tiny dev mode (default: false)
- `crop_size`: Input image size (default: 224)
- `impact_filter`: Filters for definitive impacts

**Model:**
- `architecture`: timm model name (default: resnet18)
- `dropout`: Dropout rate (default: 0.3)

**Training:**
- `batch_size`: Batch size (default: 32)
- `epochs`: Training epochs (default: 20)
- `learning_rate`: Learning rate (default: 0.001)

**Inference:**
- `sample_rate`: Process every N frames (default: 5)
- `min_score`: Minimum confidence threshold (default: 0.5)

Override with environment variables:
```bash
export DATA__TINY_MODE=true
export TRAINING__BATCH_SIZE=16
```

## API Usage

Start the API:
```bash
python -m src.impact_detector.api
```

Example request:
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("game_clip.mp4", "rb")}
params = {
    "sample_rate": 5,
    "min_score": 0.6,
    "annotate": True
}

response = requests.post(url, files=files, params=params)
results = response.json()

print(f"Found {results['num_detections']} impacts")
for detection in results['detections']:
    print(f"  Frame {detection['frame']}: {detection['score']:.3f}")
```

## Evaluation

Evaluate trained model:
```bash
make evaluate
```

Outputs:
- Classification report (precision/recall/F1)
- Confusion matrix (`outputs/evaluation/confusion_matrix.png`)
- PR curve (`outputs/evaluation/pr_curve.png`)
- Predictions CSV

## ONNX Export

Export for faster inference:
```bash
make export-onnx
```

Use ONNX model:
```yaml
# configs/base.yaml
inference:
  use_onnx: true
  onnx_path: "models/model.onnx"
```

## Docker Deployment

### Build Image

```bash
make docker-build
```

### Run Locally

```bash
make docker-run
```

Or manually:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  nfl-impact-detector:latest
```

### Deploy to Render

1. Push to GitHub
2. Connect repository in Render dashboard
3. Use `render.yaml` configuration
4. Deploy!

Or use Render Blueprint:
```bash
# render.yaml is already configured
git push origin main
```

### Deploy to Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Launch app (uses fly.toml)
flyctl launch

# Deploy
flyctl deploy
```

## Development

### Run Tests

```bash
make test
```

### Lint Code

```bash
make lint
```

### Format Code

```bash
make format
```

### Clean Generated Files

```bash
make clean
```

## Compute Requirements

**Training:**
- Full dataset: GPU recommended (4-8 hours on T4)
- Tiny mode: CPU okay (5-10 minutes)

**Inference:**
- CPU: ~2-5 seconds per frame (with sample_rate=5)
- GPU: ~0.5-1 second per frame

**Memory:**
- Training: 8GB+ RAM, 4GB+ VRAM (GPU)
- Inference: 4GB+ RAM

**Storage:**
- Dataset: ~50GB (videos)
- Preprocessed crops: 5-10GB
- Models: ~100MB

## Tiny Mode for Development

Tiny mode uses a small subset for rapid iteration:

```yaml
# configs/tiny.yaml
data:
  tiny_mode: true
  tiny_videos: 3
  tiny_max_samples: 200

training:
  epochs: 5
  batch_size: 16
```

Use for:
- Testing pipeline end-to-end
- Debugging
- Fast iteration on code changes
- CI/CD testing

**Not for production!** Train on full dataset for real use.

## Dataset Notes

The NFL Impact Detection dataset includes:
- Video footage (sideline + endzone views)
- Frame-level labels with bounding boxes
- Impact indicators and metadata

**Label filtering:** By default, we filter for "definitive impacts":
- `impact == 1`
- `confidence >= 0.5`
- `visibility >= 0.5`

Adjust in `configs/base.yaml` under `data.impact_filter`.

## Troubleshooting

**Issue: Kaggle authentication fails**
- Verify credentials at https://www.kaggle.com/settings/account
- Check `~/.kaggle/kaggle.json` permissions (should be 600)

**Issue: Out of memory during training**
- Reduce `batch_size` in config
- Use tiny mode for development
- Enable gradient accumulation (modify training script)

**Issue: No impacts detected**
- Lower `min_score` threshold
- Reduce `sample_rate` (process more frames)
- Check model is trained (not random weights)

**Issue: Video processing is slow**
- Increase `sample_rate` (trade-off: may miss impacts)
- Use ONNX model for faster inference
- Use GPU if available

## CI/CD

GitHub Actions automatically:
- Runs linting (ruff)
- Runs tests (pytest)
- Type checks (mypy)
- Builds Docker image

See `.github/workflows/ci.yml`

## License

This project is for educational purposes. The NFL Impact Detection dataset is owned by Kaggle/NFL and subject to competition rules.

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Run `make lint` and `make test`
5. Submit pull request

## Acknowledgments

- **Dataset:** Kaggle NFL Impact Detection competition
- **Models:** PyTorch, timm, ultralytics YOLOv8
- **Frameworks:** FastAPI, Streamlit

---

**Ready to detect some impacts? Start with `make install && make download-data`!**
