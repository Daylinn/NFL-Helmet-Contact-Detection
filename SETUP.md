# Setup Instructions

## Project Built Successfully! ✓

The project structure is complete and ready. Here's how to set up and use it.

## Current Status

- ✓ Project structure created (32 files)
- ✓ All Python modules in place
- ✓ Configuration system tested and working
- ✓ Makefile verified
- ✓ Git repository configured
- ⚠️ Full dependencies need installation (see below)

## Quick Start

### Step 1: Install Dependencies

The full installation includes PyTorch, OpenCV, and other ML libraries (~3-5GB). Choose one option:

**Option A: Full Install (Recommended)**
```bash
pip install -e .
```
*This takes 10-20 minutes depending on your internet speed*

**Option B: Install with Make**
```bash
make install
```

**Option C: Install Without Dev Tools (Faster)**
```bash
pip install torch torchvision timm opencv-python pillow numpy pandas \
    pyarrow fastapi uvicorn python-multipart streamlit ultralytics \
    pyyaml tqdm scikit-learn matplotlib seaborn onnx onnxruntime \
    requests kaggle
```

### Step 2: Verify Installation

```bash
python verify_install.py
```

You should see all green checkmarks.

### Step 3: Set Up Kaggle Credentials

```bash
# Option A: Environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option B: Create credentials file
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Get your API key from: https://www.kaggle.com/settings/account

### Step 4: Download Data

```bash
make download-data
```

This downloads the NFL Impact Detection dataset (~50GB). It will take a while depending on your connection.

### Step 5: Quick Test (Tiny Mode)

Test the pipeline end-to-end with a small dataset:

```bash
# Build preprocessed cache (tiny mode: 3 videos, 200 samples)
make build-cache-tiny

# Train model (5 epochs, ~5-10 minutes on CPU)
make train-tiny

# Check that model was created
ls -lh models/best.pt
```

### Step 6: Run the App

```bash
# Option A: Streamlit UI (recommended for demos)
make streamlit
# Open http://localhost:8501 in your browser

# Option B: FastAPI (for programmatic access)
make api
# API docs at http://localhost:8000/docs
```

## Full Pipeline (Production)

Once you've tested with tiny mode:

```bash
# 1. Build full preprocessed cache
make build-cache

# 2. Train on full dataset (requires GPU, 4-8 hours)
make train

# 3. Evaluate model
make evaluate

# 4. Export to ONNX (optional, for faster inference)
make export-onnx

# 5. Deploy (see deployment section below)
```

## Project Structure

```
NFL-Helmet-Contact-Detection/
├── src/impact_detector/       # Main package
│   ├── __init__.py
│   ├── config.py              # Configuration management ✓
│   ├── data_utils.py          # Data loading utilities
│   ├── dataset.py             # PyTorch datasets
│   ├── model.py               # Model architecture
│   ├── inference.py           # Inference pipeline
│   └── api.py                 # FastAPI service
├── scripts/                   # Standalone scripts
│   ├── download_data.py       # Kaggle data download
│   ├── build_crops_cache.py   # Preprocessing
│   ├── train.py               # Training
│   ├── evaluate.py            # Evaluation
│   └── export_onnx.py         # ONNX export
├── configs/                   # YAML configurations ✓
│   ├── base.yaml
│   └── tiny.yaml
├── tests/                     # pytest tests
├── data/                      # Data directory
│   ├── raw/                   # Downloaded videos
│   └── processed/             # Preprocessed crops
├── models/                    # Model checkpoints
├── logs/                      # Training logs
├── outputs/                   # Evaluation outputs
├── app.py                     # Streamlit UI
├── Makefile                   # Common commands ✓
├── Dockerfile                 # Docker config
├── pyproject.toml             # Dependencies
└── README.md                  # Full documentation

✓ = Tested and working
```

## Available Make Targets

```bash
make help              # Show all commands
make install           # Install dependencies
make download-data     # Download Kaggle dataset
make build-cache       # Build preprocessed cache
make build-cache-tiny  # Build cache (tiny mode)
make train             # Train model
make train-tiny        # Train model (tiny mode)
make evaluate          # Evaluate model
make export-onnx       # Export to ONNX
make api               # Run FastAPI server
make streamlit         # Run Streamlit UI
make test              # Run tests
make lint              # Lint code
make format            # Format code
make clean             # Clean generated files
make docker-build      # Build Docker image
make docker-run        # Run Docker container
```

## Docker Deployment

```bash
# Build image
make docker-build

# Run locally
docker run -p 8000:8000 -v $(pwd)/models:/app/models nfl-impact-detector:latest

# Deploy to Render (push to GitHub first)
# Uses render.yaml configuration

# Deploy to Fly.io
flyctl launch
flyctl deploy
```

## Troubleshooting

**Issue: pip install taking too long**
- Solution: Use faster mirrors or install core packages individually
- PyTorch is the largest (~2GB), consider CPU-only version for testing

**Issue: Kaggle authentication fails**
- Solution: Check credentials at https://www.kaggle.com/settings/account
- Verify ~/.kaggle/kaggle.json permissions (should be 600)

**Issue: Out of memory during training**
- Solution: Use tiny mode or reduce batch size in configs/base.yaml

**Issue: ModuleNotFoundError**
- Solution: Make sure you ran `pip install -e .` or `make install`

## Next Steps

1. **Test with tiny mode** to validate the pipeline works
2. **Download full dataset** if you have space (~50GB)
3. **Train full model** if you have GPU access
4. **Deploy** to Render or Fly.io for production use

## Support

- Full documentation: See README.md
- Configuration options: See configs/base.yaml
- API examples: See README.md "API Usage" section

---

**Status: Ready to go! Start with `make install` then `make download-data`**
