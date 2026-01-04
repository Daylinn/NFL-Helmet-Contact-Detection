# Quick Start Guide

Get the NFL Helmet Contact Detection API running in 5 minutes.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- 2GB free disk space
- Basic command line knowledge

## Step 1: Build the Docker Image

```bash
cd helmet_contact_detection
docker build -t helmet-contact-detection:latest .
```

This will take 2-3 minutes on first build.

## Step 2: Run the Container

```bash
docker run -p 8000:8000 helmet-contact-detection:latest
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Note:** You'll see a warning about missing model weights - this is expected!

## Step 3: Test the API

Open a new terminal and run:

```bash
# Test health endpoint
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "model_loaded": false,
  "version": "1.0.0"
}
```

## Step 4: View API Documentation

Open your browser and visit:

**http://localhost:8000/docs**

You'll see the interactive Swagger UI where you can:
- View all endpoints
- Test the API directly
- See request/response schemas

## What's Next?

### Option A: Test Without Real Model (Demo Mode)

The API will start but predictions will return 503 errors until you add model weights.

### Option B: Add Model Weights

1. **Get trained YOLO weights** (see `scripts/download_kaggle_instructions.md`)
2. **Place at:** `models/weights.pt`
3. **Rebuild:** `docker build -t helmet-contact-detection:latest .`
4. **Rerun:** `docker run -p 8000:8000 helmet-contact-detection:latest`

Now `model_loaded` should be `true`!

### Option C: Test with Pretrained Base Model

For quick testing (not accurate for helmets):

```bash
# Install ultralytics locally
pip install ultralytics

# Download pretrained YOLOv8
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('models/weights.pt')"

# Rebuild and run
docker build -t helmet-contact-detection:latest .
docker run -p 8000:8000 helmet-contact-detection:latest
```

## Running the Test Script

```bash
# Install test dependencies
pip install requests pillow numpy

# Run test
python test_api.py
```

## Common Commands

### Stop the container
```bash
# Find container ID
docker ps

# Stop it
docker stop <container_id>
```

### View logs
```bash
docker logs <container_id>
```

### Run with custom port
```bash
docker run -p 9000:8000 helmet-contact-detection:latest
# API will be at http://localhost:9000
```

### Run with volume-mounted weights
```bash
docker run \
  -p 8000:8000 \
  -v /path/to/your/weights.pt:/app/models/weights.pt:ro \
  helmet-contact-detection:latest
```

## Example API Calls

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict Frame
```bash
curl -X POST \
  http://localhost:8000/predict_frame \
  -F "file=@path/to/image.jpg"
```

### Predict Clip
```bash
curl -X POST \
  "http://localhost:8000/predict_clip?max_frames=30" \
  -F "file=@path/to/video.mp4"
```

## Troubleshooting

### "Connection refused"
- Ensure the container is running: `docker ps`
- Check the correct port is exposed: `-p 8000:8000`

### "Model not loaded" warning
- Expected if `models/weights.pt` is missing
- See "Option B" above to add weights

### Container exits immediately
- Check logs: `docker logs <container_id>`
- Ensure Dockerfile is in current directory

### Slow inference
- Normal on CPU (100-300ms per frame)
- For faster inference, use GPU-enabled Docker image

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [scripts/download_kaggle_instructions.md](scripts/download_kaggle_instructions.md) for dataset info
3. Review the code in `app/` to understand the implementation
4. Customize the contact prediction logic in `app/inference.py`

## Getting Help

- Check the [README.md](README.md) troubleshooting section
- Review API docs at http://localhost:8000/docs
- Inspect container logs: `docker logs <container_id>`

Happy detecting! 🏈
