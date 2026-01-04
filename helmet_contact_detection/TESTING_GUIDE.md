# Docker Testing Guide - NFL Helmet Contact Detection

Complete guide to build, run, and test the Helmet Contact Detection API with Docker.

## Prerequisites

- Docker installed and running
- Terminal/command prompt access
- Basic familiarity with cURL or ability to use a browser

## Quick Start (5 Minutes)

### Step 1: Download a Pretrained Model

First, we need model weights. You have two options:

**Option A: Download YOLOv8n (Quick Demo)**

```bash
# Install ultralytics if you don't have it
pip install ultralytics

# Download and save a pretrained YOLOv8n model
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); import os; os.makedirs('models', exist_ok=True); model.save('models/weights.pt'); print('✓ Model saved to models/weights.pt')"
```

**Note:** This is a general object detection model (trained on COCO dataset), not specifically trained on NFL helmets. It will detect objects but won't be accurate for helmet-specific detection. It's perfect for testing the API functionality.

**Option B: Use Your Own Trained Weights**

If you have trained YOLO weights for helmet detection:
```bash
cp /path/to/your/weights.pt models/weights.pt
```

### Step 2: Build the Docker Image

```bash
cd helmet_contact_detection
docker build -t helmet-contact-detection:latest .
```

This will take 2-3 minutes on first build.

### Step 3: Run the Container

**Option A: With built-in weights (if you placed weights.pt in models/ before building)**

```bash
docker run -p 8000:8000 helmet-contact-detection:latest
```

**Option B: With volume-mounted weights (recommended)**

```bash
docker run \
  -p 8000:8000 \
  -v "$(pwd)/models/weights.pt:/app/models/weights.pt:ro" \
  helmet-contact-detection:latest
```

You should see output like:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Model loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Test the API

Keep the container running and open a new terminal.

## Testing Commands

### 1. Root Endpoint - API Information

```bash
curl http://localhost:8000/
```

**Expected Output:**
```json
{
  "name": "NFL Helmet Contact Detection API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "docs": "/docs",
    "predict_frame": "/predict_frame",
    "predict_clip": "/predict_clip"
  }
}
```

### 2. Health Check - Service Status

```bash
curl http://localhost:8000/health
```

**Expected Output (with weights loaded):**
```json
{
  "status": "healthy",
  "weights_loaded": true,
  "message": "Model weights loaded and ready for inference",
  "version": "1.0.0"
}
```

**Expected Output (without weights):**
```json
{
  "status": "healthy",
  "weights_loaded": false,
  "message": "Model weights not loaded - predictions will fail until weights.pt is provided",
  "version": "1.0.0"
}
```

### 3. Interactive Documentation

Open in your browser:
```
http://localhost:8000/docs
```

This provides:
- Interactive API documentation
- Try-it-out functionality for all endpoints
- Request/response schemas
- Example values

### 4. Predict Frame - Single Image Detection

First, create a test image or download one:

```bash
# Create a simple test image
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (640, 480), 'white')
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 200, 200], fill='blue', outline='red', width=3)
draw.rectangle([300, 200, 400, 300], fill='green', outline='red', width=3)
img.save('test_frame.jpg')
print('✓ Test image created: test_frame.jpg')
"
```

Then test the prediction endpoint:

```bash
curl -X POST \
  http://localhost:8000/predict_frame \
  -F "file=@test_frame.jpg" \
  | python3 -m json.tool
```

**Expected Output (example):**
```json
{
  "helmets": [
    {
      "bbox": {
        "x1": 152.3,
        "y1": 98.7,
        "x2": 198.5,
        "y2": 145.2
      },
      "confidence": 0.87,
      "class_name": "helmet",
      "helmet_id": 0
    },
    {
      "bbox": {
        "x1": 305.1,
        "y1": 203.4,
        "x2": 395.8,
        "y2": 295.6
      },
      "confidence": 0.92,
      "class_name": "helmet",
      "helmet_id": 1
    }
  ],
  "contacts": [
    {
      "helmet_1_id": 0,
      "helmet_2_id": 1,
      "contact_probability": 0.15,
      "distance": 187.3,
      "overlap_iou": 0.0
    }
  ],
  "frame_has_contact": false,
  "inference_time_ms": 142.5
}
```

**Key Metrics Returned:**
- `helmets`: Array of detected helmets with bounding boxes and confidence scores
- `contacts`: Potential helmet-to-helmet contacts with probability scores
- `frame_has_contact`: Boolean indicating if contact detected (probability > 0.5)
- `inference_time_ms`: Processing time in milliseconds

### 5. Predict Clip - Video Analysis

If you have a short video file:

```bash
curl -X POST \
  "http://localhost:8000/predict_clip?max_frames=10" \
  -F "file=@test_video.mp4" \
  | python3 -m json.tool
```

**Expected Output (example):**
```json
{
  "total_frames": 120,
  "frames_analyzed": 10,
  "contact_frames": [3, 7],
  "max_contact_probability": 0.89,
  "average_helmets_per_frame": 8.3,
  "inference_time_ms": 1523.7
}
```

**Key Metrics Returned:**
- `total_frames`: Total frames in video
- `frames_analyzed`: Number of frames sampled and analyzed
- `contact_frames`: Frame indices where contact was detected
- `max_contact_probability`: Highest contact probability across all frames
- `average_helmets_per_frame`: Average number of helmets detected per frame
- `inference_time_ms`: Total processing time

## Performance Benchmarking

### Test Inference Speed

```bash
# Single frame test
time curl -X POST \
  http://localhost:8000/predict_frame \
  -F "file=@test_frame.jpg" \
  -o /dev/null -s
```

### Load Testing (Optional)

Using Apache Bench:
```bash
# 100 requests with 10 concurrent
ab -n 100 -c 10 -p test_frame.jpg \
  -T 'multipart/form-data; boundary=---boundary' \
  http://localhost:8000/predict_frame
```

## Viewing Logs

### Real-time logs:
```bash
docker logs -f <container_id>
```

### Get container ID:
```bash
docker ps
```

Example log output:
```
INFO:     Frame prediction: 2 helmets, 1 potential contacts, time: 145.23ms
INFO:     Frame prediction: 5 helmets, 3 potential contacts, time: 187.45ms
```

## Troubleshooting

### Container won't start
```bash
# Check for errors
docker logs <container_id>

# Verify weights file exists
ls -lh models/weights.pt
```

### Predictions return 503
This means weights aren't loaded:
```bash
# Check health endpoint
curl http://localhost:8000/health

# Verify weights are mounted correctly
docker exec <container_id> ls -lh /app/models/weights.pt
```

### Slow inference
- Expected on CPU (100-300ms per frame)
- For faster inference, use a GPU-enabled system
- Try smaller images or lower resolution

## Stopping the Container

```bash
# Find container ID
docker ps

# Stop gracefully
docker stop <container_id>

# Or force kill
docker kill <container_id>
```

## Demo Metrics Summary

When running the demo, you should see:

**API Health:**
- Service status: healthy
- Weights loaded: true/false
- Version: 1.0.0

**Detection Metrics (per frame):**
- Number of helmets detected
- Confidence scores (0.0 - 1.0)
- Bounding box coordinates
- Helmet IDs for tracking

**Contact Metrics:**
- Contact probability (0.0 - 1.0)
- Distance between helmet centers (pixels)
- Intersection over Union (IoU)
- Contact classification (true/false based on >0.5 threshold)

**Performance Metrics:**
- Inference time per frame (milliseconds)
- Total processing time for clips
- Frames processed per second

## Next Steps

1. **Test with real NFL footage**: Use actual game clips
2. **Fine-tune detection**: Adjust CONFIDENCE_THRESHOLD environment variable
3. **Train custom model**: Use NFL helmet dataset for better accuracy
4. **Add visualization**: Draw bounding boxes on output images
5. **Deploy to cloud**: Use AWS, GCP, or Azure for production deployment

## Example Full Test Session

```bash
# 1. Build
docker build -t helmet-contact-detection:latest .

# 2. Run
docker run -d -p 8000:8000 \
  -v "$(pwd)/models/weights.pt:/app/models/weights.pt:ro" \
  --name helmet-demo \
  helmet-contact-detection:latest

# 3. Test root
curl http://localhost:8000/

# 4. Check health
curl http://localhost:8000/health

# 5. View docs
open http://localhost:8000/docs  # macOS
# or
xdg-open http://localhost:8000/docs  # Linux
# or visit in browser

# 6. Test prediction
curl -X POST http://localhost:8000/predict_frame \
  -F "file=@test_frame.jpg" | python3 -m json.tool

# 7. View logs
docker logs -f helmet-demo

# 8. Stop
docker stop helmet-demo && docker rm helmet-demo
```

That's it! You now have a working NFL Helmet Contact Detection API running in Docker with real-time metrics.
