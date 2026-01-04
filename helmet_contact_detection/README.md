# NFL Helmet Contact Detection API

A FastAPI-based MVP demo for detecting helmets and predicting contact/impact events in NFL game footage using YOLO-based object detection.

## Features

- **Helmet Detection**: YOLOv8-based helmet detection in images and video frames
- **Contact Prediction**: Geometric heuristics to identify potential helmet-to-helmet contacts
- **RESTful API**: FastAPI endpoints for single-frame and video clip inference
- **Docker Deployment**: CPU-optimized containerized deployment
- **Type Safety**: Pydantic schemas for request/response validation
- **Health Checks**: Basic health endpoint and error handling

## Quick Start

### Prerequisites

- Docker installed on your system
- (Optional) Trained YOLO weights for helmet detection

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t helmet-contact-detection:latest .

# Run the container
docker run -p 8000:8000 helmet-contact-detection:latest
```

The API will be available at `http://localhost:8000`

### Interactive API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check

**GET** `/health`

Check service health and model load status.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "weights_loaded": true,
  "message": "Model weights loaded and ready for inference",
  "version": "1.0.0"
}
```

### Single Frame Prediction

**POST** `/predict_frame`

Detect helmets and predict contacts in a single image.

```bash
curl -X POST \
  http://localhost:8000/predict_frame \
  -F "file=@path/to/frame.jpg"
```

**Response:**
```json
{
  "helmets": [
    {
      "bbox": {
        "x1": 120.5,
        "y1": 80.2,
        "x2": 180.3,
        "y2": 140.1
      },
      "confidence": 0.92,
      "class_name": "helmet",
      "helmet_id": 0
    }
  ],
  "contacts": [
    {
      "helmet_1_id": 0,
      "helmet_2_id": 1,
      "contact_probability": 0.85,
      "distance": 45.2,
      "overlap_iou": 0.15
    }
  ],
  "frame_has_contact": true,
  "inference_time_ms": 124.5
}
```

### Video Clip Prediction

**POST** `/predict_clip?max_frames=30`

Analyze a video clip and detect contact events across frames.

```bash
curl -X POST \
  "http://localhost:8000/predict_clip?max_frames=30" \
  -F "file=@path/to/clip.mp4"
```

**Response:**
```json
{
  "total_frames": 120,
  "frames_analyzed": 30,
  "contact_frames": [5, 12, 18],
  "max_contact_probability": 0.92,
  "average_helmets_per_frame": 8.5,
  "inference_time_ms": 3420.1
}
```

## Project Structure

```
helmet_contact_detection/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── inference.py      # Model inference logic
│   ├── schemas.py        # Pydantic models
│   └── utils.py          # Helper functions
├── models/
│   └── weights.pt        # YOLO model weights (placeholder)
├── scripts/
│   ├── extract_frames.py              # Frame extraction utility
│   └── download_kaggle_instructions.md # Dataset guide
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container definition
├── .dockerignore       # Docker build exclusions
└── README.md           # This file
```

## Development Setup

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

Configure the service with environment variables:

```bash
export MODEL_PATH=/app/models/weights.pt
export CONFIDENCE_THRESHOLD=0.25
export PORT=8000
```

Or create a `.env` file:

```env
MODEL_PATH=/app/models/weights.pt
CONFIDENCE_THRESHOLD=0.25
PORT=8000
```

## Model Setup

### Option 1: Use Placeholder (Demo Mode)

The API will start without weights but predictions will return HTTP 503 errors until real weights are provided.

### Option 2: Train Your Own Model

See `scripts/download_kaggle_instructions.md` for:
- Downloading the NFL Helmet Contact Detection dataset
- Training a YOLO model on helmet detection
- Exporting weights in the correct format

### Option 3: Use Pre-trained Weights

If you have access to pre-trained YOLO weights:

1. Place the `.pt` file at `models/weights.pt`
2. Ensure the model was trained on helmet detection
3. Rebuild the Docker image

## Adding Weights

### Option A: Build weights into the image

Place your `weights.pt` file in the `models/` directory before building:

```bash
# Copy your weights
cp /path/to/your/weights.pt models/weights.pt

# Build the image
docker build -t helmet-contact-detection:latest .

# Run
docker run -p 8000:8000 helmet-contact-detection:latest
```

### Option B: Mount weights at runtime (recommended)

Keep weights separate and mount them as a volume:

```bash
# Build the image once (no weights needed)
docker build -t helmet-contact-detection:latest .

# Run with volume-mounted weights
docker run \
  -p 8000:8000 \
  -v /path/to/your/weights.pt:/app/models/weights.pt:ro \
  helmet-contact-detection:latest
```

This approach allows you to:
- Update weights without rebuilding the image
- Keep weights out of version control
- Use different weights for different runs

## Utilities

### Extract Frames from Video

Use the provided script to extract frames for testing:

```bash
python scripts/extract_frames.py \
  path/to/video.mp4 \
  output_frames/ \
  --fps 1 \
  --max-frames 100
```

**Arguments:**
- `video_path`: Path to input video
- `output_dir`: Directory to save frames
- `--fps`: Frames per second to extract (default: 1.0)
- `--max-frames`: Maximum frames to extract (optional)

## Docker Usage

### Build Image

```bash
docker build -t helmet-contact-detection:latest .
```

### Run Container

**Basic:**
```bash
docker run -p 8000:8000 helmet-contact-detection:latest
```

**With Custom Weights:**
```bash
docker run \
  -p 8000:8000 \
  -v /path/to/your/weights.pt:/app/models/weights.pt:ro \
  helmet-contact-detection:latest
```

**With Environment Variables:**
```bash
docker run \
  -p 8000:8000 \
  -e CONFIDENCE_THRESHOLD=0.3 \
  helmet-contact-detection:latest
```

### View Logs

```bash
docker logs <container_id>
```

### Stop Container

```bash
docker stop <container_id>
```

## Testing the API

### Using cURL

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predict frame:**
```bash
curl -X POST \
  http://localhost:8000/predict_frame \
  -F "file=@test_frame.jpg" \
  | jq
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict frame
with open("test_frame.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict_frame",
        files=files
    )
    print(response.json())
```

## Architecture

### Detection Pipeline

1. **Image Input**: Accept JPEG/PNG image via FastAPI
2. **YOLO Detection**: Run YOLOv8 inference for helmet bounding boxes
3. **Contact Analysis**: Analyze helmet pairs using:
   - Intersection over Union (IoU)
   - Center-to-center distance
   - Geometric heuristics
4. **Response**: Return structured JSON with detections and predictions

### Contact Prediction Methodology

The current implementation uses geometric heuristics:

- **IoU Threshold**: Helmets with IoU > 0.1 are candidates
- **Distance Threshold**: Centers within 100 pixels are candidates
- **Probability Calculation**: Weighted combination of IoU and distance

**Future Enhancements:**
- Temporal features (velocity, acceleration)
- Multi-frame context
- Learned contact classifier
- Player tracking integration

## Performance Notes

Inference speed varies based on hardware, image resolution, and model size. CPU inference is functional but slower than GPU.

**Optimization Tips:**
- Use GPU for faster inference (requires CUDA-enabled Docker image)
- Reduce image resolution for faster processing
- Adjust `max_frames` parameter for video clips
- Use smaller YOLO models (e.g., YOLOv8n vs YOLOv8x)

## Limitations

1. **Geometric Heuristics**: Contact prediction uses simple geometric rules, not learned features
2. **Single-frame Context**: Does not use temporal information across frames
3. **CPU-only**: Docker image optimized for CPU; GPU version would be significantly faster
4. **No Tracking**: Helmet IDs are per-frame, not tracked across video

## Troubleshooting

### Model Not Loading

**Symptom:** API starts but predictions fail with "Model not loaded"

**Solution:**
- Ensure `models/weights.pt` exists and is a valid YOLO model
- Check file permissions
- Review container logs: `docker logs <container_id>`

### Out of Memory Errors

**Symptom:** Container crashes during video processing

**Solution:**
- Reduce `max_frames` parameter
- Increase Docker memory limit
- Process shorter video clips

### Slow Inference

**Symptom:** Predictions take several seconds

**Solution:**
- This is expected on CPU
- Consider GPU deployment for faster inference
- Reduce image resolution
- Use smaller YOLO model (e.g., YOLOv8n instead of YOLOv8x)

## License

This project is provided as-is for educational and demonstration purposes.

The NFL dataset is subject to Kaggle competition rules. Please review the dataset terms before use.

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add GPU support
- [ ] Implement temporal tracking
- [ ] Add video output with visualizations
- [ ] Improve contact prediction with learned classifier
- [ ] Add batch processing endpoint
- [ ] Integration with player tracking data

## Contact

For questions or issues, please open an issue in the repository.

## Acknowledgments

- NFL for providing the dataset
- Ultralytics for the YOLO framework
- FastAPI team for the excellent web framework
