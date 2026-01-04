# Quick Demo - Copy & Paste Commands

Complete command sequence to get the NFL Helmet Contact Detection API running with Docker.

## Prerequisites

Ensure you have:
- Docker installed and running
- Python 3.8+ (for downloading test model)
- Terminal/command prompt open

## Step 1: Navigate to Project Directory

```bash
cd helmet_contact_detection
```

## Step 2: Download a Test Model

```bash
# Install ultralytics (if not already installed)
pip install ultralytics

# Download YOLOv8n model and save as weights.pt
python3 scripts/download_pretrained_yolo.py
```

**Expected output:**
```
Downloading YOLOv8N model...
Saving to models/weights.pt...
✓ Success! Model saved to models/weights.pt
```

## Step 3: Build Docker Image

```bash
docker build -t helmet-contact-detection:latest .
```

**Build time:** ~2-3 minutes

## Step 4: Run Container

```bash
docker run -d -p 8000:8000 \
  -v "$(pwd)/models/weights.pt:/app/models/weights.pt:ro" \
  --name helmet-demo \
  helmet-contact-detection:latest
```

**Verify it's running:**
```bash
docker ps
```

You should see `helmet-demo` in the list.

## Step 5: Test the API

### Test 1: Root endpoint
```bash
curl http://localhost:8000/
```

**Expected:** JSON with API info and endpoint links

### Test 2: Health check
```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{
  "status": "healthy",
  "weights_loaded": true,
  "message": "Model weights loaded and ready for inference",
  "version": "1.0.0"
}
```

### Test 3: View interactive docs
Open in your browser:
```
http://localhost:8000/docs
```

### Test 4: Create test image
```bash
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

### Test 5: Run prediction
```bash
curl -X POST http://localhost:8000/predict_frame \
  -F "file=@test_frame.jpg" \
  | python3 -m json.tool
```

**Expected:** JSON response with:
- `helmets`: Array of detected objects with bounding boxes
- `contacts`: Potential contacts between objects
- `frame_has_contact`: Boolean (true/false)
- `inference_time_ms`: Processing time

## Step 6: View Logs

```bash
docker logs -f helmet-demo
```

Press Ctrl+C to stop viewing logs (container keeps running)

## Step 7: Stop Demo

```bash
docker stop helmet-demo
docker rm helmet-demo
```

---

## Complete One-Liner Test Sequence

Once the container is running, test all endpoints:

```bash
echo "=== Testing Root ===" && \
curl -s http://localhost:8000/ | python3 -m json.tool && \
echo -e "\n=== Testing Health ===" && \
curl -s http://localhost:8000/health | python3 -m json.tool && \
echo -e "\n=== Creating Test Image ===" && \
python3 -c "from PIL import Image, ImageDraw; img = Image.new('RGB', (640, 480), 'white'); draw = ImageDraw.Draw(img); draw.rectangle([100, 100, 200, 200], fill='blue'); draw.rectangle([300, 200, 400, 300], fill='green'); img.save('test_frame.jpg'); print('✓ Image created')" && \
echo -e "\n=== Testing Prediction ===" && \
curl -s -X POST http://localhost:8000/predict_frame -F "file=@test_frame.jpg" | python3 -m json.tool
```

---

## Metrics You'll See

**Health Metrics:**
- Service status
- Model weights loaded (true/false)
- Status message
- API version

**Detection Metrics (per prediction):**
- Number of objects detected
- Confidence scores (0.0-1.0)
- Bounding box coordinates (x1, y1, x2, y2)
- Object IDs

**Contact Metrics:**
- Contact probability (0.0-1.0)
- Distance between objects (pixels)
- Intersection over Union (IoU)
- Frame has contact classification

**Performance Metrics:**
- Inference time in milliseconds
- Typical range: 100-300ms on CPU

---

## Troubleshooting

**Issue:** Container won't start
```bash
docker logs helmet-demo
```

**Issue:** Port 8000 already in use
```bash
# Use different port
docker run -d -p 8080:8000 \
  -v "$(pwd)/models/weights.pt:/app/models/weights.pt:ro" \
  --name helmet-demo \
  helmet-contact-detection:latest

# Then access at http://localhost:8080
```

**Issue:** Health shows weights_loaded: false
```bash
# Check if weights file exists
ls -lh models/weights.pt

# Verify volume mount
docker exec helmet-demo ls -lh /app/models/weights.pt
```

**Issue:** Slow predictions
- Normal on CPU (100-300ms per frame)
- For faster inference, use GPU-enabled Docker image

---

## What You're Seeing

The pretrained YOLOv8n model detects **general objects** (people, cars, sports balls, etc.) from the COCO dataset, not specifically NFL helmets.

For actual helmet detection, you would need to:
1. Download the NFL Helmet Contact Detection dataset from Kaggle
2. Train YOLOv8 on helmet images
3. Export the trained weights as `weights.pt`

This demo shows the **API functionality and metrics** are working correctly. With a helmet-trained model, you'd see accurate helmet detections and contact predictions from NFL game footage.

---

## Next Steps

1. ✓ Verify API is working with test model
2. Train custom model on NFL dataset (see `scripts/download_kaggle_instructions.md`)
3. Replace `models/weights.pt` with trained weights
4. Test with real NFL game footage
5. Deploy to production environment

---

**Need more details?** See `TESTING_GUIDE.md` for comprehensive testing instructions.
