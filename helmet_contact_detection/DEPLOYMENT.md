# Deployment Guide - NFL Helmet Contact Detection

Complete guide for deploying the NFL Helmet Contact Detection API to production.

## Prerequisites

- Docker installed and working ✓
- GitHub account
- Cloud platform account (Railway, Render, AWS, or GCP)
- Model weights (`models/weights.pt`)

## Deployment Options

### Option 1: Railway (Recommended for Quick Deploy)

**Pros:** Fastest deployment, free tier, automatic HTTPS
**Time:** ~5 minutes

#### Steps:

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy to Railway**
   - Go to https://railway.app
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Dockerfile and deploys!

3. **Add environment variables** (in Railway dashboard)
   ```
   MODEL_PATH=/app/models/weights.pt
   CONFIDENCE_THRESHOLD=0.25
   PORT=8000
   ```

4. **Get your public URL**
   - Railway provides: `https://your-app.railway.app`
   - Test: `curl https://your-app.railway.app/health`

**Note:** You'll need to bake weights into the Docker image for Railway (no volume mounts).

---

### Option 2: Render.com

**Pros:** Great free tier, easy setup, good docs
**Time:** ~10 minutes

#### Steps:

1. **Create `render.yaml` in project root**
   ```yaml
   services:
     - type: web
       name: helmet-detection-api
       env: docker
       plan: free
       healthCheckPath: /health
       envVars:
         - key: MODEL_PATH
           value: /app/models/weights.pt
         - key: CONFIDENCE_THRESHOLD
           value: 0.25
   ```

2. **Push to GitHub**
   ```bash
   git add render.yaml
   git commit -m "Add Render config"
   git push origin main
   ```

3. **Deploy on Render**
   - Go to https://render.com
   - New → Web Service
   - Connect GitHub repository
   - Render auto-deploys from `render.yaml`

4. **Access your app**
   - URL: `https://your-app.onrender.com`

---

### Option 3: Google Cloud Run

**Pros:** Serverless, pay-per-use, auto-scaling
**Time:** ~15 minutes

#### Steps:

1. **Install Google Cloud CLI**
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate and setup**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth configure-docker
   ```

3. **Build and push container**
   ```bash
   # Tag image for Google Container Registry
   docker tag helmet-contact-detection:latest \
     gcr.io/YOUR_PROJECT_ID/helmet-detection:latest

   # Push to GCR
   docker push gcr.io/YOUR_PROJECT_ID/helmet-detection:latest
   ```

4. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy helmet-detection \
     --image gcr.io/YOUR_PROJECT_ID/helmet-detection:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8000 \
     --memory 2Gi \
     --set-env-vars="MODEL_PATH=/app/models/weights.pt,CONFIDENCE_THRESHOLD=0.25"
   ```

5. **Get your URL**
   ```bash
   gcloud run services describe helmet-detection \
     --region us-central1 \
     --format="value(status.url)"
   ```

---

### Option 4: AWS ECS with Fargate

**Pros:** Industry standard, great for resume
**Time:** ~30 minutes

#### Steps:

1. **Install AWS CLI**
   ```bash
   # macOS
   brew install awscli

   # Configure
   aws configure
   ```

2. **Create ECR repository**
   ```bash
   aws ecr create-repository --repository-name helmet-detection
   ```

3. **Build and push to ECR**
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin \
     YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

   # Tag and push
   docker tag helmet-contact-detection:latest \
     YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/helmet-detection:latest

   docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/helmet-detection:latest
   ```

4. **Create ECS Task Definition** (use AWS Console or CLI)
   - Container: Your ECR image
   - Memory: 2GB
   - CPU: 1 vCPU
   - Port: 8000
   - Environment variables: MODEL_PATH, CONFIDENCE_THRESHOLD

5. **Create ECS Service**
   - Launch type: Fargate
   - Task definition: Your task
   - Desired tasks: 1
   - Load balancer: Application Load Balancer
   - Public IP: Enable

6. **Access via ALB DNS name**

---

## Important: Model Weights for Deployment

For cloud deployment, you need to **bake weights into the Docker image** (no volume mounts):

1. **Ensure weights are in `models/` before building:**
   ```bash
   ls -lh models/weights.pt
   ```

2. **Rebuild Docker image with weights:**
   ```bash
   docker build -t helmet-contact-detection:latest .
   ```

3. **Verify weights are in the image:**
   ```bash
   docker run helmet-contact-detection:latest ls -lh /app/models/
   ```

---

## Post-Deployment Checklist

- [ ] Health check returns 200: `curl https://your-url/health`
- [ ] Weights loaded: Check `"weights_loaded": true`
- [ ] Test prediction: Upload sample image
- [ ] Check logs for errors
- [ ] Set up custom domain (optional)
- [ ] Add monitoring (optional)

---

## Monitoring & Logging

### Railway
- Built-in logs in dashboard
- Metrics automatically tracked

### Render
- Logs in dashboard
- Can integrate with external monitoring

### Google Cloud Run
```bash
# View logs
gcloud run services logs read helmet-detection --region us-central1

# Monitor requests
gcloud run services describe helmet-detection --region us-central1
```

### AWS ECS
- CloudWatch Logs automatically enabled
- View in CloudWatch console

---

## Cost Estimates

| Platform | Free Tier | Paid (approx) |
|----------|-----------|---------------|
| Railway | 500 hrs/month | $5-20/month |
| Render | 750 hrs/month | $7-25/month |
| Cloud Run | Free tier generous | $5-15/month |
| AWS ECS | Limited free tier | $10-30/month |

---

## Next Steps

1. Choose deployment platform
2. Build Docker image with weights
3. Push to GitHub
4. Deploy following platform-specific steps
5. Test deployed API
6. Add to portfolio with public URL!

---

## Troubleshooting

**Container won't start:**
- Check logs for errors
- Verify PORT environment variable
- Ensure weights file exists in image

**503 errors:**
- Model not loading
- Check weights_loaded in /health
- Review startup logs

**Slow performance:**
- CPU inference is slow (~200-300ms/frame)
- Consider GPU instances for production
- Increase memory allocation

**Out of memory:**
- Increase container memory (2GB recommended)
- Reduce max_frames for video processing
- Use smaller YOLO model (yolov8n)

---

## Portfolio Documentation

Add this to your README:

```markdown
## 🌐 Live Demo

**API Endpoint:** https://your-deployed-url.com

**Try it:**
```bash
# Health check
curl https://your-url.com/health

# Interactive docs
https://your-url.com/docs
```

**Tech Stack:**
- FastAPI (Python web framework)
- Docker (containerization)
- YOLOv8 (object detection)
- [Platform] (deployment)
- NFL Impact Detection dataset
```
