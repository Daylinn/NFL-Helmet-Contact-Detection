"""
FastAPI application for NFL Helmet Contact Detection.
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.inference import HelmetContactDetector
from app.schemas import (
    HealthResponse,
    FramePredictionResponse,
    ClipPredictionResponse,
    ErrorResponse
)
from app.utils import load_image_from_bytes, load_video_from_bytes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global detector instance
detector: HelmetContactDetector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup: Load model
    global detector

    model_path = os.getenv("MODEL_PATH", "/app/models/weights.pt")
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

    logger.info("Initializing Helmet Contact Detector...")
    detector = HelmetContactDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )

    try:
        detector.load_model()
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"Model weights not found: {e}")
        logger.warning("API will start but predictions will fail until weights are provided")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but predictions will fail")

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down Helmet Contact Detector...")


# Initialize FastAPI app
app = FastAPI(
    title="NFL Helmet Contact Detection API",
    description="API for detecting helmets and predicting contact/impact events in NFL footage",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """
    Root endpoint with API information and links.

    Returns:
        JSON with API name and endpoint links
    """
    return {
        "name": "NFL Helmet Contact Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict_frame": "/predict_frame",
            "predict_clip": "/predict_clip"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status and model load state
    """
    weights_loaded = detector.weights_loaded if detector else False

    if weights_loaded:
        message = "Model weights loaded and ready for inference"
    else:
        message = "Model weights not loaded - predictions will fail until weights.pt is provided"

    return HealthResponse(
        status="healthy",
        weights_loaded=weights_loaded,
        message=message,
        version="1.0.0"
    )


@app.post("/predict_frame", response_model=FramePredictionResponse)
async def predict_frame(file: UploadFile = File(...)):
    """
    Predict helmet detections and contacts for a single frame/image.

    Args:
        file: Image file (JPEG, PNG, etc.)

    Returns:
        FramePredictionResponse with detections and contact predictions

    Raises:
        HTTPException: If model not loaded or processing fails
    """
    if not detector or not detector.weights_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model weights not loaded. Please provide weights.pt file - see README for instructions."
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Load and process image
        image = load_image_from_bytes(image_bytes)

        # Run inference
        result = detector.predict_frame(image)

        logger.info(
            f"Frame prediction: {len(result.helmets)} helmets, "
            f"{len(result.contacts)} potential contacts, "
            f"time: {result.inference_time_ms:.2f}ms"
        )

        return result

    except ValueError as e:
        logger.error(f"Invalid image: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_clip", response_model=ClipPredictionResponse)
async def predict_clip(
    file: UploadFile = File(...),
    max_frames: int = 30
):
    """
    Predict helmet contacts for a video clip.

    Samples frames uniformly and runs detection on each.

    Args:
        file: Video file (MP4, AVI, etc.)
        max_frames: Maximum number of frames to analyze (default: 30)

    Returns:
        ClipPredictionResponse with aggregated predictions

    Raises:
        HTTPException: If model not loaded or processing fails
    """
    if not detector or not detector.weights_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model weights not loaded. Please provide weights.pt file - see README for instructions."
        )

    if max_frames < 1 or max_frames > 100:
        raise HTTPException(
            status_code=400,
            detail="max_frames must be between 1 and 100"
        )

    try:
        # Read video bytes
        video_bytes = await file.read()

        # Load and extract frames
        frames = load_video_from_bytes(video_bytes)

        # Run inference
        result = detector.predict_clip(frames, max_frames=max_frames)

        logger.info(
            f"Clip prediction: {result.total_frames} total frames, "
            f"{result.frames_analyzed} analyzed, "
            f"{len(result.contact_frames)} contact frames, "
            f"time: {result.inference_time_ms:.2f}ms"
        )

        return result

    except ValueError as e:
        logger.error(f"Invalid video: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom exception handler for general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
