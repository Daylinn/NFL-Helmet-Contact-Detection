"""Streamlit web UI for helmet impact detection."""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.impact_detector.config import load_config
from src.impact_detector.inference import run_inference

# Page config
st.set_page_config(
    page_title="NFL Helmet Impact Detection",
    page_icon="🏈",
    layout="wide",
)

# Load config
@st.cache_resource
def get_config():
    return load_config()


config = get_config()


# Title
st.title("🏈 NFL Helmet Impact Detection")
st.markdown(
    """
    Upload NFL game footage to automatically detect helmet-to-helmet impacts.
    The system uses computer vision to identify helmets and classify potential impacts.
    """
)

# Sidebar settings
st.sidebar.header("Settings")

sample_rate = st.sidebar.slider(
    "Sample Rate (process every N frames)",
    min_value=1,
    max_value=30,
    value=config.inference.sample_rate,
    help="Higher values = faster processing but may miss impacts",
)

min_score = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=config.inference.min_score,
    step=0.05,
    help="Minimum confidence threshold for reporting impacts",
)

annotate = st.sidebar.checkbox(
    "Generate Annotated Video",
    value=True,
    help="Create output video with impact detections highlighted",
)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"],
        help="Upload NFL game footage (MP4, AVI, or MOV)",
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if st.button("🔍 Detect Impacts", type="primary"):
            with st.spinner("Processing video... This may take a few minutes."):
                # Update config
                config.inference.sample_rate = sample_rate
                config.inference.min_score = min_score

                # Prepare output path
                output_path = None
                if annotate:
                    output_path = tmp_path.replace(".mp4", "_annotated.mp4")

                try:
                    # Run inference
                    detections, annotated_path = run_inference(
                        tmp_path,
                        config,
                        annotate=annotate,
                        output_path=output_path,
                    )

                    # Store results in session state
                    st.session_state["detections"] = detections
                    st.session_state["annotated_path"] = annotated_path

                    st.success(f"✓ Processing complete! Found {len(detections)} impacts.")

                except Exception as e:
                    st.error(f"Error during processing: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with col2:
    st.header("Results")

    if "detections" in st.session_state:
        detections = st.session_state["detections"]

        if detections:
            # Summary metrics
            st.metric("Total Impacts Detected", len(detections))

            # Detections table
            st.subheader("Impact Detections")

            df = pd.DataFrame(detections)
            df["time_min:sec"] = df["time_sec"].apply(
                lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
            )
            df["bbox"] = df["bbox"].apply(lambda x: f"({x[0]}, {x[1]}, {x[2]}, {x[3]})")

            st.dataframe(
                df[["frame", "time_min:sec", "bbox", "score"]].style.format({"score": "{:.3f}"}),
                use_container_width=True,
            )

            # Download JSON
            json_data = json.dumps(detections, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_data,
                file_name="impact_detections.json",
                mime="application/json",
            )

            # Show annotated video
            if "annotated_path" in st.session_state and st.session_state["annotated_path"]:
                st.subheader("Annotated Video")
                st.video(st.session_state["annotated_path"])

                # Download annotated video
                with open(st.session_state["annotated_path"], "rb") as f:
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=f,
                        file_name="annotated_output.mp4",
                        mime="video/mp4",
                    )

        else:
            st.info("No impacts detected in this video.")

    else:
        st.info("Upload a video and click 'Detect Impacts' to see results here.")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### About
    This app uses:
    - **YOLOv8** for helmet detection
    - **ResNet** classifier for impact prediction
    - Trained on NFL Impact Detection dataset

    ### Model Status
    """
)

# Check model status
model_path = config.inference.onnx_path if config.inference.use_onnx else config.inference.model_path
model_exists = Path(model_path).exists()

if model_exists:
    st.sidebar.success(f"✓ Model loaded: {Path(model_path).name}")
else:
    st.sidebar.error(f"✗ Model not found: {model_path}")
    st.sidebar.warning("Please train a model first using `make train`")
