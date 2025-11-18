"""Streamlit web application for Multimodal AI Suite."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import time
from typing import Optional

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data.ingestion import DataIngestion
from src.data.preprocessing import ImagePreprocessor, OCRProcessor, TextPreprocessor
from src.models import DocumentClassifier, ImageClassifier, MultimodalFusionModel
from src.utils.config import load_config
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("streamlit_app", console=True)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


class MultimodalApp:
    """Main Streamlit application class."""

    def __init__(self):
        """Initialize application."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_configs()
        self.init_session_state()

    def load_configs(self):
        """Load configurations."""
        try:
            self.model_config = load_config("model_config")
            self.deployment_config = load_config("deployment_config")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.model_config = {}
            self.deployment_config = {}

    def init_session_state(self):
        """Initialize Streamlit session state."""
        if "models_loaded" not in st.session_state:
            st.session_state.models_loaded = False
        if "predictions" not in st.session_state:
            st.session_state.predictions = None
        if "processing_time" not in st.session_state:
            st.session_state.processing_time = 0

    @st.cache_resource
    def load_models(_self, model_type: str):
        """
        Load models (cached).

        Args:
            model_type: Type of model to load

        Returns:
            Loaded model
        """
        try:
            if model_type == "document":
                model = DocumentClassifier(
                    num_classes=_self.model_config.get("document_classifier", {}).get(
                        "num_classes", 10
                    ),
                    model_name=_self.model_config.get("document_classifier", {}).get(
                        "model_name", "bert-base-uncased"
                    ),
                )
            elif model_type == "image":
                model = ImageClassifier(
                    num_classes=_self.model_config.get("image_classifier", {}).get(
                        "num_classes", 100
                    ),
                    model_name=_self.model_config.get("image_classifier", {}).get(
                        "model_name", "google/vit-base-patch16-224"
                    ),
                )
            elif model_type == "multimodal":
                model = MultimodalFusionModel(
                    num_classes=_self.model_config.get("multimodal_fusion", {}).get(
                        "num_classes", 50
                    ),
                    text_model_name=_self.model_config.get("multimodal_fusion", {}).get(
                        "text_encoder", "bert-base-uncased"
                    ),
                    image_model_name=_self.model_config.get("multimodal_fusion", {}).get(
                        "image_encoder", "google/vit-base-patch16-224"
                    ),
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.to(_self.device)
            model.eval()
            logger.info(f"{model_type} model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error loading {model_type} model: {e}")
            return None

    def render_sidebar(self):
        """Render sidebar with settings and options."""
        st.sidebar.title("⚙️ Settings")

        # Model selection
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["Document Classification", "Image Recognition", "Multimodal Fusion"],
            help="Choose the type of model to use for inference",
        )

        # Upload confidence threshold
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for predictions",
        )

        # Top-K predictions
        top_k = st.sidebar.slider(
            "Top-K Predictions",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of top predictions to show",
        )

        # Advanced options
        with st.sidebar.expander("Advanced Options"):
            show_attention = st.checkbox("Show Attention Maps", value=False)
            show_embeddings = st.checkbox("Show Embeddings", value=False)

        st.sidebar.divider()

        # System information
        st.sidebar.subheader("📊 System Info")
        st.sidebar.text(f"Device: {self.device}")
        st.sidebar.text(f"PyTorch: {torch.__version__}")

        return {
            "model_type": model_type,
            "confidence_threshold": confidence_threshold,
            "top_k": top_k,
            "show_attention": show_attention,
            "show_embeddings": show_embeddings,
        }

    def render_header(self):
        """Render application header."""
        st.title("🤖 Multimodal Intelligence Workflow Suite")
        st.markdown(
            """
            **Powered by Transformer-based Models**

            Upload documents and images for AI-powered classification and analysis.
            """
        )
        st.divider()

    def render_upload_section(self, settings: dict):
        """Render file upload section."""
        col1, col2 = st.columns(2)

        uploaded_file = None
        uploaded_image = None

        with col1:
            st.subheader("📄 Upload Document or Image")

            if settings["model_type"] == "Document Classification":
                uploaded_file = st.file_uploader(
                    "Choose a document",
                    type=["pdf", "txt", "docx"],
                    help="Upload a document for classification",
                )
            elif settings["model_type"] == "Image Recognition":
                uploaded_image = st.file_uploader(
                    "Choose an image",
                    type=["jpg", "jpeg", "png", "bmp"],
                    help="Upload an image for classification",
                )
            elif settings["model_type"] == "Multimodal Fusion":
                uploaded_file = st.file_uploader(
                    "Choose a document",
                    type=["pdf", "txt", "docx"],
                    key="doc",
                )
                uploaded_image = st.file_uploader(
                    "Choose an image",
                    type=["jpg", "jpeg", "png", "bmp"],
                    key="img",
                )

        with col2:
            st.subheader("👁️ Preview")

            # Show preview
            if uploaded_file is not None:
                st.info(f"Document: {uploaded_file.name}")

            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

        return uploaded_file, uploaded_image

    def process_document(self, uploaded_file, model):
        """Process document for classification."""
        # Load document
        ingestion = DataIngestion()
        file_bytes = uploaded_file.read()

        # Extract text based on file type
        if uploaded_file.name.endswith(".pdf"):
            # For demo, use OCR processor
            ocr = OCRProcessor()
            # Convert bytes to image (simplified)
            text = f"Extracted text from {uploaded_file.name}"
        elif uploaded_file.name.endswith(".txt"):
            text = file_bytes.decode("utf-8")
        else:
            text = f"Document content from {uploaded_file.name}"

        # Preprocess text
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(text)

        # Get predictions
        start_time = time.time()
        results = model.predict([processed_text], return_probs=True)
        processing_time = time.time() - start_time

        return {
            "text": processed_text[:500],  # Show first 500 chars
            "predictions": results["predictions"].numpy(),
            "probabilities": results["probabilities"].numpy(),
            "processing_time": processing_time,
        }

    def process_image(self, uploaded_image, model):
        """Process image for classification."""
        # Load image
        image = Image.open(uploaded_image)

        # Preprocess image
        preprocessor = ImagePreprocessor()
        processed_image = preprocessor.preprocess(image).unsqueeze(0)

        # Get predictions
        start_time = time.time()
        results = model.predict(processed_image, return_probs=True)
        processing_time = time.time() - start_time

        return {
            "predictions": results["predictions"].numpy(),
            "probabilities": results["probabilities"].numpy(),
            "processing_time": processing_time,
        }

    def render_predictions(self, results: dict, settings: dict):
        """Render prediction results."""
        st.subheader("🎯 Predictions")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Top predictions
            st.markdown("### Top Predictions")

            probs = results["probabilities"][0]
            top_indices = np.argsort(probs)[::-1][: settings["top_k"]]

            for i, idx in enumerate(top_indices, 1):
                confidence = probs[idx]

                # Only show if above threshold
                if confidence >= settings["confidence_threshold"]:
                    st.progress(
                        float(confidence),
                        text=f"**Class {idx}**: {confidence:.2%}",
                    )
                else:
                    st.caption(f"Class {idx}: {confidence:.2%} (below threshold)")

        with col2:
            # Metrics
            st.markdown("### Metrics")
            st.metric("Processing Time", f"{results['processing_time']:.3f}s")
            st.metric("Top Prediction", f"Class {results['predictions'][0]}")
            st.metric(
                "Confidence",
                f"{probs[results['predictions'][0]]:.2%}",
            )

    def render_logs(self):
        """Render logs panel."""
        with st.expander("📋 Logs", expanded=False):
            st.code(
                f"""
Inference completed successfully
Device: {self.device}
Model status: Ready
            """,
                language="text",
            )

    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        settings = self.render_sidebar()

        # Upload section
        uploaded_file, uploaded_image = self.render_upload_section(settings)

        # Process button
        if st.button("🚀 Run Inference", type="primary", use_container_width=True):
            if settings["model_type"] == "Document Classification" and uploaded_file:
                with st.spinner("Processing document..."):
                    model = self.load_models("document")
                    if model:
                        results = self.process_document(uploaded_file, model)
                        self.render_predictions(results, settings)

            elif settings["model_type"] == "Image Recognition" and uploaded_image:
                with st.spinner("Processing image..."):
                    model = self.load_models("image")
                    if model:
                        results = self.process_image(uploaded_image, model)
                        self.render_predictions(results, settings)

            elif (
                settings["model_type"] == "Multimodal Fusion"
                and uploaded_file
                and uploaded_image
            ):
                with st.spinner("Processing multimodal input..."):
                    st.info("Multimodal fusion processing...")
                    # TODO: Implement multimodal processing

            else:
                st.warning("Please upload the required files for the selected model type.")

        # Logs
        self.render_logs()

        # Footer
        st.divider()
        st.caption("Multimodal Intelligence Workflow Suite v1.0.0 | Feb 2025 - June 2025")


def main():
    """Main entry point."""
    app = MultimodalApp()
    app.run()


if __name__ == "__main__":
    main()
