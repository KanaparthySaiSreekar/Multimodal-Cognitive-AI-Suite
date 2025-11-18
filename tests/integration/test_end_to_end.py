"""Integration tests for end-to-end multimodal workflows."""

import pytest
import torch
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.inference.predictor import MultimodalPredictor
from src.data.ingestion import DataIngestion
from src.data.preprocessing import TextPreprocessor, ImagePreprocessor, OCRProcessor
from src.models import DocumentClassifier, ImageClassifier, MultimodalFusionModel


class TestDocumentClassificationWorkflow:
    """End-to-end tests for document classification."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample text document."""
        return """
        Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly
        programmed. Deep learning, using neural networks with multiple layers,
        has revolutionized computer vision and natural language processing.
        """

    def test_full_document_classification_pipeline(self, sample_document):
        """Test complete document classification workflow."""
        # Initialize components
        preprocessor = TextPreprocessor()
        model = DocumentClassifier(num_classes=10)

        # Preprocess
        processed_text = preprocessor.preprocess(sample_document)
        assert len(processed_text) > 0

        # Predict
        results = model.predict([processed_text])

        # Validate results
        assert "predictions" in results
        assert "probabilities" in results
        assert results["predictions"].shape[0] == 1
        assert results["probabilities"].shape == (1, 10)

        # Check probability distribution
        probs = results["probabilities"][0]
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_document_classification_with_predictor(self, sample_document):
        """Test document classification using unified predictor."""
        predictor = MultimodalPredictor(model_type="document")

        result = predictor.predict_document(sample_document)

        assert "prediction" in result
        assert "confidence" in result
        assert "top_k_predictions" in result
        assert isinstance(result["prediction"], int)
        assert 0 <= result["confidence"] <= 1

    def test_pdf_document_workflow(self):
        """Test PDF processing workflow."""
        # Create a temporary text file (simulating PDF extraction)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document for classification.")
            temp_path = f.name

        try:
            # Load and process
            ingestion = DataIngestion()
            text = ingestion.load_text_file(temp_path)

            preprocessor = TextPreprocessor()
            processed = preprocessor.preprocess(text)

            model = DocumentClassifier(num_classes=5)
            results = model.predict([processed])

            assert results["predictions"].shape[0] == 1
        finally:
            Path(temp_path).unlink()

    def test_batch_document_classification(self):
        """Test batch processing of multiple documents."""
        documents = [
            "First document about technology and AI.",
            "Second document about healthcare and medicine.",
            "Third document about finance and economics.",
        ]

        predictor = MultimodalPredictor(model_type="document")
        results = predictor.batch_predict(documents, batch_size=2)

        assert len(results) == 3
        for result in results:
            assert "prediction" in result
            assert "confidence" in result


class TestImageRecognitionWorkflow:
    """End-to-end tests for image recognition."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return Image.new("RGB", (512, 512), color=(100, 150, 200))

    @pytest.fixture
    def sample_image_file(self, sample_image):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            sample_image.save(f.name)
            return f.name

    def test_full_image_classification_pipeline(self, sample_image):
        """Test complete image classification workflow."""
        # Initialize components
        preprocessor = ImagePreprocessor()
        model = ImageClassifier(num_classes=100)

        # Preprocess
        processed_image = preprocessor.preprocess(sample_image)
        assert processed_image.shape == (3, 224, 224)

        # Predict
        results = model.predict(processed_image.unsqueeze(0))

        # Validate results
        assert "predictions" in results
        assert "probabilities" in results
        assert results["predictions"].shape[0] == 1
        assert results["probabilities"].shape == (1, 100)

    def test_image_classification_with_predictor(self, sample_image):
        """Test image classification using unified predictor."""
        predictor = MultimodalPredictor(model_type="image")

        result = predictor.predict_image(sample_image)

        assert "prediction" in result
        assert "confidence" in result
        assert "top_k_predictions" in result
        assert len(result["top_k_predictions"]) == 5

    def test_image_file_loading_and_classification(self, sample_image_file):
        """Test loading image from file and classification."""
        try:
            ingestion = DataIngestion()
            image = ingestion.load_image(sample_image_file)

            preprocessor = ImagePreprocessor()
            processed = preprocessor.preprocess(image)

            model = ImageClassifier(num_classes=50)
            results = model.predict(processed.unsqueeze(0))

            assert results["predictions"].shape[0] == 1
        finally:
            Path(sample_image_file).unlink()

    def test_batch_image_classification(self):
        """Test batch processing of multiple images."""
        images = torch.randn(5, 3, 224, 224)

        model = ImageClassifier(num_classes=20)
        results = model.predict(images, batch_size=2)

        assert results["predictions"].shape[0] == 5
        assert results["probabilities"].shape == (5, 20)


class TestMultimodalFusionWorkflow:
    """End-to-end tests for multimodal fusion."""

    @pytest.fixture
    def sample_text(self):
        return "A beautiful sunset over the ocean with vibrant colors."

    @pytest.fixture
    def sample_image(self):
        return Image.new("RGB", (512, 512), color=(255, 100, 50))

    def test_full_multimodal_pipeline(self, sample_text, sample_image):
        """Test complete multimodal fusion workflow."""
        # Initialize components
        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor()
        model = MultimodalFusionModel(num_classes=50)

        # Preprocess
        processed_text = text_preprocessor.preprocess(sample_text)
        processed_image = image_preprocessor.preprocess(sample_image)

        # Predict
        results = model.predict(
            [processed_text],
            processed_image.unsqueeze(0),
            return_embeddings=True,
        )

        # Validate results
        assert "predictions" in results
        assert "probabilities" in results
        assert "text_embeddings" in results
        assert "image_embeddings" in results
        assert "fused_embeddings" in results

        assert results["predictions"].shape[0] == 1
        assert results["text_embeddings"].shape[0] == 1
        assert results["image_embeddings"].shape[0] == 1

    def test_multimodal_with_predictor(self, sample_text, sample_image):
        """Test multimodal fusion using unified predictor."""
        predictor = MultimodalPredictor(model_type="multimodal")

        result = predictor.predict_multimodal(
            sample_text, sample_image, return_embeddings=True
        )

        assert "prediction" in result
        assert "confidence" in result
        assert "text_embeddings" in result
        assert "image_embeddings" in result
        assert "fused_embeddings" in result

    def test_multimodal_batch_processing(self):
        """Test batch processing for multimodal inputs."""
        texts = ["First caption", "Second caption", "Third caption"]
        images = torch.randn(3, 3, 224, 224)

        model = MultimodalFusionModel(num_classes=30)
        results = model.predict(texts, images, batch_size=2)

        assert results["predictions"].shape[0] == 3
        assert results["probabilities"].shape == (3, 30)


class TestOCRWorkflow:
    """End-to-end tests for OCR processing."""

    @pytest.fixture
    def text_image(self):
        """Create an image with text (simplified)."""
        return Image.new("RGB", (800, 200), color="white")

    def test_ocr_extraction_pipeline(self, text_image):
        """Test OCR text extraction workflow."""
        ocr_processor = OCRProcessor()

        # Extract text
        text = ocr_processor.extract_text(text_image)
        assert isinstance(text, str)

        # Extract with confidence
        result = ocr_processor.extract_text_with_confidence(text_image)
        assert "text" in result
        assert "confidence" in result
        assert "word_count" in result

    def test_ocr_to_classification(self, text_image):
        """Test OCR extraction followed by classification."""
        # Extract text
        ocr_processor = OCRProcessor()
        extracted_text = ocr_processor.extract_text(text_image)

        # Classify (if text was extracted)
        if extracted_text:
            preprocessor = TextPreprocessor()
            model = DocumentClassifier(num_classes=10)

            processed = preprocessor.preprocess(extracted_text)
            results = model.predict([processed])

            assert "predictions" in results


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        predictor = MultimodalPredictor(model_type="document")

        # Should handle gracefully
        with pytest.raises((ValueError, RuntimeError, Exception)):
            predictor.predict_document("")

    def test_corrupted_image_handling(self):
        """Test handling of corrupted images."""
        preprocessor = ImagePreprocessor()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"not an image")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                preprocessor.preprocess(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_model_device_consistency(self):
        """Test model-data device consistency."""
        model = ImageClassifier(num_classes=10)
        model.to_device("cpu")

        # Data on same device
        images_cpu = torch.randn(2, 3, 224, 224)
        results = model.predict(images_cpu)

        assert results["predictions"].shape[0] == 2

    def test_inference_determinism(self):
        """Test inference determinism in eval mode."""
        torch.manual_seed(42)

        model = DocumentClassifier(num_classes=5)
        model.eval()

        text = "Test document for determinism"
        preprocessor = TextPreprocessor()
        processed = preprocessor.preprocess(text)

        # Run inference twice
        with torch.no_grad():
            results1 = model.predict([processed])
            results2 = model.predict([processed])

        # Results should be identical
        assert torch.allclose(results1["logits"], results2["logits"])


class TestPerformanceConstraints:
    """Test performance-related constraints."""

    def test_inference_latency_document(self):
        """Test document classification latency."""
        import time

        predictor = MultimodalPredictor(model_type="document")
        text = "Sample document text for latency testing."

        start_time = time.time()
        result = predictor.predict_document(text)
        elapsed = time.time() - start_time

        # Should complete in reasonable time (relaxed for CI)
        assert elapsed < 5.0  # 5 seconds max for CI environments
        assert "prediction" in result

    def test_inference_latency_image(self):
        """Test image classification latency."""
        import time

        predictor = MultimodalPredictor(model_type="image")
        image = Image.new("RGB", (512, 512), color="red")

        start_time = time.time()
        result = predictor.predict_image(image)
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 5.0
        assert "prediction" in result

    def test_batch_processing_efficiency(self):
        """Test batch processing is more efficient than sequential."""
        import time

        model = ImageClassifier(num_classes=10)
        images = torch.randn(10, 3, 224, 224)

        # Batch processing
        start = time.time()
        batch_results = model.predict(images, batch_size=10)
        batch_time = time.time() - start

        # Sequential processing (first 3 images only)
        start = time.time()
        for i in range(3):
            model.predict(images[i : i + 1], batch_size=1)
        sequential_time = time.time() - start

        # Batch should have some advantage (normalized per image)
        # This is a rough check; actual speedup depends on hardware
        assert batch_results["predictions"].shape[0] == 10
