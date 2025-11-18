"""End-to-end smoke tests for inference endpoints.

Fast smoke tests that validate critical paths work end-to-end.
Designed to run in CI/CD pipelines before deployment.

Target: Complete all smoke tests in <30 seconds.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.predictor import MultimodalPredictor
from src.models.document_classifier import DocumentClassifier
from src.models.image_classifier import ImageClassifier
from src.models.multimodal_fusion import MultimodalFusionModel
from src.utils.errors import MultimodalError


class TestDocumentInferenceSmoke:
    """Smoke tests for document classification."""

    def test_document_classification_happy_path(self):
        """Test basic document classification works end-to-end."""
        # Given: A simple text document
        text = "This is a test document about artificial intelligence and machine learning."

        # When: We classify it
        predictor = MultimodalPredictor(model_type="document")
        start_time = time.time()
        result = predictor.predict_document(text)
        latency_ms = (time.time() - start_time) * 1000

        # Then: We get a valid response
        assert result is not None
        assert "prediction" in result
        assert "confidence" in result
        assert isinstance(result["prediction"], (int, str))
        assert 0 <= result["confidence"] <= 1

        # And: Latency is reasonable (smoke test threshold)
        assert latency_ms < 5000, f"Latency {latency_ms:.0f}ms exceeds smoke test threshold"

    def test_document_classification_response_schema(self):
        """Test that response follows expected JSON schema."""
        text = "Sample document for schema validation."
        predictor = MultimodalPredictor(model_type="document")
        result = predictor.predict_document(text)

        # Validate schema
        required_fields = ["prediction", "confidence"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Validate types
        assert isinstance(result["prediction"], (int, str, np.int64))
        assert isinstance(result["confidence"], (float, np.float64))

    def test_document_classification_batch(self):
        """Test batch document classification."""
        texts = [
            "First document about technology.",
            "Second document about healthcare.",
            "Third document about finance.",
        ]

        predictor = MultimodalPredictor(model_type="document")
        results = predictor.predict_batch_documents(texts)

        assert len(results) == len(texts)
        for result in results:
            assert "prediction" in result
            assert "confidence" in result

    def test_empty_document_handling(self):
        """Test that empty documents are handled gracefully."""
        predictor = MultimodalPredictor(model_type="document")

        with pytest.raises(MultimodalError) as exc_info:
            predictor.predict_document("")

        assert exc_info.value.code == "VAL_002"


class TestImageInferenceSmoke:
    """Smoke tests for image classification."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing."""
        # Create 224x224 RGB image with random noise
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

    def test_image_classification_happy_path(self, sample_image):
        """Test basic image classification works end-to-end."""
        # When: We classify an image
        predictor = MultimodalPredictor(model_type="image")
        start_time = time.time()
        result = predictor.predict_image(sample_image)
        latency_ms = (time.time() - start_time) * 1000

        # Then: We get a valid response
        assert result is not None
        assert "prediction" in result
        assert "confidence" in result
        assert isinstance(result["prediction"], (int, str))
        assert 0 <= result["confidence"] <= 1

        # And: Latency is reasonable
        assert latency_ms < 5000, f"Latency {latency_ms:.0f}ms exceeds smoke test threshold"

    def test_image_classification_response_schema(self, sample_image):
        """Test that response follows expected JSON schema."""
        predictor = MultimodalPredictor(model_type="image")
        result = predictor.predict_image(sample_image)

        # Validate schema
        required_fields = ["prediction", "confidence"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Validate types
        assert isinstance(result["prediction"], (int, str, np.int64))
        assert isinstance(result["confidence"], (float, np.float64))

    def test_image_from_file_path(self, sample_image, tmp_path):
        """Test loading and classifying image from file path."""
        # Save image to temp file
        image_path = tmp_path / "test_image.png"
        sample_image.save(image_path)

        # Classify from path
        predictor = MultimodalPredictor(model_type="image")
        result = predictor.predict_image_from_path(str(image_path))

        assert "prediction" in result
        assert "confidence" in result

    def test_grayscale_image_conversion(self):
        """Test that grayscale images are auto-converted to RGB."""
        # Create grayscale image
        gray_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        gray_image = Image.fromarray(gray_array, mode="L")

        predictor = MultimodalPredictor(model_type="image")
        result = predictor.predict_image(gray_image)

        assert "prediction" in result
        assert "confidence" in result


class TestMultimodalInferenceSmoke:
    """Smoke tests for multimodal fusion."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

    def test_multimodal_fusion_happy_path(self, sample_image):
        """Test multimodal fusion works end-to-end."""
        # Given: Text and image
        text = "This is a product description for testing."

        # When: We perform multimodal classification
        predictor = MultimodalPredictor(model_type="multimodal")
        start_time = time.time()
        result = predictor.predict_multimodal(text, sample_image)
        latency_ms = (time.time() - start_time) * 1000

        # Then: We get a valid response
        assert result is not None
        assert "prediction" in result
        assert "confidence" in result
        assert isinstance(result["prediction"], (int, str))
        assert 0 <= result["confidence"] <= 1

        # And: Latency is reasonable
        assert latency_ms < 6000, f"Latency {latency_ms:.0f}ms exceeds smoke test threshold"

    def test_multimodal_response_schema(self, sample_image):
        """Test multimodal response schema."""
        text = "Sample multimodal input."
        predictor = MultimodalPredictor(model_type="multimodal")
        result = predictor.predict_multimodal(text, sample_image)

        # Validate schema
        required_fields = ["prediction", "confidence"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


class TestErrorHandlingSmoke:
    """Smoke tests for error handling and recovery."""

    def test_invalid_model_type(self):
        """Test that invalid model type raises appropriate error."""
        with pytest.raises((ValueError, MultimodalError)):
            MultimodalPredictor(model_type="invalid_type")

    def test_corrupted_image_handling(self, tmp_path):
        """Test handling of corrupted image files."""
        # Create a corrupted "image" file
        corrupted_path = tmp_path / "corrupted.jpg"
        corrupted_path.write_bytes(b"NOT_AN_IMAGE")

        predictor = MultimodalPredictor(model_type="image")

        with pytest.raises(MultimodalError) as exc_info:
            predictor.predict_image_from_path(str(corrupted_path))

        # Should have a proper error code
        assert hasattr(exc_info.value, "code")
        assert exc_info.value.code.startswith("IMG_")

    def test_oversized_text_handling(self):
        """Test handling of extremely long text documents."""
        # Create a document longer than token limit (512 tokens ~= 2000 chars)
        oversized_text = "word " * 10000  # ~50k characters

        predictor = MultimodalPredictor(model_type="document")

        # Should either truncate gracefully or raise validation error
        try:
            result = predictor.predict_document(oversized_text)
            assert "prediction" in result  # Truncated successfully
        except MultimodalError as e:
            assert e.code == "VAL_001"  # Validation error


class TestLatencySmokeTest:
    """Smoke tests for latency requirements."""

    def test_cold_start_latency(self):
        """Test that cold start completes within reasonable time."""
        start_time = time.time()
        predictor = MultimodalPredictor(model_type="document")
        cold_start_time = (time.time() - start_time) * 1000

        # Cold start should complete within 10 seconds
        assert cold_start_time < 10000, (
            f"Cold start took {cold_start_time:.0f}ms, exceeds 10s threshold"
        )

    def test_warm_inference_latency(self):
        """Test that warm inference meets latency targets."""
        predictor = MultimodalPredictor(model_type="document")

        # Warm up
        predictor.predict_document("Warmup request.")

        # Measure warm latency
        latencies = []
        for _ in range(5):
            start_time = time.time()
            predictor.predict_document("Test document for latency measurement.")
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

        # Smoke test: warm latency should be reasonable
        # (Production target is <600ms, smoke test uses 2000ms)
        assert avg_latency < 2000, f"Average latency {avg_latency:.0f}ms too high"
        assert p95_latency < 3000, f"P95 latency {p95_latency:.0f}ms too high"


class TestHealthCheckSmoke:
    """Smoke tests for system health checks."""

    def test_model_loadable(self):
        """Test that all model types can be loaded."""
        model_types = ["document", "image", "multimodal"]

        for model_type in model_types:
            predictor = MultimodalPredictor(model_type=model_type)
            assert predictor is not None
            assert predictor.model is not None

    def test_device_detection(self):
        """Test that device (CPU/GPU) is detected correctly."""
        predictor = MultimodalPredictor(model_type="document")

        # Should have a device
        assert hasattr(predictor, "device") or hasattr(predictor.model, "device")

        # Device should be valid
        device = getattr(predictor, "device", None) or getattr(predictor.model, "device", None)
        if device:
            assert device in [torch.device("cpu"), torch.device("cuda")]

    def test_predictor_serializable_response(self):
        """Test that responses are JSON serializable."""
        text = "Test document for serialization."
        predictor = MultimodalPredictor(model_type="document")
        result = predictor.predict_document(text)

        # Should be JSON serializable
        json_str = json.dumps(result, default=str)
        assert json_str is not None

        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed["prediction"] is not None


# Pytest configuration for smoke tests
pytestmark = pytest.mark.smoke


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
