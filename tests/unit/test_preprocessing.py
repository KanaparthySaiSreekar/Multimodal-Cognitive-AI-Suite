"""Unit tests for data preprocessing components."""

import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
from pathlib import Path

from src.data.preprocessing import TextPreprocessor, ImagePreprocessor, OCRProcessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_lowercase_conversion(self, preprocessor):
        """Test text lowercasing."""
        text = "Hello WORLD"
        result = preprocessor.preprocess(text)
        assert result == "hello world"

    def test_special_chars_removal(self, preprocessor):
        """Test special character removal."""
        text = "Hello@#$%World!"
        result = preprocessor.preprocess(text)
        assert "@#$%" not in result
        assert "hello" in result.lower()

    def test_whitespace_normalization(self, preprocessor):
        """Test extra whitespace removal."""
        text = "Hello    World   Test"
        result = preprocessor.preprocess(text)
        assert "  " not in result

    def test_empty_string_handling(self, preprocessor):
        """Test empty string input."""
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess("   ") == ""

    def test_url_removal(self, preprocessor):
        """Test URL removal from text."""
        text = "Check this http://example.com and www.test.com"
        result = preprocessor.clean_text(text)
        assert "http://example.com" not in result
        assert "www.test.com" not in result

    def test_email_removal(self, preprocessor):
        """Test email address removal."""
        text = "Contact me at test@example.com for info"
        result = preprocessor.clean_text(text)
        assert "test@example.com" not in result

    def test_tokenization(self, preprocessor):
        """Test simple tokenization."""
        text = "Hello World Test"
        tokens = preprocessor.tokenize(text)
        assert tokens == ["Hello", "World", "Test"]

    def test_unicode_handling(self, preprocessor):
        """Test Unicode character handling."""
        text = "Hello 世界 Test"
        result = preprocessor.preprocess(text)
        assert result is not None


class TestImagePreprocessor:
    """Test suite for ImagePreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        return ImagePreprocessor(image_size=(224, 224))

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return Image.new("RGB", (512, 512), color="red")

    def test_image_resize(self, preprocessor, sample_image):
        """Test image resizing."""
        tensor = preprocessor.preprocess(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_normalization(self, preprocessor, sample_image):
        """Test image normalization."""
        tensor = preprocessor.preprocess(sample_image)
        # Check if normalized (values should not be in [0, 255])
        assert tensor.max() < 10.0
        assert tensor.min() > -10.0

    def test_pil_image_input(self, preprocessor, sample_image):
        """Test PIL Image input."""
        result = preprocessor.preprocess(sample_image)
        assert isinstance(result, torch.Tensor)

    def test_numpy_array_input(self, preprocessor):
        """Test numpy array input."""
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = preprocessor.preprocess(img_array)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_file_path_input(self, preprocessor, sample_image):
        """Test file path input."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            sample_image.save(f.name)
            result = preprocessor.preprocess(f.name)
            Path(f.name).unlink()

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_denormalization(self, preprocessor, sample_image):
        """Test image denormalization."""
        tensor = preprocessor.preprocess(sample_image)
        denorm = preprocessor.denormalize(tensor)

        assert isinstance(denorm, np.ndarray)
        assert denorm.shape == (224, 224, 3)
        assert denorm.dtype == np.uint8
        assert denorm.min() >= 0
        assert denorm.max() <= 255

    def test_grayscale_conversion(self, preprocessor):
        """Test grayscale image conversion."""
        gray_image = Image.new("L", (512, 512), color=128)
        result = preprocessor.preprocess(gray_image)
        assert result.shape == (3, 224, 224)  # Should convert to RGB


class TestOCRProcessor:
    """Test suite for OCRProcessor."""

    @pytest.fixture
    def ocr_processor(self):
        return OCRProcessor()

    @pytest.fixture
    def text_image(self):
        """Create an image with text."""
        img = Image.new("RGB", (800, 400), color="white")
        # Note: This is a blank image; real tests would need text rendering
        return img

    def test_ocr_initialization(self, ocr_processor):
        """Test OCR processor initialization."""
        assert ocr_processor.languages == "eng"
        assert ocr_processor.preprocess_image is True

    def test_extract_text_with_pil_image(self, ocr_processor, text_image):
        """Test text extraction from PIL Image."""
        result = ocr_processor.extract_text(text_image)
        assert isinstance(result, str)

    def test_extract_text_with_confidence(self, ocr_processor, text_image):
        """Test text extraction with confidence scores."""
        result = ocr_processor.extract_text_with_confidence(text_image)

        assert "text" in result
        assert "confidence" in result
        assert "word_count" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 100

    def test_image_preprocessing_for_ocr(self, ocr_processor):
        """Test image preprocessing pipeline."""
        img = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
        processed = ocr_processor._preprocess_for_ocr(img)

        # Should be grayscale after preprocessing
        assert len(processed.shape) == 2
        assert processed.dtype == np.uint8

    def test_orientation_detection(self, ocr_processor, text_image):
        """Test orientation detection."""
        angle = ocr_processor.detect_orientation(text_image)
        assert isinstance(angle, (int, float))
        assert angle in [0, 90, 180, 270]

    def test_image_rotation(self, ocr_processor):
        """Test image rotation."""
        img = np.random.randint(0, 255, (400, 800), dtype=np.uint8)
        rotated = ocr_processor.rotate_image(img, 90)

        assert isinstance(rotated, np.ndarray)
        # Note: rotation might change dimensions


class TestPreprocessingEdgeCases:
    """Test edge cases and error handling."""

    def test_text_preprocessor_none_input(self):
        """Test handling of None input."""
        preprocessor = TextPreprocessor()
        with pytest.raises((TypeError, AttributeError)):
            preprocessor.preprocess(None)

    def test_image_preprocessor_invalid_path(self):
        """Test handling of invalid file path."""
        preprocessor = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor.preprocess("/nonexistent/image.jpg")

    def test_image_preprocessor_corrupted_image(self):
        """Test handling of corrupted image data."""
        preprocessor = ImagePreprocessor()

        # Create a file with invalid image data
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"corrupted data")
            temp_path = f.name

        with pytest.raises(Exception):
            preprocessor.preprocess(temp_path)

        Path(temp_path).unlink()

    def test_very_long_text(self):
        """Test handling of very long text."""
        preprocessor = TextPreprocessor()
        long_text = "word " * 100000  # 100k words

        result = preprocessor.preprocess(long_text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multilingual_text(self):
        """Test multilingual text processing."""
        preprocessor = TextPreprocessor(lowercase=False)

        # Mix of languages
        text = "Hello 世界 Привет مرحبا"
        result = preprocessor.preprocess(text)
        assert result is not None
        assert len(result) > 0
