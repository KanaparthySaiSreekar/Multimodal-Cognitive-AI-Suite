"""Pytest configuration and shared fixtures."""

import pytest
import torch
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample document for testing the multimodal AI system."


@pytest.fixture
def sample_texts():
    """Multiple sample texts."""
    return [
        "First test document about machine learning.",
        "Second test document about computer vision.",
        "Third test document about natural language processing.",
    ]


@pytest.fixture
def sample_image():
    """Sample PIL image."""
    return Image.new("RGB", (512, 512), color=(100, 150, 200))


@pytest.fixture
def sample_images():
    """Multiple sample images."""
    return [
        Image.new("RGB", (512, 512), color="red"),
        Image.new("RGB", (512, 512), color="green"),
        Image.new("RGB", (512, 512), color="blue"),
    ]


@pytest.fixture
def sample_image_tensor():
    """Sample image tensor."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_image_file(sample_image, temp_dir):
    """Temporary image file."""
    img_path = temp_dir / "test_image.jpg"
    sample_image.save(img_path)
    return img_path


@pytest.fixture
def temp_text_file(sample_text, temp_dir):
    """Temporary text file."""
    txt_path = temp_dir / "test_document.txt"
    txt_path.write_text(sample_text)
    return txt_path


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        "model": {
            "num_classes": 10,
            "hidden_size": 768,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-4,
        },
    }


# Mark tests based on requirements
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
