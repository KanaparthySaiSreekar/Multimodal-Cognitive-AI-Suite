"""Unit tests for model components."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

from src.models import DocumentClassifier, ImageClassifier, MultimodalFusionModel


class TestDocumentClassifier:
    """Test suite for DocumentClassifier."""

    @pytest.fixture
    def model(self):
        return DocumentClassifier(num_classes=5, model_name="bert-base-uncased")

    @pytest.fixture
    def sample_texts(self):
        return [
            "This is a test document about machine learning.",
            "Another sample text for classification testing.",
        ]

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.num_classes == 5
        assert model.model_name == "bert-base-uncased"
        assert hasattr(model, "bert")
        assert hasattr(model, "classifier")

    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 2
        seq_length = 128

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, 5)

    def test_predict(self, model, sample_texts):
        """Test prediction method."""
        results = model.predict(sample_texts, batch_size=2)

        assert "predictions" in results
        assert "probabilities" in results
        assert "logits" in results

        assert results["predictions"].shape[0] == len(sample_texts)
        assert results["probabilities"].shape == (len(sample_texts), 5)

    def test_prediction_probabilities_sum_to_one(self, model, sample_texts):
        """Test that probabilities sum to 1."""
        results = model.predict(sample_texts)
        probs = results["probabilities"]

        for prob_dist in probs:
            assert torch.isclose(prob_dist.sum(), torch.tensor(1.0), atol=1e-5)

    def test_freeze_bert(self, model):
        """Test BERT freezing."""
        model.freeze_bert()

        for param in model.bert.parameters():
            assert param.requires_grad is False

        # Classifier should still be trainable
        for param in model.classifier.parameters():
            assert param.requires_grad is True

    def test_unfreeze_bert(self, model):
        """Test BERT unfreezing."""
        model.freeze_bert()
        model.unfreeze_bert()

        for param in model.bert.parameters():
            assert param.requires_grad is True

    def test_save_and_load(self, model):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pth"

            # Save model
            model.save_model(save_path)
            assert save_path.exists()

            # Load model
            new_model = DocumentClassifier(num_classes=5)
            new_model.load_model(save_path)

            # Compare parameters
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_parameter_count(self, model):
        """Test parameter counting."""
        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_model_size(self, model):
        """Test model size calculation."""
        size_mb = model.get_model_size()
        assert size_mb > 0
        assert isinstance(size_mb, float)


class TestImageClassifier:
    """Test suite for ImageClassifier."""

    @pytest.fixture
    def model(self):
        return ImageClassifier(num_classes=10)

    @pytest.fixture
    def sample_images(self):
        batch_size = 2
        return torch.randn(batch_size, 3, 224, 224)

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.num_classes == 10
        assert hasattr(model, "vit")
        assert hasattr(model, "classifier")

    def test_forward_pass(self, model, sample_images):
        """Test forward pass."""
        outputs = model(pixel_values=sample_images)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 10)

    def test_predict(self, model, sample_images):
        """Test prediction method."""
        results = model.predict(sample_images, batch_size=2)

        assert "predictions" in results
        assert "probabilities" in results
        assert results["predictions"].shape[0] == 2
        assert results["probabilities"].shape == (2, 10)

    def test_attention_map_generation(self, model):
        """Test attention map generation."""
        single_image = torch.randn(1, 3, 224, 224)
        attention_map = model.get_attention_map(single_image)

        assert isinstance(attention_map, torch.Tensor)
        assert attention_map.dim() >= 3  # Should have batch, heads, patches dimensions

    def test_freeze_vit(self, model):
        """Test ViT freezing."""
        model.freeze_vit()

        for param in model.vit.parameters():
            assert param.requires_grad is False

    def test_device_movement(self, model):
        """Test moving model to different devices."""
        # Test CPU
        model.to_device("cpu")
        assert next(model.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model.to_device("cuda")
            assert next(model.parameters()).device.type == "cuda"


class TestMultimodalFusionModel:
    """Test suite for MultimodalFusionModel."""

    @pytest.fixture
    def model(self):
        return MultimodalFusionModel(num_classes=8, fusion_method="attention")

    @pytest.fixture
    def sample_inputs(self):
        batch_size = 2
        seq_length = 128

        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
        }

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.num_classes == 8
        assert model.fusion_method == "attention"
        assert hasattr(model, "text_encoder")
        assert hasattr(model, "image_encoder")

    def test_forward_pass(self, model, sample_inputs):
        """Test forward pass."""
        outputs = model(**sample_inputs)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 8)

    def test_forward_with_embeddings(self, model, sample_inputs):
        """Test forward pass with embedding extraction."""
        outputs = model(**sample_inputs, return_embeddings=True)

        assert "logits" in outputs
        assert "text_embeddings" in outputs
        assert "image_embeddings" in outputs
        assert "fused_embeddings" in outputs

    def test_predict(self, model):
        """Test prediction method."""
        texts = ["Sample text one", "Sample text two"]
        images = torch.randn(2, 3, 224, 224)

        results = model.predict(texts, images, batch_size=2)

        assert "predictions" in results
        assert "probabilities" in results
        assert results["predictions"].shape[0] == 2

    def test_freeze_encoders(self, model):
        """Test encoder freezing."""
        model.freeze_encoders()

        for param in model.text_encoder.parameters():
            assert param.requires_grad is False

        for param in model.image_encoder.parameters():
            assert param.requires_grad is False

    def test_fusion_methods(self):
        """Test different fusion methods."""
        fusion_methods = ["concatenation", "attention", "cross_attention"]

        for method in fusion_methods:
            model = MultimodalFusionModel(num_classes=5, fusion_method=method)

            # Test forward pass works
            batch_size = 1
            inputs = {
                "input_ids": torch.randint(0, 1000, (batch_size, 64)),
                "attention_mask": torch.ones(batch_size, 64),
                "pixel_values": torch.randn(batch_size, 3, 224, 224),
            }

            outputs = model(**inputs)
            assert outputs["logits"].shape == (batch_size, 5)


class TestModelEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_num_classes(self):
        """Test initialization with invalid number of classes."""
        with pytest.raises((ValueError, AssertionError)):
            DocumentClassifier(num_classes=0)

    def test_batch_size_one(self):
        """Test models with batch size of 1."""
        model = ImageClassifier(num_classes=5)
        single_image = torch.randn(1, 3, 224, 224)

        outputs = model(pixel_values=single_image)
        assert outputs["logits"].shape == (1, 5)

    def test_large_batch_size(self):
        """Test models with large batch size."""
        model = ImageClassifier(num_classes=5)
        large_batch = torch.randn(64, 3, 224, 224)

        outputs = model(pixel_values=large_batch)
        assert outputs["logits"].shape == (64, 5)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        model = DocumentClassifier(num_classes=3)
        model.train()

        input_ids = torch.randint(0, 1000, (2, 64))
        attention_mask = torch.ones(2, 64)
        labels = torch.tensor([0, 1])

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs["logits"], labels)

        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode(self):
        """Test model in evaluation mode."""
        model = ImageClassifier(num_classes=5)
        model.eval()

        images = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            outputs1 = model(pixel_values=images)
            outputs2 = model(pixel_values=images)

        # Outputs should be deterministic in eval mode
        assert torch.allclose(outputs1["logits"], outputs2["logits"])
