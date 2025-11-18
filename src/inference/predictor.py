"""Unified predictor for multimodal inference."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

from ..data.ingestion import DataIngestion
from ..data.preprocessing import ImagePreprocessor, OCRProcessor, TextPreprocessor
from ..models import DocumentClassifier, ImageClassifier, MultimodalFusionModel
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalPredictor:
    """Unified predictor for all model types."""

    def __init__(
        self,
        model_type: str = "multimodal",
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor.

        Args:
            model_type: Type of model ('document', 'image', or 'multimodal')
            model_path: Path to saved model weights
            device: Device to run inference on
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load configurations
        self.config = load_config("model_config")

        # Initialize preprocessors
        self.text_preprocessor = TextPreprocessor()
        self.image_preprocessor = ImagePreprocessor()
        self.ocr_processor = OCRProcessor()
        self.data_ingestion = DataIngestion()

        # Load model
        self.model = self._load_model(model_path)
        logger.info(f"Predictor initialized with {model_type} model on {self.device}")

    def _load_model(self, model_path: Optional[Union[str, Path]]):
        """Load model based on type."""
        if self.model_type == "document":
            config = self.config.get("document_classifier", {})
            model = DocumentClassifier(
                num_classes=config.get("num_classes", 10),
                model_name=config.get("model_name", "bert-base-uncased"),
                config=config,
            )
        elif self.model_type == "image":
            config = self.config.get("image_classifier", {})
            model = ImageClassifier(
                num_classes=config.get("num_classes", 100),
                model_name=config.get("model_name", "google/vit-base-patch16-224"),
                config=config,
            )
        elif self.model_type == "multimodal":
            config = self.config.get("multimodal_fusion", {})
            model = MultimodalFusionModel(
                num_classes=config.get("num_classes", 50),
                text_model_name=config.get("text_encoder", "bert-base-uncased"),
                image_model_name=config.get("image_encoder", "google/vit-base-patch16-224"),
                config=config,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load weights if provided
        if model_path:
            model.load_model(model_path)

        model.to(self.device)
        model.eval()

        return model

    def predict_document(
        self,
        document: Union[str, Path],
        return_attention: bool = False,
    ) -> Dict:
        """
        Predict on document.

        Args:
            document: Path to document or text string
            return_attention: Whether to return attention weights

        Returns:
            Prediction results
        """
        # Load document if path
        if isinstance(document, (str, Path)) and Path(document).exists():
            doc_path = Path(document)

            if doc_path.suffix.lower() == ".pdf":
                doc_data = self.data_ingestion.load_pdf(doc_path)
                text = doc_data["text"]
            elif doc_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                # Use OCR for images
                text = self.ocr_processor.extract_text(doc_path)
            else:
                text = self.data_ingestion.load_text_file(doc_path)
        else:
            text = str(document)

        # Preprocess
        processed_text = self.text_preprocessor.preprocess(text)

        # Predict
        results = self.model.predict([processed_text], return_probs=True)

        output = {
            "text": text[:500],  # Preview
            "prediction": int(results["predictions"][0]),
            "confidence": float(results["probabilities"][0].max()),
            "top_k_predictions": self._get_top_k(results["probabilities"][0]),
        }

        # Add attention if requested
        if return_attention and hasattr(self.model, "get_attention_weights"):
            attention, tokens = self.model.get_attention_weights(text)
            output["attention"] = attention.numpy().tolist()
            output["tokens"] = tokens

        return output

    def predict_image(
        self,
        image: Union[str, Path, Image.Image],
        return_attention: bool = False,
    ) -> Dict:
        """
        Predict on image.

        Args:
            image: Path to image or PIL Image
            return_attention: Whether to return attention map

        Returns:
            Prediction results
        """
        # Preprocess image
        processed_image = self.image_preprocessor.preprocess(image)

        # Predict
        results = self.model.predict(
            processed_image.unsqueeze(0), return_probs=True
        )

        output = {
            "prediction": int(results["predictions"][0]),
            "confidence": float(results["probabilities"][0].max()),
            "top_k_predictions": self._get_top_k(results["probabilities"][0]),
        }

        # Add attention if requested
        if return_attention and hasattr(self.model, "visualize_attention"):
            attention_map = self.model.visualize_attention(processed_image.unsqueeze(0))
            output["attention_map"] = attention_map.numpy().tolist()

        return output

    def predict_multimodal(
        self,
        document: Union[str, Path],
        image: Union[str, Path, Image.Image],
        return_embeddings: bool = False,
    ) -> Dict:
        """
        Predict on document-image pair.

        Args:
            document: Path to document or text string
            image: Path to image or PIL Image
            return_embeddings: Whether to return embeddings

        Returns:
            Prediction results
        """
        # Process document
        if isinstance(document, (str, Path)) and Path(document).exists():
            doc_data = self.data_ingestion.load_pdf(document)
            text = doc_data["text"]
        else:
            text = str(document)

        processed_text = self.text_preprocessor.preprocess(text)

        # Process image
        processed_image = self.image_preprocessor.preprocess(image)

        # Predict
        results = self.model.predict(
            [processed_text],
            processed_image.unsqueeze(0),
            return_probs=True,
            return_embeddings=return_embeddings,
        )

        output = {
            "prediction": int(results["predictions"][0]),
            "confidence": float(results["probabilities"][0].max()),
            "top_k_predictions": self._get_top_k(results["probabilities"][0]),
        }

        if return_embeddings:
            output["text_embeddings"] = results["text_embeddings"][0].numpy().tolist()
            output["image_embeddings"] = results["image_embeddings"][0].numpy().tolist()
            output["fused_embeddings"] = results["fused_embeddings"][0].numpy().tolist()

        return output

    def predict(self, *args, **kwargs) -> Dict:
        """
        Predict using appropriate method based on model type.

        Returns:
            Prediction results
        """
        if self.model_type == "document":
            return self.predict_document(*args, **kwargs)
        elif self.model_type == "image":
            return self.predict_image(*args, **kwargs)
        elif self.model_type == "multimodal":
            return self.predict_multimodal(*args, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_top_k(self, probabilities: torch.Tensor, k: int = 5) -> List[Dict]:
        """Get top-k predictions."""
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        top_indices = probabilities.argsort()[-k:][::-1]

        return [
            {"class": int(idx), "confidence": float(probabilities[idx])}
            for idx in top_indices
        ]

    def batch_predict(
        self, items: List, batch_size: int = 8
    ) -> List[Dict]:
        """
        Batch prediction.

        Args:
            items: List of items to predict on
            batch_size: Batch size for processing

        Returns:
            List of prediction results
        """
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            for item in batch:
                result = self.predict(item)
                results.append(result)

        return results
