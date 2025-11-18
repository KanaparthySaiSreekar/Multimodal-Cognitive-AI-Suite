"""Image classification model using Vision Transformer."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel

from .base_model import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImageClassifier(BaseModel):
    """Vision Transformer-based image classification model."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "google/vit-base-patch16-224",
        hidden_size: int = 768,
        dropout: float = 0.1,
        config: Optional[Dict] = None,
    ):
        """
        Initialize image classifier.

        Args:
            num_classes: Number of output classes
            model_name: Pretrained ViT model name
            hidden_size: Hidden layer size
            dropout: Dropout rate
            config: Additional configuration
        """
        if config is None:
            config = {
                "num_classes": num_classes,
                "model_name": model_name,
                "hidden_size": hidden_size,
                "dropout": dropout,
            }

        super().__init__(config)

        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained Vision Transformer
        logger.info(f"Loading ViT model: {model_name}")
        self.vit = AutoModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classifier weights
        self._init_weights()

        logger.info(f"Image classifier initialized with {num_classes} classes")

    def _init_weights(self):
        """Initialize classification head weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_embeddings: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pixel_values: Image pixel values
            return_embeddings: Whether to return embeddings
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with logits and optionally embeddings/attention
        """
        # ViT forward pass
        outputs = self.vit(
            pixel_values=pixel_values,
            output_attentions=return_attention,
        )

        # Get [CLS] token representation
        pooled_output = outputs.pooler_output

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)

        result = {"logits": logits}

        if return_embeddings:
            result["embeddings"] = pooled_output
            result["last_hidden_state"] = outputs.last_hidden_state

        if return_attention:
            result["attentions"] = outputs.attentions

        return result

    def predict(
        self,
        images: torch.Tensor,
        batch_size: int = 16,
        return_probs: bool = True,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict on batch of images.

        Args:
            images: Batch of image tensors
            batch_size: Batch size for inference
            return_probs: Whether to return probabilities
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with predictions
        """
        self.eval()
        all_logits = []
        all_embeddings = [] if return_embeddings else None

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size].to(self.device)

                # Forward pass
                outputs = self.forward(
                    pixel_values=batch_images,
                    return_embeddings=return_embeddings,
                )

                all_logits.append(outputs["logits"].cpu())

                if return_embeddings:
                    all_embeddings.append(outputs["embeddings"].cpu())

        # Concatenate results
        logits = torch.cat(all_logits, dim=0)

        results = {"logits": logits}

        if return_probs:
            results["probabilities"] = torch.softmax(logits, dim=-1)

        results["predictions"] = torch.argmax(logits, dim=-1)

        if return_embeddings:
            results["embeddings"] = torch.cat(all_embeddings, dim=0)

        return results

    def get_attention_map(
        self, pixel_values: torch.Tensor, layer: int = -1
    ) -> torch.Tensor:
        """
        Get attention map for visualization.

        Args:
            pixel_values: Image pixel values
            layer: Which layer's attention to return (-1 for last layer)

        Returns:
            Attention weights
        """
        self.eval()

        with torch.no_grad():
            outputs = self.vit(
                pixel_values=pixel_values.to(self.device),
                output_attentions=True,
            )

        # Get attention weights for specified layer
        attention = outputs.attentions[layer]

        return attention.cpu()

    def visualize_attention(
        self, pixel_values: torch.Tensor, layer: int = -1, head: int = 0
    ) -> torch.Tensor:
        """
        Create attention visualization.

        Args:
            pixel_values: Image pixel values
            layer: Which layer's attention to visualize
            head: Which attention head to visualize

        Returns:
            Attention map reshaped for visualization
        """
        attention = self.get_attention_map(pixel_values, layer)

        # Get attention for specific head
        # Shape: [batch, heads, patches, patches]
        attn_head = attention[0, head]

        # Get attention from CLS token to all patches
        cls_attention = attn_head[0, 1:]  # Skip CLS token itself

        # Reshape to 2D grid (assuming 14x14 patches for 224x224 image)
        grid_size = int(cls_attention.shape[0] ** 0.5)
        attn_map = cls_attention.reshape(grid_size, grid_size)

        return attn_map

    def freeze_vit(self):
        """Freeze ViT encoder parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
        logger.info("ViT encoder frozen")

    def unfreeze_vit(self):
        """Unfreeze ViT encoder parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True
        logger.info("ViT encoder unfrozen")

    def freeze_vit_layers(self, num_layers: int):
        """
        Freeze first N layers of ViT.

        Args:
            num_layers: Number of layers to freeze
        """
        # Freeze embeddings
        for param in self.vit.embeddings.parameters():
            param.requires_grad = False

        # Freeze first N encoder layers
        for layer in self.vit.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        logger.info(f"Froze first {num_layers} ViT layers")


class CNNImageClassifier(BaseModel):
    """Alternative CNN-based image classifier (ResNet-style)."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.1,
        config: Optional[Dict] = None,
    ):
        """
        Initialize CNN image classifier.

        Args:
            num_classes: Number of output classes
            backbone: CNN backbone architecture
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
            config: Additional configuration
        """
        if config is None:
            config = {
                "num_classes": num_classes,
                "backbone": backbone,
                "pretrained": pretrained,
                "dropout": dropout,
            }

        super().__init__(config)

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Load backbone
        logger.info(f"Loading CNN backbone: {backbone}")

        if backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights

            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            feature_dim = 2048
        elif backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove original classifier
        self.backbone.fc = nn.Identity()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        logger.info(f"CNN image classifier initialized with {num_classes} classes")

    def forward(
        self, x: torch.Tensor, return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with logits and optionally embeddings
        """
        # Extract features
        features = self.backbone(x)

        # Classification
        logits = self.classifier(features)

        result = {"logits": logits}

        if return_embeddings:
            result["embeddings"] = features

        return result

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("CNN backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("CNN backbone unfrozen")
