"""Multimodal fusion model combining text and image modalities."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor

from .base_model import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalFusionModel(BaseModel):
    """Multimodal model fusing text and image representations."""

    def __init__(
        self,
        num_classes: int,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "google/vit-base-patch16-224",
        fusion_method: str = "attention",
        hidden_size: int = 768,
        dropout: float = 0.2,
        fusion_dropout: float = 0.3,
        config: Optional[Dict] = None,
    ):
        """
        Initialize multimodal fusion model.

        Args:
            num_classes: Number of output classes
            text_model_name: Pretrained text encoder name
            image_model_name: Pretrained image encoder name
            fusion_method: Fusion method ('concatenation', 'attention', 'cross_attention')
            hidden_size: Hidden layer size
            dropout: Dropout rate
            fusion_dropout: Dropout rate for fusion layer
            config: Additional configuration
        """
        if config is None:
            config = {
                "num_classes": num_classes,
                "text_model_name": text_model_name,
                "image_model_name": image_model_name,
                "fusion_method": fusion_method,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "fusion_dropout": fusion_dropout,
            }

        super().__init__(config)

        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.hidden_size = hidden_size

        # Text encoder
        logger.info(f"Loading text encoder: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Image encoder
        logger.info(f"Loading image encoder: {image_model_name}")
        self.image_encoder = AutoModel.from_pretrained(image_model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_name)

        # Dropouts
        self.text_dropout = nn.Dropout(dropout)
        self.image_dropout = nn.Dropout(dropout)
        self.fusion_dropout = nn.Dropout(fusion_dropout)

        # Fusion mechanism
        if fusion_method == "concatenation":
            self._init_concatenation_fusion()
        elif fusion_method == "attention":
            self._init_attention_fusion()
        elif fusion_method == "cross_attention":
            self._init_cross_attention_fusion()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        logger.info(f"Multimodal fusion model initialized with {fusion_method} fusion")

    def _init_concatenation_fusion(self):
        """Initialize simple concatenation-based fusion."""
        # Classifier takes concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            self.fusion_dropout,
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def _init_attention_fusion(self):
        """Initialize attention-based fusion."""
        # Attention mechanism
        self.text_attention = nn.Linear(self.hidden_size, 1)
        self.image_attention = nn.Linear(self.hidden_size, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            self.fusion_dropout,
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def _init_cross_attention_fusion(self):
        """Initialize cross-attention fusion."""
        # Cross-attention layers
        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, dropout=0.1
        )
        self.image_to_text_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, dropout=0.1
        )

        # Layer norms
        self.text_norm = nn.LayerNorm(self.hidden_size)
        self.image_norm = nn.LayerNorm(self.hidden_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            self.fusion_dropout,
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Text input IDs
            attention_mask: Text attention mask
            pixel_values: Image pixel values
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with logits and optionally embeddings
        """
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_features = self.text_dropout(text_outputs.pooler_output)

        # Encode image
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = self.image_dropout(image_outputs.pooler_output)

        # Fusion
        if self.fusion_method == "concatenation":
            fused_features = self._concatenation_fusion(text_features, image_features)
        elif self.fusion_method == "attention":
            fused_features = self._attention_fusion(text_features, image_features)
        elif self.fusion_method == "cross_attention":
            fused_features = self._cross_attention_fusion(
                text_outputs.last_hidden_state,
                image_outputs.last_hidden_state,
                text_features,
                image_features,
            )

        # Classification
        logits = self.classifier(fused_features)

        result = {"logits": logits}

        if return_embeddings:
            result["text_embeddings"] = text_features
            result["image_embeddings"] = image_features
            result["fused_embeddings"] = fused_features

        return result

    def _concatenation_fusion(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Simple concatenation fusion."""
        return torch.cat([text_features, image_features], dim=-1)

    def _attention_fusion(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Attention-weighted fusion."""
        # Calculate attention weights
        text_attn = torch.sigmoid(self.text_attention(text_features))
        image_attn = torch.sigmoid(self.image_attention(image_features))

        # Normalize attention weights
        total_attn = text_attn + image_attn
        text_attn = text_attn / total_attn
        image_attn = image_attn / total_attn

        # Weighted features
        weighted_text = text_features * text_attn
        weighted_image = image_features * image_attn

        # Concatenate weighted features
        return torch.cat([weighted_text, weighted_image], dim=-1)

    def _cross_attention_fusion(
        self,
        text_hidden: torch.Tensor,
        image_hidden: torch.Tensor,
        text_pooled: torch.Tensor,
        image_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention fusion."""
        # Transpose for multi-head attention (seq_len, batch, hidden)
        text_hidden = text_hidden.transpose(0, 1)
        image_hidden = image_hidden.transpose(0, 1)

        # Text attending to image
        text_attended, _ = self.text_to_image_attention(
            query=text_hidden,
            key=image_hidden,
            value=image_hidden,
        )
        text_attended = self.text_norm(text_attended[0])  # Take CLS token

        # Image attending to text
        image_attended, _ = self.image_to_text_attention(
            query=image_hidden,
            key=text_hidden,
            value=text_hidden,
        )
        image_attended = self.image_norm(image_attended[0])  # Take CLS token

        # Combine attended features with pooled features
        text_combined = text_pooled + text_attended
        image_combined = image_pooled + image_attended

        # Concatenate
        return torch.cat([text_combined, image_combined], dim=-1)

    def predict(
        self,
        texts: list,
        images: torch.Tensor,
        batch_size: int = 8,
        return_probs: bool = True,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict on batch of text-image pairs.

        Args:
            texts: List of text strings
            images: Batch of image tensors
            batch_size: Batch size for inference
            return_probs: Whether to return probabilities
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with predictions
        """
        self.eval()
        all_logits = []
        all_embeddings = {} if return_embeddings else None

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_images = images[i : i + batch_size].to(self.device)

                # Tokenize texts
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=batch_images,
                    return_embeddings=return_embeddings,
                )

                all_logits.append(outputs["logits"].cpu())

                if return_embeddings:
                    for key in ["text_embeddings", "image_embeddings", "fused_embeddings"]:
                        if key not in all_embeddings:
                            all_embeddings[key] = []
                        all_embeddings[key].append(outputs[key].cpu())

        # Concatenate results
        logits = torch.cat(all_logits, dim=0)

        results = {"logits": logits}

        if return_probs:
            results["probabilities"] = torch.softmax(logits, dim=-1)

        results["predictions"] = torch.argmax(logits, dim=-1)

        if return_embeddings:
            for key, embeddings in all_embeddings.items():
                results[key] = torch.cat(embeddings, dim=0)

        return results

    def freeze_encoders(self):
        """Freeze both text and image encoders."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        logger.info("Text and image encoders frozen")

    def unfreeze_encoders(self):
        """Unfreeze both text and image encoders."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        logger.info("Text and image encoders unfrozen")

    def freeze_text_encoder(self):
        """Freeze only text encoder."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        logger.info("Text encoder frozen")

    def freeze_image_encoder(self):
        """Freeze only image encoder."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        logger.info("Image encoder frozen")
