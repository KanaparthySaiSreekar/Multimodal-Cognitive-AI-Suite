"""Document classification model using BERT."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .base_model import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DocumentClassifier(BaseModel):
    """BERT-based document classification model."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.1,
        max_length: int = 512,
        config: Optional[Dict] = None,
    ):
        """
        Initialize document classifier.

        Args:
            num_classes: Number of output classes
            model_name: Pretrained BERT model name
            hidden_size: Hidden layer size
            dropout: Dropout rate
            max_length: Maximum sequence length
            config: Additional configuration
        """
        if config is None:
            config = {
                "num_classes": num_classes,
                "model_name": model_name,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "max_length": max_length,
            }

        super().__init__(config)

        self.num_classes = num_classes
        self.model_name = model_name
        self.max_length = max_length

        # Load pretrained BERT
        logger.info(f"Loading BERT model: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classifier weights
        self._init_weights()

        logger.info(f"Document classifier initialized with {num_classes} classes")

    def _init_weights(self):
        """Initialize classification head weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with logits and optionally embeddings
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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

        return result

    def predict(
        self,
        texts: list,
        batch_size: int = 8,
        return_probs: bool = True,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict on batch of texts.

        Args:
            texts: List of text strings
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
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
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

    def get_attention_weights(
        self, text: str, layer: int = -1
    ) -> Tuple[torch.Tensor, list]:
        """
        Get attention weights for visualization.

        Args:
            text: Input text
            layer: Which layer's attention to return (-1 for last layer)

        Returns:
            Tuple of (attention_weights, tokens)
        """
        self.eval()

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Forward pass with output attentions
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Get attention weights
        attention = outputs.attentions[layer]

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return attention[0].cpu(), tokens

    def freeze_bert(self):
        """Freeze BERT encoder parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
        logger.info("BERT encoder frozen")

    def unfreeze_bert(self):
        """Unfreeze BERT encoder parameters."""
        for param in self.bert.parameters():
            param.requires_grad = True
        logger.info("BERT encoder unfrozen")

    def freeze_bert_layers(self, num_layers: int):
        """
        Freeze first N layers of BERT.

        Args:
            num_layers: Number of layers to freeze
        """
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze first N encoder layers
        for layer in self.bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        logger.info(f"Froze first {num_layers} BERT layers")
