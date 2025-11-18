"""Model evaluation utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.metrics import MetricsCalculator, AverageMeter

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate model performance on validation/test data."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        metrics_calculator: Optional[MetricsCalculator] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to run evaluation on
            metrics_calculator: Metrics calculator instance
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_calculator = metrics_calculator

        self.model.to(self.device)

    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.

        Args:
            dataloader: DataLoader for evaluation data
            criterion: Loss criterion (optional)
            return_predictions: Whether to return predictions

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        losses = AverageMeter("Loss")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                inputs, labels = self._prepare_batch(batch)

                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                # Calculate loss if criterion provided
                if criterion is not None:
                    loss = criterion(logits, labels)
                    losses.update(loss.item(), labels.size(0))

                # Get predictions
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Convert to arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        results = {"loss": losses.avg} if criterion is not None else {}

        if self.metrics_calculator:
            metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds)
            results.update(metrics)

            # Add per-class metrics
            per_class_metrics = self.metrics_calculator.calculate_per_class_metrics(
                all_labels, all_preds
            )
            results["per_class_metrics"] = per_class_metrics

            # Add confusion matrix
            conf_matrix = self.metrics_calculator.get_confusion_matrix(all_labels, all_preds)
            results["confusion_matrix"] = conf_matrix.tolist()

        # Return predictions if requested
        if return_predictions:
            results["predictions"] = all_preds
            results["labels"] = all_labels
            results["probabilities"] = all_probs

        return results

    def _prepare_batch(self, batch: Union[tuple, Dict]) -> tuple:
        """
        Prepare batch for evaluation.

        Args:
            batch: Input batch

        Returns:
            Tuple of (inputs, labels)
        """
        if isinstance(batch, dict):
            labels = batch.pop("labels").to(self.device)
            inputs = {k: v.to(self.device) for k, v in batch.items()}
        else:
            *inputs_list, labels = batch
            labels = labels.to(self.device)

            if len(inputs_list) == 1:
                inputs = {"pixel_values": inputs_list[0].to(self.device)}
            else:
                inputs = {
                    "input_ids": inputs_list[0].to(self.device),
                    "attention_mask": inputs_list[1].to(self.device),
                }
                if len(inputs_list) > 2:
                    inputs["pixel_values"] = inputs_list[2].to(self.device)

        return inputs, labels

    def evaluate_with_threshold(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate with custom confidence threshold.

        Args:
            dataloader: DataLoader for evaluation data
            threshold: Confidence threshold

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_confident = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating with threshold"):
                inputs, labels = self._prepare_batch(batch)

                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                # Get predictions and confidence
                probs = torch.softmax(logits, dim=-1)
                max_probs, preds = torch.max(probs, dim=-1)

                # Check if above threshold
                confident = (max_probs >= threshold).cpu().numpy()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confident.extend(confident)

        # Calculate metrics for confident predictions only
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confident = np.array(all_confident)

        confident_preds = all_preds[all_confident]
        confident_labels = all_labels[all_confident]

        results = {
            "total_samples": len(all_labels),
            "confident_samples": np.sum(all_confident),
            "confidence_rate": np.mean(all_confident),
        }

        if len(confident_preds) > 0 and self.metrics_calculator:
            metrics = self.metrics_calculator.calculate_metrics(
                confident_labels, confident_preds
            )
            results.update(metrics)

        return results

    def save_results(self, results: Dict[str, Any], save_path: Union[str, Path]):
        """
        Save evaluation results to file.

        Args:
            results: Evaluation results
            save_path: Path to save results
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(save_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {save_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary of model names and instances
            dataloader: DataLoader for evaluation data

        Returns:
            Dictionary of results for each model
        """
        results = {}

        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")

            # Temporarily swap model
            original_model = self.model
            self.model = model
            self.model.to(self.device)

            # Evaluate
            model_results = self.evaluate(dataloader)
            results[name] = model_results

            # Restore original model
            self.model = original_model

        return results
