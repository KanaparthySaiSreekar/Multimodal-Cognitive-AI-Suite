"""Metrics calculation utilities for model evaluation."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


class MetricsCalculator:
    """Calculate and track evaluation metrics."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Macro and micro averages
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

        return metrics

    def get_confusion_matrix(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
    ) -> np.ndarray:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return confusion_matrix(y_true, y_pred)

    def get_classification_report(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        as_dict: bool = True,
    ) -> Union[str, Dict]:
        """
        Generate classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            as_dict: Whether to return as dictionary

        Returns:
            Classification report
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=as_dict,
            zero_division=0,
        )

    def calculate_per_class_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of per-class metrics
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        return per_class_metrics

    def calculate_top_k_accuracy(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_prob: Union[np.ndarray, torch.Tensor],
        k: int = 5,
    ) -> float:
        """
        Calculate top-k accuracy.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            k: Number of top predictions to consider

        Returns:
            Top-k accuracy
        """
        y_true = self._to_numpy(y_true)
        y_prob = self._to_numpy(y_prob)

        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

        return float(correct.mean())

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return np.array(x)
        return x


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


class MetricsTracker:
    """Track metrics across training epochs."""

    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """
        Update metrics for current epoch.

        Args:
            metrics: Dictionary of metric values
            epoch: Optional epoch number
        """
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

    def get_best(self, metric_name: str, mode: str = "max") -> tuple:
        """
        Get best value for a metric.

        Args:
            metric_name: Name of the metric
            mode: 'max' or 'min'

        Returns:
            Tuple of (best_value, best_epoch)
        """
        if metric_name not in self.metrics_history:
            return None, None

        values = self.metrics_history[metric_name]
        if mode == "max":
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        return values[best_idx], best_idx

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric_name not in self.metrics_history:
            return None
        return self.metrics_history[metric_name][-1]

    def get_history(self, metric_name: str) -> Optional[List[float]]:
        """Get full history for a metric."""
        return self.metrics_history.get(metric_name)
