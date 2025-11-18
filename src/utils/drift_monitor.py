"""Model drift detection and monitoring utilities."""

import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .logger import get_logger
from .metrics import MetricsCalculator

logger = get_logger(__name__)


class DriftDetector:
    """Detect model drift and trigger retraining when necessary."""

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        performance_threshold: float = 0.05,
        save_dir: Optional[str] = None,
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Number of recent predictions to track
            drift_threshold: Threshold for distribution drift
            performance_threshold: Threshold for performance degradation
            save_dir: Directory to save drift reports
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.save_dir = Path(save_dir) if save_dir else Path("./drift_reports")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Tracking queues
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.labels = deque(maxlen=window_size)

        # Baseline statistics
        self.baseline_distribution = None
        self.baseline_confidence_mean = None
        self.baseline_confidence_std = None
        self.baseline_accuracy = None

        # Drift history
        self.drift_events = []

    def set_baseline(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ):
        """
        Set baseline statistics for drift detection.

        Args:
            predictions: Baseline predictions
            confidences: Baseline confidence scores
            labels: Ground truth labels (if available)
        """
        # Store prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        self.baseline_distribution = dict(zip(unique.tolist(), counts.tolist()))

        # Store confidence statistics
        self.baseline_confidence_mean = np.mean(confidences)
        self.baseline_confidence_std = np.std(confidences)

        # Store accuracy if labels available
        if labels is not None:
            self.baseline_accuracy = np.mean(predictions == labels)

        logger.info("Baseline statistics set for drift detection")
        logger.info(f"Baseline confidence: {self.baseline_confidence_mean:.4f} ± {self.baseline_confidence_std:.4f}")
        if labels is not None:
            logger.info(f"Baseline accuracy: {self.baseline_accuracy:.4f}")

    def add_prediction(
        self,
        prediction: int,
        confidence: float,
        label: Optional[int] = None,
    ):
        """
        Add a new prediction for drift monitoring.

        Args:
            prediction: Predicted class
            confidence: Prediction confidence
            label: True label (if available)
        """
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        if label is not None:
            self.labels.append(label)

    def detect_distribution_drift(self) -> Tuple[bool, float]:
        """
        Detect drift in prediction distribution using KL divergence.

        Returns:
            Tuple of (is_drift, drift_score)
        """
        if self.baseline_distribution is None:
            logger.warning("Baseline not set. Cannot detect drift.")
            return False, 0.0

        if len(self.predictions) < 100:  # Minimum samples needed
            return False, 0.0

        # Current distribution
        unique, counts = np.unique(list(self.predictions), return_counts=True)
        current_distribution = dict(zip(unique.tolist(), counts.tolist()))

        # Calculate KL divergence
        kl_div = self._kl_divergence(self.baseline_distribution, current_distribution)

        is_drift = kl_div > self.drift_threshold

        if is_drift:
            logger.warning(f"Distribution drift detected! KL divergence: {kl_div:.4f}")

        return is_drift, kl_div

    def detect_confidence_drift(self) -> Tuple[bool, float]:
        """
        Detect drift in confidence scores.

        Returns:
            Tuple of (is_drift, z_score)
        """
        if self.baseline_confidence_mean is None:
            return False, 0.0

        if len(self.confidences) < 100:
            return False, 0.0

        # Current confidence mean
        current_mean = np.mean(list(self.confidences))

        # Calculate Z-score
        z_score = abs(current_mean - self.baseline_confidence_mean) / (
            self.baseline_confidence_std + 1e-8
        )

        is_drift = z_score > 3.0  # 3 standard deviations

        if is_drift:
            logger.warning(
                f"Confidence drift detected! Z-score: {z_score:.4f}, "
                f"Current mean: {current_mean:.4f}, Baseline: {self.baseline_confidence_mean:.4f}"
            )

        return is_drift, z_score

    def detect_performance_drift(self) -> Tuple[bool, float]:
        """
        Detect drift in model performance.

        Returns:
            Tuple of (is_drift, accuracy_drop)
        """
        if self.baseline_accuracy is None or len(self.labels) < 100:
            return False, 0.0

        # Current accuracy
        current_accuracy = np.mean(
            np.array(list(self.predictions))[-len(self.labels) :]
            == np.array(list(self.labels))
        )

        # Calculate accuracy drop
        accuracy_drop = self.baseline_accuracy - current_accuracy

        is_drift = accuracy_drop > self.performance_threshold

        if is_drift:
            logger.warning(
                f"Performance drift detected! Accuracy drop: {accuracy_drop:.4f} "
                f"(from {self.baseline_accuracy:.4f} to {current_accuracy:.4f})"
            )

        return is_drift, accuracy_drop

    def check_all_drift(self) -> Dict:
        """
        Check all types of drift.

        Returns:
            Dictionary with drift detection results
        """
        dist_drift, kl_div = self.detect_distribution_drift()
        conf_drift, z_score = self.detect_confidence_drift()
        perf_drift, acc_drop = self.detect_performance_drift()

        results = {
            "timestamp": time.time(),
            "num_samples": len(self.predictions),
            "distribution_drift": {
                "detected": dist_drift,
                "kl_divergence": float(kl_div),
            },
            "confidence_drift": {
                "detected": conf_drift,
                "z_score": float(z_score),
            },
            "performance_drift": {
                "detected": perf_drift,
                "accuracy_drop": float(acc_drop),
            },
            "any_drift": dist_drift or conf_drift or perf_drift,
        }

        # Log drift event
        if results["any_drift"]:
            self.drift_events.append(results)
            self._save_drift_report(results)

        return results

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.

        Returns:
            Tuple of (should_retrain, reason)
        """
        drift_status = self.check_all_drift()

        if not drift_status["any_drift"]:
            return False, "No drift detected"

        reasons = []

        if drift_status["distribution_drift"]["detected"]:
            reasons.append(
                f"Distribution drift (KL: {drift_status['distribution_drift']['kl_divergence']:.4f})"
            )

        if drift_status["confidence_drift"]["detected"]:
            reasons.append(
                f"Confidence drift (Z: {drift_status['confidence_drift']['z_score']:.4f})"
            )

        if drift_status["performance_drift"]["detected"]:
            reasons.append(
                f"Performance drift (Drop: {drift_status['performance_drift']['accuracy_drop']:.4f})"
            )

        reason = "; ".join(reasons)
        logger.warning(f"Retraining recommended: {reason}")

        return True, reason

    def _kl_divergence(
        self, baseline: Dict, current: Dict, epsilon: float = 1e-10
    ) -> float:
        """Calculate KL divergence between two distributions."""
        # Get all classes
        all_classes = set(baseline.keys()) | set(current.keys())

        # Normalize distributions
        baseline_total = sum(baseline.values())
        current_total = sum(current.values())

        kl_div = 0.0

        for cls in all_classes:
            p = baseline.get(cls, epsilon) / baseline_total
            q = current.get(cls, epsilon) / current_total

            kl_div += p * np.log(p / q)

        return kl_div

    def _save_drift_report(self, results: Dict):
        """Save drift report to file."""
        timestamp = int(results["timestamp"])
        report_path = self.save_dir / f"drift_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Drift report saved to {report_path}")

    def get_summary(self) -> Dict:
        """Get summary of drift monitoring."""
        return {
            "window_size": self.window_size,
            "current_samples": len(self.predictions),
            "drift_events": len(self.drift_events),
            "baseline_set": self.baseline_distribution is not None,
            "last_drift": self.drift_events[-1] if self.drift_events else None,
        }


class RetrainingTrigger:
    """Trigger automated model retraining based on drift detection."""

    def __init__(
        self,
        drift_detector: DriftDetector,
        retrain_callback: callable,
        cooldown_period: int = 86400,  # 24 hours
    ):
        """
        Initialize retraining trigger.

        Args:
            drift_detector: Drift detector instance
            retrain_callback: Function to call for retraining
            cooldown_period: Minimum time between retraining (seconds)
        """
        self.drift_detector = drift_detector
        self.retrain_callback = retrain_callback
        self.cooldown_period = cooldown_period

        self.last_retrain_time = 0
        self.retrain_history = []

    def check_and_trigger(self) -> bool:
        """
        Check drift and trigger retraining if necessary.

        Returns:
            True if retraining was triggered
        """
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_retrain_time < self.cooldown_period:
            logger.info("Retraining cooldown period active")
            return False

        # Check if retraining is needed
        should_retrain, reason = self.drift_detector.should_retrain()

        if not should_retrain:
            return False

        # Trigger retraining
        logger.info(f"Triggering model retraining. Reason: {reason}")

        try:
            self.retrain_callback(reason=reason)

            self.last_retrain_time = current_time
            self.retrain_history.append(
                {
                    "timestamp": current_time,
                    "reason": reason,
                    "status": "success",
                }
            )

            logger.info("Model retraining completed successfully")
            return True

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

            self.retrain_history.append(
                {
                    "timestamp": current_time,
                    "reason": reason,
                    "status": "failed",
                    "error": str(e),
                }
            )

            return False

    def get_history(self) -> List[Dict]:
        """Get retraining history."""
        return self.retrain_history
