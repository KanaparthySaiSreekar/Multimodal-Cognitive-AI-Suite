"""Prometheus metrics exporter for Multimodal AI System.

Exposes runtime metrics for monitoring, alerting, and observability.
Integrates with Prometheus, Grafana, and CloudWatch.

Metrics Categories:
    - Request metrics: throughput, latency, errors
    - Model metrics: inference time, batch size, confidence scores
    - Resource metrics: CPU, GPU, memory usage
    - Drift metrics: distribution shifts, performance degradation

Usage:
    # Install: pip install prometheus-client

    # Start metrics server:
    from observability.metrics import start_metrics_server
    start_metrics_server(port=9090)

    # Instrument code:
    from observability.metrics import metrics_registry
    metrics_registry.track_inference(model_type="document", latency_ms=234, success=True)

    # Access metrics:
    curl http://localhost:9090/metrics
"""

import functools
import os
import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
    start_http_server,
)


class MetricsRegistry:
    """Central registry for all Prometheus metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics registry.

        Args:
            registry: Optional custom registry (useful for testing)
        """
        self.registry = registry or CollectorRegistry()

        # ===== Request Metrics =====
        self.inference_requests_total = Counter(
            "inference_requests_total",
            "Total number of inference requests",
            ["model_type", "status"],
            registry=self.registry,
        )

        self.inference_latency_seconds = Histogram(
            "inference_latency_seconds",
            "Inference request latency in seconds",
            ["model_type"],
            buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0),
            registry=self.registry,
        )

        self.inference_latency_summary = Summary(
            "inference_latency_summary_seconds",
            "Summary of inference latency",
            ["model_type"],
            registry=self.registry,
        )

        self.error_count_total = Counter(
            "error_count_total",
            "Total number of errors by type",
            ["error_code", "model_type"],
            registry=self.registry,
        )

        # ===== Model Performance Metrics =====
        self.model_confidence_scores = Histogram(
            "model_confidence_scores",
            "Distribution of model confidence scores",
            ["model_type"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
            registry=self.registry,
        )

        self.prediction_distribution = Counter(
            "prediction_distribution_total",
            "Distribution of predicted classes",
            ["model_type", "predicted_class"],
            registry=self.registry,
        )

        self.batch_size = Histogram(
            "batch_size",
            "Distribution of batch sizes",
            ["model_type"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
            registry=self.registry,
        )

        # ===== Resource Metrics =====
        self.input_size_bytes = Histogram(
            "input_size_bytes",
            "Size of input data in bytes",
            ["input_type"],
            buckets=(1024, 10240, 102400, 1048576, 10485760),  # 1KB to 10MB
            registry=self.registry,
        )

        self.gpu_utilization = Gauge(
            "gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_id"],
            registry=self.registry,
        )

        self.gpu_memory_used_bytes = Gauge(
            "gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["gpu_id"],
            registry=self.registry,
        )

        self.cpu_utilization = Gauge(
            "cpu_utilization_percent",
            "CPU utilization percentage",
            registry=self.registry,
        )

        self.memory_used_bytes = Gauge(
            "memory_used_bytes",
            "System memory used in bytes",
            registry=self.registry,
        )

        # ===== Drift Detection Metrics =====
        self.drift_score = Gauge(
            "drift_score",
            "Current drift detection score",
            ["metric_type"],  # distribution, confidence, performance
            registry=self.registry,
        )

        self.drift_alerts_total = Counter(
            "drift_alerts_total",
            "Total number of drift alerts triggered",
            ["alert_type"],
            registry=self.registry,
        )

        self.model_accuracy = Gauge(
            "model_accuracy",
            "Current model accuracy (when ground truth available)",
            ["model_type"],
            registry=self.registry,
        )

        # ===== System Info =====
        self.model_info = Info(
            "model_info",
            "Information about loaded models",
            registry=self.registry,
        )

        self.system_info = Info(
            "system_info",
            "System configuration information",
            registry=self.registry,
        )

        # ===== Health Metrics =====
        self.health_status = Gauge(
            "health_status",
            "Health status (1=healthy, 0=unhealthy)",
            ["component"],
            registry=self.registry,
        )

        self.last_successful_prediction_timestamp = Gauge(
            "last_successful_prediction_timestamp",
            "Unix timestamp of last successful prediction",
            ["model_type"],
            registry=self.registry,
        )

        # Initialize system info
        self._set_system_info()

    def _set_system_info(self):
        """Set static system information."""
        import platform
        import torch

        self.system_info.info(
            {
                "python_version": platform.python_version(),
                "platform": platform.system(),
                "pytorch_version": torch.__version__,
                "cuda_available": str(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            }
        )

    def track_inference(
        self,
        model_type: str,
        latency_ms: float,
        success: bool,
        confidence: Optional[float] = None,
        predicted_class: Optional[str] = None,
        error_code: Optional[str] = None,
        batch_size: int = 1,
    ):
        """
        Track a single inference request.

        Args:
            model_type: Type of model (document, image, multimodal)
            latency_ms: Inference latency in milliseconds
            success: Whether inference succeeded
            confidence: Model confidence score (0-1)
            predicted_class: Predicted class label
            error_code: Error code if failed
            batch_size: Number of samples in batch
        """
        status = "success" if success else "failure"
        self.inference_requests_total.labels(model_type=model_type, status=status).inc()

        latency_seconds = latency_ms / 1000.0
        self.inference_latency_seconds.labels(model_type=model_type).observe(latency_seconds)
        self.inference_latency_summary.labels(model_type=model_type).observe(latency_seconds)

        if success:
            self.last_successful_prediction_timestamp.labels(model_type=model_type).set(time.time())

            if confidence is not None:
                self.model_confidence_scores.labels(model_type=model_type).observe(confidence)

            if predicted_class is not None:
                self.prediction_distribution.labels(
                    model_type=model_type, predicted_class=str(predicted_class)
                ).inc()

            if batch_size > 0:
                self.batch_size.labels(model_type=model_type).observe(batch_size)

        else:
            if error_code:
                self.error_count_total.labels(error_code=error_code, model_type=model_type).inc()

    def track_input_size(self, input_type: str, size_bytes: int):
        """Track input data size."""
        self.input_size_bytes.labels(input_type=input_type).observe(size_bytes)

    def update_resource_metrics(
        self,
        cpu_percent: Optional[float] = None,
        memory_bytes: Optional[int] = None,
        gpu_percent: Optional[Dict[int, float]] = None,
        gpu_memory_bytes: Optional[Dict[int, int]] = None,
    ):
        """
        Update resource utilization metrics.

        Args:
            cpu_percent: CPU utilization percentage
            memory_bytes: Memory usage in bytes
            gpu_percent: Dict mapping GPU ID to utilization percentage
            gpu_memory_bytes: Dict mapping GPU ID to memory usage in bytes
        """
        if cpu_percent is not None:
            self.cpu_utilization.set(cpu_percent)

        if memory_bytes is not None:
            self.memory_used_bytes.set(memory_bytes)

        if gpu_percent:
            for gpu_id, percent in gpu_percent.items():
                self.gpu_utilization.labels(gpu_id=str(gpu_id)).set(percent)

        if gpu_memory_bytes:
            for gpu_id, mem_bytes in gpu_memory_bytes.items():
                self.gpu_memory_used_bytes.labels(gpu_id=str(gpu_id)).set(mem_bytes)

    def update_drift_metrics(
        self,
        distribution_drift: Optional[float] = None,
        confidence_drift: Optional[float] = None,
        performance_drift: Optional[float] = None,
    ):
        """
        Update drift detection metrics.

        Args:
            distribution_drift: Distribution drift score (e.g., KL divergence)
            confidence_drift: Confidence drift score (e.g., Z-score)
            performance_drift: Performance drift score (accuracy delta)
        """
        if distribution_drift is not None:
            self.drift_score.labels(metric_type="distribution").set(distribution_drift)

        if confidence_drift is not None:
            self.drift_score.labels(metric_type="confidence").set(confidence_drift)

        if performance_drift is not None:
            self.drift_score.labels(metric_type="performance").set(performance_drift)

    def trigger_drift_alert(self, alert_type: str):
        """Record a drift alert."""
        self.drift_alerts_total.labels(alert_type=alert_type).inc()

    def update_model_accuracy(self, model_type: str, accuracy: float):
        """Update current model accuracy."""
        self.model_accuracy.labels(model_type=model_type).set(accuracy)

    def set_model_info(self, model_type: str, model_name: str, version: str, num_classes: int):
        """Set model information."""
        self.model_info.info(
            {
                "model_type": model_type,
                "model_name": model_name,
                "version": version,
                "num_classes": str(num_classes),
            }
        )

    def update_health(self, component: str, is_healthy: bool):
        """Update component health status."""
        self.health_status.labels(component=component).set(1 if is_healthy else 0)

    @contextmanager
    def track_latency(self, model_type: str):
        """
        Context manager for tracking latency.

        Usage:
            with metrics_registry.track_latency("document"):
                result = model.predict(input)
        """
        start_time = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.inference_latency_seconds.labels(model_type=model_type).observe(latency_ms / 1000.0)

    def get_metrics_text(self) -> bytes:
        """Get metrics in Prometheus text format."""
        return generate_latest(self.registry)


# Global metrics registry instance
metrics_registry = MetricsRegistry()


def start_metrics_server(port: int = 9090, addr: str = "0.0.0.0"):
    """
    Start HTTP server to expose Prometheus metrics.

    Args:
        port: Port to listen on (default: 9090)
        addr: Address to bind to (default: 0.0.0.0)
    """
    print(f"Starting Prometheus metrics server on {addr}:{port}")
    start_http_server(port=port, addr=addr, registry=metrics_registry.registry)
    print(f"Metrics available at http://{addr}:{port}/metrics")


def track_inference_decorator(model_type: str):
    """
    Decorator for automatically tracking inference latency.

    Usage:
        @track_inference_decorator("document")
        def predict_document(text):
            return model.predict(text)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_code = None

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_code = getattr(e, "code", "UNKNOWN")
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                metrics_registry.track_inference(
                    model_type=model_type,
                    latency_ms=latency_ms,
                    success=success,
                    error_code=error_code,
                )

        return wrapper

    return decorator


class ResourceMonitor:
    """Monitor and report system resource usage."""

    def __init__(self, metrics_registry: MetricsRegistry, update_interval: int = 10):
        """
        Initialize resource monitor.

        Args:
            metrics_registry: Metrics registry to update
            update_interval: Update interval in seconds
        """
        self.metrics_registry = metrics_registry
        self.update_interval = update_interval
        self._stop = False

    def start(self):
        """Start monitoring in background thread."""
        import threading

        def monitor_loop():
            import psutil

            try:
                import torch
                has_torch = True
            except ImportError:
                has_torch = False

            while not self._stop:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                self.metrics_registry.update_resource_metrics(
                    cpu_percent=cpu_percent,
                    memory_bytes=memory.used,
                )

                # GPU metrics (if available)
                if has_torch and torch.cuda.is_available():
                    gpu_percent = {}
                    gpu_memory = {}

                    for i in range(torch.cuda.device_count()):
                        # Memory usage
                        mem_allocated = torch.cuda.memory_allocated(i)
                        gpu_memory[i] = mem_allocated

                        # Note: torch doesn't provide GPU utilization directly
                        # For production, use nvidia-ml-py3 (pynvml)

                    self.metrics_registry.update_resource_metrics(
                        gpu_memory_bytes=gpu_memory,
                    )

                time.sleep(self.update_interval)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    def stop(self):
        """Stop monitoring."""
        self._stop = True


# Example Grafana dashboard configuration (JSON)
GRAFANA_DASHBOARD_TEMPLATE = """
{
  "dashboard": {
    "title": "Multimodal AI System Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(inference_requests_total[5m])"
        }]
      },
      {
        "title": "P95 Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, inference_latency_seconds_bucket)"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(error_count_total[5m])"
        }]
      },
      {
        "title": "Drift Score",
        "targets": [{
          "expr": "drift_score"
        }]
      }
    ]
  }
}
"""


if __name__ == "__main__":
    # Example usage
    print("Starting metrics server...")
    start_metrics_server(port=9090)

    # Simulate some metrics
    print("Simulating metrics...")
    for i in range(10):
        metrics_registry.track_inference(
            model_type="document",
            latency_ms=200 + i * 10,
            success=True,
            confidence=0.9 - i * 0.01,
            predicted_class=f"class_{i % 3}",
        )
        time.sleep(0.5)

    print("\nMetrics available at http://localhost:9090/metrics")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping metrics server...")
