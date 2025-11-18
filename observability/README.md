# Observability & Metrics

Prometheus metrics exporter for production monitoring and observability.

## Quick Start

### Start Metrics Server

```python
from observability.metrics import start_metrics_server, metrics_registry

# Start metrics endpoint
start_metrics_server(port=9090)

# Metrics available at http://localhost:9090/metrics
```

### Instrument Code

```python
from observability.metrics import metrics_registry

# Track inference
metrics_registry.track_inference(
    model_type="document",
    latency_ms=234.5,
    success=True,
    confidence=0.92,
    predicted_class="technology",
    batch_size=1
)

# Update resource metrics
metrics_registry.update_resource_metrics(
    cpu_percent=45.2,
    memory_bytes=2147483648,
    gpu_percent={0: 78.5},
    gpu_memory_bytes={0: 4294967296}
)

# Track drift
metrics_registry.update_drift_metrics(
    distribution_drift=0.08,
    confidence_drift=0.12,
    performance_drift=0.02
)
```

### Use Decorator

```python
from observability.metrics import track_inference_decorator

@track_inference_decorator("document")
def predict_document(text):
    return model.predict(text)
```

## Metrics Exposed

### Request Metrics
- `inference_requests_total{model_type, status}` - Total inference requests
- `inference_latency_seconds{model_type}` - Inference latency histogram
- `error_count_total{error_code, model_type}` - Error counts

### Model Metrics
- `model_confidence_scores{model_type}` - Confidence score distribution
- `prediction_distribution_total{model_type, predicted_class}` - Prediction counts
- `batch_size{model_type}` - Batch size distribution

### Resource Metrics
- `cpu_utilization_percent` - CPU usage
- `memory_used_bytes` - Memory usage
- `gpu_utilization_percent{gpu_id}` - GPU utilization
- `gpu_memory_used_bytes{gpu_id}` - GPU memory usage

### Drift Metrics
- `drift_score{metric_type}` - Drift scores (distribution, confidence, performance)
- `drift_alerts_total{alert_type}` - Drift alert counts
- `model_accuracy{model_type}` - Current model accuracy

## Integration

### Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'multimodal-api'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

### Grafana Dashboards

Import dashboards from `observability/grafana/`:
- Main dashboard: Request metrics, latency, errors
- Resource dashboard: CPU, memory, GPU usage
- Drift dashboard: Model drift detection and alerts

### Alerts

Example Prometheus alert rules:

```yaml
groups:
  - name: multimodal_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, inference_latency_seconds_bucket) > 0.6
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above target ({{ $value }}s)"

      - alert: ModelDrift
        expr: drift_score{metric_type="distribution"} > 0.1
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Model drift detected (score: {{ $value }})"
```

## Resource Monitoring

Start automatic resource monitoring:

```python
from observability.metrics import ResourceMonitor, metrics_registry

monitor = ResourceMonitor(metrics_registry, update_interval=10)
monitor.start()

# Metrics updated every 10 seconds
# Stop with: monitor.stop()
```

## Testing

```bash
# Start metrics server
python -m observability.metrics

# In another terminal, check metrics
curl http://localhost:9090/metrics

# Simulate traffic
python tests/e2e/test_smoke_inference.py
```

## Production Deployment

### Docker

```dockerfile
EXPOSE 9090
CMD ["python", "-m", "observability.metrics"]
```

### Kubernetes

```yaml
apiVersion: v1
kind: Service
metadata:
  name: multimodal-metrics
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
spec:
  ports:
    - port: 9090
      name: metrics
```

## See Also

- [RUNBOOK.md](../docs/RUNBOOK.md) - Operational procedures
- [Load Testing](../load/locustfile.py) - Performance testing
- [Drift Monitoring](../src/utils/drift_monitor.py) - Drift detection
