"""Locust load testing for Multimodal AI System.

Industry-standard load testing using Locust framework.
Tests HTTP endpoints under realistic user traffic patterns.

Usage:
    # Install: pip install locust

    # Run with web UI:
    locust -f load/locustfile.py --host=http://localhost:8501

    # Run headless (1000 users, 100 spawn rate, 5 min):
    locust -f load/locustfile.py --host=http://localhost:8501 \
           --users 1000 --spawn-rate 100 --run-time 5m --headless

    # Distributed testing:
    locust -f load/locustfile.py --master
    locust -f load/locustfile.py --worker --master-host=<master-ip>

Performance Targets:
    - P95 latency: < 600ms
    - P99 latency: < 1000ms
    - Error rate: < 0.5%
    - Throughput: > 100 req/s (single instance)
"""

import base64
import io
import json
import random
import time
from typing import Dict, List

import numpy as np
from locust import HttpUser, TaskSet, between, task
from PIL import Image


class DocumentClassificationTasks(TaskSet):
    """Task set for document classification endpoints."""

    # Sample documents for realistic testing
    SAMPLE_DOCUMENTS = [
        "Artificial intelligence is transforming industries worldwide.",
        "The quarterly financial report shows a 15% increase in revenue.",
        "Patient presents with acute respiratory symptoms and fever.",
        "Breaking news: Major policy changes announced by government officials.",
        "Machine learning algorithms require large datasets for training.",
        "Stock market analysis indicates bullish trends in tech sector.",
        "Clinical trial results demonstrate significant efficacy improvements.",
        "Sports update: Team wins championship after dramatic overtime.",
        "Deep neural networks achieve state-of-the-art performance.",
        "Economic forecast predicts moderate growth for next quarter.",
    ]

    @task(3)
    def classify_short_document(self):
        """Test classification of short documents (most common use case)."""
        document = random.choice(self.SAMPLE_DOCUMENTS)

        with self.client.post(
            "/api/v1/classify/document",
            json={"text": document},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "prediction" in result and "confidence" in result:
                        response.success()
                    else:
                        response.failure("Missing required fields in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def classify_long_document(self):
        """Test classification of longer documents."""
        # Generate longer document (500-1000 words)
        long_doc = " ".join(random.choices(self.SAMPLE_DOCUMENTS, k=50))

        with self.client.post(
            "/api/v1/classify/document",
            json={"text": long_doc},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def batch_classify_documents(self):
        """Test batch document classification."""
        documents = random.sample(self.SAMPLE_DOCUMENTS, k=5)

        with self.client.post(
            "/api/v1/classify/document/batch",
            json={"texts": documents},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    results = response.json()
                    if len(results) == len(documents):
                        response.success()
                    else:
                        response.failure("Batch size mismatch")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")


class ImageClassificationTasks(TaskSet):
    """Task set for image classification endpoints."""

    def on_start(self):
        """Initialize test images when user starts."""
        self.test_images = self._generate_test_images()

    def _generate_test_images(self, count: int = 10) -> List[bytes]:
        """Generate synthetic test images."""
        images = []
        for _ in range(count):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Convert to bytes
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images.append(buffer.getvalue())

        return images

    @task(3)
    def classify_image(self):
        """Test single image classification."""
        image_bytes = random.choice(self.test_images)

        # Encode as base64 for JSON API
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        with self.client.post(
            "/api/v1/classify/image",
            json={"image": image_b64},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "prediction" in result and "confidence" in result:
                        response.success()
                    else:
                        response.failure("Missing required fields")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def classify_image_multipart(self):
        """Test image classification with multipart/form-data."""
        image_bytes = random.choice(self.test_images)

        files = {"file": ("test_image.png", image_bytes, "image/png")}

        with self.client.post(
            "/api/v1/classify/image/upload",
            files=files,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class MultimodalFusionTasks(TaskSet):
    """Task set for multimodal fusion endpoints."""

    SAMPLE_DOCUMENTS = DocumentClassificationTasks.SAMPLE_DOCUMENTS

    def on_start(self):
        """Initialize test data."""
        self.test_images = self._generate_test_images()

    def _generate_test_images(self, count: int = 5) -> List[bytes]:
        """Generate synthetic test images."""
        images = []
        for _ in range(count):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images.append(buffer.getvalue())
        return images

    @task
    def classify_multimodal(self):
        """Test multimodal classification (text + image)."""
        text = random.choice(self.SAMPLE_DOCUMENTS)
        image_bytes = random.choice(self.test_images)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        with self.client.post(
            "/api/v1/classify/multimodal",
            json={"text": text, "image": image_b64},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "prediction" in result and "confidence" in result:
                        response.success()
                    else:
                        response.failure("Missing required fields")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")


class HealthCheckTasks(TaskSet):
    """Health check and monitoring endpoint tasks."""

    @task(10)
    def health_check(self):
        """Test health check endpoint (frequent lightweight requests)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: HTTP {response.status_code}")

    @task(2)
    def metrics_endpoint(self):
        """Test metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics failed: HTTP {response.status_code}")


class DocumentUser(HttpUser):
    """Simulated user focusing on document classification."""

    tasks = [DocumentClassificationTasks]
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    weight = 3  # 60% of users


class ImageUser(HttpUser):
    """Simulated user focusing on image classification."""

    tasks = [ImageClassificationTasks]
    wait_time = between(2, 5)  # Wait 2-5 seconds between requests
    weight = 1  # 20% of users


class MultimodalUser(HttpUser):
    """Simulated user focusing on multimodal tasks."""

    tasks = [MultimodalFusionTasks]
    wait_time = between(3, 6)  # Wait 3-6 seconds between requests
    weight = 1  # 20% of users


class MixedWorkloadUser(HttpUser):
    """Realistic user with mixed workload."""

    wait_time = between(1, 4)

    @task(5)
    def document_task(self):
        """Document classification (most common)."""
        document = random.choice(DocumentClassificationTasks.SAMPLE_DOCUMENTS)
        self.client.post("/api/v1/classify/document", json={"text": document})

    @task(2)
    def image_task(self):
        """Image classification."""
        # Generate quick test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        self.client.post("/api/v1/classify/image", json={"image": image_b64})

    @task(1)
    def multimodal_task(self):
        """Multimodal classification."""
        text = random.choice(DocumentClassificationTasks.SAMPLE_DOCUMENTS)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        self.client.post("/api/v1/classify/multimodal", json={"text": text, "image": image_b64})

    @task(10)
    def health_check(self):
        """Regular health checks."""
        self.client.get("/health")


class StressTestUser(HttpUser):
    """Aggressive user for stress testing."""

    wait_time = between(0.1, 0.5)  # Minimal wait time
    weight = 0  # Not included in normal load, use explicitly

    tasks = [DocumentClassificationTasks, ImageClassificationTasks]


# Custom event hooks for detailed logging
from locust import events


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log when test starts."""
    print("\n" + "=" * 80)
    print("LOAD TEST STARTED")
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("=" * 80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log summary when test completes."""
    print("\n" + "=" * 80)
    print("LOAD TEST COMPLETED")

    stats = environment.stats.total

    print(f"\nRequests: {stats.num_requests}")
    print(f"Failures: {stats.num_failures}")
    print(f"Error Rate: {(stats.num_failures / stats.num_requests * 100) if stats.num_requests > 0 else 0:.2f}%")
    print(f"\nLatency:")
    print(f"  Mean: {stats.avg_response_time:.0f}ms")
    print(f"  Median: {stats.median_response_time:.0f}ms")
    print(f"  P95: {stats.get_response_time_percentile(0.95):.0f}ms")
    print(f"  P99: {stats.get_response_time_percentile(0.99):.0f}ms")
    print(f"  Max: {stats.max_response_time:.0f}ms")
    print(f"\nThroughput: {stats.total_rps:.2f} req/s")

    # Check if SLOs are met
    print("\n" + "-" * 80)
    print("SLO VALIDATION:")

    p95 = stats.get_response_time_percentile(0.95)
    p99 = stats.get_response_time_percentile(0.99)
    error_rate = (stats.num_failures / stats.num_requests * 100) if stats.num_requests > 0 else 0

    slo_p95 = "✓ PASS" if p95 < 600 else "✗ FAIL"
    slo_p99 = "✓ PASS" if p99 < 1000 else "✗ FAIL"
    slo_errors = "✓ PASS" if error_rate < 0.5 else "✗ FAIL"

    print(f"  P95 latency < 600ms: {slo_p95} ({p95:.0f}ms)")
    print(f"  P99 latency < 1000ms: {slo_p99} ({p99:.0f}ms)")
    print(f"  Error rate < 0.5%: {slo_errors} ({error_rate:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import subprocess
    import sys

    print("Starting Locust load testing...")
    print("\nUsage examples:")
    print("  1. With web UI: locust -f load/locustfile.py --host=http://localhost:8501")
    print("  2. Headless: locust -f load/locustfile.py --host=http://localhost:8501 --users 100 --spawn-rate 10 --run-time 2m --headless")
    print("\nStarting web UI on http://localhost:8089...")

    subprocess.run(["locust", "-f", __file__, "--host", "http://localhost:8501"])
