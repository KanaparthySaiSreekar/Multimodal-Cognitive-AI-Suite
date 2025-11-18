"""Load testing for multimodal AI system.

Tests system performance under concurrent load to verify latency claims.
Target: <600ms per request under normal load.
"""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import torch
from PIL import Image
import numpy as np

from src.inference.predictor import MultimodalPredictor


class LoadTester:
    """Load testing framework for multimodal system."""

    def __init__(self, model_type: str = "document"):
        """
        Initialize load tester.

        Args:
            model_type: Type of model to test
        """
        self.model_type = model_type
        self.predictor = MultimodalPredictor(model_type=model_type)
        self.results = []

    def single_request_document(self, text: str) -> Dict:
        """Execute single document classification request."""
        start_time = time.time()

        try:
            result = self.predictor.predict_document(text)
            latency = (time.time() - start_time) * 1000  # Convert to ms

            return {
                "success": True,
                "latency_ms": latency,
                "prediction": result.get("prediction"),
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {"success": False, "latency_ms": latency, "error": str(e)}

    def single_request_image(self, image: Image.Image) -> Dict:
        """Execute single image classification request."""
        start_time = time.time()

        try:
            result = self.predictor.predict_image(image)
            latency = (time.time() - start_time) * 1000

            return {
                "success": True,
                "latency_ms": latency,
                "prediction": result.get("prediction"),
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {"success": False, "latency_ms": latency, "error": str(e)}

    def single_request_multimodal(self, text: str, image: Image.Image) -> Dict:
        """Execute single multimodal request."""
        start_time = time.time()

        try:
            result = self.predictor.predict_multimodal(text, image)
            latency = (time.time() - start_time) * 1000

            return {
                "success": True,
                "latency_ms": latency,
                "prediction": result.get("prediction"),
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {"success": False, "latency_ms": latency, "error": str(e)}

    def load_test_sequential(
        self, num_requests: int = 100, test_data=None
    ) -> Dict:
        """
        Run sequential load test.

        Args:
            num_requests: Number of requests to send
            test_data: Test data (text, images, or both)

        Returns:
            Dictionary with test results
        """
        print(f"\nRunning sequential load test with {num_requests} requests...")

        results = []

        for i in range(num_requests):
            if self.model_type == "document":
                text = test_data or f"Test document number {i} for load testing."
                result = self.single_request_document(text)
            elif self.model_type == "image":
                image = test_data or Image.new("RGB", (512, 512), color="red")
                result = self.single_request_image(image)
            else:  # multimodal
                text = f"Test caption {i}"
                image = Image.new("RGB", (512, 512), color="blue")
                result = self.single_request_multimodal(text, image)

            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")

        return self._analyze_results(results)

    def load_test_concurrent(
        self,
        num_requests: int = 100,
        concurrency: int = 10,
        test_data=None,
    ) -> Dict:
        """
        Run concurrent load test.

        Args:
            num_requests: Total number of requests
            concurrency: Number of concurrent requests
            test_data: Test data

        Returns:
            Dictionary with test results
        """
        print(
            f"\nRunning concurrent load test: {num_requests} requests, "
            f"{concurrency} concurrent..."
        )

        results = []

        def make_request(request_id):
            if self.model_type == "document":
                text = test_data or f"Concurrent test document {request_id}"
                return self.single_request_document(text)
            elif self.model_type == "image":
                image = test_data or Image.new("RGB", (512, 512), color="green")
                return self.single_request_image(image)
            else:
                text = f"Concurrent caption {request_id}"
                image = Image.new("RGB", (512, 512), color="yellow")
                return self.single_request_multimodal(text, image)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(make_request, i) for i in range(num_requests)
            ]

            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_requests} requests")

        return self._analyze_results(results)

    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze load test results."""
        latencies = [r["latency_ms"] for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        if not latencies:
            return {
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failures),
                "error": "No successful requests",
            }

        analysis = {
            "total_requests": len(results),
            "successful_requests": len(latencies),
            "failed_requests": len(failures),
            "success_rate": len(latencies) / len(results) * 100,
            "latency_stats": {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            },
            "target_compliance": {
                "target_latency_ms": 600,
                "requests_under_target": sum(1 for l in latencies if l < 600),
                "percent_under_target": sum(1 for l in latencies if l < 600)
                / len(latencies)
                * 100,
            },
        }

        if failures:
            analysis["failures"] = failures[:5]  # Show first 5 failures

        return analysis

    def print_report(self, results: Dict):
        """Print formatted test report."""
        print("\n" + "=" * 70)
        print(f"LOAD TEST REPORT - {self.model_type.upper()} MODEL")
        print("=" * 70)

        print(f"\nTotal Requests: {results['total_requests']}")
        print(f"Successful: {results['successful_requests']}")
        print(f"Failed: {results['failed_requests']}")
        print(f"Success Rate: {results['success_rate']:.2f}%")

        if "latency_stats" in results:
            print("\nLatency Statistics (ms):")
            stats = results["latency_stats"]
            print(f"  Min:      {stats['min_ms']:.2f}")
            print(f"  Max:      {stats['max_ms']:.2f}")
            print(f"  Mean:     {stats['mean_ms']:.2f}")
            print(f"  Median:   {stats['median_ms']:.2f}")
            print(f"  P95:      {stats['p95_ms']:.2f}")
            print(f"  P99:      {stats['p99_ms']:.2f}")
            print(f"  StdDev:   {stats['stdev_ms']:.2f}")

            print("\nTarget Compliance (< 600ms):")
            target = results["target_compliance"]
            print(f"  Target Latency: {target['target_latency_ms']}ms")
            print(f"  Requests Under Target: {target['requests_under_target']}")
            print(f"  Percentage: {target['percent_under_target']:.2f}%")

            if target["percent_under_target"] >= 95:
                print("  ✓ PASSED - 95%+ requests under target")
            else:
                print("  ✗ FAILED - Less than 95% requests under target")

        if "failures" in results:
            print("\nSample Failures:")
            for failure in results["failures"]:
                print(f"  - {failure.get('error', 'Unknown error')}")

        print("\n" + "=" * 70 + "\n")


def run_document_load_test():
    """Run load test for document classification."""
    tester = LoadTester(model_type="document")

    # Sequential test
    sequential_results = tester.load_test_sequential(num_requests=50)
    tester.print_report(sequential_results)

    # Concurrent test
    concurrent_results = tester.load_test_concurrent(
        num_requests=50, concurrency=5
    )
    tester.print_report(concurrent_results)


def run_image_load_test():
    """Run load test for image classification."""
    tester = LoadTester(model_type="image")

    # Sequential test
    sequential_results = tester.load_test_sequential(num_requests=30)
    tester.print_report(sequential_results)

    # Concurrent test
    concurrent_results = tester.load_test_concurrent(
        num_requests=30, concurrency=3
    )
    tester.print_report(concurrent_results)


def run_multimodal_load_test():
    """Run load test for multimodal fusion."""
    tester = LoadTester(model_type="multimodal")

    # Sequential test
    sequential_results = tester.load_test_sequential(num_requests=20)
    tester.print_report(sequential_results)

    # Concurrent test
    concurrent_results = tester.load_test_concurrent(
        num_requests=20, concurrency=2
    )
    tester.print_report(concurrent_results)


def run_all_load_tests():
    """Run all load tests."""
    print("\n" + "=" * 70)
    print("MULTIMODAL AI SYSTEM - COMPREHENSIVE LOAD TESTING")
    print("=" * 70)

    # Document classification
    print("\n[1/3] Document Classification Load Test")
    run_document_load_test()

    # Image classification
    print("\n[2/3] Image Classification Load Test")
    run_image_load_test()

    # Multimodal fusion
    print("\n[3/3] Multimodal Fusion Load Test")
    run_multimodal_load_test()

    print("\n" + "=" * 70)
    print("ALL LOAD TESTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run specific test or all tests
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "document":
            run_document_load_test()
        elif test_type == "image":
            run_image_load_test()
        elif test_type == "multimodal":
            run_multimodal_load_test()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python load_test.py [document|image|multimodal]")
    else:
        run_all_load_tests()
