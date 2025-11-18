"""Resource profiling and benchmarking for multimodal AI system.

Profiles GPU vs CPU performance, memory usage, batch size limits.
"""

import gc
import json
import os
import time
from typing import Dict, List

import numpy as np
import psutil
import torch
from PIL import Image

# Import models
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DocumentClassifier, ImageClassifier, MultimodalFusionModel
from src.data.preprocessing import TextPreprocessor, ImagePreprocessor


class ResourceProfiler:
    """Profile resource usage for models."""

    def __init__(self):
        self.results = []
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> Dict:
        """Get current memory usage."""
        mem = self.process.memory_info()
        return {
            "rss_mb": mem.rss / 1024 / 1024,  # Resident set size
            "vms_mb": mem.vms / 1024 / 1024,  # Virtual memory size
        }

    def get_gpu_memory_usage(self) -> Dict:
        """Get GPU memory usage if available."""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }

    def profile_inference(
        self,
        model,
        input_data,
        device: str = "cpu",
        num_warmup: int = 5,
        num_iterations: int = 50,
        batch_size: int = 1,
    ) -> Dict:
        """
        Profile inference performance.

        Args:
            model: Model to profile
            input_data: Input data (text, images, or both)
            device: Device to run on
            num_warmup: Number of warmup iterations
            num_iterations: Number of timed iterations
            batch_size: Batch size for inference

        Returns:
            Profiling results
        """
        model.to(device)
        model.eval()

        # Clear cache
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        gc.collect()

        # Record initial memory
        initial_memory = self.get_memory_usage()
        initial_gpu_memory = self.get_gpu_memory_usage()

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                if isinstance(model, DocumentClassifier):
                    model.predict([input_data] * batch_size)
                elif isinstance(model, ImageClassifier):
                    model.predict(input_data.repeat(batch_size, 1, 1, 1))

        # Clear cache again
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies = []

        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()

                if isinstance(model, DocumentClassifier):
                    model.predict([input_data] * batch_size)
                elif isinstance(model, ImageClassifier):
                    model.predict(input_data.repeat(batch_size, 1, 1, 1))

                if device == "cuda":
                    torch.cuda.synchronize()

                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)

        # Record peak memory
        peak_memory = self.get_memory_usage()
        peak_gpu_memory = self.get_gpu_memory_usage()

        results = {
            "device": device,
            "batch_size": batch_size,
            "latency_ms": {
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "std": float(np.std(latencies)),
            },
            "throughput_samples_per_sec": batch_size / (np.mean(latencies) / 1000),
            "memory": {
                "initial_rss_mb": initial_memory["rss_mb"],
                "peak_rss_mb": peak_memory["rss_mb"],
                "delta_rss_mb": peak_memory["rss_mb"] - initial_memory["rss_mb"],
            },
        }

        if device == "cuda":
            results["gpu_memory"] = {
                "initial_allocated_mb": initial_gpu_memory.get("allocated_mb", 0),
                "peak_allocated_mb": peak_gpu_memory.get("max_allocated_mb", 0),
                "delta_allocated_mb": peak_gpu_memory.get("max_allocated_mb", 0)
                - initial_gpu_memory.get("allocated_mb", 0),
            }

        return results

    def find_max_batch_size(
        self,
        model,
        input_generator,
        device: str = "cpu",
        start_batch: int = 1,
        max_batch: int = 128,
    ) -> int:
        """
        Find maximum batch size before OOM.

        Args:
            model: Model to test
            input_generator: Function that generates input of given batch size
            device: Device to test on
            start_batch: Starting batch size
            max_batch: Maximum batch size to test

        Returns:
            Maximum successful batch size
        """
        model.to(device)
        model.eval()

        max_successful = 0
        current_batch = start_batch

        while current_batch <= max_batch:
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    inputs = input_generator(current_batch)

                    if isinstance(model, DocumentClassifier):
                        model.predict(inputs)
                    elif isinstance(model, ImageClassifier):
                        model.predict(inputs)

                max_successful = current_batch
                print(f"  Batch size {current_batch}: SUCCESS")

                current_batch *= 2

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower():
                    print(f"  Batch size {current_batch}: OOM")
                    break
                else:
                    raise

        return max_successful


def profile_document_classifier():
    """Profile document classification model."""
    print("\n" + "=" * 70)
    print("PROFILING: Document Classifier")
    print("=" * 70)

    profiler = ResourceProfiler()
    model = DocumentClassifier(num_classes=10)

    sample_text = "This is a sample document for profiling resource usage and performance benchmarking."

    # CPU Profiling
    print("\n--- CPU Profiling ---")
    for batch_size in [1, 4, 8, 16]:
        print(f"\nBatch size: {batch_size}")
        results = profiler.profile_inference(
            model, sample_text, device="cpu", batch_size=batch_size
        )

        print(f"  Mean latency: {results['latency_ms']['mean']:.2f}ms")
        print(f"  P95 latency: {results['latency_ms']['p95']:.2f}ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/s")
        print(f"  Memory delta: {results['memory']['delta_rss_mb']:.2f}MB")

    # GPU Profiling (if available)
    if torch.cuda.is_available():
        print("\n--- GPU Profiling ---")
        for batch_size in [1, 4, 8, 16, 32]:
            print(f"\nBatch size: {batch_size}")
            results = profiler.profile_inference(
                model, sample_text, device="cuda", batch_size=batch_size
            )

            print(f"  Mean latency: {results['latency_ms']['mean']:.2f}ms")
            print(f"  P95 latency: {results['latency_ms']['p95']:.2f}ms")
            print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/s")
            print(f"  GPU memory delta: {results['gpu_memory']['delta_allocated_mb']:.2f}MB")

        # Find max batch size
        print("\n--- Finding Maximum Batch Size (GPU) ---")

        def input_gen(batch_size):
            return [sample_text] * batch_size

        max_batch = profiler.find_max_batch_size(
            model, input_gen, device="cuda", max_batch=256
        )
        print(f"\nMaximum batch size: {max_batch}")


def profile_image_classifier():
    """Profile image classification model."""
    print("\n" + "=" * 70)
    print("PROFILING: Image Classifier")
    print("=" * 70)

    profiler = ResourceProfiler()
    model = ImageClassifier(num_classes=100)

    preprocessor = ImagePreprocessor()
    sample_image = Image.new("RGB", (512, 512), color="red")
    processed_image = preprocessor.preprocess(sample_image)

    # CPU Profiling
    print("\n--- CPU Profiling ---")
    for batch_size in [1, 2, 4, 8]:
        print(f"\nBatch size: {batch_size}")
        results = profiler.profile_inference(
            model, processed_image, device="cpu", batch_size=batch_size
        )

        print(f"  Mean latency: {results['latency_ms']['mean']:.2f}ms")
        print(f"  P95 latency: {results['latency_ms']['p95']:.2f}ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/s")
        print(f"  Memory delta: {results['memory']['delta_rss_mb']:.2f}MB")

    # GPU Profiling (if available)
    if torch.cuda.is_available():
        print("\n--- GPU Profiling ---")
        for batch_size in [1, 4, 8, 16, 32, 64]:
            print(f"\nBatch size: {batch_size}")
            results = profiler.profile_inference(
                model, processed_image, device="cuda", batch_size=batch_size
            )

            print(f"  Mean latency: {results['latency_ms']['mean']:.2f}ms")
            print(f"  P95 latency: {results['latency_ms']['p95']:.2f}ms")
            print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/s")
            print(f"  GPU memory delta: {results['gpu_memory']['delta_allocated_mb']:.2f}MB")

        # Find max batch size
        print("\n--- Finding Maximum Batch Size (GPU) ---")

        def input_gen(batch_size):
            return processed_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        max_batch = profiler.find_max_batch_size(
            model, input_gen, device="cuda", max_batch=256
        )
        print(f"\nMaximum batch size: {max_batch}")


def generate_report():
    """Generate comprehensive profiling report."""
    print("\n" + "=" * 70)
    print("RESOURCE PROFILING SUMMARY")
    print("=" * 70)

    print("\nSystem Information:")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    print(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\n" + "=" * 70)


def main():
    """Main profiling entry point."""
    print("\nMultimodal AI System - Resource Profiling")

    generate_report()

    # Profile each model type
    profile_document_classifier()
    profile_image_classifier()

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
