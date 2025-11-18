#!/usr/bin/env python
"""Example usage of the Multimodal AI Suite."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import MultimodalPredictor


def example_document_classification():
    """Example of document classification."""
    print("\n=== Document Classification Example ===\n")

    # Initialize predictor
    predictor = MultimodalPredictor(model_type="document")

    # Example text
    sample_text = """
    Artificial Intelligence (AI) has transformed many industries, from healthcare to finance.
    Machine learning models can now process vast amounts of data and make accurate predictions.
    Deep learning, a subset of machine learning, uses neural networks with multiple layers
    to learn complex patterns in data.
    """

    # Predict
    result = predictor.predict_document(sample_text)

    print(f"Text preview: {result['text'][:100]}...")
    print(f"\nPredicted class: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nTop-5 predictions:")
    for pred in result['top_k_predictions']:
        print(f"  Class {pred['class']}: {pred['confidence']:.2%}")


def example_image_classification():
    """Example of image classification."""
    print("\n=== Image Classification Example ===\n")

    # Initialize predictor
    predictor = MultimodalPredictor(model_type="image")

    print("Note: This example requires an actual image file.")
    print("Usage: predictor.predict_image('path/to/image.jpg')")


def example_multimodal_fusion():
    """Example of multimodal fusion."""
    print("\n=== Multimodal Fusion Example ===\n")

    # Initialize predictor
    predictor = MultimodalPredictor(model_type="multimodal")

    print("Note: This example requires both a document and an image.")
    print("Usage: predictor.predict_multimodal('path/to/document.pdf', 'path/to/image.jpg')")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("  Multimodal Intelligence Workflow Suite - Examples")
    print("=" * 60)

    # Run examples
    example_document_classification()
    example_image_classification()
    example_multimodal_fusion()

    print("\n" + "=" * 60)
    print("  Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
