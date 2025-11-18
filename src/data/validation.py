"""Data validation utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate data quality and format."""

    def __init__(self):
        self.validation_results = []

    def validate_text(
        self,
        text: str,
        min_length: int = 10,
        max_length: Optional[int] = None,
        required_words: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate text data.

        Args:
            text: Input text
            min_length: Minimum text length
            max_length: Maximum text length
            required_words: List of required words

        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "text_length": len(text),
            "word_count": len(text.split()),
        }

        # Check if text is empty
        if not text or not text.strip():
            results["valid"] = False
            results["errors"].append("Text is empty")
            return results

        # Check minimum length
        if len(text) < min_length:
            results["valid"] = False
            results["errors"].append(f"Text length {len(text)} is below minimum {min_length}")

        # Check maximum length
        if max_length and len(text) > max_length:
            results["warnings"].append(f"Text length {len(text)} exceeds maximum {max_length}")

        # Check for required words
        if required_words:
            text_lower = text.lower()
            missing_words = [word for word in required_words if word.lower() not in text_lower]
            if missing_words:
                results["warnings"].append(f"Missing required words: {missing_words}")

        return results

    def validate_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        min_size: Optional[tuple] = None,
        max_size: Optional[tuple] = None,
        allowed_modes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate image data.

        Args:
            image: Input image
            min_size: Minimum image size (width, height)
            max_size: Maximum image size (width, height)
            allowed_modes: Allowed image modes

        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image)
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"Failed to load image: {e}")
                return results
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Get image properties
        width, height = image.size
        mode = image.mode

        results["width"] = width
        results["height"] = height
        results["mode"] = mode
        results["format"] = image.format

        # Check minimum size
        if min_size:
            min_width, min_height = min_size
            if width < min_width or height < min_height:
                results["valid"] = False
                results["errors"].append(
                    f"Image size ({width}x{height}) is below minimum ({min_width}x{min_height})"
                )

        # Check maximum size
        if max_size:
            max_width, max_height = max_size
            if width > max_width or height > max_height:
                results["warnings"].append(
                    f"Image size ({width}x{height}) exceeds maximum ({max_width}x{max_height})"
                )

        # Check image mode
        if allowed_modes and mode not in allowed_modes:
            results["valid"] = False
            results["errors"].append(f"Image mode '{mode}' not in allowed modes: {allowed_modes}")

        return results

    def validate_dataset(
        self,
        data: Union[pd.DataFrame, List[Dict]],
        required_columns: Optional[List[str]] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate dataset format and content.

        Args:
            data: Input dataset
            required_columns: List of required columns
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Convert to DataFrame if list of dicts
        if isinstance(data, list):
            data = pd.DataFrame(data)

        results["num_samples"] = len(data)
        results["columns"] = list(data.columns)

        # Check if dataset is empty
        if len(data) == 0:
            results["valid"] = False
            results["errors"].append("Dataset is empty")
            return results

        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                results["valid"] = False
                results["errors"].append(f"Missing required columns: {missing_columns}")

        # Check text column
        if text_column:
            if text_column not in data.columns:
                results["valid"] = False
                results["errors"].append(f"Text column '{text_column}' not found")
            else:
                # Check for missing values
                missing_count = data[text_column].isna().sum()
                if missing_count > 0:
                    results["warnings"].append(
                        f"{missing_count} missing values in text column"
                    )

                # Check for empty strings
                empty_count = (data[text_column].astype(str).str.strip() == "").sum()
                if empty_count > 0:
                    results["warnings"].append(
                        f"{empty_count} empty strings in text column"
                    )

        # Check label column
        if label_column:
            if label_column not in data.columns:
                results["valid"] = False
                results["errors"].append(f"Label column '{label_column}' not found")
            else:
                # Get label distribution
                label_counts = data[label_column].value_counts()
                results["num_classes"] = len(label_counts)
                results["class_distribution"] = label_counts.to_dict()

                # Check for class imbalance
                if len(label_counts) > 1:
                    max_count = label_counts.max()
                    min_count = label_counts.min()
                    imbalance_ratio = max_count / min_count

                    if imbalance_ratio > 10:
                        results["warnings"].append(
                            f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})"
                        )
                    elif imbalance_ratio > 5:
                        results["warnings"].append(
                            f"Class imbalance detected (ratio: {imbalance_ratio:.2f})"
                        )

        return results

    def validate_file_path(self, file_path: Union[str, Path], must_exist: bool = True) -> Dict[str, Any]:
        """
        Validate file path.

        Args:
            file_path: Path to validate
            must_exist: Whether file must exist

        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        file_path = Path(file_path)

        results["path"] = str(file_path)
        results["exists"] = file_path.exists()

        # Check if file exists
        if must_exist and not file_path.exists():
            results["valid"] = False
            results["errors"].append(f"File does not exist: {file_path}")
            return results

        if file_path.exists():
            results["is_file"] = file_path.is_file()
            results["is_dir"] = file_path.is_dir()
            results["suffix"] = file_path.suffix

            if file_path.is_file():
                results["size_bytes"] = file_path.stat().st_size

        return results

    def validate_batch(
        self,
        items: List[Any],
        validation_fn: callable,
        stop_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate batch of items.

        Args:
            items: List of items to validate
            validation_fn: Validation function to apply
            stop_on_error: Whether to stop on first error

        Returns:
            Batch validation results
        """
        results = {
            "total": len(items),
            "valid": 0,
            "invalid": 0,
            "errors": [],
            "warnings": [],
        }

        for i, item in enumerate(items):
            try:
                item_result = validation_fn(item)

                if item_result.get("valid", True):
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    if item_result.get("errors"):
                        results["errors"].append({
                            "index": i,
                            "errors": item_result["errors"]
                        })

                    if stop_on_error:
                        break

                if item_result.get("warnings"):
                    results["warnings"].append({
                        "index": i,
                        "warnings": item_result["warnings"]
                    })

            except Exception as e:
                results["invalid"] += 1
                results["errors"].append({
                    "index": i,
                    "errors": [str(e)]
                })

                if stop_on_error:
                    break

        return results

    def log_validation_results(self, results: Dict[str, Any], name: str = "Validation"):
        """
        Log validation results.

        Args:
            results: Validation results dictionary
            name: Name for logging
        """
        if results.get("valid", True):
            logger.info(f"{name}: PASSED")
        else:
            logger.error(f"{name}: FAILED")

        if results.get("errors"):
            for error in results["errors"]:
                logger.error(f"  Error: {error}")

        if results.get("warnings"):
            for warning in results["warnings"]:
                logger.warning(f"  Warning: {warning}")
