"""Data preprocessing utilities for text and images."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from torchvision import transforms


class TextPreprocessor:
    """Text preprocessing for document classification."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_extra_whitespace: bool = True,
    ):
        """
        Initialize text preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_special_chars: Remove special characters
            remove_extra_whitespace: Remove extra whitespace
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_extra_whitespace = remove_extra_whitespace

    def preprocess(self, text: str) -> str:
        """
        Preprocess text.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove special characters but keep basic punctuation
        if self.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s.,!?;:-]", "", text)

        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted patterns.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove extra newlines
        text = re.sub(r"\n+", " ", text)

        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Simple word tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return text.split()


class ImagePreprocessor:
    """Image preprocessing for image classification."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
    ):
        """
        Initialize image preprocessor.

        Args:
            image_size: Target image size (height, width)
            mean: Normalization mean
            std: Normalization std
            augment: Whether to apply augmentation
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment

        # Basic transforms
        self.basic_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        # Augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.augment_transform = self.basic_transform

    def preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image.

        Args:
            image: Input image (path, PIL Image, or numpy array)

        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        # Apply transforms
        if self.augment:
            return self.augment_transform(image)
        else:
            return self.basic_transform(image)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor for visualization.

        Args:
            tensor: Normalized image tensor

        Returns:
            Denormalized image as numpy array
        """
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)

        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to numpy and transpose
        image = tensor.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)

        return image


class OCRProcessor:
    """OCR processing for document images."""

    def __init__(
        self,
        languages: List[str] = ["eng"],
        config: str = "--oem 3 --psm 6",
        preprocess: bool = True,
    ):
        """
        Initialize OCR processor.

        Args:
            languages: List of OCR languages
            config: Tesseract configuration
            preprocess: Whether to preprocess images before OCR
        """
        self.languages = "+".join(languages)
        self.config = config
        self.preprocess_image = preprocess

    def extract_text(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        """
        Extract text from image using OCR.

        Args:
            image: Input image

        Returns:
            Extracted text
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess image for better OCR
        if self.preprocess_image:
            image = self._preprocess_for_ocr(image)

        # Extract text using Tesseract
        text = pytesseract.image_to_string(
            image, lang=self.languages, config=self.config
        )

        return text.strip()

    def extract_text_with_confidence(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Union[str, float]]:
        """
        Extract text with confidence scores.

        Args:
            image: Input image

        Returns:
            Dictionary with text and confidence
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if self.preprocess_image:
            image = self._preprocess_for_ocr(image)

        # Get detailed OCR data
        data = pytesseract.image_to_data(
            image, lang=self.languages, config=self.config, output_type=pytesseract.Output.DICT
        )

        # Extract text and calculate average confidence
        texts = [word for word in data["text"] if word.strip()]
        confidences = [
            float(conf) for conf, word in zip(data["conf"], data["text"]) if word.strip() and conf != "-1"
        ]

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            "text": " ".join(texts),
            "confidence": avg_confidence,
            "word_count": len(texts),
        }

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Optional: dilation and erosion to remove noise
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return processed

    def detect_orientation(self, image: Union[str, Path, Image.Image, np.ndarray]) -> float:
        """
        Detect image orientation.

        Args:
            image: Input image

        Returns:
            Rotation angle
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Get orientation info from Tesseract
        try:
            osd = pytesseract.image_to_osd(image)
            angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
            return angle
        except Exception:
            return 0

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            Rotated image
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
