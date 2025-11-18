"""Error taxonomy and standardized error handling.

Provides deterministic error codes and user-facing error messages.
"""

from enum import Enum
from typing import Dict, Optional


class ErrorCategory(Enum):
    """Error categories."""

    DOCUMENT_PROCESSING = "DOC"
    IMAGE_PROCESSING = "IMG"
    OCR = "OCR"
    MULTIMODAL = "MULTI"
    SYSTEM = "SYS"
    AUTHENTICATION = "AUTH"
    RATE_LIMIT = "RATE"
    VALIDATION = "VAL"


class ErrorCode(Enum):
    """Standardized error codes with descriptions."""

    # Document Processing Errors (DOC_xxx)
    DOC_001 = ("Corrupt or unsupported PDF file", "Document could not be read. Please ensure it's a valid PDF file.")
    DOC_002 = ("Empty document", "No text could be extracted from the document.")
    DOC_003 = ("Document too large", "Document exceeds maximum size limit.")
    DOC_004 = ("Unsupported language", "Document language is not supported.")
    DOC_005 = ("Invalid document format", "Unsupported document format. Supported formats: PDF, DOCX, TXT.")

    # Image Processing Errors (IMG_xxx)
    IMG_001 = ("Corrupt or invalid image", "Image file is corrupted or not a valid image format.")
    IMG_002 = ("Image size invalid", "Image is too large or too small for processing.")
    IMG_003 = ("Low image quality", "Image quality is too low for reliable classification.")
    IMG_004 = ("Unsupported image format", "Image format not supported. Use JPG, PNG, or BMP.")

    # OCR Errors (OCR_xxx)
    OCR_001 = ("Text orientation detection failed", "Could not determine text orientation.")
    OCR_002 = ("Low contrast text", "Text contrast is too low for accurate OCR.")
    OCR_003 = ("Unsupported OCR language", "Language not supported for OCR.")
    OCR_004 = ("Handwriting detection", "Handwritten text detected - accuracy may be reduced.")
    OCR_005 = ("OCR processing failed", "OCR extraction failed. Please check image quality.")

    # Multimodal Errors (MULTI_xxx)
    MULTI_001 = ("Text-image mismatch", "Text and image appear to be unrelated.")
    MULTI_002 = ("Missing modality", "Either text or image is missing for multimodal processing.")
    MULTI_003 = ("Fusion failed", "Multimodal fusion processing failed.")

    # System Errors (SYS_xxx)
    SYS_001 = ("Out of memory", "System ran out of memory. Try reducing batch size or input size.")
    SYS_002 = ("Model loading failed", "Failed to load model. Please contact support.")
    SYS_003 = ("Inference timeout", "Request timed out. Input may be too complex.")
    SYS_004 = ("GPU error", "GPU processing error. Falling back to CPU.")
    SYS_005 = ("Internal server error", "An internal error occurred. Please try again.")

    # Authentication Errors (AUTH_xxx)
    AUTH_001 = ("Invalid token", "Authentication token is invalid.")
    AUTH_002 = ("Token expired", "Authentication token has expired.")
    AUTH_003 = ("Unauthorized access", "You do not have permission to access this resource.")
    AUTH_004 = ("Missing authentication", "Authentication credentials are required.")

    # Rate Limit Errors (RATE_xxx)
    RATE_001 = ("Rate limit exceeded", "Too many requests. Please slow down.")
    RATE_002 = ("Daily quota exceeded", "Daily request quota exceeded.")

    # Validation Errors (VAL_xxx)
    VAL_001 = ("Invalid input", "Input validation failed.")
    VAL_002 = ("Missing required field", "Required field is missing.")
    VAL_003 = ("Invalid data type", "Data type is incorrect.")
    VAL_004 = ("Value out of range", "Value is outside acceptable range.")

    def __init__(self, technical_message: str, user_message: str):
        self.technical_message = technical_message
        self.user_message = user_message
        self.code = self.name


class MultimodalError(Exception):
    """Base exception for multimodal AI system."""

    def __init__(
        self,
        error_code: ErrorCode,
        details: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize error.

        Args:
            error_code: Standardized error code
            details: Additional details
            correlation_id: Request correlation ID
        """
        self.error_code = error_code
        self.code = error_code.code
        self.technical_message = error_code.technical_message
        self.user_message = error_code.user_message
        self.details = details
        self.correlation_id = correlation_id

        message = f"[{self.code}] {self.technical_message}"
        if details:
            message += f": {details}"

        super().__init__(message)

    def to_dict(self) -> Dict:
        """Convert error to dictionary for API responses."""
        return {
            "error": {
                "code": self.code,
                "message": self.user_message,
                "details": self.details,
                "correlation_id": self.correlation_id,
            }
        }

    def to_user_response(self) -> Dict:
        """Get user-friendly error response."""
        return {
            "success": False,
            "error_code": self.code,
            "message": self.user_message,
            "correlation_id": self.correlation_id,
        }


class DocumentProcessingError(MultimodalError):
    """Document processing errors."""

    pass


class ImageProcessingError(MultimodalError):
    """Image processing errors."""

    pass


class OCRError(MultimodalError):
    """OCR processing errors."""

    pass


class MultimodalFusionError(MultimodalError):
    """Multimodal fusion errors."""

    pass


class SystemError(MultimodalError):
    """System-level errors."""

    pass


class AuthenticationError(MultimodalError):
    """Authentication errors."""

    pass


class RateLimitError(MultimodalError):
    """Rate limiting errors."""

    def __init__(
        self,
        error_code: ErrorCode,
        wait_time: float,
        details: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(error_code, details, correlation_id)
        self.wait_time = wait_time

    def to_dict(self) -> Dict:
        """Include wait time in response."""
        result = super().to_dict()
        result["error"]["wait_time_seconds"] = self.wait_time
        return result


class ValidationError(MultimodalError):
    """Input validation errors."""

    pass


def get_error_documentation() -> Dict:
    """Get complete error code documentation."""
    docs = {}

    for error_code in ErrorCode:
        category = error_code.name.split("_")[0]
        code = error_code.name

        if category not in docs:
            docs[category] = {}

        docs[category][code] = {
            "code": code,
            "technical_message": error_code.technical_message,
            "user_message": error_code.user_message,
        }

    return docs


# Error recovery strategies
ERROR_RECOVERY_STRATEGIES = {
    ErrorCode.DOC_001: [
        "Ensure the PDF file is not corrupted",
        "Try opening the PDF in a PDF reader to verify it's valid",
        "Re-export or re-save the PDF from the original source",
    ],
    ErrorCode.DOC_002: [
        "Check if the document contains scanned images instead of text",
        "Use OCR processing for image-based documents",
        "Verify the document is not blank",
    ],
    ErrorCode.IMG_001: [
        "Verify the image file is not corrupted",
        "Try re-saving the image in a different format",
        "Check the image opens in an image viewer",
    ],
    ErrorCode.IMG_002: [
        "Resize the image to recommended dimensions (224x224 to 2048x2048)",
        "Reduce image resolution if too large",
        "Use a higher resolution image if too small",
    ],
    ErrorCode.OCR_005: [
        "Improve image quality (increase resolution, reduce noise)",
        "Ensure text is clearly visible and has good contrast",
        "Rotate image to correct orientation",
    ],
    ErrorCode.SYS_001: [
        "Reduce batch size",
        "Process smaller inputs",
        "Try again when system resources are available",
    ],
    ErrorCode.RATE_001: [
        "Wait before sending next request",
        "Reduce request frequency",
        "Upgrade to higher rate limit tier if available",
    ],
}


def get_recovery_strategy(error_code: ErrorCode) -> Optional[List[str]]:
    """Get recovery strategy for error code."""
    return ERROR_RECOVERY_STRATEGIES.get(error_code)
