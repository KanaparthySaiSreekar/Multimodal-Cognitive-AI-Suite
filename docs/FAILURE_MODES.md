# Failure Modes and Edge Cases

This document details known failure modes, edge cases, and mitigation strategies for the Multimodal Intelligence Workflow Suite.

---

## Table of Contents

- [Document Processing Failures](#document-processing-failures)
- [Image Processing Failures](#image-processing-failures)
- [OCR Edge Cases](#ocr-edge-cases)
- [Multimodal Fusion Failures](#multimodal-fusion-failures)
- [System-Level Failures](#system-level-failures)
- [Mitigation Strategies](#mitigation-strategies)

---

## Document Processing Failures

### 1. Corrupt PDF Files

**Symptoms:**
- PyPDF2 raises `PdfReadError`
- Incomplete text extraction
- Garbled characters

**Root Causes:**
- Malformed PDF structure
- Encrypted PDFs without password
- PDF version incompatibility
- Corrupted file headers

**Mitigation:**
```python
# Error code: DOC_001
try:
    doc_data = ingestion.load_pdf(file_path, method="pdfplumber")
except Exception as e:
    # Fallback to alternative method
    try:
        doc_data = ingestion.load_pdf(file_path, method="pypdf2")
    except Exception:
        raise DocumentProcessingError("DOC_001", "Corrupt or unsupported PDF")
```

**Recovery Actions:**
1. Try alternative PDF parsing library
2. Attempt PDF repair with external tool
3. Fall back to OCR if PDF contains scanned images
4. Return error code DOC_001 to user

---

### 2. Empty Documents

**Symptoms:**
- Zero-length text extraction
- Classification fails with empty input

**Root Causes:**
- PDF with only images (no text layer)
- Blank pages
- Whitespace-only documents

**Mitigation:**
```python
# Error code: DOC_002
if not text or len(text.strip()) == 0:
    # Try OCR as fallback
    images = convert_pdf_to_images(file_path)
    text = ocr_processor.extract_text(images[0])

    if not text:
        raise DocumentProcessingError("DOC_002", "Empty document - no extractable text")
```

---

### 3. Extremely Long Documents

**Symptoms:**
- Out of memory errors
- Tokenization exceeds max length (512 tokens)
- Slow processing

**Root Causes:**
- Large documents (100+ pages)
- BERT token limit constraints

**Mitigation:**
```python
# Error code: DOC_003
if len(text) > MAX_LENGTH:
    # Chunking strategy
    chunks = split_into_chunks(text, chunk_size=512, overlap=50)
    predictions = []

    for chunk in chunks:
        result = model.predict([chunk])
        predictions.append(result)

    # Aggregate predictions (majority vote or averaging)
    final_prediction = aggregate_predictions(predictions)
```

---

### 4. Multilingual Documents

**Symptoms:**
- Poor classification accuracy
- Gibberish in extracted text
- Encoding errors

**Supported Languages:**
- English (primary)
- Spanish, French, German (partial support)
- Chinese, Arabic, Russian (limited support)

**Limitations:**
- BERT model trained primarily on English
- Mixed-language documents may confuse classifier
- Right-to-left languages (Arabic, Hebrew) require special handling

**Mitigation:**
```python
# Error code: DOC_004
detected_language = detect_language(text)

if detected_language not in SUPPORTED_LANGUAGES:
    logger.warning(f"Unsupported language: {detected_language}")
    # Option 1: Translate to English
    # Option 2: Use multilingual model
    # Option 3: Return warning to user
```

---

## Image Processing Failures

### 1. Corrupted Image Files

**Symptoms:**
- PIL raises `UnidentifiedImageError`
- Partial image loading
- Incorrect dimensions

**Root Causes:**
- Truncated files
- Invalid file headers
- Unsupported formats

**Mitigation:**
```python
# Error code: IMG_001
try:
    image = Image.open(file_path)
    image.verify()  # Verify it's a valid image
    image = Image.open(file_path)  # Reload after verify
except Exception as e:
    raise ImageProcessingError("IMG_001", f"Corrupt image: {str(e)}")
```

---

### 2. Extreme Image Sizes

**Symptoms:**
- Out of memory during preprocessing
- Slow inference
- CUDA OOM errors

**Problematic Sizes:**
- Very large: > 4000x4000 pixels
- Very small: < 50x50 pixels
- Extreme aspect ratios: > 10:1

**Mitigation:**
```python
# Error code: IMG_002
width, height = image.size

if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
    # Downscale before processing
    scale_factor = MAX_IMAGE_SIZE / max(width, height)
    new_size = (int(width * scale_factor), int(height * scale_factor))
    image = image.resize(new_size, Image.LANCZOS)

if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
    raise ImageProcessingError("IMG_002", "Image too small for reliable classification")
```

---

### 3. Low-Quality Images

**Symptoms:**
- Low confidence scores
- Inconsistent predictions
- Blurry or pixelated inputs

**Quality Issues:**
- Low resolution (< 100x100)
- High compression artifacts
- Excessive noise
- Poor lighting

**Indicators:**
```python
# Error code: IMG_003
def assess_image_quality(image):
    # Check resolution
    if image.size[0] * image.size[1] < 10000:
        return "LOW_RESOLUTION"

    # Check for blur (variance of Laplacian)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 100:
        return "BLURRY"

    return "OK"
```

**Mitigation:**
- Return quality warning to user
- Apply image enhancement (denoising, sharpening)
- Flag low-confidence predictions

---

### 4. Grayscale Images

**Symptoms:**
- Model expects 3-channel RGB
- Dimension mismatch errors

**Mitigation:**
```python
# Auto-convert grayscale to RGB
if image.mode != "RGB":
    image = image.convert("RGB")
```

---

## OCR Edge Cases

### 1. Handwritten Text

**Accuracy:** ~50-70% (vs 95%+ for printed text)

**Challenges:**
- Varied handwriting styles
- Cursive text
- Overlapping characters

**Mitigation:**
- Use specialized handwriting recognition model
- Return confidence scores to user
- Flag handwritten sections

---

### 2. Rotated or Skewed Documents

**Symptoms:**
- Low OCR accuracy
- Garbled text
- Missing words

**Mitigation:**
```python
# Error code: OCR_001
angle = ocr_processor.detect_orientation(image)

if angle != 0:
    image = ocr_processor.rotate_image(image, -angle)
    logger.info(f"Auto-rotated image by {-angle} degrees")
```

---

### 3. Multi-Column Layouts

**Symptoms:**
- Text order scrambled
- Cross-column text concatenation

**Problematic Layouts:**
- Newspapers
- Academic papers
- Brochures

**Mitigation:**
- Use layout analysis before OCR
- Process columns separately
- Return structured output

---

### 4. Low Contrast Text

**Symptoms:**
- Characters not detected
- Word boundaries unclear

**Examples:**
- Light gray text on white background
- Faded documents
- Watermarks

**Mitigation:**
```python
# Error code: OCR_002
def enhance_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return binary
```

---

### 5. Multilingual OCR

**Supported Languages:**
- English (primary)
- French, German, Spanish (good support)
- Chinese, Japanese, Korean (requires specific models)
- Arabic, Hebrew (right-to-left text)

**Limitations:**
```python
# Error code: OCR_003
ocr_processor = OCRProcessor(languages=["eng", "fra"])  # English + French

# For unsupported languages:
if detected_language not in SUPPORTED_OCR_LANGUAGES:
    raise OCRError("OCR_003", f"Language not supported: {detected_language}")
```

---

## Multimodal Fusion Failures

### 1. Mismatched Text-Image Pairs

**Symptoms:**
- Low fusion accuracy
- Contradictory signals
- Confused predictions

**Examples:**
- Image of cat with text "This is a dog"
- Unrelated image-caption pairs

**Detection:**
```python
# Error code: MULTI_001
text_conf = text_prediction["confidence"]
image_conf = image_prediction["confidence"]
fusion_conf = fusion_prediction["confidence"]

if fusion_conf < min(text_conf, image_conf) * 0.5:
    logger.warning("Possible text-image mismatch detected")
```

---

### 2. Missing Modality

**Symptoms:**
- One modality unavailable
- Dimension mismatch

**Mitigation:**
```python
# Error code: MULTI_002
if text is None or len(text.strip()) == 0:
    # Fall back to image-only classification
    return image_classifier.predict(image)

if image is None:
    # Fall back to text-only classification
    return document_classifier.predict(text)
```

---

## System-Level Failures

### 1. Out of Memory (OOM)

**Triggers:**
- Large batch sizes
- High-resolution images
- Concurrent requests
- GPU memory exhaustion

**Mitigation:**
```python
# Error code: SYS_001
try:
    result = model.predict(batch)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Reduce batch size and retry
        torch.cuda.empty_cache()
        result = model.predict(batch, batch_size=batch_size // 2)
```

---

### 2. Model Loading Failures

**Triggers:**
- Missing model files
- Corrupted checkpoints
- Version incompatibility

**Mitigation:**
```python
# Error code: SYS_002
try:
    model = DocumentClassifier.from_pretrained(model_path)
except Exception as e:
    # Fall back to default pretrained model
    logger.error(f"Failed to load custom model: {e}")
    model = DocumentClassifier.from_pretrained("bert-base-uncased")
```

---

### 3. Inference Timeout

**Triggers:**
- Very large inputs
- Resource contention
- Network latency (for remote inference)

**Mitigation:**
```python
# Error code: SYS_003
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Inference timeout exceeded")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30-second timeout

try:
    result = model.predict(input_data)
    signal.alarm(0)  # Cancel alarm
except TimeoutError:
    raise InferenceError("SYS_003", "Inference timeout - input too complex")
```

---

## Mitigation Strategies

### General Principles

1. **Validate Inputs Early**
   - Check file types, sizes, formats
   - Detect potential issues before processing
   - Return clear error codes

2. **Graceful Degradation**
   - Fall back to simpler methods
   - Return partial results when possible
   - Maintain service availability

3. **User Feedback**
   - Provide actionable error messages
   - Include error codes for debugging
   - Suggest remediation steps

4. **Monitoring and Alerts**
   - Log all failures with context
   - Track failure rates by type
   - Alert on anomalous error patterns

5. **Retry Logic**
   - Implement exponential backoff
   - Retry with degraded parameters
   - Limit retry attempts

### Error Code Format

```
[PREFIX]_[NUMBER]

Prefixes:
- DOC: Document processing
- IMG: Image processing
- OCR: OCR-related
- MULTI: Multimodal fusion
- SYS: System-level

Example: DOC_001, IMG_002, OCR_003
```

### Logging Best Practices

```python
logger.error(
    f"[{error_code}] {error_message}",
    extra={
        "error_code": error_code,
        "file_path": file_path,
        "file_size": file_size,
        "user_id": user_id,
        "correlation_id": correlation_id,
    }
)
```

---

## Testing Edge Cases

All edge cases should have corresponding unit or integration tests:

```python
# tests/unit/test_edge_cases.py
def test_corrupt_pdf():
    with pytest.raises(DocumentProcessingError) as exc_info:
        load_pdf("corrupt.pdf")
    assert exc_info.value.code == "DOC_001"

def test_empty_document():
    with pytest.raises(DocumentProcessingError) as exc_info:
        process_document("")
    assert exc_info.value.code == "DOC_002"

def test_oversized_image():
    large_image = Image.new("RGB", (10000, 10000))
    with pytest.raises(ImageProcessingError) as exc_info:
        process_image(large_image)
    assert exc_info.value.code == "IMG_002"
```

---

## User-Facing Documentation

For user-facing error documentation, see:
- `docs/API_ERRORS.md` - API error codes and meanings
- `docs/TROUBLESHOOTING.md` - Common issues and solutions

For operational runbooks, see:
- `docs/RUNBOOKS.md` - Incident response procedures
