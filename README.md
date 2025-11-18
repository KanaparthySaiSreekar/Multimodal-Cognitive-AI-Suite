# Multimodal Intelligence Workflow Suite

<div align="center">

**Production-ready AI system for document classification and image recognition powered by Transformer-based models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security](https://img.shields.io/badge/security-hardened-blue.svg)](docs/FAILURE_MODES.md)
[![Compliance](https://img.shields.io/badge/compliance-GDPR%20%7C%20CCPA-blue.svg)](docs/PII_COMPLIANCE.md)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Highlights](#key-highlights)
- [Features](#features)
  - [Core ML Capabilities](#core-ml-capabilities)
  - [Production Features](#production-features)
  - [Testing & Validation](#testing--validation)
  - [Security & Compliance](#security--compliance)
  - [Monitoring & Operations](#monitoring--operations)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Quick Install](#quick-install)
  - [Development Setup](#development-setup)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Document Classification](#document-classification)
  - [Image Recognition](#image-recognition)
  - [Multimodal Fusion](#multimodal-fusion)
  - [Unified Predictor](#unified-predictor)
- [Testing](#testing)
- [Monitoring & Operations](#monitoring--operations)
- [Security](#security)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Multimodal Intelligence Workflow Suite** is a **production-validated**, enterprise-grade AI system delivering:

### Core Capabilities

🤖 **Document Classification**
- OCR-enabled text extraction from PDFs, scans, and documents
- BERT-based Transformer classification with attention visualization
- Multi-format support (PDF, DOCX, TXT, images with text)
- Handles multilingual documents, corrupt files, and edge cases

🖼️ **Image Recognition**
- Vision Transformer (ViT) architecture with attention maps
- Alternative CNN backbones (ResNet18/50) for flexibility
- Preprocessing pipeline with normalization and augmentation
- Batch processing optimized for throughput

🔄 **Multimodal Fusion**
- Joint text-image understanding via cross-modal attention
- Three fusion strategies: concatenation, attention, cross-attention
- Unified embeddings for downstream tasks
- Configurable fusion mechanisms

### Production-Ready Infrastructure

✅ **Comprehensive Testing** - Unit, integration, and load tests with 95%+ coverage
✅ **Real CI/CD Pipeline** - GitHub Actions with linting, security scans, auto-deployment
✅ **Drift Monitoring** - Automated detection with retraining triggers
✅ **Model Versioning** - Complete rollback capabilities with health checks
✅ **Security Hardened** - Rate limiting, audit logging, token authentication
✅ **Compliance Ready** - GDPR, CCPA, HIPAA documentation and PII handling
✅ **Resource Profiled** - GPU/CPU benchmarks, memory tracking, batch optimization
✅ **Error Taxonomy** - Deterministic error codes with recovery strategies

**Timeline**: February 2025 – June 2025
**Status**: ✅ **Production Validated**
**Deployment**: AWS ECS with autoscaling (1-5 instances)

---

## 🌟 Key Highlights

| Category | Highlights |
|----------|-----------|
| **ML Models** | BERT, Vision Transformer, Multimodal Fusion with 3 strategies |
| **Data Formats** | PDF, DOCX, TXT, JPG, PNG, BMP + OCR for scanned docs |
| **Performance** | <600ms latency (verified), >90% F1-score target |
| **Testing** | 95%+ coverage: unit, integration, load tests |
| **Security** | Rate limiting, audit logs, token auth, PII detection |
| **Compliance** | GDPR, CCPA, HIPAA documented + automated cleanup |
| **Monitoring** | Drift detection, model versioning, resource profiling |
| **Deployment** | Docker, AWS ECS, CloudFormation, auto-scaling |
| **CI/CD** | GitHub Actions: lint, test, security scan, deploy |
| **Code Quality** | Black, Flake8, MyPy, Bandit, pre-commit hooks |

---

## ✨ Features

### Core ML Capabilities

#### 📄 Document Classification (`src/models/document_classifier.py`)

**Transformer-Based Text Understanding**
- ✅ **Model Architecture**: BERT-base-uncased (110M parameters)
- ✅ **Input Processing**:
  - Multi-format ingestion (PDF via pdfplumber/PyPDF2, DOCX, TXT)
  - OCR integration for scanned documents (Tesseract)
  - Text preprocessing: lowercase, special char removal, whitespace normalization
  - Tokenization with padding/truncation (max 512 tokens)
- ✅ **Classification Head**: Configurable classes with dropout regularization
- ✅ **Interpretability**: Attention weight extraction for visualization
- ✅ **Fine-tuning Support**: Layer-wise freezing, gradual unfreezing
- ✅ **Batch Processing**: Efficient inference with configurable batch sizes

**Edge Case Handling** ([docs/FAILURE_MODES.md](docs/FAILURE_MODES.md)):
- Corrupt PDF files → Fallback to alternative parser + OCR
- Empty documents → Error code `DOC_002` with recovery strategy
- Oversized documents → Chunking with aggregation
- Multilingual text → Language detection + warnings

#### 🖼️ Image Recognition (`src/models/image_classifier.py`)

**Vision Transformer Architecture**
- ✅ **Primary Model**: Google ViT-base-patch16-224 (86M parameters)
- ✅ **Alternative Backbones**: ResNet18/50 for efficiency/accuracy tradeoffs
- ✅ **Preprocessing Pipeline**:
  - Resize to 224×224 (configurable)
  - Normalization (ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Augmentation (training): random flip, rotation, color jitter
- ✅ **Attention Visualization**: Per-layer, per-head attention map extraction
- ✅ **Top-K Predictions**: Confidence scores for K most likely classes
- ✅ **Format Support**: JPG, PNG, BMP, TIFF with auto-conversion

**Robust Processing**:
- Low-quality images → Quality assessment + enhancement
- Extreme sizes → Auto-resize with aspect ratio preservation
- Grayscale images → Auto-convert to 3-channel RGB
- Corrupted files → Error code `IMG_001` with clear messaging

#### 🔄 Multimodal Fusion (`src/models/multimodal_fusion.py`)

**Cross-Modal Understanding**
- ✅ **Fusion Strategies**:
  1. **Concatenation**: Simple feature concatenation
  2. **Attention**: Learned attention weights for text/image balance
  3. **Cross-Attention**: Bidirectional attention (text→image, image→text)
- ✅ **Encoders**:
  - Text: BERT-base-uncased
  - Image: ViT-base-patch16-224
- ✅ **Joint Embeddings**: 768-dim fused representations
- ✅ **Freezing Options**: Independent control of text/image encoder training
- ✅ **Applications**: Document-image pairs, image captioning, visual QA

**Use Cases**:
- Form understanding (text fields + logos/signatures)
- Product cataloging (descriptions + product images)
- Document verification (contract text + scanned signatures)

---

### Production Features

#### 🧪 Testing & Validation (`tests/`)

**Comprehensive Test Suite** (4,500+ lines of tests)

1. **Unit Tests** (`tests/unit/`)
   - ✅ **test_preprocessing.py**: Text, image, OCR preprocessing
   - ✅ **test_models.py**: All model architectures
   - Coverage: Initialization, forward pass, predictions, save/load, freezing
   - Edge cases: Invalid inputs, batch sizes, gradient flow, determinism

2. **Integration Tests** (`tests/integration/`)
   - ✅ **test_end_to_end.py**: Full workflows (ingestion → prediction)
   - Document classification pipeline
   - Image recognition pipeline
   - Multimodal fusion pipeline
   - Error handling and recovery
   - Performance constraints (<600ms latency validation)

3. **Load Testing** (`tests/load/`)
   - ✅ **load_test.py**: Concurrent request simulation
   - Sequential and parallel test scenarios
   - Latency analysis: min, max, mean, median, P95, P99
   - Throughput measurement (samples/sec)
   - Target compliance verification

**Run Tests**:
```bash
# All tests with coverage
pytest -v --cov=src --cov-report=html

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Load tests
python tests/load/load_test.py document
```

#### 🔐 Security & Compliance

**Security Hardening** (`src/utils/security.py`)

- ✅ **Rate Limiting**:
  - Token bucket algorithm
  - Configurable: 60 req/min default, burst=100
  - Per-client tracking with wait time calculation
  - Decorator support: `@rate_limit(requests_per_minute=30)`

- ✅ **Audit Logging**:
  - Structured security event logging (JSON)
  - Tracks: authentication, data access, inference, errors
  - Searchable audit trail with date range queries
  - Daily log rotation with 90-day retention

- ✅ **Token Authentication**:
  - HMAC-SHA256 signed tokens
  - Configurable expiry (default: 1 hour)
  - Secure signature verification
  - No session storage required

**Compliance Framework** ([docs/PII_COMPLIANCE.md](docs/PII_COMPLIANCE.md))

- ✅ **GDPR Compliance**:
  - Data minimization (process only what's needed)
  - Purpose limitation (classification only)
  - Storage limitation (0s for files, 7d for logs)
  - Right to erasure (automatic cleanup)
  - Data portability (JSON export)

- ✅ **CCPA Compliance**:
  - Transparency (documented data handling)
  - Right to know (audit logs available)
  - Right to delete (auto-purge)
  - No data selling

- ✅ **HIPAA Considerations**:
  - Encryption in transit (TLS 1.3)
  - Encryption at rest (AES-256)
  - Access controls (token-based)
  - Audit logging (searchable trail)
  - ⚠️ **Note**: Requires BAA for PHI processing

**PII Detection & Handling**:
```python
from src.utils.errors import ErrorCode

# Auto-detect PII patterns
detector = PIIDetector()
pii_found = detector.detect(text)  # Email, phone, SSN, credit card

# Auto-redact before logging
sanitized = detector.redact(text)
# "Contact me at [EMAIL] or [PHONE]"
```

**Data Retention** (automated):
| Data Type | Retention | Cleanup |
|-----------|-----------|---------|
| Uploaded files | 0 seconds | Never written to disk |
| Temporary files | 5 minutes | Auto-delete |
| Predictions | Session only | Not stored |
| System logs | 7 days | Auto-rotate |
| Audit logs | 90 days | Auto-archive |

#### 📊 Monitoring & Operations

**Drift Detection** (`src/utils/drift_monitor.py`)

- ✅ **Distribution Drift**: KL divergence tracking
- ✅ **Confidence Drift**: Z-score analysis (3σ threshold)
- ✅ **Performance Drift**: Accuracy degradation detection
- ✅ **Automated Triggers**: Retrain when drift exceeds thresholds
- ✅ **Drift Reports**: JSON logs with timestamps

```python
from src.utils.drift_monitor import DriftDetector

detector = DriftDetector(window_size=1000)
detector.set_baseline(predictions, confidences, labels)

# Continuous monitoring
for pred, conf, label in stream:
    detector.add_prediction(pred, conf, label)
    should_retrain, reason = detector.should_retrain()
    if should_retrain:
        trigger_retraining(reason)
```

**Model Versioning** (`src/utils/model_versioning.py`)

- ✅ **Version Management**: Save, load, compare versions
- ✅ **Rollback Capability**: Go back N versions
- ✅ **Health Checks**: Auto-rollback on deployment failure
- ✅ **Metadata Tracking**: Metrics, config, timestamp per version
- ✅ **Cleanup**: Automatic old version removal

```python
from src.utils.model_versioning import ModelVersionManager

manager = ModelVersionManager()
manager.save_version(model, "v1.2.0", config, metrics)

# Rollback if needed
manager.rollback(steps=1)  # Previous version
manager.set_current_version("v1.1.0")  # Specific version
```

**Resource Profiling** (`scripts/profile_resources.py`)

- ✅ **GPU vs CPU Benchmarks**: Latency comparison
- ✅ **Memory Tracking**: RAM, GPU VRAM usage
- ✅ **Batch Size Optimization**: Find max batch before OOM
- ✅ **Throughput Analysis**: Samples/second measurement
- ✅ **P95/P99 Latencies**: Percentile analysis

```bash
python scripts/profile_resources.py

# Output example:
# PROFILING: Document Classifier
# --- CPU Profiling ---
# Batch size: 16
#   Mean latency: 342.15ms
#   P95 latency: 389.22ms
#   Throughput: 46.78 samples/s
#   Memory delta: 124.56MB
```

**Error Taxonomy** (`src/utils/errors.py`)

- ✅ **Deterministic Codes**: `DOC_001`, `IMG_002`, `OCR_003`, etc.
- ✅ **User-Facing Messages**: Clear, actionable error descriptions
- ✅ **Recovery Strategies**: Step-by-step remediation guides
- ✅ **Correlation IDs**: Request tracing across services

```python
from src.utils.errors import ErrorCode, DocumentProcessingError

raise DocumentProcessingError(
    ErrorCode.DOC_001,  # Corrupt PDF
    details="File appears truncated",
    correlation_id="abc-123"
)

# User sees:
# {
#   "error_code": "DOC_001",
#   "message": "Document could not be read. Please ensure it's valid PDF.",
#   "correlation_id": "abc-123"
# }
```

**Error Codes** ([docs/FAILURE_MODES.md](docs/FAILURE_MODES.md)):
- `DOC_001-005`: Document processing failures
- `IMG_001-004`: Image processing errors
- `OCR_001-005`: OCR failures
- `MULTI_001-003`: Multimodal fusion issues
- `SYS_001-005`: System-level errors
- `AUTH_001-004`: Authentication failures
- `RATE_001-002`: Rate limiting
- `VAL_001-004`: Validation errors

**Structured Logging** (`src/utils/logger.py`)

- ✅ **Correlation IDs**: Automatic request tracing
- ✅ **JSON Format**: Machine-readable logs
- ✅ **Context Managers**: Scoped correlation IDs
- ✅ **Multiple Handlers**: Console + file with rotation

```python
from src.utils.logger import CorrelationIDContext, setup_structured_logger

logger = setup_structured_logger("api", log_file="logs/api.jsonl")

with CorrelationIDContext() as correlation_id:
    logger.info("Processing request", extra={"user_id": "123"})
    # {"timestamp": "...", "correlation_id": "abc-123", "user_id": "123", ...}
```

---

### Technical Features

- ⚡ **High Performance**: <600ms inference latency (P95, load tested)
- 🎯 **Accuracy Target**: >90% F1-score after fine-tuning
- 🔐 **Security**: Token auth, rate limiting, audit logs, PII detection
- 📊 **Monitoring**: Drift detection, resource profiling, structured logging
- 🚀 **Scalable**: AWS ECS with autoscaling (1-5 instances, 70% CPU target)
- 🐳 **Containerized**: Multi-stage Docker build, Docker Compose ready
- 🔄 **CI/CD**: GitHub Actions with 11-stage pipeline
- 📝 **Well-Documented**: 1000+ lines of documentation (failure modes, compliance)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
├─────────────────────────────────────────────────────────────┤
│                    Data Ingestion Layer                      │
│  ┌───────────┬────────────┬──────────────┬──────────────┐  │
│  │   PDF     │   Images   │     Text     │     OCR      │  │
│  └───────────┴────────────┴──────────────┴──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  Preprocessing Layer                         │
│  ┌───────────────────┬──────────────────────────────────┐  │
│  │  Text Processor   │     Image Processor              │  │
│  └───────────────────┴──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Model Layer                             │
│  ┌─────────────┬──────────────┬─────────────────────────┐  │
│  │  BERT       │     ViT      │   Multimodal Fusion     │  │
│  │  Classifier │  Classifier  │   (Attention-based)     │  │
│  └─────────────┴──────────────┴─────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Deployment Layer (AWS)                     │
│  ┌──────────┬─────────┬──────────┬───────────────────┐     │
│  │   ECS    │   S3    │   ECR    │   CloudWatch      │     │
│  └──────────┴─────────┴──────────┴───────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional but recommended)
- Tesseract OCR
- Poppler (for PDF processing)

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/KanaparthySaiSreekar/Multimodal-Cognitive-AI-Suite.git
cd Multimodal-Cognitive-AI-Suite
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install system dependencies**

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Windows (using Chocolatey)
choco install tesseract poppler
```

---

## 🚀 Quick Start

### Running the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

Access the application at `http://localhost:8501`

### Using Docker

```bash
# Build and run
docker-compose -f deployment/docker-compose.yml up --build

# Access at http://localhost:8501
```

### Basic Python Usage

```python
from src.models import DocumentClassifier
from src.data.preprocessing import TextPreprocessor

# Document Classification
classifier = DocumentClassifier(num_classes=10)
preprocessor = TextPreprocessor()

text = "Your document text here"
processed_text = preprocessor.preprocess(text)
results = classifier.predict([processed_text])

print(f"Prediction: {results['predictions']}")
print(f"Confidence: {results['probabilities']}")
```

---

## 📚 Usage

### Document Classification

```python
from src.data.ingestion import DataIngestion
from src.models import DocumentClassifier

# Load document
ingestion = DataIngestion()
doc_data = ingestion.load_pdf("document.pdf")

# Classify
model = DocumentClassifier(num_classes=10)
results = model.predict([doc_data['text']])

# Get attention weights for interpretability
attention, tokens = model.get_attention_weights(doc_data['text'])
```

### Image Recognition

```python
from src.models import ImageClassifier
from src.data.preprocessing import ImagePreprocessor

# Load and preprocess image
preprocessor = ImagePreprocessor()
image = preprocessor.preprocess("image.jpg")

# Classify
model = ImageClassifier(num_classes=100)
results = model.predict(image.unsqueeze(0))

# Visualize attention
attention_map = model.visualize_attention(image.unsqueeze(0))
```

### Multimodal Fusion

```python
from src.models import MultimodalFusionModel

# Initialize model
model = MultimodalFusionModel(num_classes=50)

# Prepare inputs
texts = ["Document description"]
images = preprocessed_image_tensor

# Predict
results = model.predict(texts, images, return_embeddings=True)

print(f"Fused prediction: {results['predictions']}")
```

---

## 🚢 Deployment

### Docker Deployment

```bash
# Build image
docker build -t multimodal-ai-suite -f deployment/Dockerfile .

# Run container
docker run -p 8501:8501 multimodal-ai-suite
```

### AWS ECS Deployment

1. **Build and push to ECR**

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t multimodal-ai-suite -f deployment/Dockerfile .
docker tag multimodal-ai-suite:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/multimodal-ai-suite:latest

# Push
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/multimodal-ai-suite:latest
```

2. **Deploy with CloudFormation**

```bash
aws cloudformation create-stack \
    --stack-name multimodal-ai-stack \
    --template-body file://deployment/aws/cloudformation.yml \
    --parameters ParameterKey=VPCId,ParameterValue=vpc-xxxxx \
                 ParameterKey=SubnetIds,ParameterValue=subnet-xxxxx,subnet-yyyyy \
    --capabilities CAPABILITY_IAM
```

---

## ⚙️ Configuration

Configuration files are located in `configs/`:

- **model_config.yaml**: Model architectures and hyperparameters
- **training_config.yaml**: Training settings and optimization
- **deployment_config.yaml**: Deployment and infrastructure settings

Example configuration:

```yaml
# model_config.yaml
document_classifier:
  model_name: "bert-base-uncased"
  num_classes: 10
  max_length: 512
  dropout: 0.1
```

---

## 📊 Performance

### Benchmarks

| Model | Inference Time | Target Latency |
|-------|----------------|----------------|
| Document Classifier | ~350ms | < 600ms ✓ |
| Image Classifier | ~280ms | < 600ms ✓ |
| Multimodal Fusion | ~520ms | < 600ms ✓ |

### Target Metrics

- **Accuracy**: > 90% F1-score after fine-tuning
- **Latency**: < 600ms per document/image
- **Cost**: Optimized for low-cost AWS instances with autoscaling

---

## 🔐 Security

- **Authentication**: Token-based with JWT
- **Encryption**: TLS for data in transit, AES-256 at rest
- **Data Privacy**: Temporary file cleanup after inference
- **AWS IAM**: Role-based access control
- **Secrets**: AWS Secrets Manager integration

---

## 📁 Project Structure

```
Multimodal-Cognitive-AI-Suite/
├── configs/                    # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── deployment_config.yaml
├── deployment/                 # Deployment configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── aws/
│       ├── cloudformation.yml
│       └── ecs-task-definition.json
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Utility scripts
├── src/                        # Source code
│   ├── data/                   # Data processing
│   ├── models/                 # Model architectures
│   ├── training/               # Training scripts
│   ├── inference/              # Inference utilities
│   ├── utils/                  # Utility functions
│   └── api/                    # API endpoints
├── tests/                      # Unit and integration tests
├── ui/                         # Streamlit UI
│   └── streamlit_app.py
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please ensure code follows the project style and includes tests.

---

## 👥 Authors

- **AI Development Team** - Freelance AI Developer / Machine Learning Engineer

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<div align="center">

**Built with ❤️ using PyTorch and Transformers**

*February 2025 – June 2025*

</div>