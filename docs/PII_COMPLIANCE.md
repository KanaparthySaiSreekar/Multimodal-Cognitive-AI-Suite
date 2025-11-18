# PII Compliance and Data Privacy

This document outlines PII (Personally Identifiable Information) handling, data privacy measures, and compliance requirements for the Multimodal Intelligence Workflow Suite.

---

## Table of Contents

- [Overview](#overview)
- [PII Data Categories](#pii-data-categories)
- [Data Processing Principles](#data-processing-principles)
- [PII Detection and Handling](#pii-detection-and-handling)
- [Data Retention](#data-retention)
- [Compliance Frameworks](#compliance-frameworks)
- [Security Measures](#security-measures)
- [User Rights](#user-rights)
- [Incident Response](#incident-response)

---

## Overview

The Multimodal AI Suite processes documents and images that may contain PII. This document establishes policies and procedures to ensure compliant handling of sensitive data.

### Scope

This policy applies to all data processed by the system, including:
- Uploaded documents (PDF, DOCX, TXT)
- Images (JPG, PNG, BMP)
- OCR-extracted text
- Model predictions and metadata
- Audit logs and system logs

---

## PII Data Categories

### Direct Identifiers

The system may encounter documents containing:

| Category | Examples | Risk Level |
|----------|----------|------------|
| Names | Full names, signatures | HIGH |
| Identification Numbers | SSN, passport numbers, driver's license | CRITICAL |
| Financial Information | Credit card numbers, bank accounts | CRITICAL |
| Contact Information | Email, phone, address | HIGH |
| Biometric Data | Photos, fingerprints | CRITICAL |
| Medical Information | Health records, prescriptions | CRITICAL |

### Indirect Identifiers

| Category | Examples | Risk Level |
|----------|----------|------------|
| Demographic Information | Age, gender, ethnicity | MEDIUM |
| Employment Information | Job title, employer | LOW |
| Educational Records | Degrees, transcripts | MEDIUM |
| Geographic Data | Zip codes, coordinates | LOW |

---

## Data Processing Principles

### 1. Data Minimization

**Policy:** Only process data necessary for classification/recognition tasks.

**Implementation:**
```python
# Example: Redact PII before storing
def sanitize_document(text: str) -> str:
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # Remove SSN patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

    return text
```

### 2. Purpose Limitation

**Policy:** Data used only for stated purposes (classification, recognition).

**Prohibited Uses:**
- Marketing or advertising
- Profiling for discrimination
- Sale to third parties
- Unauthorized research

### 3. Storage Limitation

**Policy:** Temporary storage only, automatic purging.

**Implementation:**
```python
# Automatic file cleanup after inference
import atexit
import tempfile

class TemporaryFileManager:
    def __init__(self):
        self.temp_files = []
        atexit.register(self.cleanup)

    def create_temp_file(self, suffix=None):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self.temp_files.append(temp_file.name)
        return temp_file

    def cleanup(self):
        for file_path in self.temp_files:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
```

---

## PII Detection and Handling

### Automated PII Detection

**Tools:**
- Regular expressions for common PII patterns
- Named Entity Recognition (NER) models
- Luhn algorithm for credit card validation

**Example Implementation:**
```python
import re
from typing import List, Dict

class PIIDetector:
    """Detect common PII patterns in text."""

    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }

    def detect(self, text: str) -> List[Dict]:
        detections = []

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detections.append({
                    'type': pii_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                })

        return detections

    def redact(self, text: str) -> str:
        for pii_type, pattern in self.PATTERNS.items():
            text = re.sub(pattern, f'[{pii_type.upper()}]', text)
        return text
```

### PII Handling Workflow

1. **Detection:** Scan inputs for PII during preprocessing
2. **Flagging:** Mark documents containing PII
3. **Redaction:** Remove or mask PII before logging
4. **Notification:** Alert users if PII detected
5. **Cleanup:** Purge temporary files immediately after processing

---

## Data Retention

### Retention Periods

| Data Type | Retention Period | Justification |
|-----------|-----------------|---------------|
| Uploaded Files | 0 seconds* | Processed in memory, deleted immediately |
| Temporary Files | 5 minutes | Automatic cleanup |
| Prediction Results | User session only | Not stored server-side |
| System Logs | 7 days | Operational monitoring |
| Audit Logs | 90 days | Security and compliance |
| Error Logs | 30 days | Debugging and improvement |

*Files never written to persistent storage unless explicitly required

### Automatic Purging

```python
# Example: Cleanup old logs
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_old_logs(log_dir: str, retention_days: int):
    log_path = Path(log_dir)
    cutoff_date = datetime.now() - timedelta(days=retention_days)

    for log_file in log_path.glob("*.log"):
        file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

        if file_mtime < cutoff_date:
            log_file.unlink()
            print(f"Deleted old log: {log_file}")

# Run daily
cleanup_old_logs("./logs", retention_days=7)
cleanup_old_logs("./audit_logs", retention_days=90)
```

---

## Compliance Frameworks

### GDPR (General Data Protection Regulation)

**Applicability:** EU residents' data

**Compliance Measures:**
- ✅ Data minimization
- ✅ Purpose limitation
- ✅ Storage limitation
- ✅ Right to erasure (automatic cleanup)
- ✅ Data portability (user owns predictions)
- ✅ Pseudonymization (correlation IDs instead of personal identifiers)

### CCPA (California Consumer Privacy Act)

**Applicability:** California residents

**Compliance Measures:**
- ✅ Transparency (this document)
- ✅ Right to know what data is collected
- ✅ Right to delete (automatic purging)
- ✅ No selling of personal data

### HIPAA (Health Insurance Portability and Accountability Act)

**Applicability:** Health-related documents

**Compliance Measures:**
- ✅ Encryption in transit (TLS)
- ✅ Encryption at rest (if storage is enabled)
- ✅ Access controls (token-based auth)
- ✅ Audit logging
- ⚠️ **Note:** System should not be used for PHI without additional BAA

### SOC 2 Type II

**Controls:**
- Security: Encryption, access controls, audit logs
- Availability: High availability deployment
- Processing Integrity: Data validation, error handling
- Confidentiality: No data sharing, secure deletion
- Privacy: This compliance document

---

## Security Measures

### Encryption

**In Transit:**
- TLS 1.3 for all communications
- Certificate pinning for critical endpoints

**At Rest:**
- AES-256 encryption for any persisted data
- Encrypted EBS volumes on AWS

### Access Controls

**Authentication:**
- Token-based authentication (JWT)
- Automatic token expiry (1 hour default)

**Authorization:**
- Role-based access control (RBAC)
- Principle of least privilege

**Audit Logging:**
```python
# Every data access is logged
audit_logger.log_data_access(
    user_id=user_id,
    resource_type="document",
    resource_id=document_id,
    action="classify",
    correlation_id=correlation_id,
)
```

### Secure Deletion

```python
import os

def secure_delete(file_path: str, passes: int = 3):
    """Securely delete file by overwriting before deletion."""
    if not os.path.exists(file_path):
        return

    file_size = os.path.getsize(file_path)

    with open(file_path, "ba+", buffering=0) as f:
        for _ in range(passes):
            f.seek(0)
            f.write(os.urandom(file_size))

    os.remove(file_path)
```

---

## User Rights

### Right to Access

Users can request:
- What data was processed
- How data was used
- Audit logs for their requests

### Right to Rectification

Users can:
- Correct inaccurate data
- Update incomplete data

### Right to Erasure

**Implementation:**
- Automatic deletion after processing
- Manual deletion on request
- Permanent removal from backups within 30 days

### Right to Data Portability

Users can:
- Export their prediction results
- Receive data in machine-readable format (JSON)

### Right to Object

Users can:
- Opt-out of processing
- Request human review of automated decisions

---

## Incident Response

### PII Breach Protocol

1. **Detection:** Automated alerts for unauthorized access
2. **Containment:** Immediate system isolation
3. **Assessment:** Determine scope and impact
4. **Notification:** Notify affected users within 72 hours
5. **Remediation:** Fix vulnerability, update policies
6. **Documentation:** Document incident and response

### Breach Notification Template

```
Subject: Data Security Incident Notification

Dear [User],

We are writing to inform you of a security incident that may have affected your data.

Incident Details:
- Date: [Date]
- Type: [Type of breach]
- Data Affected: [Categories of data]

Actions Taken:
- [Remediation steps]

Your Next Steps:
- [Recommended actions]

Contact Information:
- security@example.com
- 1-800-XXX-XXXX

We sincerely apologize for this incident.
```

---

## Data Processing Agreement (DPA)

For enterprise customers processing sensitive data, a DPA is required:

### Key Provisions

1. **Processing Instructions:** Data processed only per customer instructions
2. **Confidentiality:** All personnel bound by confidentiality
3. **Security:** Appropriate technical and organizational measures
4. **Sub-processors:** List of sub-processors (AWS, etc.)
5. **Data Transfers:** Mechanisms for international transfers
6. **Audits:** Right to audit security measures
7. **Termination:** Data deletion upon contract termination

---

## Compliance Checklist

### Pre-Deployment

- [ ] PII detection enabled
- [ ] Automatic cleanup configured
- [ ] Encryption enabled (transit and rest)
- [ ] Access controls configured
- [ ] Audit logging enabled
- [ ] Retention policies configured
- [ ] Privacy policy published
- [ ] DPA template prepared

### Operational

- [ ] Regular security audits
- [ ] Log review (weekly)
- [ ] Access review (monthly)
- [ ] Penetration testing (quarterly)
- [ ] Compliance training (annually)
- [ ] Policy review (annually)

### Incident Response

- [ ] Incident response plan documented
- [ ] Response team identified
- [ ] Communication templates prepared
- [ ] Legal contacts established

---

## Contact Information

**Data Protection Officer:**
- Email: dpo@example.com
- Phone: 1-800-XXX-XXXX

**Security Team:**
- Email: security@example.com
- Emergency: security-emergency@example.com

**Legal:**
- Email: legal@example.com

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-02-01 | Initial version | AI Development Team |

---

**Last Updated:** November 2025
**Next Review:** February 2026
