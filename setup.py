"""Setup configuration for Multimodal Intelligence Workflow Suite."""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
README = Path(__file__).parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

# Read requirements
REQUIREMENTS = Path(__file__).parent / "requirements.txt"
requirements = []
if REQUIREMENTS.exists():
    with open(REQUIREMENTS, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-ai-suite",
    version="1.0.0",
    author="AI Development Team",
    author_email="dev@multimodal-ai.com",
    description="Multimodal AI system for document classification and image recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KanaparthySaiSreekar/Multimodal-Cognitive-AI-Suite",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.3",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-train=training.train_document:main",
            "multimodal-serve=api.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "multimodal",
        "ai",
        "machine-learning",
        "deep-learning",
        "computer-vision",
        "nlp",
        "document-classification",
        "image-recognition",
        "transformers",
        "bert",
        "vision-transformer",
    ],
)
