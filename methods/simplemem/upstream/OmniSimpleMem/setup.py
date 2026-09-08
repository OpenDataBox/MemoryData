"""
Setup script for OmniMem.
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="omnimem",
    version="0.1.0",
    author="OmniMem Team",
    description="Unified Multimodal Memory for Lifelong AI Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "tiktoken>=0.5.0",
        "rank_bm25>=0.2.2",
        "sentence-transformers>=2.2.0",
        "nltk>=3.8.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "server": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "python-multipart>=0.0.5",
        ],
        "audio": [
            "soundfile>=0.12.0",
            "librosa>=0.10.0",
        ],
        "visual": [
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "open_clip_torch>=2.20.0",
            "Pillow>=9.0.0",
        ],
        "vector": [
            "faiss-cpu>=1.7.0",
        ],
        "all": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "python-multipart>=0.0.5",
            "soundfile>=0.12.0",
            "librosa>=0.10.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "faiss-cpu>=1.7.0",
            "Pillow>=9.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omnimem-server=omni_memory.app:main",
        ],
    },
)
