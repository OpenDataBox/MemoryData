"""
Multimodal Processors for Omni-Memory.

Processors handle ingestion of different modalities with entropy-triggered
filtering for efficient storage.
"""

from omni_memory.processors.text_processor import TextProcessor
from omni_memory.processors.image_processor import ImageProcessor
from omni_memory.processors.audio_processor import AudioProcessor
from omni_memory.processors.video_processor import VideoProcessor
from omni_memory.processors.base import BaseProcessor, ProcessingResult

__all__ = [
    "TextProcessor",
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "BaseProcessor",
    "ProcessingResult",
]
