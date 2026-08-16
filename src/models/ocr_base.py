"""
src/models/ocr_base.py
=======================
Abstract base interface for all OCR models in the pipeline.

Any OCR model — TrOCR, CNN character reconstruction, or a future ensemble —
must implement this interface. The pipeline only calls recognize_line() and
never imports a concrete model class directly; it uses model_registry.py
to get the active model.

Adding a new OCR backend later:
    1. Create a new class inheriting from OCRModel
    2. Implement recognize_line()
    3. Register it in model_registry.py
    4. Update config.yaml active_ocr_model
    → No other file needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

# PIL is optional at import time (lazy import in subclasses)
try:
    from PIL import Image as PILImage
    _PIL_IMAGE_TYPE = PILImage.Image
except ImportError:
    _PIL_IMAGE_TYPE = None

# Type alias used in signatures
ImageInput = Union[str, Path, "_PIL_IMAGE_TYPE"]


class OCRModel(ABC):
    """Abstract interface all OCR models must implement.

    The pipeline interacts exclusively with this interface, never with
    concrete model classes, enabling clean model substitution via config.
    """

    @abstractmethod
    def recognize_line(self, image: ImageInput) -> tuple[str, float]:
        """Transcribe a single prescription line image.

        Parameters
        ----------
        image : Path-like (str or Path) to an image file,
                or a PIL Image object.

        Returns
        -------
        (text, confidence) : tuple
            text       — predicted transcription string (may be empty)
            confidence — float in [0.0, 1.0]; higher means more confident.
                         Use 0.0 when no confidence estimate is available.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier string for this model, e.g. 'trocr' or 'cnn'.
        Used in pipeline result dicts and dashboard display.
        """
        ...

    @property
    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready for inference.
        Subclasses may override; default returns True.
        """
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"
