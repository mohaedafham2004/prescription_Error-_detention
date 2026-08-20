"""
src/models/ner_base.py
=======================
Abstract Base Class for Clinical Named Entity Recognition (NER) models.
"""

from __future__ import annotations

try:
    from member2_nlp.ner_model.ner_base import NERModel
except ImportError:
    from abc import ABC, abstractmethod
    from typing import Dict, List, Any

    class NERModel(ABC):
        """Abstract base class for prescription Named Entity Recognition models."""

        @abstractmethod
        def extract_entities(self, text: str) -> Dict[str, List[str]]:
            """Extract clinical entities (MEDICINE, DOSAGE, FREQUENCY, DURATION) from text."""
            pass

        @property
        @abstractmethod
        def model_name(self) -> str:
            """Human-readable name of the NER model."""
            pass

        @property
        def is_ready(self) -> bool:
            """Return True if the model weights and pipelines are loaded."""
            return True
