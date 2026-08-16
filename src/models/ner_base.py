"""
src/models/ner_base.py
=======================
Abstract base interface for all NER models in the pipeline.

Any NER model — custom spaCy, Hugging Face Clinical NER, or a future
ensemble — must implement this interface. The pipeline uses model_registry.py
to get the active model; it never imports a concrete class directly.

Adding a new NER backend later:
    1. Create a new class inheriting from NERModel
    2. Implement extract_entities() and model_name
    3. Register it in model_registry.py (get_ner_model factory)
    4. Update config.yaml active_ner_model
    → No other file needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


# ── Canonical entity key schema ───────────────────────────────────────────────
# All NERModel implementations MUST return a dict with exactly these keys.
# Values are lists of extracted text strings (may be empty lists).
#
#   "MEDICINE"  — drug / medicine name
#   "DOSAGE"    — amount + unit  (e.g. "500mg", "2 tablets")
#   "FREQUENCY" — how often      (e.g. "twice daily", "TID")
#   "DURATION"  — how long       (e.g. "7 days", "2 weeks")

ENTITY_KEYS: List[str] = ["MEDICINE", "DOSAGE", "FREQUENCY", "DURATION"]


def empty_entities() -> Dict[str, List[str]]:
    """Return the canonical empty entity dict."""
    return {k: [] for k in ENTITY_KEYS}


class NERModel(ABC):
    """Abstract interface all NER models must implement.

    The pipeline interacts exclusively with this interface, never with
    concrete model classes, enabling clean model substitution via config.
    """

    @abstractmethod
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract prescription entities from a text string.

        Parameters
        ----------
        text : Full prescription text (may be multi-line).

        Returns
        -------
        dict with exactly the keys defined in ENTITY_KEYS:
            {
                "MEDICINE":  ["Amoxicillin"],
                "DOSAGE":    ["500mg"],
                "FREQUENCY": ["twice daily"],
                "DURATION":  ["7 days"],
            }
        All keys must always be present (use empty list if not found).
        Values are deduplicated lists of plain text strings.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier string for this model, e.g. 'spacy' or 'hf_clinical'.
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
