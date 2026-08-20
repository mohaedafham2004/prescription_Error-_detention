"""
src/models/hf_ner_infer.py
===========================
HuggingFace Clinical NER model wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ner_base import NERModel

try:
    from member2_nlp.ner_model.hf_ner_infer import HFClinicalNERModel
except ImportError:
    class HFClinicalNERModel(NERModel):
        """Fallback HuggingFace Clinical NER model."""

        def __init__(self, model_name: str = "Posos/ClinicalNER"):
            self._model_name = model_name

        @property
        def model_name(self) -> str:
            return f"hf ({self._model_name})"

        @property
        def is_ready(self) -> bool:
            return True

        def extract_entities(self, text: str) -> Dict[str, List[str]]:
            return {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
