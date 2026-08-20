"""
src/models/ner_infer.py
========================
spaCy NER model wrapper for prescription clinical entity extraction.
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
    from member2_nlp.ner_model.ner_infer import SpacyNERModel
except ImportError:
    class SpacyNERModel(NERModel):
        """Fallback spaCy NER model when member2_nlp is uninstalled/separate branch."""

        def __init__(self, model_dir: str = "models/ner_model"):
            self.model_dir = model_dir
            self.nlp = None
            self._is_ready = False
            self._load_spacy()

        def _load_spacy(self):
            try:
                import spacy
                path = Path(self.model_dir)
                if path.exists():
                    self.nlp = spacy.load(str(path))
                    self._is_ready = True
                else:
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                        self._is_ready = True
                    except Exception:
                        self.nlp = None
            except Exception:
                self.nlp = None

        @property
        def model_name(self) -> str:
            return "spacy (custom medical NER)"

        @property
        def is_ready(self) -> bool:
            return self._is_ready

        def extract_entities(self, text: str) -> Dict[str, List[str]]:
            entities = {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
            if not text or not self.nlp:
                return entities

            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    lbl = ent.label_.upper()
                    if lbl in entities and ent.text not in entities[lbl]:
                        entities[lbl].append(ent.text)
            except Exception:
                pass

            return entities
