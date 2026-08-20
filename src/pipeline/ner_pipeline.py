"""
src/pipeline/ner_pipeline.py
=============================
NER stage pipeline runner with graceful fallback when member2_nlp is on a separate branch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.model_registry import get_ner_model

try:
    from member2_nlp.ner_pipeline.ner_pipeline import run_ner_pipeline, NERPipelineResult
except ImportError:
    def run_ner_pipeline(text: str, cfg: dict) -> dict:
        """Run NER entity extraction on OCR transcribed text."""
        empty_entities = {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
        if not text:
            return {
                "entities": empty_entities,
                "raw_spans": [],
                "missing": ["MEDICINE", "DOSAGE", "FREQUENCY", "DURATION"],
                "ner_available": False,
            }

        try:
            ner_model = get_ner_model(cfg)
            if ner_model and ner_model.is_ready:
                entities = ner_model.extract_entities(text)
                missing = [k for k, v in entities.items() if not v]
                return {
                    "entities": entities,
                    "raw_spans": [],
                    "missing": missing,
                    "ner_available": True,
                    "ner_model_used": ner_model.model_name,
                }
        except Exception as e:
            pass

        return {
            "entities": empty_entities,
            "raw_spans": [],
            "missing": ["MEDICINE", "DOSAGE", "FREQUENCY", "DURATION"],
            "ner_available": False,
        }
