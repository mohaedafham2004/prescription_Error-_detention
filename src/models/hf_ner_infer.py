"""
src/models/hf_ner_infer.py
===========================
Hugging Face ClinicalNER model wrapper implementing the NERModel interface.

Uses Posos/ClinicalNER, a BERT-based token classification model fine-tuned on
French (and some English) clinical text. Because it was trained on clinical
records it picks up pharmaceutical entities robustly from prescription text.

Label mapping (model → our schema)
-----------------------------------
The Posos/ClinicalNER model outputs these entity types:
    DRUG        → MEDICINE
    STRENGTH    → DOSAGE     ⚠ See note below
    DOSAGE      → DOSAGE     ⚠ See note below
    FREQUENCY   → FREQUENCY
    DURATION    → DURATION
    FORM        → (dropped — not in our schema; printed as a warning if seen)

⚠ STRENGTH vs DOSAGE NOTE:
    The model distinguishes between STRENGTH (concentration: e.g. "500mg")
    and DOSAGE (quantity to take: e.g. "2 tablets"). Our schema merges both
    into DOSAGE. This is a pragmatic simplification — if you need to separate
    them for clinical safety purposes, extend NERModel to include a
    "STRENGTH" key and update the error-detection rules accordingly.
    For now both are combined under DOSAGE with a log note.

Usage
-----
    from src.models.hf_ner_infer import HFClinicalNERModel

    ner = HFClinicalNERModel()
    entities = ner.extract_entities("Amoxicillin 500mg twice daily for 7 days")
    # {"MEDICINE": ["Amoxicillin"], "DOSAGE": ["500mg"], ...}

Configuration
-------------
    # config.yaml
    active_ner_model: "hf_clinical"

    # Optional — override the HF model identifier:
    hf_ner_model_name: "Posos/ClinicalNER"
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ner_base import NERModel, ENTITY_KEYS, empty_entities

log = logging.getLogger(__name__)

# ── Default HF model identifier ───────────────────────────────────────────────
DEFAULT_HF_MODEL = "Posos/ClinicalNER"

# ── Label mapping: HF model label → our canonical key ─────────────────────────
# Any label not in this map is silently logged and skipped.
_LABEL_MAP: Dict[str, str] = {
    "DRUG":      "MEDICINE",
    "STRENGTH":  "DOSAGE",     # ⚠ merged with DOSAGE — see module docstring
    "DOSAGE":    "DOSAGE",
    "FREQUENCY": "FREQUENCY",
    "DURATION":  "DURATION",
    # "FORM": (dropped — packaging form e.g. "tablet", "capsule" is not in our schema)
}


class HFClinicalNERModel(NERModel):
    """Hugging Face Posos/ClinicalNER wrapper implementing the NERModel interface.

    Lazy-loads the tokenizer and model from Hugging Face Hub on first use,
    using the transformers `pipeline("ner", aggregation_strategy="simple")`
    for word-level entity grouping.

    Parameters
    ----------
    model_name : HF Hub repo id (default: "Posos/ClinicalNER").
    device     : -1 for CPU (default), 0 for first CUDA GPU.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_HF_MODEL,
        device: int = -1,
    ):
        self._model_name = model_name
        self._device = device
        self._pipeline = None
        self._loaded = False

    # ── NERModel interface ─────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return "hf_clinical"

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract prescription entities using Posos/ClinicalNER.

        Returns
        -------
        dict with keys MEDICINE, DOSAGE, FREQUENCY, DURATION.
        All keys always present; values are deduplicated text lists.
        """
        self._ensure_loaded()

        if not text or not text.strip():
            return empty_entities()

        try:
            raw = self._pipeline(text)
        except Exception as e:
            log.warning("[HFClinicalNER] Inference failed: %s", e)
            return empty_entities()

        return self._map_results(raw)

    # ── Lazy Loading ───────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForTokenClassification,
                pipeline as hf_pipeline,
            )
        except ImportError:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers"
            )

        print(f"  [HFClinicalNER] Loading model from Hugging Face Hub: {self._model_name} ...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForTokenClassification.from_pretrained(self._model_name)

            self._pipeline = hf_pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",   # groups subword tokens into words
                device=self._device,
            )
            self._loaded = True
            print(f"  [HFClinicalNER] Model loaded successfully.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load HF Clinical NER model '{self._model_name}': {e}\n"
                "Check your internet connection or set HF_TOKEN for private repos."
            ) from e

    # ── Label Mapping ──────────────────────────────────────────────────────────

    def _map_results(self, raw_entities: list) -> Dict[str, List[str]]:
        """Map HF pipeline output to our canonical entity dict.

        The pipeline returns a list of dicts like:
            {"entity_group": "DRUG", "word": "Amoxicillin", "score": 0.99, ...}

        ⚠ STRENGTH and DOSAGE are both merged into DOSAGE.
        FORM is logged but dropped (not in our schema).
        """
        result: Dict[str, List[str]] = empty_entities()
        strength_seen: List[str] = []   # collect separately for the merge note

        for ent in raw_entities:
            raw_label = ent.get("entity_group", ent.get("entity", "")).upper()
            word = ent.get("word", "").strip()
            if not word:
                continue

            if raw_label == "STRENGTH":
                strength_seen.append(word)
                canon_key = "DOSAGE"
            elif raw_label in _LABEL_MAP:
                canon_key = _LABEL_MAP[raw_label]
            elif raw_label == "FORM":
                log.debug("[HFClinicalNER] FORM entity dropped (not in schema): %r", word)
                continue
            else:
                log.debug("[HFClinicalNER] Unknown label %r for word %r — skipped", raw_label, word)
                continue

            if word not in result[canon_key]:
                result[canon_key].append(word)

        if strength_seen:
            # ⚠ STRENGTH entities merged into DOSAGE — flag for manual review
            log.debug(
                "[HFClinicalNER] STRENGTH entities merged into DOSAGE: %s. "
                "If you need to distinguish drug concentration from administered "
                "quantity, extend NERModel schema to include a STRENGTH key.",
                strength_seen,
            )

        return result

    # ── Convenience: full result dict (mirrors SpacyNERModel.extract()) ─────────

    def extract(self, text: str) -> Dict:
        """Full result dict matching SpacyNERModel.extract() shape for symmetry."""
        structured = self.extract_entities(text)
        missing = [k for k in ENTITY_KEYS if not structured[k]]
        return {
            "text":       text,
            "entities":   [],   # raw span list not available from HF pipeline here
            "structured": structured,
            "missing":    missing,
        }
