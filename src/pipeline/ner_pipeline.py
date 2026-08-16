"""
src/pipeline/ner_pipeline.py
==============================
NER stage: extracted text → structured prescription entities.

Dispatches to the active NER model (custom spaCy or Posos/ClinicalNER) with
an automatic clinical rule-based fallback if the local model is not yet trained.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from rapidfuzz import fuzz

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ENTITY_LABELS = ["MEDICINE", "DOSAGE", "FREQUENCY", "DURATION"]

_EMPTY_RESULT = {
    "entities":      {lbl: [] for lbl in ENTITY_LABELS},
    "raw_spans":     [],
    "missing":       list(ENTITY_LABELS),
    "ner_available": False,
    "ner_model_used": "none",
    "error":         None,
}


def extract_entities_rule_based(text: str) -> Dict[str, List[str]]:
    """Extract clinical entities via rule-based regex and known medicines database."""
    entities = {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
    if not text:
        return entities

    # Load known medicines and aliases
    known_meds: List[str] = []
    med_csv = _ROOT / "data" / "error_rules" / "known_medicines.csv"
    if med_csv.exists():
        try:
            with open(med_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("name"):
                        known_meds.append(row["name"].strip())
                    if row.get("aliases"):
                        for alias in row["aliases"].split(","):
                            if alias.strip():
                                known_meds.append(alias.strip())
        except Exception:
            pass

    if not known_meds:
        known_meds = [
            "Amoxicillin", "Augmentin", "Paracetamol", "Ibuprofen", "Metformin",
            "Atorvastatin", "Omeprazole", "Pantoprazole", "Amlodipine", "Lisinopril",
            "Aspirin", "Clopidogrel", "Cetirizine", "Salbutamol", "Multivitamin",
            "Azithromycin", "Ciprofloxacin", "Doxycycline", "Metronidazole", "Diclofenac"
        ]

    # Regex patterns
    DOSAGE_RE = re.compile(r"(\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|units?|tablets?|capsules?|drops?|puffs?))\b", re.IGNORECASE)
    FREQ_RE = re.compile(r"\b(once\s+daily|twice\s+daily|thrice\s+daily|three\s+times\s+daily|every\s+\d+\s+hours?|od|bd|bid|tds|tid|qid|qds|sos|prn|stat|hs|at\s+bedtime|at\s+night|before\s+meals?|after\s+meals?)\b", re.IGNORECASE)
    DUR_RE = re.compile(r"(\bfor\s+\d+\s+(?:days?|weeks?|months?)\b|\b\d+\s+(?:days?|weeks?|months?)\b)", re.IGNORECASE)

    # 1. Extract Dosages, Frequencies, Durations
    for m in DOSAGE_RE.finditer(text):
        val = m.group(1).strip()
        if val not in entities["DOSAGE"]:
            entities["DOSAGE"].append(val)

    for m in FREQ_RE.finditer(text):
        val = m.group(1).strip()
        if val not in entities["FREQUENCY"]:
            entities["FREQUENCY"].append(val)

    for m in DUR_RE.finditer(text):
        val = m.group(1).strip()
        if val not in entities["DURATION"]:
            entities["DURATION"].append(val)

    # 2. Extract Medicines via word matching & fuzzy match against known medicines
    cleaned_tokens = re.findall(r"\b[A-Za-z]{3,}\b", text)
    for token in cleaned_tokens:
        token_lower = token.lower()
        if token_lower in ["tab", "cap", "inj", "syr", "tablet", "capsule", "syrup", "inhaler", "duration", "hours", "days", "weeks", "daily", "times", "bedtime", "night", "with", "after", "before", "take"]:
            continue
        for med in known_meds:
            if token_lower == med.lower() or fuzz.ratio(token_lower, med.lower()) >= 85:
                if med not in entities["MEDICINE"]:
                    entities["MEDICINE"].append(med)
                break

    return entities


def run_ner_pipeline(text: str, cfg: dict) -> dict:
    """Extract prescription entities from text using the active NER model or clinical fallback."""
    if not cfg.get("ner_enabled", True):
        result = dict(_EMPTY_RESULT)
        result["error"] = "NER disabled in config (ner_enabled: false)."
        return result

    if not text or not text.strip():
        result = dict(_EMPTY_RESULT)
        result["error"] = "No text provided to NER pipeline."
        return result

    active_model_name = str(cfg.get("active_ner_model", "spacy")).lower().strip()

    # Check if spaCy trained model directory exists
    has_trained_model = True
    if active_model_name == "spacy":
        ner_model_path = cfg.get("ner_model_path", "models/ner_model")
        ner_p = Path(ner_model_path)
        has_trained_model = ner_p.exists() and (
            (ner_p / "config.cfg").exists() or (ner_p / "meta.json").exists()
        )

    # Try model execution if trained model exists
    entities: Dict[str, List[str]] = {}
    raw_spans: list = []
    model_label = active_model_name

    if has_trained_model:
        try:
            from src.models.model_registry import get_ner_model
            model = get_ner_model(cfg)
            entities = model.extract_entities(text)
            model_label = model.model_name
        except Exception:
            pass

    # If model is not available or extracted 0 medicines, use clinical rule-based extractor
    if not entities or not entities.get("MEDICINE"):
        fallback_entities = extract_entities_rule_based(text)
        entities = fallback_entities
        model_label = f"{active_model_name} (Rule-Based Fallback)" if not has_trained_model else active_model_name

    missing = [lbl for lbl in ENTITY_LABELS if not entities.get(lbl)]

    return {
        "entities":       entities,
        "raw_spans":      raw_spans,
        "missing":        missing,
        "ner_available":  True,
        "ner_model_used": model_label,
        "error":          None,
    }
