"""
prescription_ocr_pipeline/src/postprocess.py
=============================================
Post-processing module:
  1. Rule/regex extraction of medicine, dosage, frequency, duration
  2. Fuzzy-match drug names against drug_dictionary.json
  3. Structured output normalisation (entity dict)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[3]
for p in [str(_ROOT), str(_PIPELINE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Drug dictionary ---------------------------------------------------------
_DICT_PATH = _PIPELINE_ROOT / "data" / "drug_dictionary.json"
_DRUG_DICT: Dict[str, List[str]] = {}
_ALL_DRUG_NAMES: List[str] = []  # canonical + alias flat list


def _load_drug_dict() -> None:
    global _DRUG_DICT, _ALL_DRUG_NAMES
    if _ALL_DRUG_NAMES:
        return
    try:
        with open(_DICT_PATH, "r", encoding="utf-8") as f:
            _DRUG_DICT = json.load(f)
        for canonical, aliases in _DRUG_DICT.items():
            _ALL_DRUG_NAMES.append(canonical.lower())
            _ALL_DRUG_NAMES.extend([a.lower() for a in aliases])
    except FileNotFoundError:
        pass  # graceful – no fuzzy matching


# ── Regex patterns ----------------------------------------------------------
_DOSAGE_PATTERN = re.compile(
    r"\b(\d{1,4}(?:\.\d+)?)\s*"
    r"(mg|mcg|ug|µg|ml|g|tab(?:let)?s?|cap(?:sule)?s?|drops?|puff?s?|units?|iu|mEq)\b",
    re.IGNORECASE,
)

_FREQUENCY_KEYWORDS = {
    r"\bOD\b|\bonce daily\b|\b1[-\s]?times?\b": "OD",
    r"\bBD\b|\bBID\b|\btwice daily\b|\b2[-\s]?times?\b": "BD",
    r"\bTDS\b|\bTID\b|\bthree times\b|\b3[-\s]?times?\b": "TDS",
    r"\bQID\b|\bfour times\b|\b4[-\s]?times?\b": "QID",
    r"\bSOS\b|\bas needed\b|\bPRN\b|\bwhen required\b": "PRN",
    r"\bHS\b|\bat (bed)?night\b|\bbedtime\b": "HS",
    r"\bstat\b|\bimmediately\b|\bonce\b": "STAT",
    r"\bevery\s+(\d+)\s*h(?:ours?)?\b": "every {n}h",
    r"\bweekly\b|\bonce a week\b": "Weekly",
}

_DURATION_PATTERN = re.compile(
    r"\bfor\s+(\d+)\s*(days?|weeks?|months?)\b"
    r"|\b(\d+)\s*(days?|weeks?|months?)\s*(?:only|course|supply|course)?\b",
    re.IGNORECASE,
)


def _extract_dosage(text: str) -> Optional[str]:
    m = _DOSAGE_PATTERN.search(text)
    if m:
        return m.group(0).strip()
    return None


def _extract_frequency(text: str) -> Optional[str]:
    for pattern, label in _FREQUENCY_KEYWORDS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            if "{n}" in label:
                return label.replace("{n}", m.group(1))
            return label
    return None


def _extract_duration(text: str) -> Optional[str]:
    m = _DURATION_PATTERN.search(text)
    if m:
        groups = [g for g in m.groups() if g is not None]
        if len(groups) >= 2:
            return f"{groups[0]} {groups[1]}"
    return None


def _fuzzy_match_drug(word: str, threshold: int = 70) -> Optional[str]:
    """Return canonical drug name if fuzzy score ≥ threshold, else None."""
    _load_drug_dict()
    if not _ALL_DRUG_NAMES:
        return None
    try:
        from rapidfuzz import process as fuzz_process, fuzz
        match = fuzz_process.extractOne(
            word.lower(),
            _ALL_DRUG_NAMES,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if match is None:
            return None
        matched_alias = match[0]
        # Resolve alias → canonical
        for canonical, aliases in _DRUG_DICT.items():
            if matched_alias == canonical.lower() or matched_alias in [a.lower() for a in aliases]:
                return canonical
        return matched_alias.capitalize()
    except ImportError:
        return None


def _extract_medicines(text: str, threshold: int = 70) -> List[str]:
    """Extract all drug names from text using token fuzzy matching."""
    _load_drug_dict()
    if not _ALL_DRUG_NAMES:
        # Fallback: capitalised tokens
        return [w for w in text.split() if w[0].isupper() and len(w) > 3]

    words = re.findall(r"[A-Za-z]+(?:\s+[A-Za-z]+)?", text)
    found: List[str] = []
    seen: set = set()
    for phrase in words:
        if len(phrase) < 4:
            continue
        canonical = _fuzzy_match_drug(phrase, threshold)
        if canonical and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    return found


def postprocess(
    ocr_result: Dict[str, Any],
    fuzzy_threshold: int = 70,
) -> Dict[str, Any]:
    """
    Convert raw OCR output into structured prescription entities.

    Parameters
    ----------
    ocr_result      : {"text": str, "confidence": float, "engine": str}
    fuzzy_threshold : Minimum rapidfuzz score to accept a drug match.

    Returns
    -------
    {
        "medicines"   : List[str],      # canonical drug names
        "dosage"      : Optional[str],
        "frequency"   : Optional[str],
        "duration"    : Optional[str],
        "raw_text"    : str,
        "confidence"  : float,
        "engine"      : str,
    }
    """
    text = ocr_result.get("text", "") or ""
    confidence = ocr_result.get("confidence", 0.0)
    engine = ocr_result.get("engine", "unknown")

    medicines = _extract_medicines(text, fuzzy_threshold)
    dosage = _extract_dosage(text)
    frequency = _extract_frequency(text)
    duration = _extract_duration(text)

    return {
        "medicines": medicines,
        "dosage": dosage,
        "frequency": frequency,
        "duration": duration,
        "raw_text": text,
        "confidence": confidence,
        "engine": engine,
    }
