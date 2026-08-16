"""
prescription_ocr_pipeline/src/evaluate.py
==========================================
Field-level accuracy evaluation:

  - Medicine detection  : Fuzzy-match each predicted drug against GT list (score ≥ threshold)
  - Dosage / Frequency / Duration : Normalise strings + fuzzy compare

Outputs structured per-image breakdown and aggregate summary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[3]
for p in [str(_ROOT), str(_PIPELINE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Normalisation helpers ---------------------------------------------------

def _normalise_strength(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", "", s)          # "500 mg" → "500mg"
    s = s.replace("mcg", "ug")         # unify µg
    s = s.replace("µg", "ug")
    return s


def _normalise_freq(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    # Canonical forms
    mappings = {
        "once daily": "od", "od": "od",
        "twice daily": "bd", "bd": "bd", "bid": "bd",
        "tds": "tds", "tid": "tds", "three times": "tds",
        "qid": "qid", "four times": "qid",
        "prn": "prn", "sos": "prn", "as needed": "prn",
        "hs": "hs", "at night": "hs", "bedtime": "hs",
        "stat": "stat",
        "weekly": "weekly",
    }
    for key, norm in mappings.items():
        if key in s:
            return norm
    return s


def _normalise_duration(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("days", "day").replace("weeks", "week").replace("months", "month")
    return s


def _fuzzy_score(a: str, b: str) -> float:
    """Return 0–1 fuzzy similarity between two strings."""
    try:
        from rapidfuzz import fuzz
        return fuzz.token_sort_ratio(a, b) / 100.0
    except ImportError:
        # Jaccard fallback
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


def _medicine_score(
    predicted: List[str],
    gt_meds: List[Dict[str, Any]],
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Compute precision, recall, F1 for medicine detection.
    Each GT drug must be matched to exactly one predicted drug.
    """
    gt_names = [m["drug"].lower() for m in gt_meds]
    pred_names = [p.lower() for p in predicted]

    matched_gt = set()
    matched_pred = set()

    for i, pn in enumerate(pred_names):
        for j, gn in enumerate(gt_names):
            if j in matched_gt:
                continue
            if _fuzzy_score(pn, gn) >= threshold:
                matched_gt.add(j)
                matched_pred.add(i)
                break

    tp = len(matched_pred)
    fp = len(pred_names) - tp
    fn = len(gt_names) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def _field_match(
    predicted: Optional[str],
    gt_value: Optional[str],
    normalise_fn,
    threshold: float = 0.70,
) -> bool:
    if not predicted and not gt_value:
        return True
    if not predicted or not gt_value:
        return False
    return _fuzzy_score(normalise_fn(predicted), normalise_fn(gt_value)) >= threshold


# ── Per-image evaluation ---------------------------------------------------

def evaluate_image(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    fuzzy_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Evaluate one prescription's predictions against its ground truth.

    Parameters
    ----------
    prediction   : postprocess() output dict
    ground_truth : one entry from ground_truth.json

    Returns
    -------
    {
        "image"        : str,
        "engine"       : str,
        "confidence"   : float,
        "medicine"     : {"precision", "recall", "f1"},
        "dosage_hit"   : bool,
        "frequency_hit": bool,
        "duration_hit" : bool,
        "field_score"  : float,   # mean of medicine_f1, dosage, freq, duration (0–1)
    }
    """
    gt_meds = ground_truth.get("medications", [])

    med_scores = _medicine_score(
        prediction.get("medicines", []),
        gt_meds,
        threshold=fuzzy_threshold,
    )

    # For dosage/freq/duration use the FIRST GT medication entry as reference
    first_gt = gt_meds[0] if gt_meds else {}

    dosage_hit = _field_match(
        prediction.get("dosage"),
        first_gt.get("strength"),
        _normalise_strength,
        fuzzy_threshold,
    )
    freq_hit = _field_match(
        prediction.get("frequency"),
        first_gt.get("frequency"),
        _normalise_freq,
        fuzzy_threshold,
    )
    dur_hit = _field_match(
        prediction.get("duration"),
        first_gt.get("duration"),
        _normalise_duration,
        fuzzy_threshold,
    )

    field_score = round(
        (med_scores["f1"] + float(dosage_hit) + float(freq_hit) + float(dur_hit)) / 4.0,
        3,
    )

    return {
        "image": ground_truth.get("image", "unknown"),
        "engine": prediction.get("engine", "unknown"),
        "confidence": round(prediction.get("confidence", 0.0), 3),
        "medicine": med_scores,
        "dosage_hit": dosage_hit,
        "frequency_hit": freq_hit,
        "duration_hit": dur_hit,
        "field_score": field_score,
    }


# ── Aggregate report -------------------------------------------------------

def aggregate_report(per_image_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build aggregate accuracy summary from a list of per-image evaluate_image() dicts."""
    if not per_image_results:
        return {}

    n = len(per_image_results)

    avg_med_f1 = sum(r["medicine"]["f1"] for r in per_image_results) / n
    avg_dosage = sum(int(r["dosage_hit"]) for r in per_image_results) / n
    avg_freq = sum(int(r["frequency_hit"]) for r in per_image_results) / n
    avg_dur = sum(int(r["duration_hit"]) for r in per_image_results) / n
    avg_field = sum(r["field_score"] for r in per_image_results) / n

    return {
        "n_images": n,
        "avg_medicine_f1": round(avg_med_f1, 3),
        "avg_dosage_accuracy": round(avg_dosage, 3),
        "avg_frequency_accuracy": round(avg_freq, 3),
        "avg_duration_accuracy": round(avg_dur, 3),
        "overall_field_score": round(avg_field, 3),
    }
