"""
src/pipeline/full_pipeline.py
==============================
Single entry point for the full end-to-end prescription analysis pipeline.

Call this from the dashboard, test scripts, or any downstream consumer.
No other pipeline module needs to be imported by the caller.

Pipeline
--------
    raw image
        → ocr_pipeline  (image cleanup + line segmentation + TrOCR)
        → ner_pipeline  (entity extraction: MEDICINE/DOSAGE/FREQ/DUR)
        → error_detection (fuzzy match + range check + freq format + interactions)

Output
------
{
    "extracted_text":  str,
    "lines":           [{"line_index", "text", "confidence", "model_used", ...}],
    "entities": {
        "MEDICINE":  [...],
        "DOSAGE":    [...],
        "FREQUENCY": [...],
        "DURATION":  [...],
    },
    "issues": [
        {
            "error_type": str,   # MISSPELLING | OUT_OF_RANGE | BAD_FREQUENCY |
                                 # INTERACTION | MISSING_FIELD | UNRECOGNISED
            "severity":   str,   # HIGH | MEDIUM | LOW | INFO
            "field":      str,
            "value":      str,
            "message":    str,
            "suggestion": str | None,
        }, ...
    ],
    "ocr_model_used": str,
    "ner_available":  bool,
    "n_lines":        int,
    "mean_ocr_confidence": float,
    "low_confidence_lines": [int, ...],
    "total_time_s":   float,
    "error":          str | None,   # top-level error message; None on success
}

Usage
-----
    from src.pipeline.full_pipeline import run_full_pipeline

    result = run_full_pipeline("data/raw/prescriptions/rx1.jpg")
    print(result["extracted_text"])
    print(result["entities"])
    for issue in result["issues"]:
        print(issue["severity"], issue["message"])
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_loader import load_config
from src.pipeline.ocr_pipeline import run_ocr_pipeline
from src.pipeline.ner_pipeline import run_ner_pipeline
from src.pipeline.error_detection import ErrorDetector, issues_to_dict_list


def run_full_pipeline(
    image_path: str,
    config_path: Optional[str] = None,
    cfg_override: Optional[dict] = None,
    verbose: bool = False,
    debug_dir: Optional[str] = None,
) -> dict:
    """Run the complete prescription analysis pipeline.

    Parameters
    ----------
    image_path  : Path to the raw prescription image (JPG/PNG).
    config_path : Optional explicit path to config.yaml.
                  If None, config is auto-discovered from the project root.
    verbose     : If True, print stage timing to stdout.

    Returns
    -------
    Result dict — see module docstring for the full schema.
    All exceptions are caught; on failure, "error" key contains the message
    and all other fields return safe empty defaults.
    """
    t_total = time.time()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        cfg = dict(load_config(config_path))   # copy so we don't mutate the cached dict
        if cfg_override:
            cfg.update(cfg_override)
    except Exception as e:
        return _error_result(image_path, f"Config load error: {e}", t_total)

    def _log(msg):
        if verbose:
            print(f"  [pipeline] {msg}")

    _log(f"Starting full pipeline: {image_path}")
    _log(f"  Active OCR model : {cfg.get('active_ocr_model', 'trocr')}")
    _log(f"  Active NER model : {cfg.get('active_ner_model', 'spacy')}")
    _log(f"  NER enabled      : {cfg.get('ner_enabled', True)}")

    # ── Stage 1: OCR ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        ocr = run_ocr_pipeline(image_path, cfg, debug_dir=debug_dir)
        _log(f"  OCR done in {time.time()-t0:.1f}s — "
             f"{ocr.n_lines} lines, conf={ocr.mean_confidence:.2f}")
    except Exception as e:
        return _error_result(image_path, f"OCR stage failed: {e}", t_total)

    # ── Stage 2: NER ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        ner = run_ner_pipeline(ocr.full_text, cfg)
        # If Gemini multimodal already extracted structured entities, merge them if local NER is unavailable or empty
        gemini_ents = getattr(ocr, "_gemini_entities", None)
        if gemini_ents and any(gemini_ents.values()):
            if not any(ner.get("entities", {}).values()) or not ner.get("ner_available"):
                ner["entities"] = gemini_ents
                ner["ner_available"] = True
                ner["ner_model_used"] = "gemini (multimodal vision)"

        _log(f"  NER done in {time.time()-t0:.1f}s — "
             f"available={ner['ner_available']}, "
             f"missing={ner['missing']}")
    except Exception as e:
        gemini_ents = getattr(ocr, "_gemini_entities", None)
        if gemini_ents and any(gemini_ents.values()):
            ner = {
                "entities":      gemini_ents,
                "raw_spans":     [],
                "missing":       [k for k, v in gemini_ents.items() if not v],
                "ner_available": True,
                "ner_model_used": "gemini (multimodal vision)",
            }
        else:
            ner = {
                "entities":      {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []},
                "raw_spans":     [],
                "missing":       ["MEDICINE", "DOSAGE", "FREQUENCY", "DURATION"],
                "ner_available": False,
                "error":         f"NER stage error: {e}",
            }


    # ── Stage 3: Error detection ──────────────────────────────────────────────
    t0 = time.time()
    try:
        detector = ErrorDetector(
            medicines_csv    = str(Path(cfg.get("error_rules_dir",
                                                "data/error_rules")) / "known_medicines.csv"),
            dosage_csv       = str(Path(cfg.get("error_rules_dir",
                                                "data/error_rules")) / "dosage_ranges.csv"),
            interactions_csv = str(Path(cfg.get("error_rules_dir",
                                                "data/error_rules")) / "interactions.csv"),
            classes_csv      = str(Path(cfg.get("error_rules_dir",
                                                "data/error_rules")) / "therapeutic_classes.csv"),
            fuzzy_threshold  = int(cfg.get("fuzzy_match_threshold", 80)),
        )
        issues = detector.check(ner["entities"])
        _log(f"  Error detection done in {time.time()-t0:.1f}s — "
             f"{len(issues)} issue(s)")
    except Exception as e:
        issues = []
        _log(f"  Error detection failed: {e}")

    # ── Stage 4: Pharmacological monographs ──────────────────────────────────
    from src.pipeline.drug_monographs import get_drug_monograph
    monographs = {}
    for med in ner.get("entities", {}).get("MEDICINE", []):
        try:
            mono = get_drug_monograph(med)
            if mono:
                monographs[med] = mono
        except Exception:
            pass

    # ── Stage 5: Overall risk assessment ─────────────────────────────────────
    from src.pipeline.risk_assessment import assess_risk
    issues_list = issues_to_dict_list(issues)
    has_entities = any(bool(v) for v in ner.get("entities", {}).values())
    ner_confidence = 0.94 if (ner.get("ner_available") and has_entities) else (0.80 if ner.get("ner_available") else 0.0)
    risk = assess_risk(issues_list, ocr_confidence=ocr.mean_confidence, ner_confidence=ner_confidence)

    # ── Assemble output ───────────────────────────────────────────────────────
    total = round(time.time() - t_total, 2)
    _log(f"Pipeline complete in {total}s (Risk: {risk['level'].upper()})")

    return {
        "extracted_text":       ocr.full_text,
        "lines":                [lr.to_dict() for lr in ocr.lines],
        "entities":             ner["entities"],
        "monographs":           monographs,
        "issues":               issues_list,
        "risk":                 risk,
        "ocr_model_used":       ocr.ocr_model_used,
        "ner_available":        ner["ner_available"],
        "ner_model_used":       ner.get("ner_model_used", cfg.get("active_ner_model", "spacy")),
        "ner_error":            ner.get("error"),
        "n_lines":              ocr.n_lines,
        "mean_ocr_confidence":  ocr.mean_confidence,
        "low_confidence_lines": ocr.low_confidence,
        "total_time_s":         total,
        "error":                None,
    }


def _error_result(image_path: str, message: str, t_start: float) -> dict:
    """Return a safe empty result with an error message."""
    return {
        "extracted_text":       "",
        "lines":                [],
        "entities":             {"MEDICINE": [], "DOSAGE": [],
                                 "FREQUENCY": [], "DURATION": []},
        "monographs":           {},
        "issues":               [],
        "risk":                 {
            "level": "high",
            "reason": "Pipeline processing encountered an error.",
            "message": "⚠️ Please recheck this prescription manually.",
        },
        "ocr_model_used":       "",
        "ner_available":        False,
        "ner_error":            None,
        "n_lines":              0,
        "mean_ocr_confidence":  0.0,
        "low_confidence_lines": [],
        "total_time_s":         round(time.time() - t_start, 2),
        "error":                message,
    }
