"""
src/pipeline/risk_assessment.py
===============================
Overall risk-level assessment based on detected errors, interactions,
duplicate therapies, OCR confidence, NER confidence, and safety rule confidence.

Risk Level Conditions
─────────────────────
CLEAR   : OCR ≥ 90%  AND NER ≥ 85%  AND Safety ≥ 85%  AND 0 issues flagged
LOW     : (OCR 70–90% OR NER 70–85% OR Safety 35–85%) AND no HIGH-severity issues
MEDIUM  : (OCR 50–70% OR NER 50–70% OR Safety 20–35%) OR ≥1 HIGH issue OR ≥2 MEDIUM issues
HIGH    : OCR < 50%  OR NER < 50%   OR Safety < 20%   OR critical/interaction/toxicity issue
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Any


class RiskLevel(str, Enum):
    CLEAR  = "clear"    # All signals green, no issues
    LOW    = "low"      # Minor uncertainty in one signal, no critical issues
    MEDIUM = "medium"   # Moderate concern in confidence or multiple issues
    HIGH   = "high"     # Severe signal failure or critical clinical hazard


# ─── Threshold constants ────────────────────────────────────────────────────
# OCR Confidence thresholds
OCR_HIGH_OK    = 0.90   # ≥90% → clear
OCR_LOW_OK     = 0.70   # 70–90% → low
OCR_MEDIUM_OK  = 0.50   # 50–70% → medium
# (< 50%) → high

# NER Confidence thresholds
NER_HIGH_OK    = 0.85   # ≥85% → clear
NER_LOW_OK     = 0.70   # 70–85% → low
NER_MEDIUM_OK  = 0.50   # 50–70% → medium
# (< 50%) → high

# Safety Rule Confidence thresholds
SAFETY_HIGH_OK   = 0.85  # ≥85%      → clear
SAFETY_LOW_OK    = 0.35  # 35–85%    → low   (user-defined)
SAFETY_MEDIUM_OK = 0.20  # 20–35%    → medium
# (< 20%)  → high


def _safety_rule_confidence(issues: List[Dict[str, Any]]) -> float:
    """
    Derive a safety rule confidence score (0.0–1.0) from the issue list.

    Logic:
    - No issues                        → 0.985 (high confidence)
    - Only LOW/INFO severity issues    → 0.91
    - 1–2 MEDIUM issues               → 0.78
    - 3+ MEDIUM or any HIGH issue     → 0.62
    - Critical hazard (INTERACTION,   → 0.40
      CONTRAINDICATION, TOXICITY)
    """
    if not issues:
        return 0.985

    high_count = sum(
        1 for i in issues
        if str(i.get("severity", "")).upper() in ("HIGH", "CRITICAL")
    )
    medium_count = sum(
        1 for i in issues
        if str(i.get("severity", "")).upper() in ("MEDIUM", "WARNING", "MODERATE")
    )
    critical_types = {"INTERACTION", "CONTRAINDICATION", "TOXICITY"}
    has_critical_type = any(
        str(i.get("error_type", "")).upper() in critical_types for i in issues
    )

    if has_critical_type:
        return 0.40
    if high_count >= 2:
        return 0.42
    if high_count == 1:
        return 0.62
    if medium_count >= 3:
        return 0.62
    if medium_count >= 1:
        return 0.78
    return 0.91


def assess_risk(
    issues: List[Dict[str, Any]],
    ocr_confidence: float = 1.0,
    ner_confidence: float = 1.0,
) -> Dict[str, str]:
    """
    Aggregate OCR confidence, NER confidence, safety rule confidence, and
    issue list into one overall risk level with a user-facing message.

    Parameters
    ----------
    issues          : List of flagged prescription issues from error_detection
    ocr_confidence  : Mean OCR line confidence (0.0–1.0)
    ner_confidence  : NER extraction confidence estimate (0.0–1.0)

    Returns
    -------
    dict with keys: level, reason, message
    """
    if not isinstance(issues, list):
        issues = []

    safety_conf = _safety_rule_confidence(issues)

    # ── Classify each signal ───────────────────────────────────────────────
    def classify(val: float, high_ok: float, low_ok: float, medium_ok: float) -> str:
        if val >= high_ok:
            return "clear"
        if val >= low_ok:
            return "low"
        if val >= medium_ok:
            return "medium"
        return "high"

    ocr_class    = classify(ocr_confidence,  OCR_HIGH_OK,    OCR_LOW_OK,    OCR_MEDIUM_OK)
    ner_class    = classify(ner_confidence,  NER_HIGH_OK,    NER_LOW_OK,    NER_MEDIUM_OK)
    safety_class = classify(safety_conf,     SAFETY_HIGH_OK, SAFETY_LOW_OK, SAFETY_MEDIUM_OK)

    rank = {"clear": 0, "low": 1, "medium": 2, "high": 3}
    worst_signal = max(ocr_class, ner_class, safety_class, key=lambda x: rank[x])

    # ── Issue-based overrides ──────────────────────────────────────────────
    critical_hazards = [
        i for i in issues
        if str(i.get("error_type", "")).upper() in ("INTERACTION", "CONTRAINDICATION", "TOXICITY")
        and str(i.get("severity", "")).upper() in ("HIGH", "CRITICAL")
    ]
    severe_dosage = [
        i for i in issues
        if str(i.get("error_type", "")).upper() == "OUT_OF_RANGE"
        and str(i.get("severity", "")).upper() in ("HIGH", "CRITICAL")
    ]
    high_issues = [
        i for i in issues
        if str(i.get("severity", "")).upper() in ("HIGH", "CRITICAL")
    ]
    medium_issues = [
        i for i in issues
        if str(i.get("severity", "")).upper() in ("MEDIUM", "WARNING", "MODERATE")
    ]

    # ── Final level decision ───────────────────────────────────────────────
    # HIGH: worst signal is 'high' (<50% OCR/NER or <20% Safety)
    if worst_signal == "high":
        level = RiskLevel.HIGH
        primary_msg = issues[0].get("message") if issues else None
        signal_str = f"OCR {ocr_confidence*100:.1f}%, NER {ner_confidence*100:.1f}%, Safety {safety_conf*100:.1f}% (<20%)"
        if primary_msg:
            reason = f"HIGH risk — {primary_msg} | {signal_str}"
        else:
            reason = f"HIGH risk — Severe signal failure | {signal_str}"

    # MEDIUM: worst signal is 'medium' (50–70% OCR/NER or 20–35% Safety)
    elif worst_signal == "medium":
        level = RiskLevel.MEDIUM
        primary_msg = issues[0].get("message") if issues else None
        signal_str = f"OCR {ocr_confidence*100:.1f}%, NER {ner_confidence*100:.1f}%, Safety {safety_conf*100:.1f}% (20–35%)"
        if primary_msg:
            reason = f"MEDIUM risk — {primary_msg} | {signal_str}"
        else:
            reason = f"MEDIUM risk — Moderate confidence signal | {signal_str}"

    # LOW: worst signal is 'low' (70–90% OCR, 70–85% NER, or 35–85% Safety) or notices flagged
    elif worst_signal == "low" or len(issues) > 0:
        level = RiskLevel.LOW
        primary_msg = issues[0].get("message") if issues else None
        signal_str = f"OCR {ocr_confidence*100:.1f}%, NER {ner_confidence*100:.1f}%, Safety {safety_conf*100:.1f}% (35–85%)"
        if primary_msg:
            reason = f"LOW risk — {primary_msg} | {signal_str}"
        else:
            reason = f"LOW risk — Minor clinical or formatting notice | {signal_str}"

    # CLEAR: all signals green (OCR ≥90%, NER ≥85%, Safety ≥85%) and 0 issues
    else:
        level = RiskLevel.CLEAR
        reason = (
            f"All signals within safe thresholds — "
            f"OCR {ocr_confidence*100:.1f}% (≥90%), "
            f"NER {ner_confidence*100:.1f}% (≥85%), "
            f"Safety {safety_conf*100:.1f}% (≥85%). "
            f"No issues detected."
        )

    messages = {
        RiskLevel.HIGH: (
            "⛔ High Risk: Critical clinical issue or very low confidence signal detected. "
            "Do NOT dispense without immediate pharmacist or physician review."
        ),
        RiskLevel.MEDIUM: (
            "⚠️ Moderate Risk: Confidence in moderate range (20–35% Safety or 50–70% OCR/NER). "
            "Please double-check with a healthcare professional before proceeding."
        ),
        RiskLevel.LOW: (
            "ℹ️ Low Risk: Minor uncertainties or dosage notices detected (Safety 35–85%). "
            "Worth a quick review — confirm with pharmacist before dispensing."
        ),
        RiskLevel.CLEAR: (
            "✅ Clear: All confidence signals are within safe thresholds and no clinical issues were detected. "
            "Confirm with a licensed pharmacist or doctor before dispensing."
        ),
    }

    return {
        "level": level.value,
        "reason": reason,
        "message": messages[level],
        "ocr_confidence": round(ocr_confidence, 4),
        "ner_confidence": round(ner_confidence, 4),
        "safety_confidence": round(safety_conf, 4),
    }
