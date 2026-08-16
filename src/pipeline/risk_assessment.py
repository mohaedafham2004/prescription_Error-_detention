"""
src/pipeline/risk_assessment.py
===============================
Overall risk-level assessment based on detected errors, interactions,
duplicate therapies, and OCR confidence.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Any


class RiskLevel(str, Enum):
    CLEAR = "clear"       # no issues found
    LOW = "low"           # minor issues (e.g. missing duration, low-severity note)
    MEDIUM = "medium"     # a few real issues (e.g. unclear dosage format, moderate interaction)
    HIGH = "high"         # serious issues (e.g. missing critical fields, major interaction, low OCR confidence)


def assess_risk(issues: List[Dict[str, Any]], ocr_confidence: float = 1.0) -> Dict[str, str]:
    """
    Aggregate the list of issues plus OCR confidence into one overall risk level
    with a direct, non-alarmist user-facing message.

    Parameters
    ----------
    issues : list of dicts, each with 'severity': 'high'|'medium'|'low'|'info'
    ocr_confidence : float between 0.0 and 1.0 (defaults to 1.0)

    Returns
    -------
    dict : {"level": str, "reason": str, "message": str}
    """
    if not isinstance(issues, list):
        issues = []

    # Case-insensitive severity matching
    high_count = sum(1 for i in issues if str(i.get("severity", "")).lower() in ("high", "critical"))
    medium_count = sum(1 for i in issues if str(i.get("severity", "")).lower() in ("medium", "warning", "moderate"))

    # Very low OCR confidence undermines downstream reliability
    if ocr_confidence < 0.35:
        level = RiskLevel.HIGH
        reason = "OCR confidence is very low — the extracted text itself may be unreliable."
    elif high_count >= 1:
        level = RiskLevel.HIGH
        reason = f"{high_count} critical field(s) missing or clearly abnormal."
    elif medium_count >= 2:
        level = RiskLevel.MEDIUM
        reason = f"{medium_count} field(s) need review."
    elif medium_count == 1 or len(issues) > 0:
        level = RiskLevel.LOW
        reason = "Minor issues detected."
    else:
        level = RiskLevel.CLEAR
        reason = "No issues detected in the extracted fields."

    messages = {
        RiskLevel.HIGH: "⚠️ Please recheck this prescription with your doctor or pharmacist before proceeding. Key information could not be reliably confirmed.",
        RiskLevel.MEDIUM: "⚠️ Some details in this prescription need review. Please double-check with a healthcare professional if anything looks unclear.",
        RiskLevel.LOW: "ℹ️ Minor issues detected — worth a quick check, but nothing critical.",
        RiskLevel.CLEAR: "✅ No issues detected. As always, confirm with a licensed pharmacist or doctor before use.",
    }

    return {
        "level": level.value,
        "reason": reason,
        "message": messages[level],
    }
