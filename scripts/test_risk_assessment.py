"""
scripts/test_risk_assessment.py
===============================
Unit test script for assess_risk() logic.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.pipeline.risk_assessment import assess_risk, RiskLevel


def test_risk_assessment():
    print("Testing Risk Assessment Logic...\n")

    # Case 1: No issues, high confidence
    r1 = assess_risk([], ocr_confidence=0.95)
    print("Case 1 (No issues):", r1["level"].upper())
    print("  Reason:", r1["reason"])
    print("  Message:", r1["message"])
    assert r1["level"] == RiskLevel.CLEAR.value, f"Expected clear, got {r1['level']}"
    print("  ✅ PASS\n")

    # Case 2: One high-severity issue
    issues_high = [{"field": "MEDICINE", "severity": "HIGH", "message": "Missing medicine"}]
    r2 = assess_risk(issues_high, ocr_confidence=0.90)
    print("Case 2 (One High Issue):", r2["level"].upper())
    print("  Reason:", r2["reason"])
    print("  Message:", r2["message"])
    assert r2["level"] == RiskLevel.HIGH.value, f"Expected high, got {r2['level']}"
    print("  ✅ PASS\n")

    # Case 3: Two medium-severity issues
    issues_med = [
        {"field": "DOSAGE", "severity": "medium", "message": "Dosage slightly out of range"},
        {"field": "FREQUENCY", "severity": "MEDIUM", "message": "Unusual frequency"},
    ]
    r3 = assess_risk(issues_med, ocr_confidence=0.88)
    print("Case 3 (Two Medium Issues):", r3["level"].upper())
    print("  Reason:", r3["reason"])
    print("  Message:", r3["message"])
    assert r3["level"] == RiskLevel.MEDIUM.value, f"Expected medium, got {r3['level']}"
    print("  ✅ PASS\n")

    # Case 4: Exactly one medium issue
    issues_single_med = [{"field": "DURATION", "severity": "medium", "message": "Duration missing"}]
    r4 = assess_risk(issues_single_med, ocr_confidence=0.85)
    print("Case 4 (Single Medium Issue):", r4["level"].upper())
    print("  Reason:", r4["reason"])
    print("  Message:", r4["message"])
    assert r4["level"] == RiskLevel.LOW.value, f"Expected low, got {r4['level']}"
    print("  ✅ PASS\n")

    # Case 5: Low OCR confidence (< 0.35)
    r5 = assess_risk([], ocr_confidence=0.28)
    print("Case 5 (Low OCR Confidence 0.28):", r5["level"].upper())
    print("  Reason:", r5["reason"])
    print("  Message:", r5["message"])
    assert r5["level"] == RiskLevel.HIGH.value, f"Expected high, got {r5['level']}"
    print("  ✅ PASS\n")

    print("🎉 ALL RISK ASSESSMENT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_risk_assessment()
