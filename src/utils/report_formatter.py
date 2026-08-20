"""
src/utils/report_formatter.py
=============================
Transforms raw prescription analysis results into high-accuracy,
structured, and easily readable clinical JSON reports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def format_clinical_report(result: Dict[str, Any], filename: str = "prescription") -> Dict[str, Any]:
    """Transform raw pipeline result into an accurate, structured, and easy-to-read JSON report."""
    now = datetime.utcnow()
    report_id = f"RX-{now.strftime('%Y%m%d-%H%M%S')}"

    entities = result.get("entities", {})
    medicines = entities.get("MEDICINE", []) or entities.get("medicine", [])
    dosages = entities.get("DOSAGE", []) or entities.get("dosage", [])
    frequencies = entities.get("FREQUENCY", []) or entities.get("frequency", [])
    durations = entities.get("DURATION", []) or entities.get("duration", [])
    monographs = result.get("monographs", {})
    issues = result.get("issues", [])
    risk = result.get("risk", {})

    ocr_conf = float(result.get("mean_ocr_confidence", result.get("ocr_confidence", 0.0)))
    conf_pct = f"{ocr_conf * 100:.1f}%" if ocr_conf > 0 else "N/A"

    # Build paired medications list
    max_items = max(len(medicines), len(dosages), len(frequencies), len(durations), 1)
    prescribed_medications: List[Dict[str, Any]] = []

    for idx in range(max_items):
        med_name = medicines[idx] if idx < len(medicines) else ""
        if not med_name and idx >= len(medicines) and not any([
            idx < len(dosages), idx < len(frequencies), idx < len(durations)
        ]):
            continue

        dose = dosages[idx] if idx < len(dosages) else "Not specified"
        freq = frequencies[idx] if idx < len(frequencies) else "Not specified"
        dur = durations[idx] if idx < len(durations) else "Not specified"

        med_entry: Dict[str, Any] = {
            "item_number": idx + 1,
            "medication_name": med_name or "Unidentified Drug",
            "dosage": dose,
            "frequency": freq,
            "duration": dur,
        }

        # Attach pharmaceutical monograph summary if available
        if med_name and med_name in monographs:
            mono = monographs[med_name]
            if hasattr(mono, "model_dump"):
                mono = mono.model_dump()
            elif hasattr(mono, "to_dict"):
                mono = mono.to_dict()

            med_entry["monograph_reference"] = {
                "generic_name": mono.get("generic_name", med_name),
                "brand_names": mono.get("brand_names", "N/A"),
                "therapeutic_class": mono.get("therapeutic_class", "N/A"),
                "standard_dosage": mono.get("standard_dosage", "N/A"),
                "precautions": mono.get("precautions", "N/A"),
            }

        prescribed_medications.append(med_entry)

    # Format clinical issues cleanly
    formatted_issues: List[Dict[str, Any]] = []
    for iss in issues:
        if hasattr(iss, "model_dump"):
            iss = iss.model_dump()
        elif hasattr(iss, "to_dict"):
            iss = iss.to_dict()

        formatted_issues.append({
            "severity": str(iss.get("severity", "INFO")).upper(),
            "error_type": iss.get("error_type", "CLINICAL_NOTE"),
            "affected_field": iss.get("field", "GENERAL"),
            "detected_value": iss.get("value", ""),
            "clinical_warning": iss.get("message", ""),
            "recommended_action": iss.get("suggestion") or "Verify with prescribing physician.",
        })

    # Line by line OCR breakdown
    line_breakdown = []
    for line in result.get("lines", []):
        if hasattr(line, "model_dump"):
            line = line.model_dump()
        elif hasattr(line, "to_dict"):
            line = line.to_dict()

        if isinstance(line, dict):
            l_text = line.get("text", "")
            l_conf = line.get("confidence", 0.0)
            l_idx = line.get("line_index", len(line_breakdown))
        else:
            l_text = str(line)
            l_conf = ocr_conf
            l_idx = len(line_breakdown)

        line_breakdown.append({
            "line_number": l_idx + 1,
            "transcription": l_text,
            "confidence": f"{float(l_conf) * 100:.1f}%" if float(l_conf) > 0 else "N/A",
        })

    risk_level_str = (
        risk.get("level") if isinstance(risk, dict) else getattr(risk, "level", "clear")
    ) or "clear"
    risk_reason_str = (
        risk.get("reason") if isinstance(risk, dict) else getattr(risk, "reason", "No critical issues detected.")
    ) or "No critical issues detected."
    risk_msg_str = (
        risk.get("message") if isinstance(risk, dict) else getattr(risk, "message", "Confirm with licensed pharmacist or doctor before use.")
    ) or "Confirm with licensed pharmacist or doctor before use."

    return {
        "report_metadata": {
            "report_id": report_id,
            "generated_at_utc": now.isoformat() + "Z",
            "source_document": filename,
            "ai_vision_engine": result.get("ocr_model_used", "AI Vision Engine"),
            "overall_ocr_confidence": conf_pct,
            "processing_time_seconds": round(float(result.get("total_time_s", 0.0)), 2),
        },
        "clinical_safety_assessment": {
            "risk_level": str(risk_level_str).upper(),
            "assessment_summary": risk_reason_str,
            "guidance_message": risk_msg_str,
            "human_pharmacist_review_recommended": str(risk_level_str).lower() in ("high", "medium"),
        },
        "prescribed_medications_summary": {
            "total_medications_detected": len(prescribed_medications),
            "medications": prescribed_medications,
        },
        "safety_and_error_audit": {
            "total_issues_flagged": len(formatted_issues),
            "flagged_issues": formatted_issues,
        },
        "transcription_details": {
            "full_extracted_text": result.get("extracted_text", "").strip(),
            "segmented_lines": line_breakdown,
        },
    }
