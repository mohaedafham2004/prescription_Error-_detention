"""
src/utils/pdf_generator.py
==========================
Generates formatted, high-accuracy clinical PDF reports from prescription analysis data.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

from src.utils.report_formatter import format_clinical_report


def generate_prescription_pdf(result: Dict[str, Any], filename: str = "prescription") -> bytes:
    """Generate a PDF binary buffer for the prescription analysis result."""
    report = format_clinical_report(result, filename=filename)
    meta = report["report_metadata"]
    assessment = report["clinical_safety_assessment"]
    meds = report["prescribed_medications_summary"]["medications"]
    issues = report["safety_and_error_audit"]["flagged_issues"]
    trans = report["transcription_details"]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        fontName="Helvetica",
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodySmall",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155"),
        fontName="Helvetica",
    )
    cell_bold_style = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
        fontName="Helvetica-Bold",
    )

    story = []

    # 1. Header
    story.append(Paragraph("Smart Prescription NLP — Clinical Audit Report", title_style))
    story.append(Paragraph(f"Automated AI-assisted prescription transcription and clinical safety review", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # 2. Metadata Box (2 Columns)
    meta_data = [
        [
            Paragraph(f"<b>Report ID:</b> {meta['report_id']}", body_style),
            Paragraph(f"<b>AI Engine:</b> {meta['ai_vision_engine']}", body_style),
        ],
        [
            Paragraph(f"<b>Generated:</b> {meta['generated_at_utc']}", body_style),
            Paragraph(f"<b>OCR Confidence:</b> {meta['overall_ocr_confidence']}", body_style),
        ],
        [
            Paragraph(f"<b>Source Image:</b> {meta['source_document']}", body_style),
            Paragraph(f"<b>Processing Time:</b> {meta['processing_time_seconds']}s", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Clinical Risk Banner
    risk_level = assessment["risk_level"]
    if risk_level == "HIGH":
        bg_color = colors.HexColor("#fee2e2")
        border_color = colors.HexColor("#ef4444")
        text_color = colors.HexColor("#991b1b")
    elif risk_level == "MEDIUM":
        bg_color = colors.HexColor("#fef3c7")
        border_color = colors.HexColor("#f59e0b")
        text_color = colors.HexColor("#92400e")
    elif risk_level == "LOW":
        bg_color = colors.HexColor("#e0f2fe")
        border_color = colors.HexColor("#0284c7")
        text_color = colors.HexColor("#075985")
    else:
        bg_color = colors.HexColor("#dcfce7")
        border_color = colors.HexColor("#22c55e")
        text_color = colors.HexColor("#166534")

    risk_text = f"<b>CLINICAL RISK LEVEL: {risk_level}</b><br/>{assessment['guidance_message']}<br/><font size=7.5 color='#475569'>Reason: {assessment['assessment_summary']}</font>"
    risk_para = Paragraph(risk_text, ParagraphStyle("RiskText", parent=styles["Normal"], textColor=text_color, fontSize=8.5, leading=11))
    risk_table = Table([[risk_para]], colWidths=[540])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 10))

    # 4. Prescribed Medications Table
    story.append(Paragraph("1. Prescribed Medications Regimen", section_title_style))
    if not meds:
        story.append(Paragraph("<i>No clinical medications detected in transcription.</i>", body_style))
    else:
        med_headers = [
            Paragraph("<b>#</b>", cell_bold_style),
            Paragraph("<b>Medication Name</b>", cell_bold_style),
            Paragraph("<b>Dosage</b>", cell_bold_style),
            Paragraph("<b>Frequency</b>", cell_bold_style),
            Paragraph("<b>Duration</b>", cell_bold_style),
        ]
        med_rows = [med_headers]
        for m in meds:
            med_rows.append([
                Paragraph(str(m["item_number"]), body_style),
                Paragraph(f"<b>{m['medication_name']}</b>", body_style),
                Paragraph(m["dosage"], body_style),
                Paragraph(m["frequency"], body_style),
                Paragraph(m["duration"], body_style),
            ])

        med_table = Table(med_rows, colWidths=[24, 170, 100, 120, 126])
        med_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#94a3b8")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(med_table)

    story.append(Spacer(1, 10))

    # 5. Safety and Error Audit
    story.append(Paragraph("2. Safety & Error Audit", section_title_style))
    if not issues:
        story.append(Paragraph("✅ <b>No Contraindications or Errors Flagged:</b> All recognized medications, dosages, and administration intervals conform to standard therapeutic ranges.", body_style))
    else:
        iss_headers = [
            Paragraph("<b>Severity</b>", cell_bold_style),
            Paragraph("<b>Category</b>", cell_bold_style),
            Paragraph("<b>Target</b>", cell_bold_style),
            Paragraph("<b>Clinical Issue & Recommendation</b>", cell_bold_style),
        ]
        iss_rows = [iss_headers]
        for iss in issues:
            iss_rows.append([
                Paragraph(f"<b>{iss['severity']}</b>", body_style),
                Paragraph(iss["error_type"], body_style),
                Paragraph(iss["detected_value"] or iss["affected_field"], body_style),
                Paragraph(f"{iss['clinical_warning']}<br/><font color='#0284c7'><b>Action:</b> {iss['recommended_action']}</font>", body_style),
            ])

        iss_table = Table(iss_rows, colWidths=[55, 95, 90, 300])
        iss_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#fca5a5")),
            ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#f87171")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff1f2")]),
        ]))
        story.append(iss_table)



    # 7. Footer / Disclaimer
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    disclaimer = (
        "<b>Clinical Disclaimer:</b> This report is generated by an AI-assisted optical character recognition (OCR) and NLP error detection pipeline. "
        "It is designed strictly for decision support and does not constitute a definitive medical diagnosis or replacement for a licensed pharmacist or physician verification."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=7, leading=9, textColor=colors.HexColor("#94a3b8"))))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
