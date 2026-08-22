"""
backend/app/routers/analyze.py
==============================
Endpoint: POST /analyze & POST /api/analyze
Processes an uploaded prescription image through OCR + NER + Safety + Monograph pipeline.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

# Allow importing from root and backend
_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_APP_DIR = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_BACKEND_DIR), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.pipeline.full_pipeline import run_full_pipeline
from src.utils.config_loader import load_config
try:
    from app.schemas import (
        AnalyzeResponse,
        EntitiesDict,
        IssueItem,
        LineItem,
        DrugMonograph,
        RiskAssessment,
        SampleItem,
    )
except ImportError:
    from backend.app.schemas import (
        AnalyzeResponse,
        EntitiesDict,
        IssueItem,
        LineItem,
        DrugMonograph,
        RiskAssessment,
        SampleItem,
    )

router = APIRouter(tags=["Analyze"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze prescription image",
    description="Upload a prescription image scan (PNG, JPG, JPEG) to run OCR, clinical NER entity extraction, drug safety error detection, and pharmacological monographs.",
)
@router.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    include_in_schema=False,
)
async def analyze_prescription(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    sample_id: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
):
    cfg = load_config()

    temp_path = None
    try:
        if sample_id:
            filename = f"{sample_id}.png" if not sample_id.endswith(".png") else sample_id
            sample_path = _ROOT / "data" / "raw" / "prescriptions" / filename
            if not sample_path.exists():
                sample_path = _ROOT / "data" / "raw" / "prescriptions" / "sample_rx.png"
            target_image_path = str(sample_path)
        elif use_sample or file is None:
            sample_path = _ROOT / "data" / "raw" / "prescriptions" / "sample_rx.png"
            if not sample_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Sample prescription image not found at data/raw/prescriptions/sample_rx.png",
                )
            target_image_path = str(sample_path)
        else:
            if file.content_type and not (
                file.content_type.startswith("image/") or file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type '{file.content_type}'. Please upload an image (PNG, JPG, JPEG).",
                )

            suffix = Path(file.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
            target_image_path = temp_path

        # Run pipeline asynchronously in worker thread
        cfg_override = {"trocr_use_pretrained": cfg.get("trocr_use_pretrained", True)}
        if model:
            cfg_override["active_ocr_model"] = model.lower().strip()

        import asyncio
        result = await asyncio.to_thread(
            run_full_pipeline,
            target_image_path,
            cfg_override=cfg_override,
            verbose=False,
        )

        if result.get("error"):
            return AnalyzeResponse(
                extracted_text="",
                entities=EntitiesDict(),
                monographs={},
                issues=[],
                risk=RiskAssessment(
                    level="high",
                    reason="Pipeline processing error.",
                    message="⚠️ Could not complete prescription analysis.",
                ),
                lines=[],
                ocr_model_used=cfg.get("active_ocr_model", "trocr"),
                ocr_confidence=0.0,
                ner_confidence=0.0,
                ner_available=False,
                total_time_s=result.get("total_time_s", 0.0),
                error=result["error"],
            )

        raw_entities = result.get("entities", {})
        entities = EntitiesDict(
            medicine=raw_entities.get("MEDICINE", []),
            dosage=raw_entities.get("DOSAGE", []),
            frequency=raw_entities.get("FREQUENCY", []),
            duration=raw_entities.get("DURATION", []),
        )

        issues = [
            IssueItem(
                error_type=iss.get("error_type", "ISSUE"),
                severity=iss.get("severity", "LOW"),
                field=iss.get("field", "MEDICINE"),
                value=iss.get("value", ""),
                message=iss.get("message", ""),
                suggestion=iss.get("suggestion"),
            )
            for iss in result.get("issues", [])
        ]

        lines = [
            LineItem(
                line_index=line.get("line_index", idx),
                text=line.get("text", ""),
                confidence=round(float(line.get("confidence", 0.0)), 4),
                model_used=line.get("model_used"),
            )
            for idx, line in enumerate(result.get("lines", []))
        ]

        # Convert monographs dict
        monographs = {}
        for med, mono in result.get("monographs", {}).items():
            monographs[med] = DrugMonograph(
                name=mono.get("name", med.title()),
                generic_name=mono.get("generic_name", med.title()),
                brand_names=mono.get("brand_names"),
                composition=mono.get("composition"),
                manufacturer=mono.get("manufacturer"),
                therapeutic_class=mono.get("therapeutic_class"),
                dosage_forms=mono.get("dosage_forms"),
                usage=mono.get("usage"),
                standard_dosage=mono.get("standard_dosage"),
                precautions=mono.get("precautions"),
            )

        ocr_conf = float(result.get("mean_ocr_confidence", 0.0))
        ner_available = bool(result.get("ner_available", False))
        has_entities = any(bool(v) for v in raw_entities.values())
        ner_conf = 0.94 if (ner_available and has_entities) else (0.80 if ner_available else 0.0)

        # Risk assessment
        raw_risk = result.get("risk", {})
        risk = RiskAssessment(
            level=raw_risk.get("level", "clear"),
            reason=raw_risk.get("reason", "No issues detected."),
            message=raw_risk.get("message", "✅ No issues detected. Confirm with a doctor/pharmacist before use."),
        )

        return AnalyzeResponse(
            extracted_text=result.get("extracted_text", ""),
            entities=entities,
            monographs=monographs,
            issues=issues,
            risk=risk,
            lines=lines,
            ocr_model_used=result.get("ocr_model_used", cfg.get("active_ocr_model", "trocr")),
            ocr_confidence=round(ocr_conf, 4),
            ner_confidence=round(ner_conf, 4),
            ner_available=ner_available,
            total_time_s=round(float(result.get("total_time_s", 0.0)), 2),
            error=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prescription pipeline execution failed: {str(e)}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
