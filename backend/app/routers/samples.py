"""
backend/app/routers/samples.py
==============================
Endpoints for testing with clinical sample prescriptions.
"""

import sys
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_APP_DIR = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_BACKEND_DIR), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from app.schemas import SampleItem
except ImportError:
    from backend.app.schemas import SampleItem

router = APIRouter(prefix="/api", tags=["Samples"])

SAMPLES_CATALOG = [
    {
        "id": "cardio_rx",
        "title": "Cardiology Center Prescription",
        "category": "Cardiovascular",
        "description": "Prescription for Atorvastatin 40mg, Aspirin 75mg, Clopidogrel 75mg, and Pantoprazole 40mg.",
        "filename": "cardio_rx.png",
    },
    {
        "id": "infection_rx",
        "title": "Infection & Respiratory Clinic",
        "category": "Infectious Disease",
        "description": "Prescription for Augmentin 625mg, Paracetamol 650mg, Cetirizine 10mg, and Salbutamol 100mcg.",
        "filename": "infection_rx.png",
    },
    {
        "id": "diabetic_care_rx",
        "title": "Metropolitan Diabetes Care",
        "category": "Endocrinology",
        "description": "Prescription for Metformin 500mg, Lisinopril 10mg, Amlodipine 5mg, and Ibuprofen 400mg.",
        "filename": "diabetic_care_rx.png",
    },
    {
        "id": "sample_rx",
        "title": "General Practice Clinic",
        "category": "General Medicine",
        "description": "Sample prescription scan with standard clinical notations for general OCR verification.",
        "filename": "sample_rx.png",
    },
]


@router.get(
    "/samples",
    response_model=List[SampleItem],
    summary="Get catalog of available sample prescriptions",
    description="Returns preloaded clinical sample prescriptions for testing and evaluation.",
)
async def list_samples():
    return SAMPLES_CATALOG


@router.get(
    "/sample-image/{sample_id}",
    summary="Get specific sample prescription image file",
    description="Streams a sample prescription scan by ID (cardio_rx, infection_rx, diabetic_care_rx, sample_rx).",
)
async def get_sample_by_id(sample_id: str):
    filename = f"{sample_id}.png" if not sample_id.endswith(".png") else sample_id
    sample_path = _ROOT / "data" / "raw" / "prescriptions" / filename
    if not sample_path.exists():
        sample_path = _ROOT / "data" / "raw" / "prescriptions" / "sample_rx.png"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(str(sample_path), media_type="image/png", filename=filename)


@router.get(
    "/sample-image",
    summary="Get default sample prescription image",
    description="Returns the default sample prescription image file.",
)
async def get_default_sample():
    return await get_sample_by_id("sample_rx")
