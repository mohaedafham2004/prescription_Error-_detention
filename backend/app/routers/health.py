"""
backend/app/routers/health.py
=============================
Health check endpoint for Render keep-alive and uptime monitoring.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from src.utils.config_loader import load_config

router = APIRouter(tags=["Health"])


class HealthStatus(BaseModel):
    status: str = "ok"
    active_ocr_model: str = "gemini"
    active_ner_model: str = "spacy"
    service: str = "Smart Prescription NLP API"
    version: str = "1.0.0"


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Health check ping",
    description="Returns service health status for deployment keep-alive pinging.",
)
@router.get(
    "/api/health",
    response_model=HealthStatus,
    include_in_schema=False,
)
async def health_check():
    try:
        cfg = load_config()
        active_ocr = cfg.get("active_ocr_model", "gemini")
        active_ner = cfg.get("active_ner_model", "spacy")
    except Exception:
        active_ocr = "gemini"
        active_ner = "spacy"

    return HealthStatus(
        status="ok",
        active_ocr_model=active_ocr,
        active_ner_model=active_ner,
        service="Smart Prescription NLP API",
        version="1.0.0",
    )
