"""
backend/app/routers/health.py
=============================
Health check endpoint for Render keep-alive and uptime monitoring.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(tags=["Health"])


class HealthStatus(BaseModel):
    status: str
    service: str = "Smart Prescription NLP API"
    version: str = "1.0.0"


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Health check ping",
    description="Returns service health status for deployment keep-alive pinging.",
)
async def health_check():
    return HealthStatus(
        status="healthy",
        service="Smart Prescription NLP API",
        version="1.0.0",
    )
