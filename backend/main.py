"""
backend/main.py
===============
FastAPI application entrypoint for Smart Prescription NLP API.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_loader import load_config
from backend.routes.analyze import router as analyze_router
from backend.routes.metrics import router as metrics_router
from backend.schemas import HealthResponse

# Initialize FastAPI application
app = FastAPI(
    title="Smart Prescription NLP - API",
    description="High-performance clinical OCR, NER extraction, and drug safety error detection API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:4000",
        "http://127.0.0.1:4000",
        "http://localhost:8501",
        "http://localhost:8502",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8502",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(analyze_router)
app.include_router(metrics_router)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check and active model status",
)
async def health_check():
    """Returns the operational status of the API and currently configured active models."""
    cfg = load_config()
    return HealthResponse(
        status="healthy",
        active_ocr_model=cfg.get("active_ocr_model", "trocr"),
        active_ner_model=cfg.get("active_ner_model", "spacy"),
        version="1.0.0",
        trocr_available=True,
        ner_available=bool(cfg.get("ner_enabled", True)),
    )


@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    return {
        "name": "Smart Prescription NLP API",
        "status": "online",
        "docs": "/docs",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
