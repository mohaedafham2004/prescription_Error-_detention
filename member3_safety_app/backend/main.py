"""
backend/main.py
===============
FastAPI application entrypoint for Smart Prescription NLP API.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root and member3_safety_app are in sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
_SAFETY_DIR = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_ROOT), str(_SAFETY_DIR), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config_loader import load_config
try:
    from member3_safety_app.backend.routes.analyze import router as analyze_router
    from member3_safety_app.backend.routes.metrics import router as metrics_router
    from member3_safety_app.backend.schemas import HealthResponse
except ImportError:
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
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3100",
        "http://127.0.0.1:3100",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:4000",
        "http://127.0.0.1:4000",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
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

    uvicorn.run("member3_safety_app.backend.main:app", host="127.0.0.1", port=8001, reload=True)
