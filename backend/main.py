"""
backend/main.py
===============
FastAPI application entrypoint for Smart Prescription NLP API.
Deploys directly to Render.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend, app, and project root are in sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[1]
_APP_DIR = _BACKEND_DIR / "app"

for _p in [str(_ROOT), str(_BACKEND_DIR), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from app.routers.health import router as health_router, HealthStatus
    from app.routers.analyze import router as analyze_router
    from app.routers.metrics import router as metrics_router
    from app.routers.samples import router as samples_router
except ImportError:
    from backend.app.routers.health import router as health_router, HealthStatus
    from backend.app.routers.analyze import router as analyze_router
    from backend.app.routers.metrics import router as metrics_router
    from backend.app.routers.samples import router as samples_router

app = FastAPI(
    title="Smart Prescription NLP API",
    description="Multimodal Clinical OCR, NER Entity Extraction, and Drug Safety Error Detection API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ────────────────────────────────────────────────────────
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://prescription-error-detention.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(metrics_router)
app.include_router(samples_router)


@app.get("/", response_model=HealthStatus, tags=["Root"])
async def root():
    return HealthStatus(
        status="healthy",
        service="Smart Prescription NLP API",
        version="1.0.0",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
