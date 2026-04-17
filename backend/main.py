"""
SENTINEL FastAPI Application

Serves:
  • /api/*          — backend API
  • /               — React frontend (from frontend/dist)
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import routers
from backend.api import auth, query, upload, memory, datalab, anomaly, rca

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SENTINEL Analytics",
    description="AI-powered multi-agent analytics platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow the Vite dev server and same-origin production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev
        "http://localhost:3000",
        "http://localhost:8000",
        "*",                      # development convenience (restrict in prod)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------
app.include_router(auth.router)
app.include_router(query.router)
app.include_router(upload.router)
app.include_router(memory.router)
app.include_router(datalab.router)
app.include_router(anomaly.router)
app.include_router(rca.router)


@app.get("/api/health")
async def health():
    from backend.core.namespace import sentinel_ns
    registry = sentinel_ns.get_dataset_registry() if sentinel_ns.is_initialized else {}
    return {
        "status": "ok",
        "initialized": sentinel_ns.is_initialized,
        "has_custom_data": sentinel_ns.has_custom_data,
        "dataset_count": len(registry),
        "datasets": list(registry.keys()),
    }


# ---------------------------------------------------------------------------
# Serve React frontend (production build)
# ---------------------------------------------------------------------------
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    # Serve static assets (JS, CSS, images)
    if (_FRONTEND_DIST / "assets").exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_FRONTEND_DIST / "assets")),
            name="assets",
        )

    # Serve logos (provider images referenced as /logos/...)
    if (_FRONTEND_DIST / "logos").exists():
        app.mount(
            "/logos",
            StaticFiles(directory=str(_FRONTEND_DIST / "logos")),
            name="logos",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve index.html for all non-API routes (SPA routing)."""
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404)
        index = _FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"error": "Frontend not built. Run: cd frontend && npm run build"}
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "SENTINEL API is running.",
            "hint": "Build the frontend: cd frontend && npm install && npm run build",
            "docs": "/api/docs",
        }

# ---------------------------------------------------------------------------
# Startup log
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    logger.info("=" * 60)
    logger.info("  SENTINEL Analytics API  —  http://localhost:8000")
    logger.info("  API docs: http://localhost:8000/api/docs")
    if _FRONTEND_DIST.exists():
        logger.info("  Frontend: http://localhost:8000/")
    else:
        logger.info("  Frontend not built — run: cd frontend && npm run build")
    logger.info("=" * 60)
