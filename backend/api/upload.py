"""
Data upload route.

POST /api/upload  — upload CSV / Excel / SQLite / Parquet
                    and ADD it to the active SENTINEL namespace.
                    Each dataset gets its own isolated DuckDB file so
                    queries target exactly one dataset at a time.
"""

from __future__ import annotations
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.api.schemas import UploadResponse
from backend.core.data_loader import ingest_file
from backend.core.namespace import sentinel_ns

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

_BASE_DIR   = Path(__file__).resolve().parent.parent.parent
_UPLOAD_DIR = _BASE_DIR / "data" / "uploads"

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".db", ".sqlite", ".sqlite3"}
MAX_FILE_SIZE = 500 * 1024 * 1024   # 500 MB


def _db_path_for_dataset(filename: str) -> str:
    """Generate a unique DuckDB file path for a given dataset filename."""
    safe_name = re.sub(r"[^a-z0-9_]", "_", Path(filename).stem.lower())[:50]
    if not safe_name:
        safe_name = "dataset"
    return str(_UPLOAD_DIR / f"{safe_name}.duckdb")


@router.post("/upload", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """
    Upload a dataset file and add it to SENTINEL's active data.

    Each dataset is stored in its own isolated DuckDB file.
    The frontend dropdown controls which dataset queries are routed to.

    Supported formats: CSV, Excel (.xlsx/.xls), Parquet, SQLite.
    Large files (100k+ rows) are handled via chunked streaming.
    """
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. "
                   f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1e6:.1f} MB). Maximum: 500 MB.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    os.makedirs(_UPLOAD_DIR, exist_ok=True)

    # ── Per-dataset DuckDB: each file gets its own isolated DB ──────────
    db_path = _db_path_for_dataset(filename)
    os.makedirs(Path(db_path).parent, exist_ok=True)

    # Always start fresh — delete any previous DuckDB for this filename
    for ext in ("", ".wal"):
        old = db_path + ext if ext else db_path
        if os.path.exists(old):
            try:
                os.remove(old)
                logger.info("Removed old DB file: %s", old)
            except Exception:
                pass

    try:
        result = ingest_file(content, filename, db_path, existing_con=None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)[:300]}")

    # Update the shared namespace with new schema and dataset registration
    if sentinel_ns.is_initialized:
        await sentinel_ns.update_data(
            new_con=result["con"],
            new_schema=result["schema"],
            date_min=result["date_min"],
            date_max=result["date_max"],
            filename=filename,
            new_tables=result["tables"],
            row_count=result["row_count"],
            all_tables=result.get("all_tables", result["tables"]),
        )
    else:
        logger.warning("Namespace not yet initialised — data upload deferred until configure is called")

    date_min_str = str(result["date_min"]) if result["date_min"] else None
    date_max_str = str(result["date_max"]) if result["date_max"] else None

    # Determine total datasets loaded
    dataset_count = len(sentinel_ns.get_dataset_registry()) if sentinel_ns.is_initialized else 1

    return UploadResponse(
        success=True,
        filename=filename,
        row_count=result["row_count"],
        tables=result["tables"],
        primary_table=result["primary_table"],
        columns=result["columns"],
        date_col=result.get("date_col"),
        date_min=date_min_str,
        date_max=date_max_str,
        schema_preview=result["schema"][:3000],
        dataset_count=dataset_count,
        all_tables=result.get("all_tables", result["tables"]),
    )
