"""
Auth / Provider routes.

POST /api/provider/configure  — validate key, initialise namespace
GET  /api/provider/models     — return model list for a given provider
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.api.schemas import (
    ProviderConfigRequest, ProviderConfigResponse, ModelsResponse
)
from backend.core.llm_factory import (
    validate_api_key, get_provider_models, get_defaults, PROVIDER_CATALOGUE
)
from backend.core.namespace import sentinel_ns

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/provider", tags=["provider"])

# Paths used by the shared namespace
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH     = str(_BASE_DIR / "data" / "sentinel_ecom.duckdb")
CHROMA_PATH = str(_BASE_DIR / "data" / "chroma_bge")
GRAPH_PATH  = str(_BASE_DIR / "data" / "l3_ecom.gml")


@router.post("/configure", response_model=ProviderConfigResponse)
async def configure_provider(req: ProviderConfigRequest):
    """
    1. Validate the API key by calling the LLM with a test prompt.
    2. If valid, initialise (or reconfigure) the SentinelNamespace.
    3. Return the model list so the frontend can refresh the dropdown.
    """
    # Validate first — cheap call, doesn't touch namespace
    valid, err = validate_api_key(
        req.provider, req.api_key, req.main_model, req.fast_model
    )
    if not valid:
        return ProviderConfigResponse(
            valid=False,
            provider=req.provider,
            main_model=req.main_model,
            fast_model=req.fast_model,
            error=err,
        )

    # Initialise or reconfigure namespace
    try:
        os.makedirs(Path(DB_PATH).parent, exist_ok=True)

        if not sentinel_ns.is_initialized:
            # First-time init: exec all agent files
            sentinel_ns.initialize_sync(
                api_key=req.api_key,
                provider=req.provider,
                main_model=req.main_model,
                fast_model=req.fast_model,
                db_path=DB_PATH,
                chroma_path=CHROMA_PATH,
                graph_path=GRAPH_PATH,
            )
        else:
            # Just swap LLMs — no need to re-exec all agents
            await sentinel_ns.reconfigure(
                api_key=req.api_key,
                provider=req.provider,
                main_model=req.main_model,
                fast_model=req.fast_model,
            )
    except Exception as exc:
        logger.error("Namespace init/reconfig failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"System initialisation failed: {exc}")

    models = get_provider_models(req.provider)
    return ProviderConfigResponse(
        valid=True,
        provider=req.provider,
        main_model=req.main_model,
        fast_model=req.fast_model,
        models=models,
    )


@router.get("/models", response_model=ModelsResponse)
async def get_models(provider: str):
    """Return available models for a provider."""
    if provider not in PROVIDER_CATALOGUE:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    default_main, default_fast = get_defaults(provider)
    return ModelsResponse(
        provider=provider,
        models=get_provider_models(provider),
        default_main=default_main,
        default_fast=default_fast,
    )


@router.get("/catalogue")
async def get_catalogue():
    """Return full provider catalogue (used by the login page)."""
    result = {}
    for prov, cat in PROVIDER_CATALOGUE.items():
        result[prov] = {
            "display_name": cat["display_name"],
            "default_main": cat["default_main"],
            "default_fast": cat["default_fast"],
            "models": cat["models"],
        }
    return result
