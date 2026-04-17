"""
Memory inspection routes.

GET /api/memory/stats        — L2 / L3 / L4 counts
GET /api/memory/layer/{layer} — full content of a memory layer
"""

from __future__ import annotations
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas import MemoryStats, L3GraphData
from backend.core.namespace import sentinel_ns

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats():
    """Return usage statistics for all memory layers."""
    if not sentinel_ns.is_initialized:
        return MemoryStats(
            l2={"count": 0, "pct": 0},
            l3={"nodes": 0, "edges": 0},
            l4={"count": 0, "pct": 0},
        )
    stats = sentinel_ns.get_memory_stats()
    L2_CAPACITY = 1000
    L4_CAPACITY = 100
    return MemoryStats(
        l2={
            "count": stats["l2_count"],
            "capacity": L2_CAPACITY,
            "pct": round(stats["l2_count"] / L2_CAPACITY * 100, 1),
        },
        l3={
            "nodes": stats["l3_nodes"],
            "edges": stats["l3_edges"],
        },
        l4={
            "count": stats["l4_count"],
            "capacity": L4_CAPACITY,
            "pct": round(stats["l4_count"] / L4_CAPACITY * 100, 1),
        },
    )


@router.get("/layer/l2")
async def get_l2_layer(dataset: Optional[str] = Query(None)):
    """Return all L2 episodic memory entries, optionally filtered by dataset."""
    if not sentinel_ns.is_initialized:
        return {"episodes": []}
    return {"episodes": sentinel_ns.get_l2_episodes(dataset=dataset)}


@router.get("/layer/l4")
async def get_l4_layer():
    """Return all L4 procedural SQL patterns."""
    if not sentinel_ns.is_initialized:
        return {"patterns": []}
    return {"patterns": sentinel_ns.get_l4_patterns()}


@router.get("/layer/l3", response_model=L3GraphData)
async def get_l3_layer():
    """Return L3 causal graph as node/edge lists for frontend visualisation."""
    if not sentinel_ns.is_initialized:
        return L3GraphData(nodes=[], edges=[])
    data = sentinel_ns.get_l3_graph_data()
    return L3GraphData(**data)
