"""
Query execution route.

POST /api/query  — run an analytics query through the LangGraph pipeline
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from backend.api.schemas import QueryRequest, QueryResponse, ChartItem
from backend.core.namespace import sentinel_ns
from backend.agents.chart_request_agent import generate_requested_charts, detect_chart_request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


# ---------------------------------------------------------------------------
# Python code-block extraction and chart execution
# ---------------------------------------------------------------------------
_PY_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
# Matches [label](data:image/png;base64,...)  OR  bare data:image/... URIs
_IMG_DATA_RE = re.compile(
    r"(?:\[[^\]]*\]\(data:image/[^)]+\)|data:image/[A-Za-z]+;base64,[A-Za-z0-9+/=]{20,})",
    re.DOTALL,
)


def _extract_python_blocks(text: str):
    """
    Find all ```python ... ``` fenced blocks in `text`.
    Returns (blocks: list[str], cleaned_text: str) where cleaned_text has the
    code blocks replaced by a small placeholder pill.
    """
    blocks = []
    def _replacer(m):
        code = m.group(1).strip()
        if code:
            blocks.append(code)
        return "\n> ⚡ *Interactive chart rendered above*\n"
    cleaned = _PY_BLOCK_RE.sub(_replacer, text)
    return blocks, cleaned


def _strip_image_data(text: str) -> str:
    """Remove base64 image data URIs left by matplotlib print() calls."""
    return _IMG_DATA_RE.sub("", text).strip()


def _json_safe(obj: Any) -> Any:
    """Recursively make an object JSON-serializable (handles numpy/pandas types)."""
    import datetime
    try:
        import numpy as np
        import pandas as pd
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
    except ImportError:
        pass
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _safe_dict(obj) -> Optional[Dict]:
    """Convert a result dict to JSON-safe form, handling numpy/pandas types."""
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        cleaned = _json_safe(obj)
        json.dumps(cleaned)  # verify
        return cleaned
    except Exception as exc:
        logger.warning("Could not serialize result dict: %s", exc)
        # Attempt field-by-field salvage
        salvaged = {}
        for k, v in obj.items():
            try:
                cleaned_v = _json_safe(v)
                json.dumps(cleaned_v)
                salvaged[k] = cleaned_v
            except Exception:
                salvaged[k] = str(v)[:300]
        return salvaged


def _parse_final_response(text: str, last_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull typed fields from last_state and split final_response into
    insight text vs chart-analysis section.

    Priority for chart_explanations:
      1. last_state["chart_explanations"]  — set directly by viz_agent (most reliable)
      2. Text after the ─── separator in final_response  — for RCA/anomaly/math agents
         that embed their chart analysis in final_response

    Priority for insights:
      1. Text BEFORE the ─── separator (the narrative summary)
      2. Full final_response text if no separator found
    """
    sql = last_state.get("sql_query", "") or ""
    aqp_ci = None
    sql_result_preview = None

    aqp_raw = last_state.get("aqp_ci", "")
    if aqp_raw:
        try:
            aqp_ci = json.loads(aqp_raw)
        except Exception:
            pass

    sql_result_raw = last_state.get("sql_result_json", "")
    if sql_result_raw:
        try:
            rows = json.loads(sql_result_raw)
            sql_result_preview = rows[:10] if isinstance(rows, list) else None
        except Exception:
            pass

    # ── 1. chart_explanations — use state directly (primary) ───────────
    # viz_agent writes "chart_explanations" to the LangGraph state.
    # call_ask() also supplements last_state from its own viz_agent capture,
    # so this should always be available for SQL queries even if
    # sql_response_writer fails.
    chart_explanations: str = last_state.get("chart_explanations", "") or ""

    # ── 2. insights — narrative from final_response ─────────────────────
    insights: str = text or ""

    if text:
        # Agents embed chart analysis in final_response after a ─── separator.
        # Strip it so insights contains only the narrative portion.
        sep_match = re.search(r'\n─{3,}\n', text)
        if sep_match:
            insights = text[:sep_match.start()].strip()
            # For agents that DON'T write state["chart_explanations"]
            # (RCA, anomaly, math), parse chart analysis from the text too.
            if not chart_explanations:
                rest = text[sep_match.end():].strip()
                chart_explanations = re.sub(
                    r'^(?:📊\s*)?(?:Chart-by-chart analysis[^:\n]*|Chart Analysis):?\s*',
                    '',
                    rest,
                    flags=re.IGNORECASE,
                ).strip()

    logger.debug(
        "_parse_final_response: insights=%d chars, chart_explanations=%d chars, sql=%d chars",
        len(insights), len(chart_explanations), len(sql),
    )

    return {
        "insights": insights,
        "chart_explanations": chart_explanations,
        "sql": sql if sql else None,
        "aqp_ci": aqp_ci,
        "sql_result_preview": sql_result_preview,
        "intent": last_state.get("intent", "sql_query"),
        "rca_result": last_state.get("rca_result"),
        "forecast_result": last_state.get("forecast_result"),
        "anomaly_result": last_state.get("anomaly_result"),
        "math_result": last_state.get("math_result"),
    }


def _extract_memory_info(last_state: Dict) -> Dict[str, Any]:
    memory_ctx = last_state.get("memory_context", "")
    cache_hit = bool(memory_ctx and "L2" in memory_ctx and len(memory_ctx) > 100)
    return {
        "cache_hit": cache_hit,
        "source": "L2 episodic" if cache_hit else "none",
        "context_length": len(memory_ctx),
    }


@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    """Execute an analytics query through the SENTINEL multi-agent pipeline."""
    if not sentinel_ns.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="System not initialised. Please configure your API provider first.",
        )

    # Resolve dataset tables for schema-narrowing (dataset-aware routing)
    dataset_tables: Optional[List[str]] = None
    dataset_name: str = ""
    if req.dataset:
        registry = sentinel_ns.get_dataset_registry()
        info = registry.get(req.dataset)
        if info:
            dataset_tables = info.get("tables", [])
            dataset_name = req.dataset
            logger.info(
                "Dataset-aware routing: dataset=%s tables=%s",
                dataset_name, dataset_tables,
            )

    try:
        raw = await sentinel_ns.call_ask(
            req.query,
            dataset_tables=dataset_tables,
            dataset_name=dataset_name,
        )
    except Exception as exc:
        logger.error("call_ask failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)[:500]}")

    final_response: str = raw.get("final_response", "")
    fig_queue: List = raw.get("fig_queue", [])
    last_state: Dict = raw.get("last_state", {})
    duration_ms: int = raw.get("duration_ms", 0)
    error: Optional[str] = raw.get("error")

    # ── Extract & execute Python code blocks from the response ──────────
    # These are ```python ... ``` blocks the LLM generates (e.g. matplotlib).
    # We run them through exec_python_block() to convert to interactive Plotly.
    py_code_blocks, final_response = _extract_python_blocks(final_response)
    extra_charts: List[ChartItem] = []
    if py_code_blocks:
        for code_block in py_code_blocks:
            try:
                block_charts = sentinel_ns.exec_python_block(code_block)
                for title, html in block_charts:
                    if html:
                        extra_charts.append(ChartItem(
                            title=str(title) or "Chart",
                            html=str(html),
                        ))
            except Exception as exc:
                logger.warning("exec_python_block raised: %s", exc)
    # Strip any residual base64 image data URIs from the response text
    final_response = _strip_image_data(final_response)

    # ── Generate explicitly-requested chart type ──────────────────────────
    # If the user said "show a heatmap", "plot a scatter", etc., generate it
    # from the SQL result (or fetch fresh data) regardless of which agent ran.
    if detect_chart_request(req.query):
        try:
            ns = sentinel_ns._ns  # shared namespace
            requested = generate_requested_charts(
                query=req.query,
                sql_result_json=last_state.get("sql_result_json", ""),
                run_sql_fn=ns.get("run_sql"),
                schema=ns.get("SCHEMA", ""),
                call_llm_fn=ns.get("call_llm"),
                fast_model=ns.get("FAST_MODEL", ""),
            )
            for title, html in requested:
                if html:
                    extra_charts.append(ChartItem(
                        title=str(title) or "Requested Chart",
                        html=str(html),
                    ))
                    logger.info("ChartRequestAgent added: %s", title)
        except Exception as exc:
            logger.warning("ChartRequestAgent error: %s", exc)

    # Build chart items
    charts: List[ChartItem] = []
    for item in fig_queue:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            html_str, title = item[0], item[1]
        elif isinstance(item, dict):
            html_str = item.get("html", "")
            title = item.get("title", "Chart")
        else:
            continue
        if html_str:
            charts.append(ChartItem(
                title=str(title) if title else "Chart",
                html=str(html_str),
            ))

    parsed = _parse_final_response(final_response, last_state)
    memory_info = _extract_memory_info(last_state)

    return QueryResponse(
        query=req.query,
        intent=parsed["intent"],
        sql=parsed["sql"],
        sql_result_preview=parsed["sql_result_preview"],
        aqp_ci=parsed["aqp_ci"],
        charts=charts + extra_charts,
        insights=parsed["insights"],
        chart_explanations=parsed["chart_explanations"],
        rca_result=_safe_dict(parsed["rca_result"]),
        forecast_result=_safe_dict(parsed["forecast_result"]),
        anomaly_result=_safe_dict(parsed["anomaly_result"]),
        math_result=_safe_dict(parsed["math_result"]),
        memory_info=memory_info,
        error=error,
        duration_ms=duration_ms,
    )
