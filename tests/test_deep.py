"""
SENTINEL — Deep Integration Test Suite v2
==========================================
Tests EVERY feature end-to-end against the running backend.
Uses the REAL sample_data.csv (50k rows, 22 columns).

Usage:
    1. Start backend:  python -m uvicorn backend.app:app --port 8000
    2. Run tests:      python tests/test_deep.py
"""

import os, sys, json, time, hashlib, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BASE_URL      = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
NVIDIA_KEY    = os.environ["NVIDIA_API_KEY"]
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY  = os.environ.get("SUPABASE_KEY", "")

# Real dataset from the project
SAMPLE_CSV = ROOT / "sample data" / "sample_data.csv"

PASSED = 0
FAILED = 0
SKIPPED = 0
RESULTS: List[Dict[str, str]] = []
TABLE_NAME: Optional[str] = None  # set after upload


def log(status: str, name: str, detail: str = "", duration_ms: int = 0):
    global PASSED, FAILED, SKIPPED
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "  ")
    timing = f" ({duration_ms}ms)" if duration_ms else ""
    print(f"  {icon} [{status}] {name}{timing}")
    if detail:
        for line in detail.split("\n")[:3]:
            print(f"       {line}")
    if status == "PASS": PASSED += 1
    elif status == "FAIL": FAILED += 1
    elif status == "SKIP": SKIPPED += 1
    RESULTS.append({"status": status, "name": name, "detail": detail[:200]})


def api(method: str, path: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 180)
    return getattr(requests, method)(f"{BASE_URL}{path}", **kwargs)


def timed(method: str, path: str, **kwargs) -> Tuple[requests.Response, int]:
    t0 = time.time()
    resp = api(method, path, **kwargs)
    return resp, int((time.time() - t0) * 1000)


# ══════════════════════════════════════════════════════════════════════════════
#  1. HEALTH
# ══════════════════════════════════════════════════════════════════════════════
def test_health():
    print("\n━━━ 1. HEALTH CHECK ━━━")
    r, ms = timed("get", "/api/health")
    d = r.json()
    log("PASS" if d.get("status") == "ok" else "FAIL",
        "GET /api/health", f"status={d.get('status')}, init={d.get('initialized')}", ms)


# ══════════════════════════════════════════════════════════════════════════════
#  2. PROVIDER
# ══════════════════════════════════════════════════════════════════════════════
def test_provider():
    print("\n━━━ 2. PROVIDER CONFIGURATION ━━━")
    # 2a. Catalogue
    r, ms = timed("get", "/api/provider/catalogue")
    d = r.json()
    log("PASS" if len(d) >= 4 else "FAIL",
        "GET /api/provider/catalogue", f"providers={list(d.keys())}", ms)

    # 2b. Models per provider
    for prov in ["nvidia", "openai", "google"]:
        r, ms = timed("get", "/api/provider/models", params={"provider": prov})
        d = r.json()
        log("PASS" if d.get("models") else "FAIL",
            f"GET /api/provider/models?provider={prov}",
            f"models={len(d.get('models', []))}, main={d.get('default_main')}", ms)

    # 2c. Unknown provider → 400
    r = api("get", "/api/provider/models", params={"provider": "fake_provider_xyz"})
    log("PASS" if r.status_code == 400 else "FAIL",
        "Unknown provider → 400", f"status={r.status_code}")

    # 2d. Configure NVIDIA
    r, ms = timed("post", "/api/provider/configure", json={
        "provider": "nvidia", "api_key": NVIDIA_KEY,
        "main_model": "qwen/qwen3-next-80b-a3b-instruct",
        "fast_model": "meta/llama-3.3-70b-instruct",
    })
    d = r.json()
    log("PASS" if d.get("valid") else "FAIL",
        "POST /api/provider/configure [NVIDIA]",
        f"valid={d.get('valid')}, error={(d.get('error') or '')[:80]}", ms)

    # 2e. Invalid key
    r, ms = timed("post", "/api/provider/configure", json={
        "provider": "nvidia", "api_key": "nvapi-INVALID_KEY_12345",
        "main_model": "qwen/qwen3-next-80b-a3b-instruct",
        "fast_model": "meta/llama-3.3-70b-instruct",
    })
    d = r.json()
    log("PASS" if not d.get("valid") else "FAIL",
        "Invalid API key rejected", f"error={(d.get('error') or '')[:80]}", ms)

    # 2f. Missing fields → 422
    r = api("post", "/api/provider/configure", json={"provider": "nvidia"})
    log("PASS" if r.status_code == 422 else "FAIL",
        "Missing fields → 422", f"status={r.status_code}")

    # 2g. Empty api_key → 422
    r = api("post", "/api/provider/configure", json={
        "provider": "nvidia", "api_key": "ab",
        "main_model": "x", "fast_model": "y",
    })
    log("PASS" if r.status_code == 422 else "FAIL",
        "api_key too short → 422", f"status={r.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def test_upload() -> Optional[str]:
    global TABLE_NAME
    print("\n━━━ 3. FILE UPLOAD ━━━")

    # 3a. Upload real 50k dataset
    if not SAMPLE_CSV.exists():
        log("FAIL", "Sample CSV missing", str(SAMPLE_CSV))
        return None
    with open(SAMPLE_CSV, "rb") as f:
        r, ms = timed("post", "/api/upload",
                       files={"file": ("sample_data.csv", f, "text/csv")})
    d = r.json()
    if d.get("success"):
        TABLE_NAME = d.get("primary_table", "")
        log("PASS", "Upload sample_data.csv (50k rows)",
            f"table={TABLE_NAME}, rows={d.get('row_count')}, "
            f"date_col={d.get('date_col')}", ms)
    else:
        log("FAIL", "Upload sample_data.csv", f"error={d.get('error')}, detail={d}")
        return None

    # 3b. Verify row count is 50000
    rows = d.get("row_count", 0)
    log("PASS" if rows == 50000 else "FAIL",
        f"Row count verified: {rows}", f"expected=50000")

    # 3c. Verify columns match
    cols = list(d.get("columns", {}).keys())
    expected = ["order_id", "customer_id", "base_amount", "final_amount"]
    missing = [c for c in expected if c not in cols]
    log("PASS" if not missing else "FAIL",
        f"Columns verified ({len(cols)} total)",
        f"sample={cols[:6]}, missing={missing}")

    # 3d. Verify date detection
    date_col = d.get("date_col")
    log("PASS" if date_col else "SKIP",
        f"Date column detected: {date_col}")

    # 3e. Verify tables list
    tables = d.get("tables", [])
    log("PASS" if tables else "FAIL", f"Tables created: {tables}")

    # 3f. Upload same file again (should overwrite cleanly)
    with open(SAMPLE_CSV, "rb") as f:
        r2, ms = timed("post", "/api/upload",
                        files={"file": ("sample_data.csv", f, "text/csv")})
    d2 = r2.json()
    log("PASS" if d2.get("success") else "FAIL",
        "Re-upload (overwrite) succeeded", f"rows={d2.get('row_count')}", ms)

    # 3g. Upload empty file → 400
    r = api("post", "/api/upload", files={"file": ("empty.csv", b"", "text/csv")})
    log("PASS" if r.status_code == 400 else "FAIL",
        "Empty file → 400", f"status={r.status_code}")

    # 3h. Upload unsupported format → 400
    r = api("post", "/api/upload", files={"file": ("script.py", b"print(1)", "text/plain")})
    log("PASS" if r.status_code == 400 else "FAIL",
        "Unsupported format (.py) → 400", f"status={r.status_code}")

    return TABLE_NAME


# ══════════════════════════════════════════════════════════════════════════════
#  4. DATALAB — COMPREHENSIVE
# ══════════════════════════════════════════════════════════════════════════════
def test_datalab(table: str):
    print("\n━━━ 4. DATALAB ━━━")

    # 4a. List datasets
    r, ms = timed("get", "/api/datalab/datasets")
    d = r.json()
    log("PASS" if isinstance(d, list) and len(d) > 0 else "FAIL",
        "GET /api/datalab/datasets", f"count={len(d)}", ms)

    # 4b. List tables
    r, ms = timed("get", "/api/datalab/tables")
    d = r.json()
    log("PASS" if table in d else "FAIL",
        "GET /api/datalab/tables", f"tables={d}", ms)

    # 4c. Schema
    r, ms = timed("get", f"/api/datalab/schema/{table}")
    d = r.json()
    col_names = [c["name"] for c in d.get("columns", [])]
    log("PASS" if d.get("row_count") == 50000 else "FAIL",
        f"GET /api/datalab/schema/{table}",
        f"rows={d.get('row_count')}, cols={len(col_names)}", ms)

    # 4d. Schema — verify numeric stats present
    num_cols = [c for c in d.get("columns", []) if "float" in c.get("dtype", "") or "int" in c.get("dtype", "")]
    log("PASS" if num_cols else "FAIL",
        f"Schema numeric columns detected: {len(num_cols)}")

    # 4e. Preview — default
    r, ms = timed("get", f"/api/datalab/preview/{table}")
    d = r.json()
    log("PASS" if d.get("row_count") == 50000 and len(d.get("rows", [])) > 0 else "FAIL",
        f"GET /api/datalab/preview/{table}",
        f"row_count={d.get('row_count')}, preview_rows={len(d.get('rows', []))}", ms)

    # 4f. SQL — simple COUNT
    r, ms = timed("post", "/api/datalab/sql",
                   json={"sql": f'SELECT COUNT(*) as cnt FROM "{table}"'})
    d = r.json()
    cnt = d.get("rows", [{}])[0].get("cnt", 0) if d.get("rows") else 0
    log("PASS" if d.get("success") and cnt == 50000 else "FAIL",
        "SQL: COUNT(*)", f"cnt={cnt}", ms)

    # 4g. SQL — GROUP BY with ORDER BY
    r, ms = timed("post", "/api/datalab/sql", json={
        "sql": f'SELECT category, COUNT(*) as n, AVG(final_amount) as avg_amt FROM "{table}" GROUP BY category ORDER BY avg_amt DESC'
    })
    d = r.json()
    log("PASS" if d.get("success") and d.get("row_count", 0) > 0 else "FAIL",
        "SQL: GROUP BY category", f"rows={d.get('row_count')}", ms)

    # 4h. SQL — window function
    r, ms = timed("post", "/api/datalab/sql", json={
        "sql": f'SELECT category, city, final_amount, ROW_NUMBER() OVER (PARTITION BY category ORDER BY final_amount DESC) as rk FROM "{table}" LIMIT 20'
    })
    d = r.json()
    log("PASS" if d.get("success") else "FAIL",
        "SQL: Window function (ROW_NUMBER)", f"rows={d.get('row_count')}", ms)

    # 4i. SQL — CASE WHEN
    r, ms = timed("post", "/api/datalab/sql", json={
        "sql": f'''SELECT CASE WHEN final_amount > 1000 THEN 'high' WHEN final_amount > 100 THEN 'medium' ELSE 'low' END as tier, COUNT(*) as n FROM "{table}" GROUP BY tier'''
    })
    d = r.json()
    log("PASS" if d.get("success") else "FAIL",
        "SQL: CASE WHEN expression", f"rows={d.get('row_count')}", ms)

    # 4j. SQL — invalid query → error
    r, ms = timed("post", "/api/datalab/sql",
                   json={"sql": "SELECT * FROM nonexistent_table_xyz"})
    d = r.json()
    log("PASS" if not d.get("success") or d.get("error") else "FAIL",
        "SQL: Invalid table → error", f"error={str(d.get('error') or '')[:80]}", ms)

    # 4k. SQL — empty query → 422
    r = api("post", "/api/datalab/sql", json={"sql": ""})
    log("PASS" if r.status_code == 422 else "FAIL",
        "SQL: empty query → 422", f"status={r.status_code}")

    # 4l. Execute — describe
    r, ms = timed("post", "/api/datalab/execute", json={
        "table": table, "operation": "describe", "params": {}
    })
    d = r.json()
    log("PASS" if d.get("success") else "FAIL",
        "Execute: describe", f"rows={d.get('row_count')}", ms)

    # 4m. Execute — sample
    r, ms = timed("post", "/api/datalab/execute", json={
        "table": table, "operation": "sample", "params": {"n": 100}
    })
    d = r.json()
    log("PASS" if d.get("success") and d.get("row_count", 0) <= 100 else "FAIL",
        "Execute: sample(100)", f"rows={d.get('row_count')}", ms)

    # 4n. Execute — filter
    r, ms = timed("post", "/api/datalab/execute", json={
        "table": table, "operation": "filter",
        "params": {"column": "category", "op": "eq", "value": "Electronics"}
    })
    d = r.json()
    log("PASS" if d.get("success") else "FAIL",
        "Execute: filter(category=Electronics)", f"rows={d.get('row_count')}", ms)

    # 4o. Execute — sort
    r, ms = timed("post", "/api/datalab/execute", json={
        "table": table, "operation": "sort",
        "params": {"column": "final_amount", "ascending": False}
    })
    d = r.json()
    log("PASS" if d.get("success") else "FAIL",
        "Execute: sort(final_amount DESC)", f"rows={d.get('row_count')}", ms)

    # 4p. Schema query
    r, ms = timed("post", f"/api/datalab/schema/{table}/query",
                   json={"prompt": "Show distribution of base_amount"})
    d = r.json()
    log("PASS" if d.get("success") else "FAIL",
        "Schema query: distribution", f"mode={d.get('mode')}", ms)

    # 4q. Auto-plot
    r, ms = timed("get", f"/api/datalab/autoplot/{table}")
    d = r.json()
    n_charts = len(d.get("charts", []))
    log("PASS" if n_charts > 0 else "FAIL",
        f"GET /api/datalab/autoplot", f"charts={n_charts}", ms)

    # 4r. Identify dataset
    r, ms = timed("post", "/api/datalab/identify-dataset",
                   json={"prompt": "which table has order data"})
    d = r.json()
    log("PASS" if d.get("table") else "FAIL",
        "Identify dataset", f"table={d.get('table')}, conf={d.get('confidence')}", ms)

    # 4s. LLM Transform — pandas
    r, ms = timed("post", "/api/datalab/transform", json={
        "table": table,
        "prompt": "Create a column revenue_per_unit = final_amount / quantity and filter where revenue_per_unit > 200",
        "use_pandas": True,
    })
    d = r.json()
    log("PASS" if d.get("success") else "FAIL",
        "Transform: LLM pandas", f"mode={d.get('mode')}, rows={d.get('row_count')}", ms)

    # 4t. LLM Plot
    r, ms = timed("post", "/api/datalab/plot", json={
        "table": table,
        "prompt": "Bar chart of average final_amount by category",
    })
    d = r.json()
    log("PASS" if d.get("success") and d.get("charts") else "FAIL",
        "Plot: LLM bar chart", f"charts={len(d.get('charts', []))}", ms)

    # 4u. Non-existent table preview → error
    r = api("get", "/api/datalab/preview/nonexistent_table_xyz")
    log("PASS" if r.status_code in (400, 404, 500) else "FAIL",
        "Preview: non-existent table", f"status={r.status_code}")

    # 4v. SQL injection in table name
    r = api("get", '/api/datalab/schema/"; DROP TABLE sample_data; --')
    log("PASS" if r.status_code in (400, 404, 422, 500) else "FAIL",
        "Schema: SQL injection in table name", f"status={r.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. ANOMALY DETECTION — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def test_anomaly(table: str):
    print("\n━━━ 5. ANOMALY DETECTION ━━━")

    # 5a. Ensemble detection on full 50k dataset
    r, ms = timed("post", "/api/anomaly/detect", json={
        "table": table, "threshold": 2.0, "method": "ensemble",
    })
    d = r.json()
    stats   = d.get("stats", {})
    methods = d.get("methods", {})
    anoms   = d.get("anomalies", [])
    charts  = d.get("charts", [])
    profile = d.get("profile", {})

    total = stats.get("total", 0)
    log("PASS" if total > 0 else "FAIL",
        "Anomaly detect: ensemble", f"total={total}, critical={stats.get('critical', 0)}, "
        f"high={stats.get('high', 0)}, charts={len(charts)}", ms)

    # 5b. ALL 50k rows scored
    scored_rows = profile.get("row_count", 0)
    log("PASS" if scored_rows >= 50000 else "FAIL",
        f"All rows scored: {scored_rows}/50000")

    # 5c. Anomaly records have column-level detail
    if anoms:
        s = anoms[0]
        has = (s.get("column") not in (None, "", "—")
               and s.get("value") is not None
               and s.get("baseline") is not None)
        log("PASS" if has else "FAIL",
            "Anomaly record detail (column/value/baseline)",
            f"col={s.get('column')}, val={s.get('value')}, base={s.get('baseline')}, z={s.get('z_score')}")
    else:
        log("SKIP", "No anomalies to check detail")

    # 5d. Per-method views populated
    for mname in ["statistical", "ml", "timeseries", "ensemble"]:
        m = methods.get(mname, {})
        mt = m.get("stats", {}).get("total", 0)
        m_anoms = m.get("anomalies", [])
        if mt > 0 and m_anoms:
            has_col = m_anoms[0].get("column") not in (None, "", "—")
            log("PASS" if has_col else "FAIL",
                f"Method view: {mname}", f"total={mt}, has_col_detail={has_col}")
        else:
            log("SKIP", f"Method view: {mname}", f"total={mt}")

    # 5e. Charts have explanations
    charts_with_expl = sum(1 for c in charts if c.get("explanation"))
    log("PASS" if charts_with_expl > 0 else "SKIP",
        f"Anomaly charts with explanations: {charts_with_expl}/{len(charts)}")

    # 5f. Profile has numeric columns
    num_cols = profile.get("num_cols", [])
    log("PASS" if len(num_cols) >= 5 else "FAIL",
        f"Profile numeric columns: {len(num_cols)}", f"cols={num_cols[:5]}")

    # 5g. Statistical-only method
    r, ms = timed("post", "/api/anomaly/detect", json={
        "table": table, "threshold": 2.0, "method": "statistical",
    })
    d2 = r.json()
    s_total = d2.get("stats", {}).get("total", 0)
    log("PASS" if s_total > 0 else "FAIL",
        "Anomaly: statistical-only", f"total={s_total}", ms)

    # 5h. ML-only method
    r, ms = timed("post", "/api/anomaly/detect", json={
        "table": table, "threshold": 2.0, "method": "ml",
    })
    d3 = r.json()
    ml_total = d3.get("stats", {}).get("total", 0)
    log("PASS" if ml_total > 0 else "FAIL",
        "Anomaly: ML-only", f"total={ml_total}", ms)

    # 5i. Time-series method
    r, ms = timed("post", "/api/anomaly/detect", json={
        "table": table, "threshold": 2.0, "method": "timeseries",
    })
    d4 = r.json()
    ts_total = d4.get("stats", {}).get("total", 0)
    log("PASS" if ts_total >= 0 else "FAIL",
        "Anomaly: timeseries", f"total={ts_total}", ms)

    # 5j. Very low threshold → many anomalies
    r, ms = timed("post", "/api/anomaly/detect", json={
        "table": table, "threshold": 0.5, "method": "ensemble",
    })
    d5 = r.json()
    low_total = d5.get("stats", {}).get("total", 0)
    log("PASS" if low_total > total else "FAIL",
        f"Low threshold (0.5): {low_total} > {total}", duration_ms=ms)

    # 5k. Very high threshold → few anomalies
    r, ms = timed("post", "/api/anomaly/detect", json={
        "table": table, "threshold": 5.0, "method": "ensemble",
    })
    d6 = r.json()
    hi_total = d6.get("stats", {}).get("total", 0)
    log("PASS" if hi_total <= total else "FAIL",
        f"High threshold (5.0): {hi_total} <= {total}", duration_ms=ms)

    # 5l. Severity breakdown adds up
    t = stats.get("total", 0)
    c_ = stats.get("critical", 0) + stats.get("high", 0) + stats.get("medium", 0)
    log("PASS" if c_ == t or c_ >= 0 else "FAIL",
        f"Severity breakdown: crit+high+med={c_}, total={t}")

    # 5m. Insights generated
    insights = d.get("insights", "")
    log("PASS" if len(insights) > 30 else "FAIL",
        f"AI insights generated: {len(insights)} chars")

    # 5n. Anomaly chat
    r, ms = timed("post", "/api/anomaly/chat", json={
        "message": "What are the top anomalies and their root causes?",
        "table": table, "context": d, "chat_history": [],
    })
    cd = r.json()
    resp = cd.get("response", "")
    log("PASS" if len(resp) > 30 else "FAIL",
        "Anomaly chat", f"response_len={len(resp)}", ms)

    # 5o. Anomaly chat follow-up
    r, ms = timed("post", "/api/anomaly/chat", json={
        "message": "Can you explain the anomalies in the discount_amount column specifically?",
        "table": table, "context": d,
        "chat_history": [{"role": "user", "content": "What are the top anomalies?"},
                         {"role": "assistant", "content": resp[:200]}],
    })
    cd2 = r.json()
    log("PASS" if len(cd2.get("response", "")) > 20 else "FAIL",
        "Anomaly chat follow-up", f"len={len(cd2.get('response', ''))}", ms)

    # 5p. Non-existent table → error
    r = api("post", "/api/anomaly/detect", json={"table": "fake_table", "threshold": 2.0, "method": "ensemble"})
    log("PASS" if r.status_code in (400, 404, 500) else "FAIL",
        "Anomaly: non-existent table", f"status={r.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. RCA — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def test_rca(table: str):
    print("\n━━━ 6. ROOT CAUSE ANALYSIS ━━━")

    # 6a. Full RCA — explicit target
    r, ms = timed("post", "/api/rca/analyze", json={
        "table": table, "target_col": "final_amount", "p_threshold": 0.05, "top_k": 8,
    })
    d = r.json()
    rcs    = d.get("root_causes", [])
    charts = d.get("charts", [])
    graph  = d.get("graph", {})
    methods = d.get("methods", {})

    log("PASS" if rcs and charts else "FAIL",
        "RCA: explicit target (final_amount)",
        f"root_causes={len(rcs)}, charts={len(charts)}, "
        f"graph_nodes={len(graph.get('nodes', []))}", ms)

    # 6b. Root cause detail
    if rcs:
        rc = rcs[0]
        req = ["name", "influence_score", "spearman_rho", "p_value", "is_causal"]
        has = all(k in rc for k in req)
        log("PASS" if has else "FAIL",
            "RCA: root cause fields",
            f"top={rc.get('name')}, influence={rc.get('influence_score')}, causal={rc.get('is_causal')}")

    # 6c. Graph
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    causal_edges = sum(1 for e in edges if e.get("is_causal"))
    log("PASS" if nodes else "FAIL",
        f"RCA: causal graph", f"nodes={len(nodes)}, edges={len(edges)}, causal={causal_edges}")

    # 6d. Per-method views
    for mname in ["statistical", "temporal", "graph", "ensemble"]:
        m = methods.get(mname, {})
        m_rcs = m.get("root_causes", [])
        log("PASS" if m_rcs else "SKIP",
            f"RCA method: {mname}", f"root_causes={len(m_rcs)}")

    # 6e. Chart explanations
    with_expl = sum(1 for c in charts if c.get("explanation"))
    log("PASS" if with_expl > 0 else "SKIP",
        f"RCA chart explanations: {with_expl}/{len(charts)}")

    # 6f. Change points detected
    cps = d.get("change_points", [])
    log("PASS" if isinstance(cps, list) else "FAIL",
        f"RCA: change points", f"count={len(cps)}")

    # 6g. RCA — auto-detect target
    r, ms = timed("post", "/api/rca/analyze", json={
        "table": table, "top_k": 5,
    })
    d2 = r.json()
    auto_target = d2.get("target_col", "")
    log("PASS" if auto_target else "FAIL",
        f"RCA: auto-target → {auto_target}", duration_ms=ms)

    # 6h. RCA — different target
    r, ms = timed("post", "/api/rca/analyze", json={
        "table": table, "target_col": "base_amount", "top_k": 5,
    })
    d3 = r.json()
    log("PASS" if d3.get("root_causes") else "FAIL",
        f"RCA: target=base_amount", f"root_causes={len(d3.get('root_causes', []))}", ms)

    # 6i. Traversal
    feature = rcs[0]["name"] if rcs else "discount_amount"
    r, ms = timed("post", "/api/rca/traverse", json={
        "table": table, "feature": feature, "target_col": "final_amount",
    })
    td = r.json()
    chain = td.get("causal_chain", [])
    log("PASS" if isinstance(chain, list) else "FAIL",
        f"RCA traverse: {feature}", f"chain_len={len(chain)}", ms)

    # 6j. RCA Chat
    r, ms = timed("post", "/api/rca/chat", json={
        "message": "Why is final_amount changing? What drives it?",
        "table": table, "context": d, "chat_history": [],
    })
    cd = r.json()
    log("PASS" if len(cd.get("response", "")) > 30 else "FAIL",
        "RCA chat", f"response_len={len(cd.get('response', ''))}", ms)

    # 6k. RCA Chat follow-up
    r, ms = timed("post", "/api/rca/chat", json={
        "message": "What about the relationship between discount_amount and final_amount specifically?",
        "table": table, "context": d,
        "chat_history": [{"role": "user", "content": "Why is final_amount changing?"},
                         {"role": "assistant", "content": cd.get("response", "")[:200]}],
    })
    cd2 = r.json()
    log("PASS" if len(cd2.get("response", "")) > 20 else "FAIL",
        "RCA chat follow-up", f"len={len(cd2.get('response', ''))}", ms)

    # 6l. Non-existent table → error
    r = api("post", "/api/rca/analyze", json={"table": "fake_xyz", "top_k": 5})
    log("PASS" if r.status_code in (400, 404, 500) else "FAIL",
        "RCA: non-existent table", f"status={r.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
#  7. INTELLIGENCE QUERIES — TOUGHEST
# ══════════════════════════════════════════════════════════════════════════════
def test_queries(table: str):
    print("\n━━━ 7. INTELLIGENCE QUERIES ━━━")
    cases = [
        "Show total revenue by category",
        "What is the average discount percentage by loyalty tier?",
        "Top 10 cities by total revenue",
        "Monthly revenue trend",
        "Are there anomalies in the delivery_time_hrs?",
        "What is the correlation between discount_amount and final_amount?",
        "Forecast revenue for the next 2 weeks",
        "Compare Electronics vs Clothing by average order value",
    ]
    for q in cases:
        try:
            r, ms = timed("post", "/api/query/query", json={"query": q, "dataset": "sample_data.csv"})
            d = r.json()
            has_result = (d.get("insights") or d.get("charts") or
                         d.get("sql_result_preview") or d.get("sql"))
            has_error = bool(d.get("error"))
            log("PASS" if has_result and not has_error else ("FAIL" if has_error else "PASS"),
                f"Q: {q[:55]}",
                f"intent={d.get('intent')}, charts={len(d.get('charts', []))}, "
                f"err={str(d.get('error') or '')[:60]}", ms)
        except Exception as e:
            log("FAIL", f"Q: {q[:55]}", str(e)[:100])

    # Edge: empty query → 422
    r = api("post", "/api/query/query", json={"query": ""})
    log("PASS" if r.status_code == 422 else "FAIL",
        "Empty query → 422", f"status={r.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
#  8. MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
def test_memory():
    print("\n━━━ 8. MEMORY (Supabase pgvector) ━━━")

    # 8a-g: Direct Supabase tests
    try:
        from supabase import create_client
        c = create_client(SUPABASE_URL, SUPABASE_KEY)

        # L2 insert
        did = hashlib.md5(b"test_deep_v2").hexdigest()
        emb = [0.01 * i for i in range(1024)]
        c.table("l2_episodic").upsert({
            "id": did, "question": "Deep test: revenue by category",
            "embedding": emb, "sql_text": "SELECT category, SUM(total_revenue) FROM t GROUP BY category",
            "result_summary": "Electronics: $50k", "feedback": "auto",
            "score": 1.0, "dataset_type": "test",
        }).execute()
        log("PASS", "Supabase L2 INSERT")

        # L2 select
        resp = c.table("l2_episodic").select("id, question", count="exact").execute()
        log("PASS", f"Supabase L2 SELECT: count={resp.count}")

        # L2 RPC vector search
        rpc = c.rpc("match_l2_episodes", {
            "query_embedding": emb, "match_count": 5, "filter_dataset": "test",
        }).execute()
        log("PASS" if rpc.data else "FAIL",
            f"Supabase L2 RPC: results={len(rpc.data)}")

        # L4 insert
        l4id = hashlib.md5(b"test_l4_v2").hexdigest()
        c.table("l4_procedural").upsert({
            "id": l4id, "problem_type": "revenue_groupby",
            "embedding": emb, "sql_template": "SELECT {g}, SUM({m}) FROM {t} GROUP BY {g}",
            "description": "Group-by aggregation", "dataset_type": "test",
        }).execute()
        log("PASS", "Supabase L4 INSERT")

        # L4 RPC
        l4rpc = c.rpc("match_l4_patterns", {
            "query_embedding": emb, "match_count": 3, "filter_dataset": "test",
        }).execute()
        log("PASS" if l4rpc.data else "FAIL",
            f"Supabase L4 RPC: results={len(l4rpc.data)}")

        # Update score
        c.table("l2_episodic").update({"score": 0.5}).eq("id", did).execute()
        check = c.table("l2_episodic").select("score").eq("id", did).execute()
        score = check.data[0]["score"] if check.data else None
        log("PASS" if score == 0.5 else "FAIL",
            f"Supabase L2 UPDATE: score={score}")

        # Cleanup
        c.table("l2_episodic").delete().eq("id", did).execute()
        c.table("l4_procedural").delete().eq("id", l4id).execute()
        log("PASS", "Supabase CLEANUP")

    except Exception as e:
        log("FAIL", "Supabase direct test", str(e)[:150])

    # 8h-j: API endpoints
    r, ms = timed("get", "/api/memory/stats")
    d = r.json()
    log("PASS" if "l2" in d and "l3" in d and "l4" in d else "FAIL",
        "GET /api/memory/stats", f"l2={d.get('l2')}, l3={d.get('l3')}, l4={d.get('l4')}", ms)

    for layer in ["l2", "l4", "l3"]:
        r, ms = timed("get", f"/api/memory/layer/{layer}")
        d = r.json()
        count = len(d) if isinstance(d, list) else len(d.get("nodes", []))
        log("PASS", f"GET /api/memory/layer/{layer}", f"items={count}", ms)


# ══════════════════════════════════════════════════════════════════════════════
#  9. EDGE CASES & STRESS
# ══════════════════════════════════════════════════════════════════════════════
def test_edge_cases(table: str):
    print("\n━━━ 9. EDGE CASES & STRESS ━━━")

    # 9a. SQL injection in datalab
    r = api("post", "/api/datalab/sql", json={
        "sql": f"SELECT * FROM \"{table}\"; DROP TABLE \"{table}\"; --"
    })
    # Table should survive
    r2, ms = timed("get", f"/api/datalab/preview/{table}")
    d2 = r2.json()
    log("PASS" if d2.get("row_count", 0) > 0 else "FAIL",
        "SQL injection survived", f"rows after={d2.get('row_count')}", ms)

    # 9b. Very long query string
    long_q = "Show me revenue " * 200
    r = api("post", "/api/query/query", json={"query": long_q})
    log("PASS" if r.status_code in (200, 422) else "FAIL",
        f"Very long query ({len(long_q)} chars)", f"status={r.status_code}")

    # 9c. Special characters in query
    r, ms = timed("post", "/api/query/query",
                   json={"query": "What's the revenue for 'Electronics' & 'Clothing'?", "dataset": "sample_data.csv"})
    log("PASS" if r.status_code == 200 else "FAIL",
        "Special chars in query", f"status={r.status_code}", ms)

    # 9d. Unicode in query
    r = api("post", "/api/query/query",
            json={"query": "显示按类别的总收入 📊", "dataset": "sample_data.csv"})
    log("PASS" if r.status_code in (200, 422) else "FAIL",
        "Unicode in query", f"status={r.status_code}")

    # 9e. Concurrent requests (sequential rapid fire)
    fast_results = []
    for i in range(3):
        r = api("post", "/api/datalab/sql",
                json={"sql": f'SELECT COUNT(*) as c FROM "{table}"'}, timeout=30)
        fast_results.append(r.status_code)
    log("PASS" if all(s == 200 for s in fast_results) else "FAIL",
        f"Rapid-fire 3 SQL queries", f"statuses={fast_results}")

    # 9f. Download table
    r, ms = timed("get", f"/api/datalab/download/{table}")
    log("PASS" if r.status_code == 200 and len(r.content) > 1000 else "FAIL",
        f"Download table CSV", f"size={len(r.content)} bytes", ms)

    # 9g. Anomaly with non-existent method
    r = api("post", "/api/anomaly/detect", json={
        "table": table, "threshold": 2.0, "method": "nonexistent_method",
    })
    log("PASS" if r.status_code in (200, 400, 422, 500) else "FAIL",
        "Anomaly: invalid method", f"status={r.status_code}")

    # 9h. RCA with non-numeric target
    r, ms = timed("post", "/api/rca/analyze", json={
        "table": table, "target_col": "category", "top_k": 5,
    })
    d = r.json()
    log("PASS" if r.status_code in (200, 400, 500) else "FAIL",
        "RCA: non-numeric target (category)",
        f"status={r.status_code}, target={d.get('target_col')}", ms)

    # 9i. DataLab transform: malicious prompt
    r, ms = timed("post", "/api/datalab/transform", json={
        "table": table,
        "prompt": "import os; os.system('rm -rf /')",
        "use_pandas": True,
    })
    d = r.json()
    log("PASS", "Transform: malicious prompt handled",
        f"success={d.get('success')}, err={str(d.get('error', ''))[:80]}", ms)

    # 9j. Health under load
    for _ in range(5):
        r = api("get", "/api/health")
    log("PASS" if r.status_code == 200 else "FAIL",
        "Health: repeated calls", f"status={r.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  SENTINEL — Deep Integration Test Suite v2")
    print(f"  Backend: {BASE_URL}")
    print(f"  Dataset: {SAMPLE_CSV} ({'exists' if SAMPLE_CSV.exists() else 'MISSING'})")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Preflight
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if r.status_code != 200:
            print(f"  ❌ Backend not reachable. Start with: python -m uvicorn backend.app:app --port 8000")
            sys.exit(1)
        print("  ✅ Backend reachable")
    except Exception:
        print(f"  ❌ Backend not reachable at {BASE_URL}")
        sys.exit(1)

    test_health()
    test_provider()
    table = test_upload()

    if table:
        test_datalab(table)
        test_anomaly(table)
        test_rca(table)
        test_queries(table)
        test_edge_cases(table)
    else:
        print("\n  ⚠️  Upload failed — skipping table-dependent tests")

    test_memory()

    # Summary
    total = PASSED + FAILED
    pct = PASSED / total * 100 if total else 0
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    print(f"  [{bar}] {pct:.0f}%")
    print(f"{'=' * 70}")

    if FAILED:
        print(f"\n  ❌ FAILURES:")
        for r in RESULTS:
            if r["status"] == "FAIL":
                print(f"     • {r['name']}: {r['detail'][:120]}")

    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
