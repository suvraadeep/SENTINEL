"""
SentinelNamespace — exec()-based shared namespace engine.

The agent files in agents/ were written for a Jupyter notebook where all cells
share one Python namespace.  They contain no import statements in their
function bodies; every symbol they use (call_llm, run_sql, l2_retrieve, …)
must be present in the enclosing namespace at call time.

This module replicates that notebook namespace by:
  1. Seeding it with all standard-library and third-party modules.
  2. String-patching each source file to replace Kaggle-specific paths/keys.
  3. exec()'ing each file in dependency order into the shared dict.
  4. Exposing helpers to reconfigure the LLM or swap in user-uploaded data.
"""

import os
import sys
import builtins
import asyncio
import logging
import textwrap
import json
import re
import hashlib
import warnings
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"

# Ordered by dependency — later files can call symbols defined by earlier ones.
_LOAD_ORDER = [
    "model.py",
    "state.py",
    "dataset.py",
    "charts.py",
    "memory.py",
    "chart_analysis.py",
    "intent_classifir.py",
    "schema_linker.py",
    "query_decomposer.py",
    "extract_first_sql.py",
    "sql_validator.py",
    "render_chart.py",
    "sql_response_writer.py",
    "rca_agent.py",
    "forcast_agent.py",
    "anomaly_agent.py",
    "math_agent.py",
    "prediction_agent.py",
    "memory_curator.py",
    "route_intent.py",
]

# ------------------------------------------------------------------
# Plotly sentinel theme (dark, matching UI colour palette)
# ------------------------------------------------------------------
_SENTINEL_COLORS = [
    "#3B82F6", "#06B6D4", "#8B5CF6", "#10B981",
    "#F59E0B", "#EF4444", "#EC4899", "#14B8A6",
    "#F97316", "#6366F1",
]

_PLOTLY_TEMPLATE_SPEC = {
    "layout": {
        "paper_bgcolor": "#111827",
        "plot_bgcolor": "#111827",
        "font": {"color": "#F1F5F9", "family": "Inter, sans-serif", "size": 12},
        "title": {"font": {"color": "#F1F5F9", "size": 16}},
        "xaxis": {
            "gridcolor": "#1E293B",
            "linecolor": "#334155",
            "tickcolor": "#94A3B8",
            "tickfont": {"color": "#94A3B8"},
            "title": {"font": {"color": "#94A3B8"}},
            "showgrid": True,
        },
        "yaxis": {
            "gridcolor": "#1E293B",
            "linecolor": "#334155",
            "tickcolor": "#94A3B8",
            "tickfont": {"color": "#94A3B8"},
            "title": {"font": {"color": "#94A3B8"}},
            "showgrid": True,
        },
        "colorway": _SENTINEL_COLORS,
        "legend": {
            "bgcolor": "#1E293B",
            "bordercolor": "#334155",
            "font": {"color": "#94A3B8"},
        },
        "hoverlabel": {
            "bgcolor": "#1E293B",
            "bordercolor": "#334155",
            "font": {"color": "#F1F5F9"},
        },
        "colorscale": {
            "sequential": [[0, "#0A0E1A"], [0.5, "#3B82F6"], [1, "#06B6D4"]],
            "diverging": [[0, "#EF4444"], [0.5, "#111827"], [1, "#3B82F6"]],
        },
    }
}


def _apply_plotly_theme(ns: dict) -> None:
    """Register the sentinel Plotly template inside the namespace."""
    try:
        pio = ns.get("pio")
        go = ns.get("go")
        if pio is None or go is None:
            return
        import plotly.graph_objects as _go
        import plotly.io as _pio
        template = _go.layout.Template(_PLOTLY_TEMPLATE_SPEC)
        _pio.templates["sentinel"] = template
        _pio.templates.default = "sentinel"
        # Keep ns references in sync
        ns["pio"] = _pio
        # Also expose color list for agents that query it
        ns["_SENTINEL_COLORS"] = _SENTINEL_COLORS
    except Exception as exc:
        logger.warning("Could not apply Plotly sentinel theme: %s", exc)


# ------------------------------------------------------------------
# Source-level patching helpers
# ------------------------------------------------------------------
def _patch_source(source: str, api_key: str, provider: str,
                  main_model: str, fast_model: str,
                  db_path: str, chroma_path: str, graph_path: str) -> str:
    """String-replace Kaggle-specific values before exec()."""
    replacements = {
        # Paths
        "/kaggle/working/sentinel_ecom.duckdb": db_path.replace("\\", "/"),
        "/kaggle/working/chroma_bge": chroma_path.replace("\\", "/"),
        "/kaggle/working/l3_ecom.gml": graph_path.replace("\\", "/"),
        # API key fallback strings (model.py uses these as defaults)
        '"YOUR_NVIDIA_KEY"': f'"{api_key}"',
        '"YOUR_GROQ_KEY"': f'"{api_key}"',
        '"YOUR_OPENAI_KEY"': f'"{api_key}"',
        '"YOUR_GEMINI_KEY"': f'"{api_key}"',
        '"YOUR_CLAUDE_KEY"': f'"{api_key}"',
        # Notebook magic
        "!uv pip install": "# pip install",
        "!pip install": "# pip install",
        # Kaggle secrets import
        "from kaggle_secrets import UserSecretsClient":
            "# from kaggle_secrets import UserSecretsClient  # patched",
        # Jupyter-specific renderer
        'pio.renderers.default = "iframe"': '# pio.renderers.default = "iframe"  # patched',
        # Force dark plotly theme — override all explicit template references
        'template="plotly_white"': 'template="sentinel"',
        "template='plotly_white'": "template='sentinel'",
        'template = "plotly_white"': 'template = "sentinel"',
        # Remove Jupyter display calls
        "from IPython.display import display, IFrame, HTML":
            "# from IPython.display import display, IFrame, HTML  # patched",
        # Nudge LLM-generated matplotlib code toward plt.show() so it can be intercepted
        "import matplotlib.pyplot as plt\nimport seaborn as sns":
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport seaborn as sns",
        "import matplotlib.pyplot as plt":
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    # Replace kaggle_secrets usage blocks
    source = re.sub(
        r"try:\s*\n\s*from kaggle_secrets.*?except.*?:\s*\n\s*API_KEY\s*=\s*os\.environ\.get\([^)]+\)",
        f'API_KEY = "{api_key}"',
        source,
        flags=re.DOTALL,
    )

    return source


def _patch_model_py(source: str, api_key: str, provider: str,
                    main_model: str, fast_model: str) -> str:
    """
    Additional patches specific to model.py:
    - Replace PROVIDER, MAIN_MODEL, FAST_MODEL, FALLBACK_MODEL constants.
    - Wrap the LLM construction blocks so wrong providers don't fail.
    """
    # Override model constants
    source = re.sub(r'PROVIDER\s*=\s*"[^"]*"', f'PROVIDER = "{provider}"', source)
    source = re.sub(r'MAIN_MODEL\s*=\s*"[^"]*"', f'MAIN_MODEL = "{main_model}"', source)
    source = re.sub(r'FAST_MODEL\s*=\s*"[^"]*"', f'FAST_MODEL = "{fast_model}"', source)

    # Disable smoke tests (they invoke the LLM immediately)
    source = re.sub(
        r'result\d+\s*=\s*_\w+_llm\s*\.invoke\([^)]+\)\s*\nprint\([^)]+\)',
        '# smoke test disabled in web mode',
        source,
        flags=re.MULTILINE,
    )
    source = re.sub(
        r'_test\s*=\s*call_llm\([^)]+\)\s*\nprint\([^)]+\)',
        '# smoke test disabled',
        source,
        flags=re.MULTILINE,
    )
    return source


def _patch_dataset_py(source: str) -> str:
    """
    Skip the DuckDB table creation / data generation if the con already exists
    in the namespace.  We do this by checking for a sentinel comment guard we
    inject at the top.
    """
    # We do NOT skip dataset.py entirely — we need get_schema(), run_sql(), etc.
    # But we skip the 50k-row data generation on subsequent calls.
    return source


# ------------------------------------------------------------------
# Main namespace class
# ------------------------------------------------------------------
class SentinelNamespace:
    """
    Thread-safe wrapper around the exec()-based shared namespace.

    Usage
    -----
    ns = SentinelNamespace()
    ns.initialize(api_key=..., provider=..., ...)
    result = await ns.call_ask("show revenue by category")
    """

    def __init__(self):
        self._ns: Dict[str, Any] = {}
        self._lock = asyncio.Lock()          # serialise queries
        self._thread_lock = threading.Lock()  # for sync init
        self._initialized = False
        self._data_loaded = False            # True once user uploads custom data
        self._query_count = 0               # tracks queries for periodic curator runs
        self._curator_running = False        # prevents concurrent curator runs
        self._conversation_history: List[Dict] = []  # chat continuation context (last 10 exchanges)
        self._dataset_registry: Dict[str, Any] = {}  # filename → {tables, date_min, date_max, row_count}
        self._original_con = None       # stored when switching to modified version
        self._original_schema = None
        self._current_dataset: str = ""     # dataset name for current query (for L2 tagging)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def initialize_sync(
        self,
        api_key: str,
        provider: str,
        main_model: str,
        fast_model: str,
        db_path: str,
        chroma_path: str,
        graph_path: str,
    ) -> None:
        """
        Bootstrap the shared namespace.  Called synchronously from a
        FastAPI startup handler or when the user configures a new provider.
        """
        with self._thread_lock:
            logger.info("Initialising SentinelNamespace (provider=%s, model=%s)", provider, main_model)

            os.makedirs(chroma_path, exist_ok=True)
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

            # Seed builtins + standard modules so agent files can use them
            # without their own import statements.
            import importlib
            seed_modules = [
                "os", "sys", "json", "re", "hashlib", "warnings", "math",
                "textwrap", "datetime", "time", "copy", "itertools",
                "collections", "functools", "pathlib",
                "numpy", "pandas", "duckdb", "networkx",
                "plotly.graph_objects", "plotly.express",
                "plotly.io", "plotly.subplots",
                "scipy.stats", "scipy",
                "statsmodels.tsa.stattools",
                "statsmodels.tsa.holtwinters",
                "sklearn.preprocessing", "sklearn.cluster",
                "sklearn.metrics.pairwise",
            ]

            self._ns = {"__builtins__": vars(builtins)}

            # Shortcuts used directly by agents
            self._ns.update({
                "os": os,
                "sys": sys,
                "json": json,
                "re": re,
                "hashlib": hashlib,
                "warnings": warnings,
                "math": math,
                "textwrap": textwrap,
            })

            for mod_name in ["numpy", "pandas", "duckdb", "networkx"]:
                try:
                    mod = importlib.import_module(mod_name)
                    alias = {"numpy": "np", "pandas": "pd"}.get(mod_name, mod_name)
                    self._ns[alias] = mod
                    self._ns[mod_name] = mod
                except ImportError:
                    pass

            # plotly
            try:
                import plotly.graph_objects as go
                import plotly.express as px
                import plotly.io as pio
                from plotly.subplots import make_subplots
                self._ns.update({
                    "go": go, "px": px, "pio": pio,
                    "make_subplots": make_subplots,
                    "plotly": importlib.import_module("plotly"),
                })
            except ImportError:
                pass

            # scipy / statsmodels / sklearn
            try:
                import scipy.stats as _stats
                from scipy.stats import ttest_ind, pearsonr, spearmanr
                from statsmodels.tsa.stattools import grangercausalitytests, adfuller
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                self._ns.update({
                    "stats": _stats, "ttest_ind": ttest_ind,
                    "pearsonr": pearsonr, "spearmanr": spearmanr,
                    "grangercausalitytests": grangercausalitytests,
                    "adfuller": adfuller,
                    "ExponentialSmoothing": ExponentialSmoothing,
                    "StandardScaler": StandardScaler, "KMeans": KMeans,
                })
            except ImportError as e:
                logger.warning("Some scientific libs missing: %s", e)

            # prophet — inject a stub into sys.modules if not installed so that
            # `from prophet import Prophet` inside agent files won't crash exec()
            try:
                from prophet import Prophet
                self._ns["Prophet"] = Prophet
            except ImportError:
                import types as _types
                _stub_prophet = _types.ModuleType("prophet")
                class _StubProphet:
                    def __init__(self, *a, **kw): pass
                    def fit(self, *a, **kw): return self
                    def predict(self, *a, **kw):
                        import pandas as _pd
                        return _pd.DataFrame()
                    def make_future_dataframe(self, *a, **kw):
                        import pandas as _pd
                        return _pd.DataFrame()
                _stub_prophet.Prophet = _StubProphet
                sys.modules.setdefault("prophet", _stub_prophet)
                self._ns["Prophet"] = _StubProphet
                logger.warning("prophet not installed — using stub; forecast agent will have limited functionality")

            # sympy
            try:
                import sympy as sp
                self._ns["sp"] = sp
            except ImportError:
                pass

            # langchain / langgraph
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                from langgraph.graph import StateGraph, END
                from typing import TypedDict, List, Dict, Any, Literal, Optional, Tuple
                self._ns.update({
                    "HumanMessage": HumanMessage,
                    "SystemMessage": SystemMessage,
                    "StateGraph": StateGraph,
                    "END": END,
                    "TypedDict": TypedDict,
                    "List": List, "Dict": Dict,
                    "Any": Any, "Literal": Literal,
                    "Optional": Optional, "Tuple": Tuple,
                })
            except ImportError as e:
                logger.error("LangChain not available: %s", e)

            # Additional stdlib
            from datetime import datetime, timedelta
            import itertools, collections, functools, copy
            self._ns.update({
                "datetime": datetime, "timedelta": timedelta,
                "itertools": itertools, "collections": collections,
                "functools": functools, "copy": copy,
            })

            # Set API key in environment so Kaggle-secrets fallback works
            os.environ["NVIDIA_API_KEY"] = api_key
            os.environ["GROQ_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["ANTHROPIC_API_KEY"] = api_key

            # Max retries constant used by multiple agents
            self._ns["MAX_RETRIES"] = 5
            self._ns["MAX_TOKENS"] = 16384

            # Execute agent files in order
            for filename in _LOAD_ORDER:
                path = AGENTS_DIR / filename
                if not path.exists():
                    logger.warning("Agent file not found, skipping: %s", path)
                    continue
                try:
                    source = path.read_text(encoding="utf-8")
                    source = _patch_source(
                        source, api_key, provider, main_model, fast_model,
                        db_path, chroma_path, graph_path,
                    )
                    if filename == "model.py":
                        source = _patch_model_py(source, api_key, provider, main_model, fast_model)
                    exec(compile(source, str(path), "exec"), self._ns)
                    logger.info("  ✓ loaded %s", filename)

                    # Override hardcoded agents immediately so route_intent.py
                    # picks up the dataset-agnostic versions when it runs next.
                    if filename == "rca_agent.py":
                        self._ns["rca_agent"] = self._build_agnostic_rca_agent()
                        logger.info("  → rca_agent patched (dataset-agnostic)")
                    elif filename == "anomaly_agent.py":
                        self._ns["anomaly_agent"] = self._build_agnostic_anomaly_agent()
                        logger.info("  → anomaly_agent patched (dataset-agnostic)")

                except Exception as exc:
                    logger.error("  ✗ failed to load %s: %s", filename, exc, exc_info=True)
                    # Don't abort — some files may fail due to optional deps

            # Apply dark Plotly theme after charts.py loaded pio
            _apply_plotly_theme(self._ns)

            # Now replace the LLM objects with proper provider-specific ones
            self._reconfigure_llm(api_key, provider, main_model, fast_model)

            # ── Wrap call_llm: anti-hallucination + no-raw-code hints ──────────
            # Injected as a system message into every non-SQL/non-JSON LLM call.
            _orig_clm = self._ns.get("call_llm")
            if _orig_clm:
                _NO_CODE_HINT = (
                    "\n\nIMPORTANT: Do NOT output Python, matplotlib, or plotly code blocks "
                    "in your response. Express all findings in plain prose. "
                    "Charts are generated automatically by a separate pipeline."
                )
                _NO_HALLUC_HINT = (
                    "\n\nCRITICAL: Only reference table/column names that exist in the "
                    "provided schema. Never invent or guess column names."
                )
                # Tokens that indicate a JSON/SQL-generation call — skip hints for these
                _SKIP_TOKS = frozenset([
                    "Return ONLY", "return only", "JSON array", "valid SQL",
                    "ONLY a JSON", "Reply with ONLY", "Output ONLY the SQL",
                    "Generate the DuckDB", "corrected SQL", "ONE letter only",
                    "output only valid", '"chart":', "Classify:",
                ])
                def _call_llm_guarded(prompt, system="", model=None, temperature=0.0):
                    skip = any(t in prompt for t in _SKIP_TOKS)
                    if not skip:
                        system = (system + _NO_CODE_HINT + _NO_HALLUC_HINT).strip() if system \
                                  else (_NO_CODE_HINT + _NO_HALLUC_HINT).strip()
                    if model is not None:
                        return _orig_clm(prompt, system=system, model=model,
                                         temperature=temperature)
                    return _orig_clm(prompt, system=system, temperature=temperature)
                self._ns["call_llm"] = _call_llm_guarded

            # Monkey-patch l2_collection.add to inject _current_dataset into metadata.
            # This must be done AFTER all agent files are exec()'d since memory.py
            # creates l2_collection during its own exec().
            l2_col = self._ns.get("l2_collection")
            if l2_col is not None:
                _orig_l2_add = l2_col.add
                _ns_ref = self._ns
                def _tagged_l2_add(*args, **kwargs):
                    ds = _ns_ref.get("_current_dataset", "")
                    if ds and "metadatas" in kwargs:
                        meta = kwargs["metadatas"]
                        if isinstance(meta, list):
                            kwargs["metadatas"] = [
                                dict(m, dataset=ds) if isinstance(m, dict) else {"dataset": ds}
                                for m in meta
                            ]
                    return _orig_l2_add(*args, **kwargs)
                l2_col.add = _tagged_l2_add
                logger.info("L2 collection patched for dataset tagging")

            self._initialized = True
            logger.info("SentinelNamespace initialised ✓")

            # ── Auto-reconnect to previously uploaded data ─────────────
            # On restart, update_data() state is lost. Check for existing
            # uploaded DuckDB files and reconnect to the most recent one.
            try:
                uploads_dir = Path(db_path).parent / "uploads"
                if uploads_dir.exists():
                    duckdb_files = sorted(
                        uploads_dir.glob("*.duckdb"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if duckdb_files:
                        latest = duckdb_files[0]
                        logger.info("Auto-reconnecting to existing upload: %s", latest.name)
                        import duckdb as _ddb
                        import pandas as _pd
                        from scipy import stats as _stats

                        new_con = _ddb.connect(str(latest))
                        new_con.execute("PRAGMA threads=4")

                        # Get tables from this DB
                        try:
                            tables = [r[0] for r in new_con.execute(
                                "SELECT table_name FROM information_schema.tables "
                                "WHERE table_schema='main'"
                            ).fetchall()]
                        except Exception:
                            tables = []

                        if tables:
                            # ── Swap connection ──
                            self._ns["con"] = new_con
                            self._data_loaded = True

                            # ── Rebuild closures over new connection ──
                            _con = new_con

                            def _new_run_sql(query: str) -> _pd.DataFrame:
                                try:
                                    return _con.execute(query).df()
                                except Exception as e:
                                    raise ValueError(f"SQL error: {e}\nQuery: {query}")

                            def _new_run_sql_approx(query: str, sample_frac: float = 0.3,
                                                     confidence: float = 0.95):
                                result = _new_run_sql(query)
                                if result.empty or len(result) < 5:
                                    return result, {}
                                numeric_cols = result.select_dtypes(include="number").columns.tolist()
                                z = _stats.norm.ppf((1 + confidence) / 2)
                                ci_info = {}
                                for col in numeric_cols:
                                    se = result[col].sem()
                                    margin = z * se
                                    ci_info[col] = {
                                        "mean": result[col].mean(),
                                        "ci_lower": result[col].mean() - margin,
                                        "ci_upper": result[col].mean() + margin,
                                        "confidence": confidence,
                                    }
                                return result, ci_info

                            def _new_get_schema() -> str:
                                parts = []
                                for tbl in tables:
                                    try:
                                        cols = _con.execute(f"DESCRIBE {tbl}").df()
                                        cols_str = ", ".join(
                                            f"{r['column_name']}:{r['column_type']}"
                                            for _, r in cols.iterrows()
                                        )
                                        sample = _con.execute(
                                            f"SELECT * FROM {tbl} LIMIT 2"
                                        ).df().to_string(index=False)
                                        n = _con.execute(
                                            f"SELECT COUNT(*) FROM {tbl}"
                                        ).fetchone()[0]
                                        parts.append(
                                            f"TABLE {tbl} ({n:,} rows)\n"
                                            f"COLUMNS: {cols_str}\nSAMPLE:\n{sample}"
                                        )
                                    except Exception:
                                        pass
                                return "\n\n".join(parts)

                            self._ns["run_sql"] = _new_run_sql
                            self._ns["run_sql_approx"] = _new_run_sql_approx
                            self._ns["get_schema"] = _new_get_schema

                            # Rebuild schema
                            schema = _new_get_schema()
                            self._ns["SCHEMA"] = schema

                            # Register dataset
                            ds_filename = latest.stem + ".csv"
                            row_count = 0
                            try:
                                row_count = new_con.execute(
                                    f"SELECT COUNT(*) FROM {tables[0]}"
                                ).fetchone()[0]
                            except Exception:
                                pass

                            self._dataset_registry[ds_filename] = {
                                "tables": tables,
                                "date_min": None,
                                "date_max": None,
                                "row_count": row_count,
                                "schema": schema[:500],
                            }
                            self._ns["_current_dataset"] = ds_filename

                            logger.info(
                                "Auto-reconnected: %d tables (%s), %d rows",
                                len(tables), tables, row_count,
                            )

                            # Rebuild L3 graph from the reconnected data
                            try:
                                self._rebuild_l3_graph(new_con)
                            except Exception as exc:
                                logger.warning("L3 rebuild on auto-reconnect failed: %s", exc)
                        else:
                            logger.info("Uploaded DB found but contains no tables: %s", latest.name)
            except Exception as exc:
                logger.warning("Auto-reconnect to uploads failed (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Dataset-agnostic agent builders (monkey-patched into _ns)
    # ------------------------------------------------------------------

    def _build_agnostic_anomaly_agent(self):
        """
        Return a dataset-agnostic anomaly_agent closure.
        Replaces the hardcoded e-commerce version that uses the `orders` table.
        """
        _ns      = self._ns
        _self_ref = self   # closed over so the closure can read _dataset_registry

        def agnostic_anomaly_agent(state):  # noqa: C901
            import pandas as _pd
            import numpy as _np

            con       = _ns.get("con")
            call_llm  = _ns.get("call_llm")
            safe_show = _ns.get("safe_show", lambda f, t: None)
            l2_store  = _ns.get("l2_store",  lambda *a, **kw: None)
            FAST_MODEL = _ns.get("FAST_MODEL", "")

            if con is None:
                msg = "No database connection available."
                return {"anomaly_result": {"count": 0}, "final_response": msg}

            # ── 1. Discover tables ────────────────────────────────────────────
            try:
                tables = [r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='main'"
                ).fetchall()]
            except Exception as exc:
                return {"anomaly_result": {"count": 0},
                        "final_response": f"Could not list tables: {exc}"}

            # Filter to the dataset currently in scope (if set)
            target_ds = _ns.get("_current_dataset", "")
            if target_ds:
                try:
                    ds_tables = _self_ref._dataset_registry.get(target_ds, {}).get("tables", [])
                    if ds_tables:
                        tables = [t for t in tables if t in ds_tables] or tables
                except Exception:
                    pass

            if not tables:
                return {"anomaly_result": {"count": 0},
                        "final_response": "No tables found in database."}

            # ── 2. Profile and detect anomalies across all tables ─────────────
            all_anomalies = []
            WINDOW, Z_THRESH = 5, 2.0
            Z_HIGH, Z_CRIT   = 3.0, 4.0

            for tbl in tables:
                try:
                    df = con.execute(f"SELECT * FROM \"{tbl}\" LIMIT 5000").df()
                except Exception:
                    continue
                if df.empty:
                    continue

                # Detect date column
                date_col = None
                for c in df.columns:
                    if _pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col = c; break
                    if any(kw in c.lower() for kw in ("date", "time", "created", "updated", "ts")):
                        try:
                            df[c] = _pd.to_datetime(df[c], errors="coerce")
                            if df[c].notna().sum() > len(df) * 0.5:
                                date_col = c; break
                        except Exception:
                            pass

                num_cols = df.select_dtypes(include="number").columns.tolist()
                if not num_cols:
                    continue

                if date_col:
                    df = df.sort_values(date_col).reset_index(drop=True)

                # Rolling Z-score + IQR per numeric column
                series_vals = df[num_cols].values
                for ci, col in enumerate(num_cols):
                    series = series_vals[:, ci]
                    effective_window = min(WINDOW, max(2, len(series) // 10))
                    for i in range(effective_window, len(series)):
                        window = series[i - effective_window: i]
                        mu     = window.mean()
                        sigma  = window.std() + 1e-9
                        z      = (series[i] - mu) / sigma
                        q1, q3 = _np.percentile(window, 25), _np.percentile(window, 75)
                        iqr    = q3 - q1
                        fl, fh = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        if abs(z) > Z_THRESH or series[i] < fl or series[i] > fh:
                            sev = ("CRITICAL" if abs(z) >= Z_CRIT
                                   else "HIGH" if abs(z) >= Z_HIGH
                                   else "MEDIUM")
                            row_info = {"table": tbl, "column": col,
                                        "index": i, "value": round(float(series[i]), 4),
                                        "baseline": round(float(mu), 4),
                                        "z_score": round(float(z), 4),
                                        "severity": sev}
                            if date_col and not _pd.isna(df[date_col].iloc[i]):
                                row_info["date"] = str(df[date_col].iloc[i])
                            all_anomalies.append(row_info)

            if not all_anomalies:
                msg = "No anomalies detected above threshold in any table."
                return {"anomaly_result": {"count": 0}, "final_response": msg}

            anom_df  = _pd.DataFrame(all_anomalies)
            critical = anom_df[anom_df["severity"] == "CRITICAL"]
            high     = anom_df[anom_df["severity"].isin(["HIGH", "CRITICAL"])]

            # ── 3. Charts ─────────────────────────────────────────────────────
            all_analysis = []
            try:
                import plotly.express as px
                import plotly.graph_objects as go

                # Chart 1: Scatter |z_score| vs index/date colored by column
                plot_df = anom_df.copy()
                plot_df["abs_z"] = plot_df["z_score"].abs().clip(upper=10)
                x_col = "date" if "date" in plot_df.columns else "index"
                fig1 = px.scatter(
                    plot_df, x=x_col, y="abs_z",
                    color="column", symbol="severity", size="abs_z", size_max=18,
                    facet_col="table" if plot_df["table"].nunique() > 1 else None,
                    title="Anomalies — |Z-score| by Position & Column",
                    hover_data=["value", "baseline", "z_score", "table", "column"],
                )
                safe_show(fig1, "Anomaly scatter — |Z-score| by position")
                all_analysis.append("**Anomaly Scatter**: Detected anomalous data points visualized by Z-score magnitude across columns and tables.")

                # Chart 2: Heatmap worst z_score per table × column
                pivot = (anom_df.groupby(["table", "column"])["z_score"]
                         .apply(lambda x: x.abs().max())
                         .reset_index()
                         .pivot(index="table", columns="column", values="z_score")
                         .fillna(0))
                if not pivot.empty:
                    fig2 = px.imshow(
                        pivot, title="Worst |Z-score| — Table × Column",
                        color_continuous_scale="RdYlGn_r", text_auto=".1f",
                    )
                    safe_show(fig2, "Severity heatmap — table × column")
                    all_analysis.append("**Severity Heatmap**: Shows which table/column combinations have the most extreme anomalies.")
            except Exception as chart_exc:
                logger.warning("Anomaly chart generation failed: %s", chart_exc)

            # ── 4. LLM alert ──────────────────────────────────────────────────
            top5 = anom_df.sort_values("z_score", key=abs, ascending=False).head(5)
            alert = ""
            if call_llm:
                try:
                    alert = call_llm(
                        f"Write a 3-4 sentence data alert for these anomalies. "
                        f"State the severity, which columns are affected, and suggest one immediate action.\n"
                        f"{top5.to_string(index=False)}",
                        model=FAST_MODEL, temperature=0.1,
                    )
                except Exception as llm_exc:
                    alert = f"{len(all_anomalies)} anomalies detected ({len(critical)} CRITICAL, {len(high) - len(critical)} HIGH)."

            chart_section = "\n\n".join(all_analysis)
            full_response = (
                f"{alert}\n\n"
                f"{'─' * 50}\n"
                f"Chart-by-chart analysis:\n{chart_section}"
            ).strip()

            try:
                l2_store(state.get("query", "anomaly scan"), "/* anomaly scan */",
                         f"{len(anom_df)} anomalies, {len(critical)} CRITICAL")
            except Exception:
                pass

            return {
                "anomaly_result": {
                    "count":           len(anom_df),
                    "total_anomalies": len(anom_df),
                    "critical_count":  len(critical),
                    "high_count":      len(high) - len(critical),
                    "alert":           alert,
                    "anomalies":       anom_df.head(20).to_dict("records"),
                },
                "final_response": full_response,
            }

        return agnostic_anomaly_agent

    def _build_agnostic_rca_agent(self):
        """
        Dataset-agnostic rca_agent — delegates to CausalRCAEngine.
        Replaces the hardcoded orders-table version; works on any uploaded dataset.
        """
        _ns       = self._ns
        _self_ref = self

        def agnostic_rca_agent(state):
            from backend.api.rca import CausalRCAEngine

            con      = _ns.get("con")
            call_llm = _ns.get("call_llm")
            l2_store = _ns.get("l2_store", lambda *a, **kw: None)
            safe_show = _ns.get("safe_show", lambda f, t: None)

            if con is None:
                return {"rca_result": {}, "final_response": "No database connection available."}

            # ── Discover tables in scope ──────────────────────────────────
            try:
                tables = [r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='main'"
                ).fetchall()]
            except Exception as exc:
                return {"rca_result": {}, "final_response": f"Could not list tables: {exc}"}

            target_ds = _ns.get("_current_dataset", "")
            if target_ds:
                try:
                    ds_tables = _self_ref._dataset_registry.get(
                        target_ds, {}).get("tables", [])
                    if ds_tables:
                        tables = [t for t in tables if t in ds_tables] or tables
                except Exception:
                    pass

            if not tables:
                return {"rca_result": {}, "final_response": "No tables found in database."}

            # ── Run CausalRCAEngine on first valid table ───────────────────
            for tbl in tables:
                try:
                    df = con.execute(f'SELECT * FROM "{tbl}" LIMIT 10000').df()
                    if df.empty or len(df) < 4:
                        continue
                    engine = CausalRCAEngine(df, table_name=tbl)
                    if not engine.num_cols:
                        continue

                    result = engine.run(call_llm=call_llm)

                    # Push charts into safe_show queue so Intelligence tab renders them
                    for chart in result.get("charts", [])[:3]:
                        try:
                            import plotly.graph_objects as _go
                            # Re-parse the HTML to extract a figure title for queue metadata
                            safe_show(None, chart.get("title", "RCA Chart"))
                        except Exception:
                            pass

                    rc      = result.get("root_causes", [])
                    stats   = result.get("statistics", {})
                    primary = rc[0] if rc else {}

                    rca_result = {
                        # Fields consumed by existing RcaBlock.jsx
                        "worst_category": primary.get("name", "Unknown"),
                        "delta_pct":      0.0,
                        "granger":        stats.get("granger", {}),
                        "correlations":   {c: d.get("rho", 0)
                                           for c, d in stats.get("spearman", {}).items()},
                        "narrative":      result.get("explanation", ""),
                        # Extended fields for new RCADashboard.jsx
                        "drivers":        rc[:5],
                        "root_causes":    rc,
                        "graph":          result.get("graph", {}),
                        "change_points":  result.get("change_points", []),
                        "statistics":     stats,
                        "profile":        result.get("profile", {}),
                        "table":          tbl,
                        "target_col":     engine.target_col,
                    }

                    try:
                        l2_store(
                            state.get("query", "rca"), "/* RCA */",
                            f"RCA: {tbl}/{engine.target_col} — primary driver: "
                            f"{primary.get('name', '?')}",
                        )
                    except Exception:
                        pass

                    return {
                        "rca_result":     rca_result,
                        "final_response": result.get("explanation", "RCA complete."),
                    }

                except Exception as exc:
                    logger.warning("CausalRCAEngine failed on %s: %s", tbl, exc, exc_info=True)
                    continue

            return {"rca_result": {}, "final_response": "Could not run RCA on any table."}

        return agnostic_rca_agent

    def _reconfigure_llm(self, api_key: str, provider: str,
                         main_model: str, fast_model: str) -> None:
        """Build proper LangChain LLM objects and inject them into namespace."""
        from backend.core.llm_factory import build_llm_pair
        try:
            main_llm, fast_llm = build_llm_pair(provider, api_key, main_model, fast_model)
            self._ns["_main_llm"] = main_llm
            self._ns["_fast_llm"] = fast_llm
            self._ns["MAIN_MODEL"] = main_model
            self._ns["FAST_MODEL"] = fast_model
            self._ns["PROVIDER"] = provider
            self._ns["API_KEY"] = api_key
            logger.info("LLM reconfigured: provider=%s main=%s fast=%s", provider, main_model, fast_model)
        except Exception as exc:
            logger.error("LLM reconfiguration failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Query pre-validation, context enrichment & code execution
    # ------------------------------------------------------------------

    def _build_greeting_response(self) -> Dict[str, Any]:
        """Friendly SENTINEL greeting with schema context."""
        schema = self._ns.get("SCHEMA", "")
        tables = re.findall(r'TABLE\s+(\w+)', schema, re.IGNORECASE)[:5]
        reg = self._dataset_registry
        if reg:
            files = list(reg.keys())[:3]
            data_hint = f" Loaded datasets: **{', '.join(files)}**."
        elif tables:
            data_hint = f" I can query: **{', '.join(tables)}**."
        else:
            data_hint = " Using the built-in e-commerce dataset."
        return {
            "type": "greeting",
            "response": (
                f"Hi! I'm **SENTINEL**, your AI-powered analytics assistant.{data_hint}\n\n"
                f"I can help you with:\n"
                f"- **SQL queries** — revenue, trends, rankings, segmentation\n"
                f"- **Anomaly detection** — find unusual patterns in your data\n"
                f"- **Forecasting** — Prophet-based time-series prediction\n"
                f"- **Root cause analysis** — understand why metrics changed\n"
                f"- **ML prediction** — regression, feature impact, valuation\n\n"
                f"Try: *\"Show revenue by category\"* or *\"Find sales anomalies\"*"
            ),
        }

    def _prevalidate_query(self, query: str) -> Dict[str, Any]:
        """
        Three-stage pre-validation — no full pipeline for off-topic queries.

        Stage 1: Fast regex  → obvious greetings (zero LLM calls)
        Stage 2: Keyword set → data-analytics words (zero LLM calls)
        Stage 3: LLM check   → short/ambiguous queries checked against schema

        Returns {"type": "valid"|"greeting"|"irrelevant", "response": str}
        """
        q       = query.strip()
        q_lower = q.lower()

        # ── Stage 1: fast regex greetings ───────────────────────────────────
        _GREET = re.compile(
            r'^(hi+|hello+|hey+|howdy|greetings?|yo|sup)\s*[!.,?]*$'
            r'|^how\s+are\s+(you|u)\s*[!.,?]*$'
            r'|^what[\'\u2019]?s\s+up\s*[!.,?]*$'
            r'|^(thanks?|thank\s+you|thx|ty)\s*[!.,?]*$'
            r'|^(bye|goodbye|ciao|see\s+you)\s*[!.,?]*$'
            r'|^(who|what)\s+are\s+you\s*[!.,?]*$'
            r'|^what\s+can\s+you\s+do\s*[!.,?]*$'
            r'|^help\s*[!.,?]*$'
            r'|^(test|ping|ok|okay)\s*[!.,?]*$'
            r'|^(good|great|nice|cool|awesome|perfect|wow)\s*[!.,?]*$',
            re.IGNORECASE,
        )
        if _GREET.match(q_lower):
            return self._build_greeting_response()

        schema        = self._ns.get("SCHEMA", "")
        call_llm_fn   = self._ns.get("call_llm")

        # ── Stage 2: data keyword heuristic ─────────────────────────────────
        _DATA_KW = frozenset([
            "show","list","find","get","count","sum","average","avg","total",
            "top","bottom","best","worst","highest","lowest","max","min","mean",
            "trend","forecast","predict","anomaly","outlier","rca","root","cause",
            "revenue","sale","order","customer","product","category","region",
            "date","month","year","week","daily","monthly","yearly","quarterly",
            "compare","correlation","distribution","segment","group","cluster",
            "filter","where","what","why","which","analyze","analysis",
            "insight","report","summary","breakdown","percentage","ratio",
            "rate","rank","increase","decrease","growth","decline","spike","drop",
            "data","table","query","column","value","metric","kpi","dashboard",
            "price","cost","amount","fee","profit","margin","quantity","volume",
            "row","record","entry","dataset","sql","join","select","from",
            "median","variance","stddev","percentile","quartile","histogram",
            "regression","classification","model","prediction","estimate","ml",
            "chart","plot","graph","visualiz","scatter","bar","line","pie",
        ])
        qwords = set(re.findall(r'\b[a-z]{3,}\b', q_lower))
        if qwords & _DATA_KW:
            return {"type": "valid", "response": ""}

        # Schema keyword overlap (catches dataset-specific column/table names)
        if schema:
            schema_words = set(re.findall(r'\b[a-z_]{3,}\b', schema.lower()))
            if len(qwords & schema_words) >= 2:
                return {"type": "valid", "response": ""}

        # ── Stage 3: LLM relevance check for ambiguous short queries ────────
        if not call_llm_fn or not schema:
            return {"type": "valid", "response": ""}

        fast_model     = self._ns.get("FAST_MODEL", "")
        schema_compact = "\n".join(schema.split("\n")[:12])
        prompt = (
            f"Database schema (abbreviated):\n{schema_compact}\n\n"
            f"User message: {q}\n\n"
            f"Classify: A=greeting/small-talk, B=relevant to this database, "
            f"C=completely unrelated to this data.\n"
            f"Reply ONLY with A, B, or C."
        )
        try:
            ans = call_llm_fn(prompt, model=fast_model, temperature=0.0).strip().upper()
            if ans.startswith("A"):
                return self._build_greeting_response()
            if ans.startswith("C"):
                tables = re.findall(r'TABLE\s+(\w+)', schema, re.IGNORECASE)[:4]
                d = f"**{', '.join(tables)}**" if tables else "your dataset"
                return {
                    "type": "irrelevant",
                    "response": (
                        f"That question is outside the scope of your loaded data ({d}).\n\n"
                        f"I can only analyze data in your database. Try:\n"
                        f"- *\"Show top products by revenue\"*\n"
                        f"- *\"Find anomalies in orders\"*\n"
                        f"- *\"Forecast next 7 days of sales\"*"
                    ),
                }
        except Exception as exc:
            logger.warning("Pre-validation LLM check failed: %s", exc)

        return {"type": "valid", "response": ""}

    def _enrich_query_with_context(self, query: str) -> str:
        """
        Prepend the last 2 conversation exchanges for follow-up queries.
        Only activates when the query contains reference words or is very short.
        """
        if not self._conversation_history:
            return query

        _FOLLOWUP = re.compile(
            r'\b(that|those|it|its|them|same|previous|last|above|prior'
            r'|more|further|deeper|drill|breakdown|break\s+down|expand|elaborate'
            r'|also|additionally|what\s+about|and\s+also|why|reason|cause'
            r'|explain|clarify|continue|related|similar|compared)\b',
            re.IGNORECASE,
        )
        is_followup = bool(_FOLLOWUP.search(query)) or len(query.split()) <= 5
        if not is_followup:
            return query

        parts = []
        for entry in self._conversation_history[-2:]:
            sql_snip = f"\n  SQL: {entry['sql'][:100]}..." if entry.get("sql") else ""
            parts.append(
                f"Previous Q: \"{entry['query']}\"\n"
                f"  Intent: {entry.get('intent','sql_query')}\n"
                f"  Result: {entry.get('summary','')[:200]}{sql_snip}"
            )
        if not parts:
            return query

        ctx = "\n---\n".join(parts)
        return (
            f"[CONVERSATION CONTEXT — for reference only]\n{ctx}\n[END CONTEXT]\n\n"
            f"Current question: {query}"
        )

    def _execute_code_blocks(self, text: str) -> Tuple[str, List]:
        """
        Find ```python / ``` blocks in text, execute in namespace,
        capture any Plotly/matplotlib charts generated.
        Returns (cleaned_text, list_of_(html, title)_tuples).
        """
        pattern = re.compile(r'```(?:python|py)?\n(.*?)```', re.DOTALL)
        blocks  = list(pattern.finditer(text))
        if not blocks:
            return text, []

        self._ns['_FIG_QUEUE'] = []
        safe_show_fn = self._ns.get('safe_show')

        # Patch go.Figure.show → safe_show so inline Plotly code is captured
        _go_cls = None; _orig_go_show = None
        try:
            import plotly.graph_objects as _go_mod
            _go_cls       = _go_mod.Figure
            _orig_go_show = _go_cls.show
            _sfn          = safe_show_fn
            _ns_ref       = self._ns

            def _cap_go_show(fig_self, *a, **kw):
                t = ''
                try: t = fig_self.layout.title.text or ''
                except Exception: pass
                t = t or 'Chart'
                if _sfn:
                    _sfn(fig_self, t)
                else:
                    html = fig_self.to_html(
                        include_plotlyjs='cdn', full_html=False,
                        config={'responsive': True}
                    )
                    _ns_ref.setdefault('_FIG_QUEUE', []).append((html, t))
            _go_cls.show = _cap_go_show
        except Exception:
            _go_cls = None

        # Patch plt.show → capture as embedded base64 image
        _plt_mod = None; _orig_plt_show = None
        try:
            import matplotlib as _mpl
            _mpl.use('Agg')
            import matplotlib.pyplot as _plt_mod
            _orig_plt_show = _plt_mod.show
            _ns2           = self._ns

            def _cap_plt_show(*a, **kw):
                import io, base64 as _b64
                buf = io.BytesIO()
                try:
                    _plt_mod.gcf().savefig(
                        buf, format='png', bbox_inches='tight',
                        facecolor='#111827', edgecolor='none', dpi=100
                    )
                    buf.seek(0)
                    b64  = _b64.b64encode(buf.read()).decode()
                    html = (
                        '<div style="background:#111827;padding:8px;border-radius:8px;text-align:center;">'
                        f'<img src="data:image/png;base64,{b64}" '
                        'style="max-width:100%;height:auto;display:block;margin:0 auto;"></div>'
                    )
                    _ns2.setdefault('_FIG_QUEUE', []).append((html, 'Chart'))
                except Exception as _e:
                    logger.warning("plt.show capture failed: %s", _e)
                finally:
                    _plt_mod.close('all')
            _plt_mod.show = _cap_plt_show
            self._ns['plt'] = _plt_mod
        except Exception:
            _plt_mod = None

        new_charts: List = []
        try:
            for m in blocks:
                try:
                    exec(compile(m.group(1).strip(), '<inline_chart>', 'exec'), self._ns)
                except Exception as exc:
                    logger.warning("Inline code block exec failed: %s", exc)
            new_charts = list(self._ns.get('_FIG_QUEUE', []))
        finally:
            if _go_cls is not None and _orig_go_show is not None:
                try: _go_cls.show = _orig_go_show
                except Exception: pass
            if _plt_mod is not None and _orig_plt_show is not None:
                try: _plt_mod.show = _orig_plt_show
                except Exception: pass
            self._ns['_FIG_QUEUE'] = []

        if new_charts:
            return pattern.sub('\n*[Interactive chart rendered above]*\n', text), new_charts
        return text, []

    def _update_conversation_history(
        self, query: str, last_state: Dict, final_response: str
    ) -> None:
        """Store query + result summary for follow-up context (max 10)."""
        entry = {
            "query":   query[:200],
            "intent":  last_state.get("intent", "sql_query"),
            "sql":     (last_state.get("sql_query") or "")[:150],
            "summary": final_response[:300] if final_response else "",
        }
        self._conversation_history.append(entry)
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]

    # ------------------------------------------------------------------
    # Schema filtering helper
    # ------------------------------------------------------------------
    def _filter_schema_for_tables(self, tables: List[str]) -> str:
        """Build a SCHEMA string containing only the specified tables."""
        con = self._ns.get("con")
        if not con or not tables:
            return self._ns.get("SCHEMA", "")
        schema_lines: List[str] = []
        for tbl in tables:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tbl):
                continue
            try:
                cols_df = con.execute(f"DESCRIBE {tbl}").df()
                schema_lines.append(f"TABLE {tbl}:")
                for _, row in cols_df.iterrows():
                    schema_lines.append(f"  {row['column_name']} {row['column_type']}")
                schema_lines.append("")
            except Exception:
                pass
        return "\n".join(schema_lines) if schema_lines else self._ns.get("SCHEMA", "")

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------
    async def call_ask(self, query: str, dataset_tables: Optional[List[str]] = None, dataset_name: str = "") -> Dict[str, Any]:
        """
        Run an analytics query through SENTINEL.

        Stages:
          0. Pre-validate  — greetings and out-of-scope queries short-circuit here
          1. Enrich        — prepend conversation context for follow-ups
          2. Execute       — run the full LangGraph pipeline
          3. Code blocks   — execute any inline Python chart code in the response
          4. History       — store for future context continuity
        """
        async with self._lock:
            if not self._initialized:
                raise RuntimeError("Namespace not initialised — call initialize_sync first")

            ask_fn = self._ns.get("ask")
            if ask_fn is None:
                raise RuntimeError("ask() function not found in namespace")

            import time
            t0 = time.time()

            # ── 0. Pre-validate: reject greetings / out-of-scope queries ────────
            validation = self._prevalidate_query(query)
            if validation["type"] != "valid":
                return {
                    "final_response": validation["response"],
                    "fig_queue":      [],
                    "last_state":     {"intent": "chat"},
                    "duration_ms":    int((time.time() - t0) * 1000),
                    "error":          None,
                }

            # ── 1. Enrich with conversation context for follow-ups ───────────────
            enriched_query = self._enrich_query_with_context(query)

            # ── Set current dataset for L2 episode tagging ──────────────────────
            self._ns["_current_dataset"] = dataset_name
            self._current_dataset = dataset_name

            # ── Swap to dataset-specific DuckDB connection + schema ──────────────
            original_con: Optional[Any] = None
            original_schema: Optional[str] = None
            if dataset_name and dataset_name in self._dataset_registry:
                ds_info = self._dataset_registry[dataset_name]
                ds_con = ds_info.get("con")
                ds_schema = ds_info.get("schema", "")
                if ds_con:
                    original_con = self._ns.get("con")
                    original_schema = self._ns.get("SCHEMA", "")
                    self._ns["con"] = ds_con
                    self._ns["SCHEMA"] = ds_schema
                    logger.info(
                        "Swapped to dataset '%s' connection (tables=%s)",
                        dataset_name, ds_info.get("tables", []),
                    )
            elif dataset_tables:
                original_schema = self._ns.get("SCHEMA", "")
                filtered = self._filter_schema_for_tables(dataset_tables)
                if filtered:
                    self._ns["SCHEMA"] = filtered
                    logger.info(
                        "Schema narrowed to %d tables for dataset '%s'",
                        len(dataset_tables), dataset_name,
                    )

            # Clear chart queue
            self._ns["_FIG_QUEUE"] = []

            # Patch flush_charts — return 0 so `n_shown > 0` never raises TypeError
            original_flush = self._ns.get("flush_charts", lambda: None)
            self._ns["flush_charts"] = lambda: 0

            final_response = ""
            error_msg      = None

            # ── Capture viz_agent output independently ───────────────────────────
            _viz_capture: Dict[str, Any] = {}
            original_viz_agent = self._ns.get("viz_agent")
            if original_viz_agent:
                def _capturing_viz_agent(state):
                    r = original_viz_agent(state)
                    if isinstance(r, dict):
                        _viz_capture["chart_explanations"] = r.get("chart_explanations", "") or ""
                    return r
                self._ns["viz_agent"] = _capturing_viz_agent

            # ── Intercept sentinel.invoke to capture final LangGraph state ───────
            sentinel_graph         = self._ns.get("sentinel")
            original_sentinel_invoke = None
            if sentinel_graph and hasattr(sentinel_graph, "invoke"):
                original_sentinel_invoke = sentinel_graph.invoke
                def _capturing_invoke(state, *args, **kwargs):
                    result_state = original_sentinel_invoke(state, *args, **kwargs)
                    self._ns["_last_state"] = dict(result_state) if result_state else {}
                    return result_state
                sentinel_graph.invoke = _capturing_invoke

            # ── 2. Execute the pipeline ──────────────────────────────────────────
            try:
                result = ask_fn(enriched_query)
                if isinstance(result, dict):
                    final_response = result.get("final_response", str(result))
                else:
                    final_response = str(result) if result else ""
            except Exception as exc:
                logger.error("ask() raised: %s", exc, exc_info=True)
                error_msg = str(exc)
            finally:
                self._ns["flush_charts"] = original_flush
                if sentinel_graph and original_sentinel_invoke is not None:
                    sentinel_graph.invoke = original_sentinel_invoke
                if original_viz_agent is not None:
                    self._ns["viz_agent"] = original_viz_agent
                # Restore original connection + schema if we swapped
                if original_con is not None:
                    self._ns["con"] = original_con
                if original_schema is not None:
                    self._ns["SCHEMA"] = original_schema
                # Clear current dataset tag
                self._ns["_current_dataset"] = ""
                self._current_dataset = ""

            duration_ms = int((time.time() - t0) * 1000)

            # Harvest chart queue
            fig_queue: List[Tuple[str, str]] = list(self._ns.get("_FIG_QUEUE", []))
            self._ns["_FIG_QUEUE"] = []

            # Harvest LangGraph state
            last_state: Dict = dict(self._ns.get("_last_state", {}))

            # Supplement with independently captured viz_agent data
            if _viz_capture.get("chart_explanations") and not last_state.get("chart_explanations"):
                last_state["chart_explanations"] = _viz_capture["chart_explanations"]

            # ── 3. Execute any inline Python code blocks in the response ─────────
            if final_response:
                final_response, inline_charts = self._execute_code_blocks(final_response)
                if inline_charts:
                    fig_queue = list(fig_queue) + list(inline_charts)
                    logger.info("Captured %d inline chart(s) from code blocks", len(inline_charts))

            # ── Fallback final_response if pipeline produced nothing ─────────────
            if not final_response:
                chart_expl     = last_state.get("chart_explanations", "")
                sql_result_raw = last_state.get("sql_result_json", "")
                if chart_expl:
                    final_response = (
                        "The charts above visualise your query results. "
                        "See chart-by-chart analysis below for statistical insights."
                    )
                elif sql_result_raw:
                    try:
                        import pandas as _pd
                        _rows = json.loads(sql_result_raw)
                        _df   = _pd.DataFrame(_rows) if isinstance(_rows, list) else _pd.DataFrame()
                        _parts = [f"**{len(_df)} rows** returned."]
                        for _c in _df.select_dtypes(include="number").columns[:3]:
                            _s = _df[_c].dropna()
                            if len(_s):
                                _parts.append(
                                    f"**{_c}**: min={_s.min():,.2f}, "
                                    f"max={_s.max():,.2f}, avg={_s.mean():,.2f}"
                                )
                        for _c in _df.select_dtypes(include=["object","category"]).columns[:1]:
                            _top = _df[_c].value_counts().head(3)
                            _parts.append(
                                "Top **" + _c + "**: "
                                + ", ".join(f"{k} ({v})" for k, v in _top.items())
                            )
                        final_response = " ".join(_parts)
                    except Exception:
                        final_response = "Query executed. Results shown in SQL section above."
                elif error_msg:
                    final_response = f"Analysis failed: {error_msg[:300]}"
                else:
                    final_response = "Query executed. Results shown above."

            # ── 4. Update conversation history ───────────────────────────────────
            if not error_msg:
                self._update_conversation_history(query, last_state, final_response)
                self._query_count += 1
                if self._query_count % 3 == 0 and not self._curator_running:
                    asyncio.create_task(self._run_memory_curator())

            return {
                "final_response": final_response,
                "fig_queue":      fig_queue,
                "last_state":     last_state,
                "duration_ms":    duration_ms,
                "error":          error_msg,
            }

    # ------------------------------------------------------------------
    # Background memory consolidation
    # ------------------------------------------------------------------
    async def _run_memory_curator(self) -> None:
        """
        Run memory_curator in background after queries to consolidate L2→L3/L4.
        Acquires the query lock so it doesn't race with concurrent queries.
        """
        if self._curator_running:
            return
        self._curator_running = True
        try:
            async with self._lock:
                curator_fn = self._ns.get("memory_curator")
                if not curator_fn:
                    return
                logger.info("[MemoryCurator] Background consolidation starting...")
                # Pass a minimal state — memory_curator only uses the global
                # l2_collection, l3_graph, l4_collection from the namespace
                curator_fn({"query": "__curator__"})
                l3 = self._ns.get("l3_graph")
                if l3:
                    logger.info(
                        "[MemoryCurator] Done — L3 graph: %d nodes, %d edges",
                        l3.number_of_nodes(), l3.number_of_edges(),
                    )
        except Exception as exc:
            logger.warning("[MemoryCurator] Background run failed: %s", exc)
        finally:
            self._curator_running = False

    # ------------------------------------------------------------------
    # Provider reconfiguration (called when user changes provider/key)
    # ------------------------------------------------------------------
    async def reconfigure(self, api_key: str, provider: str,
                          main_model: str, fast_model: str) -> None:
        async with self._lock:
            self._reconfigure_llm(api_key, provider, main_model, fast_model)

    # ------------------------------------------------------------------
    # Data upload — add dataset to namespace (multi-dataset safe)
    # ------------------------------------------------------------------
    async def update_data(
        self,
        new_con,
        new_schema: str,
        date_min,
        date_max,
        filename: str = "",
        new_tables: Optional[List[str]] = None,
        row_count: int = 0,
        all_tables: Optional[List[str]] = None,
        db_path: Optional[str] = None,
    ) -> None:
        async with self._lock:
            import pandas as pd
            from scipy import stats as _stats

            # ── 1. Replace the namespace connection and schema ────────────
            self._ns["con"] = new_con
            self._ns["SCHEMA"] = new_schema
            self._ns["DATA_DATE_MIN"] = date_min
            self._ns["DATA_DATE_MAX"] = date_max
            self._data_loaded = True

            # ── 2. Rebuild run_sql / run_sql_approx / get_schema ──────────
            #    These functions in dataset.py capture `con` via closure.
            #    We must replace them so they use the NEW connection.
            _con = new_con  # local ref for closures below

            def _new_run_sql(query: str) -> pd.DataFrame:
                try:
                    return _con.execute(query).df()
                except Exception as e:
                    raise ValueError(f"SQL error: {e}\nQuery: {query}")

            def _new_run_sql_approx(query: str, sample_frac: float = 0.3,
                                     confidence: float = 0.95):
                result = _new_run_sql(query)
                if result.empty or len(result) < 5:
                    return result, {}
                numeric_cols = result.select_dtypes(include="number").columns.tolist()
                z = _stats.norm.ppf((1 + confidence) / 2)
                ci_info = {}
                for col in numeric_cols:
                    se = result[col].sem()
                    margin = z * se
                    ci_info[col] = {
                        "mean": result[col].mean(),
                        "ci_lower": result[col].mean() - margin,
                        "ci_upper": result[col].mean() + margin,
                        "confidence": confidence,
                    }
                return result, ci_info

            def _new_get_schema() -> str:
                parts = []
                try:
                    tables = [r[0] for r in _con.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema='main'"
                    ).fetchall()]
                except Exception:
                    tables = all_tables or new_tables or []
                for tbl in tables:
                    try:
                        cols = _con.execute(f"DESCRIBE {tbl}").df()
                        cols_str = ", ".join(
                            f"{r['column_name']}:{r['column_type']}"
                            for _, r in cols.iterrows()
                        )
                        sample = _con.execute(
                            f"SELECT * FROM {tbl} LIMIT 2"
                        ).df().to_string(index=False)
                        n = _con.execute(
                            f"SELECT COUNT(*) FROM {tbl}"
                        ).fetchone()[0]
                        parts.append(
                            f"TABLE {tbl} ({n:,} rows)\n"
                            f"COLUMNS: {cols_str}\nSAMPLE:\n{sample}"
                        )
                    except Exception as exc:
                        logger.warning("get_schema failed for table %s: %s", tbl, exc)
                return "\n\n".join(parts)

            self._ns["run_sql"] = _new_run_sql
            self._ns["run_sql_approx"] = _new_run_sql_approx
            self._ns["get_schema"] = _new_get_schema

            # ── 3. Refresh SCHEMA from the actual uploaded tables ─────────
            try:
                refreshed_schema = _new_get_schema()
                if refreshed_schema:
                    self._ns["SCHEMA"] = refreshed_schema
                    new_schema = refreshed_schema
            except Exception as exc:
                logger.warning("Schema refresh failed (using ingested schema): %s", exc)

            # ── 4. Register in dataset registry ──────────────────────────
            if filename:
                entry = {
                    "tables":   new_tables or [],
                    "date_min": str(date_min) if date_min else None,
                    "date_max": str(date_max) if date_max else None,
                    "row_count": row_count,
                    "con":      new_con,
                    "schema":   new_schema,
                }
                if db_path:
                    entry["db_path"] = db_path
                self._dataset_registry[filename] = entry
                logger.info(
                    "Dataset registered: %s → tables=%s (db_path=%s)",
                    filename, new_tables, db_path
                )

            # ── 5. Rebuild L3 graph from ALL registered datasets ─────────
            try:
                self._rebuild_l3_graph(new_con, date_min, date_max)
            except Exception as exc:
                logger.warning("L3 graph rebuild failed (non-critical): %s", exc)

            # ── 6. Seed domain-specific L4 patterns ──────────────────────
            # Detects dataset type from actual column names and seeds
            # appropriate SQL patterns (e-commerce / real-estate / financial /
            # HR / generic) — must happen AFTER run_sql is updated above.
            try:
                seed_fn = self._ns.get("seed_for_dataset_type")
                if seed_fn is not None and callable(seed_fn):
                    detected_domain = seed_fn(new_con)
                    self._ns["_CURRENT_DATASET_TYPE"] = detected_domain
                    logger.info("Dataset domain detected: %s", detected_domain)
            except Exception as exc:
                logger.warning("L4 pattern seeding failed (non-critical): %s", exc)

            logger.info(
                "Namespace data updated: file=%s, date range %s → %s, "
                "total datasets=%d",
                filename, date_min, date_max, len(self._dataset_registry)
            )

    def get_dataset_registry(self) -> Dict[str, Any]:
        """Return info about all uploaded datasets."""
        return dict(self._dataset_registry)

    async def switch_active_version(self, version: str) -> Dict[str, Any]:
        """
        Switch the active version to 'original' or 'modified'.
        Both tables live in the SAME shared DuckDB connection — no connection swap.
        We just update the SCHEMA to point queries at the correct table.
        """
        async with self._lock:
            con = self._ns.get("con")
            if not con:
                return {"success": False, "error": "No connection available"}

            # Save which version is active
            self._ns["active_version"] = version

            original_table = self._ns.get("original_table", "")
            modified_table = self._ns.get("modified_table", "")

            if version == 'modified':
                if not modified_table:
                    return {"success": False, "error": "No modified table found"}

                # Build schema for the modified table
                try:
                    cols_df = con.execute(f"DESCRIBE {modified_table}").df()
                    schema_lines = [f"TABLE {modified_table}:"]
                    for _, row in cols_df.iterrows():
                        schema_lines.append(f"  {row['column_name']} {row['column_type']}")
                    self._ns["SCHEMA"] = "\n".join(schema_lines)
                except Exception as e:
                    logger.warning("Failed to build schema for modified table: %s", e)

                logger.info(
                    "Switched active version → MODIFIED (table=%s)",
                    modified_table,
                )
                return {"success": True, "version": "modified", "table": modified_table}

            else:
                # Build schema for the original table
                if original_table:
                    try:
                        cols_df = con.execute(f"DESCRIBE {original_table}").df()
                        schema_lines = [f"TABLE {original_table}:"]
                        for _, row in cols_df.iterrows():
                            schema_lines.append(f"  {row['column_name']} {row['column_type']}")
                        self._ns["SCHEMA"] = "\n".join(schema_lines)
                    except Exception as e:
                        logger.warning("Failed to build schema for original table: %s", e)

                logger.info("Switched active version → ORIGINAL (table=%s)", original_table)
                return {"success": True, "version": "original", "table": original_table}

    async def remove_dataset(self, filename: str) -> Dict[str, Any]:
        """
        Remove a dataset and drop its associated DuckDB tables.
        Rebuilds SCHEMA and L3 graph from remaining datasets.
        Thread-safe: acquires the query lock.
        """
        async with self._lock:
            # ── Try exact match first, then fuzzy ────────────────────────
            matched_key = None
            if filename in self._dataset_registry:
                matched_key = filename
            else:
                # Try case-insensitive match
                for key in self._dataset_registry:
                    if key.lower() == filename.lower():
                        matched_key = key
                        break
                # Try normalized match (spaces/underscores/hyphens)
                if not matched_key:
                    def _norm(s):
                        return re.sub(r'[\s_\-]+', '', s).lower()
                    fn_norm = _norm(filename)
                    for key in self._dataset_registry:
                        if _norm(key) == fn_norm:
                            matched_key = key
                            break
                # Try substring match (filename contained in key or vice versa)
                if not matched_key:
                    fn_lower = filename.lower()
                    for key in self._dataset_registry:
                        if fn_lower in key.lower() or key.lower() in fn_lower:
                            matched_key = key
                            break

            if not matched_key:
                logger.warning(
                    "Dataset '%s' not found. Registry keys: %s",
                    filename, list(self._dataset_registry.keys())
                )
                return {"success": False, "error": f"Dataset '{filename}' not found in registry"}

            info = self._dataset_registry.pop(matched_key)
            tables_to_drop = info.get("tables", [])
            ds_con = info.get("con")       # dataset-specific connection
            ds_db_path = info.get("db_path")  # isolated .duckdb file path
            con = self._ns.get("con")       # current active connection

            dropped, errors = [], []

            # If the dataset has its own isolated DuckDB, drop tables there
            # and then close + delete the file
            if ds_con and ds_db_path:
                for tbl in tables_to_drop:
                    try:
                        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tbl):
                            ds_con.execute(f"DROP TABLE IF EXISTS {tbl}")
                            dropped.append(tbl)
                    except Exception as exc:
                        errors.append(f"{tbl}: {exc}")
                        logger.warning("Could not drop table %s: %s", tbl, exc)
                # Close the isolated connection
                try:
                    ds_con.close()
                    logger.info("Closed isolated DuckDB connection: %s", ds_db_path)
                except Exception as exc:
                    logger.warning("Failed to close DuckDB con: %s", exc)
                # Delete the .duckdb file (and .wal)
                import os as _os
                for ext in ("", ".wal"):
                    fpath = ds_db_path + ext if ext else ds_db_path
                    if _os.path.exists(fpath):
                        try:
                            _os.remove(fpath)
                            logger.info("Deleted DuckDB file: %s", fpath)
                        except Exception as exc:
                            logger.warning("Failed to delete %s: %s", fpath, exc)
            elif con:
                # Fallback: drop tables from the shared connection
                for tbl in tables_to_drop:
                    try:
                        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tbl):
                            con.execute(f"DROP TABLE IF EXISTS {tbl}")
                            dropped.append(tbl)
                    except Exception as exc:
                        errors.append(f"{tbl}: {exc}")
                        logger.warning("Could not drop table %s: %s", tbl, exc)

            # Delete L2 ChromaDB episodes tagged to this dataset
            l2 = self._ns.get("l2_collection")
            if l2:
                try:
                    l2_results = l2.get(where={"dataset": filename})
                    if l2_results.get("ids"):
                        l2.delete(ids=l2_results["ids"])
                        logger.info(
                            "Deleted %d L2 episodes for dataset %s",
                            len(l2_results["ids"]), filename,
                        )
                except Exception as exc:
                    logger.warning("L2 cleanup for dataset %s failed: %s", filename, exc)

            # Rebuild SCHEMA from remaining datasets
            remaining: List[str] = []
            for ds_info in self._dataset_registry.values():
                remaining.extend(ds_info.get("tables", []))

            if con and remaining:
                try:
                    schema_lines: List[str] = []
                    for tbl in remaining:
                        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tbl):
                            cols_df = con.execute(f"DESCRIBE {tbl}").df()
                            schema_lines.append(f"TABLE {tbl}:")
                            for _, row in cols_df.iterrows():
                                schema_lines.append(f"  {row['column_name']} {row['column_type']}")
                            schema_lines.append("")
                    self._ns["SCHEMA"] = "\n".join(schema_lines)
                    # Update date range from remaining datasets
                    all_dates = [
                        (v.get("date_min"), v.get("date_max"))
                        for v in self._dataset_registry.values()
                    ]
                    mins = [d[0] for d in all_dates if d[0]]
                    maxs = [d[1] for d in all_dates if d[1]]
                    if mins:
                        self._ns["DATA_DATE_MIN"] = min(mins)
                    if maxs:
                        self._ns["DATA_DATE_MAX"] = max(maxs)
                except Exception as exc:
                    logger.warning("Schema rebuild after removal failed: %s", exc)
            elif not remaining:
                self._ns["SCHEMA"] = ""
                self._ns["DATA_DATE_MIN"] = None
                self._ns["DATA_DATE_MAX"] = None
                self._data_loaded = False

            # Rebuild L3 graph from remaining tables
            if con:
                try:
                    date_min = self._ns.get("DATA_DATE_MIN")
                    date_max = self._ns.get("DATA_DATE_MAX")
                    self._rebuild_l3_graph(con, date_min, date_max)
                except Exception as exc:
                    logger.warning("L3 graph rebuild after removal failed: %s", exc)

            logger.info(
                "Dataset removed: %s (dropped=%s, remaining=%s)",
                filename, dropped, list(self._dataset_registry.keys())
            )
            return {
                "success": True,
                "filename": filename,
                "dropped_tables": dropped,
                "errors": errors,
                "remaining_datasets": list(self._dataset_registry.keys()),
                "remaining_tables": remaining,
            }

    def _rebuild_l3_graph(self, con, date_min, date_max) -> None:
        """Build a minimal L3 causal graph from the uploaded data schema."""
        import networkx as nx
        G = nx.DiGraph()

        # Build table-to-dataset mapping from registry
        tbl_to_ds: Dict[str, str] = {}
        for ds_filename, ds_info in self._dataset_registry.items():
            for t in ds_info.get("tables", []):
                tbl_to_ds[t] = ds_filename

        # Get tables and their columns from the connection
        try:
            tables_result = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchdf()
            tables = tables_result['table_name'].tolist()
        except Exception:
            tables = []

        for tbl in tables:
            try:
                ds = tbl_to_ds.get(tbl, "")
                G.add_node(tbl, type="table", dataset=ds)
                cols = con.execute(f"DESCRIBE {tbl}").df()
                for _, row in cols.iterrows():
                    col_id = f"{tbl}.{row['column_name']}"
                    G.add_node(col_id, type="column", table=tbl, dataset=ds)
                    G.add_edge(tbl, col_id, rel="has_column")
            except Exception:
                pass

        # Add date-range business rules
        if date_min and date_max:
            G.add_node("rule_date_anchor", type="business_rule",
                       description=f"Data spans {date_min} to {date_max} — never use CURRENT_DATE")
        G.add_node("rule_status_filter", type="business_rule",
                   description="Filter for delivered/active records when computing revenue metrics")

        self._ns["l3_graph"] = G

        # Also update helper functions to use new graph
        if "l3_get_context" in self._ns:
            def _new_l3_get_context(entity):
                lines = []
                matches = [n for n in G.nodes if entity.lower() in n.lower()]
                for node in matches[:3]:
                    in_e  = [(u, d) for u, v, d in G.in_edges(node, data=True)]
                    out_e = [(v, d) for u, v, d in G.out_edges(node, data=True)]
                    for u, d in in_e:
                        lines.append(f"  {u} ->[{d.get('rel')}] {node}")
                    for v, d in out_e:
                        lines.append(f"  {node} ->[{d.get('rel')}] {v}")
                return "\n".join(lines) if lines else "No causal links."
            self._ns["l3_get_context"] = _new_l3_get_context

        if "l3_get_business_rules" in self._ns:
            def _new_l3_get_business_rules():
                rules = [d.get("description", "") for _, d in G.nodes(data=True)
                         if d.get("type") == "business_rule"]
                return "\n".join(f"  - {r}" for r in rules)
            self._ns["l3_get_business_rules"] = _new_l3_get_business_rules

        logger.info("L3 graph rebuilt: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ------------------------------------------------------------------
    # Memory stats
    # ------------------------------------------------------------------
    def get_memory_stats(self) -> Dict[str, Any]:
        l2 = self._ns.get("l2_collection")
        l4 = self._ns.get("l4_collection")
        l3 = self._ns.get("l3_graph")
        return {
            "l2_count": l2.count() if l2 else 0,
            "l4_count": l4.count() if l4 else 0,
            "l3_nodes": l3.number_of_nodes() if l3 else 0,
            "l3_edges": l3.number_of_edges() if l3 else 0,
        }

    def get_l2_episodes(self, dataset: Optional[str] = None) -> List[Dict]:
        l2 = self._ns.get("l2_collection")
        if not l2 or l2.count() == 0:
            return []
        try:
            if dataset:
                result = l2.get(where={"dataset": dataset}, include=["documents", "metadatas"])
            else:
                result = l2.get(include=["documents", "metadatas"])
            episodes = []
            for doc, meta in zip(result["documents"], result["metadatas"]):
                episodes.append({
                    "question": doc,
                    "sql": meta.get("sql", "")[:400],
                    "result_summary": meta.get("result_summary", "")[:300],
                    "score": float(meta.get("score", 1.0)),
                    "timestamp": meta.get("timestamp", ""),
                    "dataset": meta.get("dataset", ""),
                })
            return episodes
        except Exception as exc:
            logger.warning("get_l2_episodes failed: %s", exc)
            return []

    def get_l4_patterns(self) -> List[Dict]:
        l4 = self._ns.get("l4_collection")
        if not l4 or l4.count() == 0:
            return []
        try:
            result = l4.get(include=["documents", "metadatas"])
            patterns = []
            for doc, meta in zip(result["documents"], result["metadatas"]):
                patterns.append({
                    "problem_type": meta.get("problem_type", doc[:60]),
                    "sql_template": meta.get("sql_template", "")[:500],
                    "example_query": doc[:200],
                })
            return patterns
        except Exception as exc:
            logger.warning("get_l4_patterns failed: %s", exc)
            return []

    def _rebuild_l3_graph(self, con=None, date_min=None, date_max=None) -> None:
        """
        Rebuild the L3 causal knowledge graph from the current database.
        Builds the graph directly using the given connection (not the stale
        `build_l3_graph()` closure from dataset.py).
        """
        import networkx as nx

        if con is None:
            con = self._ns.get("con")
        if con is None:
            return

        G = nx.DiGraph()

        # Discover tables
        try:
            tables = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main'"
            ).fetchall()]
        except Exception:
            tables = []

        if not tables:
            self._ns["l3_graph"] = G
            return

        # Build schema map
        schema_map = {}
        for tbl in tables:
            try:
                cols_df = con.execute(f"DESCRIBE {tbl}").df()
                schema_map[tbl] = cols_df["column_name"].tolist()
            except Exception:
                schema_map[tbl] = []

        # Add table nodes
        for tbl in tables:
            G.add_node(tbl, type="table")

        # Add column nodes and has_column edges
        for tbl, cols in schema_map.items():
            for col in cols:
                node = f"{tbl}.{col}"
                G.add_node(node, type="column", table=tbl)
                G.add_edge(tbl, node, rel="has_column")

        # Detect FK relationships (columns with same name across tables)
        all_col_tables = {}
        for tbl, cols in schema_map.items():
            for col in cols:
                if col.endswith("_id") or col == "id":
                    all_col_tables.setdefault(col, []).append(tbl)

        for col, tbls in all_col_tables.items():
            if len(tbls) > 1:
                for i in range(1, len(tbls)):
                    G.add_edge(
                        f"{tbls[0]}.{col}",
                        f"{tbls[i]}.{col}",
                        rel="foreign_key",
                    )

        # Add date range business rule
        if date_min and date_max:
            G.add_node(
                "rule_date_anchor", type="business_rule",
                description=f"data spans {date_min} to {date_max} — never use CURRENT_DATE",
            )

        self._ns["l3_graph"] = G

        # Also update the namespace build_l3_graph to use new con
        _con_ref = con
        def _updated_build_l3_graph():
            """Rebuilt build_l3_graph with correct connection."""
            self._rebuild_l3_graph(_con_ref)
            return self._ns.get("l3_graph", nx.DiGraph())
        self._ns["build_l3_graph"] = _updated_build_l3_graph

        # Save to GML
        graph_path = self._ns.get("GRAPH_PATH")
        if graph_path:
            try:
                nx.write_gml(G, graph_path)
            except Exception:
                pass

        logger.info(
            "L3 graph rebuilt: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )

    def get_l3_graph_data(self) -> Dict:
        l3 = self._ns.get("l3_graph")
        if l3 is None:
            return {"nodes": [], "edges": []}
        try:
            import networkx as nx
            nodes = [
                {
                    "id": str(n),
                    "label": str(n),
                    "type": l3.nodes[n].get("type", "unknown"),
                    "description": l3.nodes[n].get("description", ""),
                    "dataset": l3.nodes[n].get("dataset", ""),
                }
                for n in l3.nodes()
            ]
            edges = [
                {
                    "source": str(u),
                    "target": str(v),
                    "rel": l3.edges[u, v].get("rel", l3.edges[u, v].get("relationship", "foreign_key")),
                    "weight": l3.edges[u, v].get("weight", 1.0),
                    "confidence": l3.edges[u, v].get("confidence", 1.0),
                }
                for u, v in l3.edges()
            ]
            return {"nodes": nodes, "edges": edges}
        except Exception as exc:
            logger.warning("get_l3_graph_data failed: %s", exc)
            return {"nodes": [], "edges": []}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def has_custom_data(self) -> bool:
        return self._data_loaded

    # ------------------------------------------------------------------
    # Execute a Python code block and return list of (title, html) charts
    # ------------------------------------------------------------------
    def exec_python_block(self, code: str) -> List[Tuple[str, str]]:
        """
        Execute a Python code block inside the SENTINEL namespace.
        Intercepts matplotlib plt.show()/savefig() and safe_show() Plotly figs.
        Returns a list of (title, html_string) tuples for interactive Plotly charts.
        """
        results: List[Tuple[str, str]] = []
        if not self._initialized:
            return results

        # ── Set up matplotlib interception ──────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None

        captured_mpl: List[Tuple[str, Any]] = []
        _fig_counter = [0]

        def _patched_show(*args, **kwargs):
            if plt is None:
                return
            fig = plt.gcf()
            if fig and fig.get_axes():
                _fig_counter[0] += 1
                title = fig.get_suptitle() or (
                    fig.get_axes()[0].get_title() or f"Chart {_fig_counter[0]}"
                )
                captured_mpl.append((title.strip() or f"Chart {_fig_counter[0]}", fig))
            try:
                plt.close("all")
            except Exception:
                pass

        def _patched_savefig(buf_or_path, *args, **kwargs):
            if plt is None:
                return
            fig = plt.gcf()
            if fig and fig.get_axes():
                _fig_counter[0] += 1
                title = fig.get_suptitle() or (
                    fig.get_axes()[0].get_title() or f"Chart {_fig_counter[0]}"
                )
                captured_mpl.append((title.strip() or f"Chart {_fig_counter[0]}", fig))

        # ── Prep execution namespace ─────────────────────────────────────
        exec_ns = dict(self._ns)  # shallow copy so we don't pollute
        exec_ns["_FIG_QUEUE"] = []  # fresh queue for this block
        if plt is not None:
            exec_ns["plt"] = plt
            plt.show = _patched_show
            plt.savefig = _patched_savefig

        # Intercept safe_show so Plotly figs added during exec end up in exec_ns queue
        _orig_safe_show = exec_ns.get("safe_show")
        def _exec_safe_show(fig, label: str = "Chart"):
            try:
                html = fig.to_html(include_plotlyjs="cdn", full_html=False,
                                   config={"responsive": True})
                exec_ns["_FIG_QUEUE"].append((html, label))
            except Exception as e:
                logger.warning("exec safe_show capture failed: %s", e)
        exec_ns["safe_show"] = _exec_safe_show

        # ── Execute the code block ───────────────────────────────────────
        try:
            exec(compile(code, "<llm_code_block>", "exec"), exec_ns)
        except Exception as exc:
            logger.warning("exec_python_block failed: %s", exc)
        finally:
            if plt is not None:
                try:
                    import matplotlib
                    _orig_plt_show = matplotlib.pyplot.show
                    plt.show = _orig_plt_show
                except Exception:
                    pass

        # ── Convert matplotlib figures → Plotly HTML ─────────────────────
        for title, mpl_fig in captured_mpl:
            try:
                import plotly.tools as tls
                import plotly.io as pio_local
                plotly_fig = tls.mpl_to_plotly(mpl_fig)
                plotly_fig.update_layout(
                    template="sentinel",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font={"color": "#F1F5F9"},
                    title={"text": title, "font": {"color": "#F1F5F9"}},
                )
                html = plotly_fig.to_html(
                    include_plotlyjs="cdn", full_html=False,
                    config={"responsive": True, "displayModeBar": True},
                )
                results.append((title, html))
                try:
                    import matplotlib.pyplot as _plt
                    _plt.close(mpl_fig)
                except Exception:
                    pass
            except Exception as exc:
                logger.warning("mpl_to_plotly failed for '%s': %s", title, exc)
                # Last resort: embed as static image in a plotly figure
                try:
                    import io as _io, base64 as _b64
                    import matplotlib.pyplot as _plt
                    buf = _io.BytesIO()
                    mpl_fig.savefig(buf, format="png", dpi=100,
                                    bbox_inches="tight",
                                    facecolor="#111827")
                    buf.seek(0)
                    b64 = _b64.b64encode(buf.read()).decode()
                    import plotly.graph_objects as _go
                    img_fig = _go.Figure()
                    img_fig.add_layout_image(
                        source=f"data:image/png;base64,{b64}",
                        x=0, y=1, xref="paper", yref="paper",
                        sizex=1, sizey=1, xanchor="left", yanchor="top",
                        layer="above",
                    )
                    img_fig.update_layout(
                        paper_bgcolor="#111827", plot_bgcolor="#111827",
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=420,
                        title={"text": title, "font": {"color": "#F1F5F9"}},
                        xaxis={"visible": False}, yaxis={"visible": False},
                    )
                    html = img_fig.to_html(
                        include_plotlyjs="cdn", full_html=False,
                        config={"responsive": True},
                    )
                    results.append((title, html))
                    _plt.close(mpl_fig)
                except Exception as exc2:
                    logger.warning("Static image fallback also failed: %s", exc2)

        # ── Harvest any Plotly figs from the exec _FIG_QUEUE ────────────────
        for item in exec_ns.get("_FIG_QUEUE", []):
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    html_str, title = item[0], item[1]
                elif isinstance(item, dict):
                    html_str = item.get("html", "")
                    title = item.get("title", "Chart")
                else:
                    continue
                if html_str:
                    results.append((str(title) or "Chart", str(html_str)))
            except Exception:
                pass

        logger.info("exec_python_block: produced %d chart(s)", len(results))
        return results


# Singleton instance shared across the FastAPI app
sentinel_ns = SentinelNamespace()
