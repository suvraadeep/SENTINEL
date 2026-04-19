"""
SENTINEL Model Configuration
Multi-provider LLM setup — provider/model/key are injected by the backend
namespace engine (backend/core/namespace.py) after this file is exec()'d.
"""
import os, json, hashlib, warnings, textwrap, re, math
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import duckdb
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

import scipy.stats as stats
from scipy.stats import ttest_ind, pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

import sympy as sp
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

print("All imports OK")

# ── Provider configuration ────────────────────────────────────────────
# These are placeholders — backend/core/namespace.py replaces them with
# real LangChain LLM objects after this file is exec()'d.
API_KEY        = os.environ.get("OPENAI_API_KEY", "")
PROVIDER       = "openai"
MAIN_MODEL     = "gpt-4o"
FAST_MODEL     = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4o-mini"

# Placeholder LLM objects — replaced by _reconfigure_llm() in namespace.py
_main_llm = None
_fast_llm = None

MAX_RETRIES = 5
MAX_TOKENS  = 16384

# ── Paths and Supabase config (patched by namespace.py to real values) ─
DB_PATH       = "/kaggle/working/sentinel_ecom.duckdb"
GRAPH_PATH    = "/kaggle/working/l3_ecom.gml"
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY  = os.environ.get("SUPABASE_KEY", "")


CHART_CODE_HINT = (
    "\n\nIMPORTANT: When writing Python code that generates charts or plots, "
    "ALWAYS use plotly.express (px) or plotly.graph_objects (go) — NEVER matplotlib. "
    "To display a chart call: safe_show(fig, 'Chart Title'). "
    "Do NOT call plt.show(), plt.savefig(), or print base64 images."
)


def call_llm(prompt: str, system: str = "", model: str = MAIN_MODEL,
             temperature: float = 0.0) -> str:
    """
    Unified LLM caller using langchain message interface.
    Automatically picks _main_llm or _fast_llm based on model arg.
    Falls back to fast_llm on any error.
    """
    llm = _fast_llm if model == FAST_MODEL else _main_llm

    if llm is None:
        return "LLM_ERROR: LLM not yet initialised"

    # Append chart-code hint so the LLM prefers Plotly over matplotlib
    effective_system = (system or "") + CHART_CODE_HINT

    messages = []
    if effective_system:
        messages.append(SystemMessage(content=effective_system))
    messages.append(HumanMessage(content=prompt))

    try:
        resp = llm.bind(temperature=temperature).invoke(messages)
        return resp.content.strip()
    except Exception as e1:
        print(f"  [LLM] {model} failed: {e1} — trying fallback")
        try:
            resp = _fast_llm.bind(temperature=temperature).invoke(messages)
            return resp.content.strip()
        except Exception as e2:
            return f"LLM_ERROR: {e2}"


print(f"Model config loaded | provider placeholder: {PROVIDER}")
