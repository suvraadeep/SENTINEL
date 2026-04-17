"""
DataLab API — live data exploration and manipulation.

Endpoints:
  GET    /api/datalab/datasets                → list all uploaded datasets
  DELETE /api/datalab/datasets/{filename}     → remove a dataset + drop its tables
  GET    /api/datalab/tables                  → list all tables in DuckDB
  GET    /api/datalab/schema/{table}          → detailed column-level schema info
  POST   /api/datalab/identify-dataset        → AI: which dataset does this prompt target?
  GET    /api/datalab/preview/{table}         → preview rows + stats for a table
  POST   /api/datalab/execute                 → run a DataLab operation (filter/sort/agg)
  POST   /api/datalab/sql                     → run arbitrary SELECT SQL
  POST   /api/datalab/transform               → NL prompt → pandas/numpy transform (+ SQL fallback)
  GET    /api/datalab/autoplot/{table}        → auto-generate 5-6 Plotly charts
  POST   /api/datalab/plot                    → NL prompt → custom Plotly chart
  GET    /api/datalab/download/{table}        → download table as CSV
"""

from __future__ import annotations
import io
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.api.schemas import (
    DatasetInfo,
    DatasetRemoveResponse,
    DataLabPreviewResponse,
    DataLabOperationRequest,
    DataLabOperationResponse,
    DataLabSqlRequest,
    DataLabSqlResponse,
    DataLabTransformV2Request,
    DataLabTransformV2Response,
    DataLabSchemaResponse,
    DataLabSchemaColumn,
    DataLabSchemaQueryRequest,
    DataLabSchemaQueryResponse,
    DataLabIdentifyRequest,
    DataLabIdentifyResponse,
    DataLabPlotRequest,
    DataLabPlotResponse,
    DataLabAutoPlotResponse,
    DataLabPromoteRequest,
    DataLabPromoteResponse,
    DataLabDropTableResponse,
)
from backend.core.namespace import sentinel_ns

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datalab", tags=["datalab"])

_MAX_PREVIEW_ROWS = 500
_MAX_SQL_ROWS = 2000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_con():
    """Get the active DuckDB connection or raise 404."""
    if not sentinel_ns.is_initialized:
        raise HTTPException(status_code=400, detail="SENTINEL not initialized. Configure a provider first.")
    con = sentinel_ns._ns.get("con")
    if con is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload a file first.")
    return con


def _get_con_for_table(table: str):
    """
    Resolve the DuckDB connection for a given table.
    Since both original and modified tables live in the same shared
    connection, this always returns the shared connection.
    """
    return _get_con()


def _safe_val(v) -> Any:
    """Convert a value to a JSON-serializable form."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def _df_to_rows(df) -> List[Dict[str, Any]]:
    """Convert a pandas DataFrame to a JSON-safe list of dicts."""
    rows = []
    for _, row in df.iterrows():
        safe_row = {}
        for col, val in row.items():
            if isinstance(val, (pd.Timestamp,)):
                safe_row[col] = str(val)
            elif hasattr(val, 'item'):  # numpy scalar
                safe_row[col] = _safe_val(val.item())
            else:
                safe_row[col] = _safe_val(val)
        rows.append(safe_row)
    return rows


def _get_all_tables(con) -> List[str]:
    """Return all user tables from the connection."""
    try:
        result = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in result]
    except Exception:
        try:
            result = con.execute("SHOW TABLES").fetchall()
            return [r[0] for r in result]
        except Exception:
            return []


def _get_table_columns(con, table: str) -> List[str]:
    """Return list of column names for a table."""
    try:
        df = con.execute(f"DESCRIBE {table}").df()
        return df["column_name"].tolist()
    except Exception:
        return []


def _build_schema_context(con, tables: List[str]) -> str:
    """Build a schema string with table → column mapping for LLM context."""
    lines: List[str] = []
    for tbl in tables:
        try:
            cols_df = con.execute(f"DESCRIBE {tbl}").df()
            lines.append(f"TABLE {tbl}:")
            for _, row in cols_df.iterrows():
                lines.append(f"  {row['column_name']} ({row['column_type']})")
            lines.append("")
        except Exception:
            pass
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL Verifier — checks column references against actual schema
# ---------------------------------------------------------------------------

def _extract_sql_columns(sql: str) -> List[str]:
    """
    Rough heuristic: extract bare identifiers from SQL that look like column refs.
    We strip SQL keywords and return lowercase candidate column names.
    """
    _SQL_KW = frozenset([
        "select", "from", "where", "group", "by", "order", "having", "limit",
        "offset", "join", "inner", "left", "right", "outer", "on", "as", "and",
        "or", "not", "in", "between", "like", "null", "is", "distinct", "count",
        "sum", "avg", "min", "max", "case", "when", "then", "else", "end",
        "with", "cte", "over", "partition", "rows", "range", "desc", "asc",
        "true", "false", "cast", "coalesce", "nullif", "greatest", "least",
        "round", "floor", "ceil", "abs", "ln", "log", "sqrt", "power", "mod",
        "current_date", "current_timestamp", "extract", "epoch", "year", "month",
        "day", "hour", "minute", "second", "interval", "date_trunc", "date_add",
        "strftime", "to_date", "now", "stddev_pop", "stddev", "variance",
        "concat", "trim", "upper", "lower", "length", "substring", "replace",
        "split_part", "string_agg", "array_agg", "row_number", "rank",
        "dense_rank", "lag", "lead", "first_value", "last_value", "ntile",
        "using", "union", "all", "except", "intersect", "insert", "into",
        "values", "update", "set", "delete", "create", "table", "view",
        "int", "integer", "bigint", "double", "float", "varchar", "text",
        "boolean", "bool", "timestamp", "date", "numeric", "decimal",
    ])
    # Remove string literals
    sql_no_strings = re.sub(r"'[^']*'", " ", sql)
    sql_no_strings = re.sub(r'"[^"]*"', " ", sql_no_strings)
    # Remove numbers
    sql_no_strings = re.sub(r'\b\d+\.?\d*\b', " ", sql_no_strings)
    # Extract identifiers
    tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql_no_strings)
    return [t.lower() for t in tokens if t.lower() not in _SQL_KW]


def _verify_sql_columns(sql: str, actual_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if SQL references columns that don't exist.
    Returns (all_valid, list_of_missing_columns).
    """
    actual_lower = {c.lower() for c in actual_columns}
    sql_refs = _extract_sql_columns(sql)
    missing = [ref for ref in sql_refs if ref not in actual_lower]
    return len(missing) == 0, missing


def _try_fix_sql(sql: str, missing_cols: List[str], actual_columns: List[str],
                 call_llm_fn, fast_model: str, base_sql: str) -> Optional[str]:
    """Ask the LLM to rewrite SQL replacing missing columns with valid ones."""
    fix_prompt = f"""The following DuckDB SQL references columns that do NOT exist.

ORIGINAL SQL:
{sql}

MISSING COLUMNS (do not use these): {missing_cols}

ACTUAL AVAILABLE COLUMNS: {actual_columns}

BASE QUERY (the data source):
{base_sql}

TASK: Rewrite the SQL to accomplish the same goal using ONLY the actual available columns.
- If a missing column can be DERIVED (e.g. discount_percentage = discount_amount / base_amount * 100), compute it inline.
- Do NOT reference any column from the MISSING list.
- Return ONLY the corrected SQL SELECT statement, no markdown, no explanation.

CORRECTED SQL:"""
    try:
        fixed = call_llm_fn(fix_prompt, model=fast_model, temperature=0.0).strip()
        fixed = re.sub(r"^```[\w]*\s*", "", fixed).strip()
        fixed = re.sub(r"\s*```\s*$", "", fixed).strip()
        return fixed if fixed.upper().startswith("SELECT") or fixed.upper().startswith("WITH") else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pandas/NumPy transform engine
# ---------------------------------------------------------------------------

_PANDAS_SYSTEM_PROMPT = """You are a Python data transformation expert using pandas and numpy.
Generate Python code that transforms a DataFrame called `df`.
The result MUST be stored in a variable called `result_df`.
Use ONLY columns that exist in the DataFrame — never invent column names.
You may use: pd (pandas), np (numpy).
Return ONLY executable Python code — no imports, no markdown fences, no explanation.

CRITICAL RULES:
- ALWAYS start with `result_df = df.copy()` to preserve ALL original rows
- NEVER use groupby().agg() to create features — use groupby().transform() instead to keep all rows
- NEVER use merge/join/pivot to create features — use .map() or .transform() to maintain row count
- The output MUST have the SAME number of rows as the input unless the user explicitly asks to filter/aggregate/reduce
- When creating derived features, ADD new columns to the full DataFrame, do NOT collapse rows"""

_PANDAS_EXAMPLES = """
EXAMPLE TRANSFORMATIONS:
- Normalize numeric col: result_df = df.copy(); result_df['price_norm'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
- Z-score standardize: result_df = df.copy(); result_df['price_std'] = (df['price'] - df['price'].mean()) / df['price'].std()
- Log transform: result_df = df.copy(); result_df['amount_log'] = np.log1p(df['amount'].clip(lower=0))
- One-hot encode: result_df = pd.get_dummies(df, columns=['category'], prefix='cat', drop_first=False)
- Bin numeric: result_df = df.copy(); result_df['price_bin'] = pd.cut(df['price'], bins=3, labels=['low','mid','high'])
- Drop null rows: result_df = df.dropna(subset=['revenue'])
- Filter rows: result_df = df[df['status'] == 'delivered'].copy()
- Create total spending per customer (KEEP ALL ROWS): result_df = df.copy(); result_df['total_spending_per_customer'] = df.groupby('customer_id')['price'].transform('sum')
- Create average per group (KEEP ALL ROWS): result_df = df.copy(); result_df['avg_category_price'] = df.groupby('category')['price'].transform('mean')
- Create feature: result_df = df.copy(); result_df['revenue_per_unit'] = df['revenue'] / df['quantity'].replace(0, np.nan)
- Create count per group (KEEP ALL ROWS): result_df = df.copy(); result_df['order_count'] = df.groupby('customer_id')['order_id'].transform('count')
- Aggregate by group (REDUCES ROWS): result_df = df.groupby('category').agg(total_revenue=('revenue','sum'), avg_price=('price','mean')).reset_index()
- Pivot (REDUCES ROWS): result_df = df.pivot_table(values='revenue', index='date', columns='category', aggfunc='sum').reset_index()
- Correlation matrix: result_df = df.select_dtypes(include='number').corr().reset_index()
- Compute percentile rank: result_df = df.copy(); result_df['price_rank'] = df['price'].rank(pct=True)
- Compute discount_pct from discount_amount and base_amount: result_df = df.copy(); result_df['discount_pct'] = df['discount_amount'] / df['base_amount'].replace(0, np.nan) * 100

WARNING: When the user says 'create a feature for X per Y', use groupby().transform(), NOT groupby().agg().
CORRECT: result_df = df.copy(); result_df['total_per_customer'] = df.groupby('customer_id')['price'].transform('sum')
WRONG:   result_df = df.groupby('customer_id')['price'].sum().reset_index()  # This DROPS rows!
"""


# ---------------------------------------------------------------------------
# Pandas verifier helpers — self-healing code generation
# ---------------------------------------------------------------------------

_ROW_REDUCING_KEYWORDS = frozenset([
    "filter", "drop", "remove", "delete", "exclude", "where", "only",
    "subset", "aggregate", "group by", "groupby", "summarize", "summarise",
    "pivot", "pivot_table", "correlation", "corr", "top ", "bottom ",
    "sample", "head", "tail", "first ", "last ", "unique", "distinct",
    "deduplicate", "dedup", "reduce", "collapse", "rollup",
])


def _is_row_reducing_prompt(prompt: str) -> bool:
    """
    Heuristic: does the user's prompt intend to reduce the number of rows?
    If not, we expect result_df to have roughly the same row count as input df.
    """
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in _ROW_REDUCING_KEYWORDS)


def _validate_pandas_result(
    df_original: pd.DataFrame,
    result_df: pd.DataFrame,
    prompt: str,
) -> Optional[str]:
    """
    Validate the pandas transform result. Returns an error message if
    the result looks wrong, None if it looks acceptable.
    """
    orig_rows = len(df_original)
    result_rows = len(result_df)

    # Empty result is always suspicious
    if result_rows == 0 and orig_rows > 0:
        return (
            f"Result has 0 rows but input had {orig_rows} rows. "
            "The code likely filtered out all data incorrectly."
        )

    # If the prompt is NOT row-reducing, flag large drops
    if not _is_row_reducing_prompt(prompt):
        if orig_rows > 0 and result_rows < orig_rows * 0.9:
            return (
                f"Result has only {result_rows} rows (input had {orig_rows}) "
                f"— {100 - round(result_rows / orig_rows * 100)}% of rows were dropped. "
                "The user did NOT ask to filter/aggregate/reduce rows. "
                "Use `result_df = df.copy()` to preserve all rows and add new columns. "
                "For per-group features, use groupby().transform() NOT groupby().agg()."
            )

    # Check for NaN explosion in new columns
    orig_cols = set(df_original.columns)
    new_cols = [c for c in result_df.columns if c not in orig_cols]
    for col in new_cols[:5]:
        nan_pct = result_df[col].isna().mean()
        if nan_pct > 0.95 and result_rows > 10:
            return (
                f"New column '{col}' is {nan_pct*100:.0f}% NaN — "
                "the computation likely failed silently. "
                "Check column references and division-by-zero handling."
            )

    return None  # looks OK


_DANGEROUS = re.compile(
    r'\b(import\s+os|import\s+subprocess|import\s+sys|exec\s*\(|eval\s*\('
    r'|__import__|open\s*\(|shutil|pathlib\.Path\.unlink|rmdir|remove)\b',
    re.IGNORECASE
)


def _replay_prior_steps(df: pd.DataFrame, prior_step_codes: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Execute prior step codes in sequence to rebuild the accumulated DataFrame.
    Returns (accumulated_df, error_msg).  On error returns (original_df, error_msg).
    """
    for i, step_code in enumerate(prior_step_codes):
        if not step_code.strip():
            continue
        exec_ns = {"df": df.copy(), "pd": pd, "np": np, "result_df": None}
        try:
            exec(compile(step_code, f"<replay_step_{i}>", "exec"), exec_ns)
            replayed = exec_ns.get("result_df")
            if replayed is not None and isinstance(replayed, pd.DataFrame):
                df = replayed
        except Exception as exc:
            logger.warning("Prior step %d replay failed: %s", i, exc)
            return df, f"Prior step {i} replay failed: {str(exc)[:200]}"
    return df, None


def _generate_pandas_transform(
    df: pd.DataFrame,
    prompt: str,
    call_llm_fn,
    fast_model: str,
    prior_step_codes: Optional[List[str]] = None,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str], str]:
    """
    Generate and execute a pandas transformation with self-healing verifier.

    If prior_step_codes is provided, ALL prior steps are executed first so
    the LLM sees the fully-accumulated DataFrame (with all previously added
    columns) rather than the raw base table.

    Returns (code, result_df, error_msg, verifier_notes).
    """
    _MAX_RETRIES = 2

    # ── Replay all prior transform steps to get the accumulated df ──────────
    if prior_step_codes:
        df, replay_err = _replay_prior_steps(df, prior_step_codes)
        if replay_err:
            logger.warning("Step replay had errors (continuing): %s", replay_err)

    # ── Build schema context (shared across retries) ────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if df[c].nunique() <= 60]

    schema_lines = [f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = round(df[col].isna().mean() * 100, 1)
        if col in num_cols:
            s = df[col].dropna()
            schema_lines.append(
                f"  {col} ({dtype}): min={s.min():.3g}, max={s.max():.3g}, "
                f"mean={s.mean():.3g}, nulls={null_pct}%"
            )
        elif col in cat_cols:
            uniq = df[col].dropna().unique()[:8].tolist()
            schema_lines.append(f"  {col} ({dtype}): unique={uniq}, nulls={null_pct}%")
        else:
            schema_lines.append(f"  {col} ({dtype}), nulls={null_pct}%")

    schema_ctx = "\n".join(schema_lines)
    sample_str = df.head(3).to_string(index=False)

    verifier_notes_parts: List[str] = []
    last_code = ""
    last_error = ""

    for attempt in range(_MAX_RETRIES + 1):
        # ── Build the LLM prompt ─────────────────────────────────────────
        if attempt == 0:
            llm_prompt = f"""{_PANDAS_SYSTEM_PROMPT}

{_PANDAS_EXAMPLES}

DATAFRAME SCHEMA:
{schema_ctx}

SAMPLE (first 3 rows):
{sample_str}

USER REQUEST: {prompt}

IMPORTANT:
- Store the final result in `result_df`
- ONLY use column names listed in DATAFRAME SCHEMA above
- If the user asks for a derived feature (like discount_percentage), compute it from existing columns
- Keep all original columns unless explicitly asked to drop them
- ALWAYS start with `result_df = df.copy()` when adding new columns
- Never use groupby/aggregate unless the user explicitly asks for aggregation or summary

PYTHON CODE:"""
        else:
            # Retry prompt with error context
            llm_prompt = f"""{_PANDAS_SYSTEM_PROMPT}

{_PANDAS_EXAMPLES}

DATAFRAME SCHEMA:
{schema_ctx}

SAMPLE (first 3 rows):
{sample_str}

USER REQUEST: {prompt}

YOUR PREVIOUS CODE WAS INCORRECT:
```python
{last_code}
```

ERROR / VALIDATION FAILURE:
{last_error}

FIX THE CODE. Rules:
- Store the final result in `result_df`
- ONLY use column names listed in DATAFRAME SCHEMA above
- Start with `result_df = df.copy()` to preserve all original rows
- Do NOT use groupby/aggregate unless the user explicitly asks for it
- The result MUST have {len(df)} rows (same as input) unless the user asked to filter/reduce
- Handle division by zero with .replace(0, np.nan) or similar

CORRECTED PYTHON CODE:"""

        try:
            code = call_llm_fn(llm_prompt, model=fast_model, temperature=0.0).strip()
        except Exception as exc:
            return "", None, f"LLM error: {str(exc)[:200]}", ""

        # Strip markdown fences
        code = re.sub(r"^```[\w]*\s*", "", code).strip()
        code = re.sub(r"\s*```\s*$", "", code).strip()

        if _DANGEROUS.search(code):
            return code, None, "Generated code contains disallowed operations.", ""

        last_code = code

        # ── Execute the code ─────────────────────────────────────────────
        exec_ns = {
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "result_df": None,
        }
        try:
            exec(compile(code, "<transform>", "exec"), exec_ns)
            result_df = exec_ns.get("result_df")
            if result_df is None:
                last_error = "Transform code did not set `result_df`."
                verifier_notes_parts.append(
                    f"⚠ Attempt {attempt + 1}: code did not produce result_df"
                )
                if attempt < _MAX_RETRIES:
                    continue
                return code, None, last_error, " → ".join(verifier_notes_parts)

            if not isinstance(result_df, pd.DataFrame):
                last_error = f"`result_df` must be a DataFrame, got {type(result_df).__name__}."
                verifier_notes_parts.append(
                    f"⚠ Attempt {attempt + 1}: result_df was {type(result_df).__name__}"
                )
                if attempt < _MAX_RETRIES:
                    continue
                return code, None, last_error, " → ".join(verifier_notes_parts)

        except Exception as exc:
            last_error = f"Execution error: {str(exc)[:400]}"
            verifier_notes_parts.append(
                f"⚠ Attempt {attempt + 1}: execution error — {str(exc)[:100]}"
            )
            if attempt < _MAX_RETRIES:
                continue
            return code, None, last_error, " → ".join(verifier_notes_parts)

        # ── Validate the result ──────────────────────────────────────────
        validation_error = _validate_pandas_result(df, result_df, prompt)
        if validation_error:
            last_error = validation_error
            verifier_notes_parts.append(
                f"⚠ Attempt {attempt + 1}: {validation_error[:120]}"
            )
            logger.info(
                "Pandas verifier (attempt %d/%d): %s",
                attempt + 1, _MAX_RETRIES + 1, validation_error,
            )
            if attempt < _MAX_RETRIES:
                continue
            # All retries exhausted and validation STILL failing —
            # return None so the caller falls back to SQL.
            verifier_notes_parts.append("All pandas retries failed validation → SQL fallback.")
            return code, None, validation_error, " → ".join(verifier_notes_parts)

        # ── Success — passed validation ──────────────────────────────────
        if attempt > 0:
            verifier_notes_parts.append(
                f"✓ Fixed on attempt {attempt + 1}"
            )
        return code, result_df, None, " → ".join(verifier_notes_parts)

    # Should not reach here, but just in case
    return last_code, None, last_error, " → ".join(verifier_notes_parts)



# ---------------------------------------------------------------------------
# Dataset identification AI
# ---------------------------------------------------------------------------

def _identify_dataset_from_prompt(
    prompt: str,
    tables: List[str],
    registry: Dict[str, Any],
    con,
    call_llm_fn,
    fast_model: str,
) -> DataLabIdentifyResponse:
    """
    Identify which dataset/table a prompt is about.
    Multi-stage:
      1. Exact match: table or filename substring in prompt
      2. Column-name match: columns unique to one table appear in prompt
      3. LLM classification: use schema context + prompt
    """
    if not tables:
        return DataLabIdentifyResponse(ambiguous=True, reason="No tables loaded", confidence=0.0)

    if len(tables) == 1:
        return DataLabIdentifyResponse(
            table=tables[0],
            confidence=1.0,
            reason="Only one table loaded",
            ambiguous=False,
        )

    prompt_lower = prompt.lower()

    # Stage 1: exact/substring match of table name or filename in prompt
    for tbl in tables:
        if tbl.lower() in prompt_lower:
            # Find which dataset owns this table
            owner = next(
                (fn for fn, info in registry.items() if tbl in info.get("tables", [])),
                None
            )
            return DataLabIdentifyResponse(
                table=tbl,
                dataset=owner,
                confidence=0.95,
                reason=f"Table name '{tbl}' found in prompt",
                ambiguous=False,
            )

    for filename in registry:
        # Use the stem of the filename (without extension)
        stem = re.sub(r'\.[^.]+$', '', filename).replace("_", " ").replace("-", " ").lower()
        if stem in prompt_lower or filename.lower() in prompt_lower:
            tables_in_ds = registry[filename].get("tables", [])
            if tables_in_ds:
                return DataLabIdentifyResponse(
                    table=tables_in_ds[0],
                    dataset=filename,
                    confidence=0.9,
                    reason=f"Dataset name '{filename}' found in prompt",
                    ambiguous=False,
                )

    # Stage 2: column-name intersection per table
    table_col_scores: Dict[str, int] = {}
    for tbl in tables:
        try:
            cols = _get_table_columns(con, tbl)
            # Count how many column names appear in the prompt
            score = sum(
                1 for c in cols
                if len(c) >= 4 and c.lower() in prompt_lower
            )
            table_col_scores[tbl] = score
        except Exception:
            table_col_scores[tbl] = 0

    max_score = max(table_col_scores.values(), default=0)
    if max_score > 0:
        best_tables = [t for t, s in table_col_scores.items() if s == max_score]
        if len(best_tables) == 1:
            best = best_tables[0]
            owner = next(
                (fn for fn, info in registry.items() if best in info.get("tables", [])),
                None
            )
            return DataLabIdentifyResponse(
                table=best,
                dataset=owner,
                confidence=0.8,
                reason=f"Column names from '{best}' match prompt ({max_score} matches)",
                ambiguous=False,
            )

    # Stage 3: LLM classification
    schema_ctx = _build_schema_context(con, tables)
    classify_prompt = f"""You must identify which database table the user's request is about.

AVAILABLE TABLES AND THEIR SCHEMAS:
{schema_ctx}

USER REQUEST: {prompt}

Which table is this request about? Reply with ONLY the table name from the list above.
If the request could apply to multiple tables, reply with the MOST relevant one.
Table name:"""

    try:
        answer = call_llm_fn(classify_prompt, model=fast_model, temperature=0.0).strip()
        # Clean up answer
        answer = answer.strip().strip('"').strip("'").split()[0] if answer else ""
        if answer in tables:
            owner = next(
                (fn for fn, info in registry.items() if answer in info.get("tables", [])),
                None
            )
            return DataLabIdentifyResponse(
                table=answer,
                dataset=owner,
                confidence=0.75,
                reason=f"LLM identified table '{answer}'",
                ambiguous=False,
                candidates=tables,
            )
    except Exception as exc:
        logger.warning("LLM dataset identification failed: %s", exc)

    # Fallback: ambiguous — return all candidates
    return DataLabIdentifyResponse(
        ambiguous=True,
        reason="Could not identify the target dataset from the prompt",
        confidence=0.0,
        candidates=tables,
    )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

_CHART_COLORS = [
    "#3B82F6", "#06B6D4", "#8B5CF6", "#10B981",
    "#F59E0B", "#EF4444", "#EC4899", "#14B8A6",
]

_DARK_LAYOUT = dict(
    paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(color="#F1F5F9", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#1E293B", tickfont=dict(color="#94A3B8"), linecolor="#334155"),
    yaxis=dict(gridcolor="#1E293B", tickfont=dict(color="#94A3B8"), linecolor="#334155"),
    legend=dict(bgcolor="#1E293B", bordercolor="#334155", font=dict(color="#94A3B8")),
    colorway=_CHART_COLORS,
    margin=dict(l=50, r=20, t=50, b=50),
)


def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(
        include_plotlyjs="cdn", full_html=False,
        config={"responsive": True, "displayModeBar": True},
    )


def _apply_dark(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(color="#F1F5F9", size=14)),
                      height=height, **_DARK_LAYOUT)
    return fig


# ===========================================================================
# ENDPOINTS
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /api/datalab/datasets
# ---------------------------------------------------------------------------
@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """Return all registered datasets with their table names and row counts."""
    _get_con()  # ensure initialized
    registry = sentinel_ns.get_dataset_registry()
    result = []
    for filename, info in registry.items():
        result.append(DatasetInfo(
            filename=filename,
            tables=info.get("tables", []),
            row_count=info.get("row_count", 0),
            date_min=info.get("date_min"),
            date_max=info.get("date_max"),
        ))
    return result


# ---------------------------------------------------------------------------
# DELETE /api/datalab/datasets/{filename}
# ---------------------------------------------------------------------------
@router.delete("/datasets/{filename:path}", response_model=DatasetRemoveResponse)
async def remove_dataset(filename: str):
    """Remove a dataset and drop its DuckDB tables. Thread-safe."""
    import urllib.parse
    filename = urllib.parse.unquote(filename)  # handle %20 etc.
    _get_con()  # ensure initialized
    result = await sentinel_ns.remove_dataset(filename)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    return DatasetRemoveResponse(
        success=True,
        filename=filename,
        dropped_tables=result.get("dropped_tables", []),
        remaining_datasets=result.get("remaining_datasets", []),
    )


# ---------------------------------------------------------------------------
# GET /api/datalab/tables
# ---------------------------------------------------------------------------
@router.get("/tables", response_model=List[str])
async def list_tables():
    """Return all tables in the active DuckDB connection plus isolated DBs."""
    con = _get_con()
    all_tbls = set(_get_all_tables(con))
    # Also include tables from isolated DuckDB connections in the registry
    registry = sentinel_ns.get_dataset_registry()
    for _fn, info in registry.items():
        for t in info.get("tables", []):
            all_tbls.add(t)
    return sorted(all_tbls)


# ---------------------------------------------------------------------------
# GET /api/datalab/schema/{table}  — detailed column-level info
# ---------------------------------------------------------------------------
@router.get("/schema/{table}", response_model=DataLabSchemaResponse)
async def get_schema(table: str):
    """Return detailed schema info: nulls, cardinality, sample values, stats."""
    con = _get_con_for_table(table)
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
        raise HTTPException(status_code=400, detail="Invalid table name.")
    if table not in _get_all_tables(con):
        raise HTTPException(status_code=404, detail=f"Table '{table}' not found.")

    try:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        # Load full data for stats (capped at 50k for performance)
        df = con.execute(f"SELECT * FROM {table} LIMIT 50000").df()

        columns: List[DataLabSchemaColumn] = []
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = round(null_count / max(len(df), 1) * 100, 2)
            unique_count = int(df[col].nunique())
            sample_raw = df[col].dropna().head(8).tolist()
            sample_values = [_safe_val(v) for v in sample_raw]

            col_info = DataLabSchemaColumn(
                name=col,
                dtype=str(df[col].dtype),
                null_count=null_count,
                null_pct=null_pct,
                unique_count=unique_count,
                sample_values=sample_values,
            )
            # Numeric stats
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna()
                if len(s):
                    col_info.min_val = _safe_val(float(s.min()))
                    col_info.max_val = _safe_val(float(s.max()))
                    col_info.mean_val = _safe_val(round(float(s.mean()), 4))
            columns.append(col_info)

        memory_mb = round(df.memory_usage(deep=True).sum() / 1e6, 3)
        return DataLabSchemaResponse(
            table=table,
            row_count=row_count,
            col_count=len(df.columns),
            columns=columns,
            memory_mb=memory_mb,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Schema info failed for %s: %s", table, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema info failed: {str(exc)[:200]}")


# ---------------------------------------------------------------------------
# POST /api/datalab/schema/{table}/query  — NL queries about the schema
# ---------------------------------------------------------------------------
_SCHEMA_PATTERNS = [
    # (keyword_patterns, code_template, description)
    (["null", "missing", "nan", "na "],
     "result_df = df.isnull().sum().rename('null_count').reset_index()\nresult_df.columns = ['column', 'null_count']\nresult_df['null_pct'] = (result_df['null_count'] / len(df) * 100).round(2)",
     "null_counts"),
    (["duplicate", "dupl"],
     "dup_count = df.duplicated().sum()\nresult_df = pd.DataFrame([{'duplicate_rows': int(dup_count), 'pct': round(dup_count/len(df)*100,2)}])",
     "duplicates"),
    (["dtype", "type", "data type", "column type"],
     "result_df = df.dtypes.reset_index()\nresult_df.columns = ['column', 'dtype']",
     "dtypes"),
    (["describe", "statistic", "summary stat", "stats"],
     "result_df = df.describe(include='all').T.reset_index()\nresult_df.rename(columns={'index':'column'}, inplace=True)",
     "describe"),
    (["unique count", "cardinality", "nunique", "unique value"],
     "result_df = df.nunique().reset_index()\nresult_df.columns = ['column', 'unique_count']",
     "unique_counts"),
    (["memory", "size", "footprint"],
     "mem = df.memory_usage(deep=True)\nresult_df = pd.DataFrame({'column': mem.index, 'bytes': mem.values})",
     "memory_usage"),
    (["correlat"],
     "result_df = df.select_dtypes(include='number').corr().reset_index()\nresult_df.rename(columns={'index':'column'}, inplace=True)",
     "correlation"),
    (["value count", "frequency", "top value", "most common"],
     "vc = {c: df[c].value_counts().head(10).to_dict() for c in df.select_dtypes(include=['object','category']).columns[:5]}\nresult_df = pd.DataFrame([{'column': k, 'top_values': str(v)} for k, v in vc.items()])",
     "value_counts"),
    (["outlier", "anomal"],
     "num = df.select_dtypes(include='number')\nrows = []\nfor c in num.columns:\n    q1,q3 = num[c].quantile(0.25), num[c].quantile(0.75)\n    iqr = q3-q1\n    out = ((num[c] < q1-1.5*iqr) | (num[c] > q3+1.5*iqr)).sum()\n    rows.append({'column':c,'outlier_count':int(out),'pct':round(out/len(df)*100,2)})\nresult_df = pd.DataFrame(rows)",
     "outliers"),
    (["skew"],
     "result_df = df.select_dtypes(include='number').skew().reset_index()\nresult_df.columns = ['column', 'skewness']",
     "skewness"),
    (["kurtosis"],
     "result_df = df.select_dtypes(include='number').kurtosis().reset_index()\nresult_df.columns = ['column', 'kurtosis']",
     "kurtosis"),
    (["shape", "dimension", "how many row", "how many col", "row count", "col count"],
     "result_df = pd.DataFrame([{'rows': len(df), 'columns': len(df.columns)}])",
     "shape"),
    (["numeric column", "numerical column", "number column"],
     "result_df = pd.DataFrame({'numeric_columns': df.select_dtypes(include='number').columns.tolist()})",
     "numeric_columns"),
    (["categorical column", "category column", "text column", "string column", "object column"],
     "result_df = pd.DataFrame({'categorical_columns': df.select_dtypes(include=['object','category']).columns.tolist()})",
     "categorical_columns"),
    (["date column", "datetime column", "time column", "temporal column"],
     "date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or str(df[c].dtype).startswith('datetime')]\nresult_df = pd.DataFrame({'date_columns': date_cols})",
     "date_columns"),
    (["min max", "minimum maximum", "range", "min and max"],
     "num = df.select_dtypes(include='number')\nresult_df = pd.DataFrame({'column': num.columns, 'min': num.min().values, 'max': num.max().values})",
     "min_max"),
    (["zero", "zero value"],
     "result_df = (df == 0).sum().reset_index()\nresult_df.columns = ['column', 'zero_count']",
     "zero_values"),
    (["constant", "single value", "no variance"],
     "cols = [c for c in df.columns if df[c].nunique() <= 1]\nresult_df = pd.DataFrame({'constant_columns': cols, 'count': len(cols)})",
     "constant_columns"),
    (["high cardinality", "many unique", "high unique"],
     "result_df = pd.DataFrame({'column': [c for c in df.columns if df[c].nunique() > 50], 'unique_count': [df[c].nunique() for c in df.columns if df[c].nunique() > 50]})",
     "high_cardinality"),
    (["sample", "head", "first row", "preview", "example"],
     "result_df = df.head(10)",
     "sample_rows"),
]


@router.post("/schema/{table}/query", response_model=DataLabSchemaQueryResponse)
async def query_schema_nl(table: str, req: DataLabSchemaQueryRequest):
    """
    Answer a natural-language question about a table's schema.
    Uses 20 hardcoded keyword-based patterns first; falls back to LLM for unknowns.
    """
    con = _get_con()
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
        raise HTTPException(status_code=400, detail="Invalid table name.")
    if table not in _get_all_tables(con):
        raise HTTPException(status_code=404, detail=f"Table '{table}' not found.")

    try:
        df = con.execute(f"SELECT * FROM {table} LIMIT 50000").df()
    except Exception as exc:
        return DataLabSchemaQueryResponse(success=False, error=f"Cannot load table: {exc}")

    prompt_lower = req.prompt.lower()

    # ── Hardcoded pattern matching ───────────────────────────────────────────
    matched_code: Optional[str] = None
    matched_mode = "hardcoded"
    for keywords, code_template, _ in _SCHEMA_PATTERNS:
        if any(kw in prompt_lower for kw in keywords):
            matched_code = code_template
            break

    # ── LLM fallback for unrecognised prompts ────────────────────────────────
    if matched_code is None:
        call_llm_fn = sentinel_ns._ns.get("call_llm")
        fast_model = sentinel_ns._ns.get("FAST_MODEL", "")
        if call_llm_fn:
            matched_mode = "llm"
            schema_ctx = "\n".join(
                f"  {col} ({df[col].dtype})" for col in df.columns
            )
            llm_prompt = f"""You are a pandas expert. The user wants to query the schema/statistics of a DataFrame.

DataFrame '{table}' ({len(df)} rows × {len(df.columns)} cols):
{schema_ctx}

USER QUESTION: {req.prompt}

Write Python code that answers the question using the DataFrame `df`.
Store the result in `result_df` (must be a pandas DataFrame).
Use only pandas/numpy — no matplotlib, no plotly.
Do NOT import anything.
Return ONLY the code, no explanations.

CODE:"""
            try:
                raw = call_llm_fn(llm_prompt, model=fast_model, temperature=0.0).strip()
                matched_code = re.sub(r"^```[\w]*\s*", "", raw).strip()
                matched_code = re.sub(r"\s*```\s*$", "", matched_code).strip()
            except Exception as exc:
                return DataLabSchemaQueryResponse(success=False, error=f"LLM error: {exc}")
        else:
            return DataLabSchemaQueryResponse(success=False, error="Unrecognized query and no LLM available.")

    if _DANGEROUS.search(matched_code):
        return DataLabSchemaQueryResponse(success=False, error="Generated code contains disallowed operations.")

    # ── Execute ───────────────────────────────────────────────────────────────
    exec_ns = {"df": df.copy(), "pd": pd, "np": np, "result_df": None}
    try:
        exec(compile(matched_code, "<schema_query>", "exec"), exec_ns)
        result_df = exec_ns.get("result_df")
        if result_df is None or not isinstance(result_df, pd.DataFrame):
            return DataLabSchemaQueryResponse(success=False, code=matched_code, error="Code did not produce a DataFrame in result_df.")
        result_json = result_df.head(200).to_json(orient="records", default_handler=str)
        fmt = "table" if len(result_df.columns) <= 8 else "json"
        return DataLabSchemaQueryResponse(
            success=True,
            code=matched_code,
            result_json=result_json,
            format=fmt,
            mode=matched_mode,
        )
    except Exception as exc:
        return DataLabSchemaQueryResponse(success=False, code=matched_code, error=f"Execution error: {str(exc)[:300]}")


# ---------------------------------------------------------------------------
# POST /api/datalab/identify-dataset
# ---------------------------------------------------------------------------
@router.post("/identify-dataset", response_model=DataLabIdentifyResponse)
async def identify_dataset(req: DataLabIdentifyRequest):
    """AI-powered: determine which table a prompt is targeting."""
    con = _get_con()
    tables = _get_all_tables(con)
    registry = sentinel_ns.get_dataset_registry()

    call_llm_fn = sentinel_ns._ns.get("call_llm")
    fast_model = sentinel_ns._ns.get("FAST_MODEL", "")

    if not call_llm_fn:
        # No LLM — do heuristic only
        call_llm_fn = None

    result = _identify_dataset_from_prompt(
        req.prompt, tables, registry, con,
        call_llm_fn or (lambda p, **kw: ""),
        fast_model,
    )
    return result


# ---------------------------------------------------------------------------
# GET /api/datalab/preview/{table}
# ---------------------------------------------------------------------------
@router.get("/preview/{table}", response_model=DataLabPreviewResponse)
async def preview_table(table: str, limit: int = 100, offset: int = 0):
    """Return a preview of a table's contents with column stats."""
    con = _get_con_for_table(table)

    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
        raise HTTPException(status_code=400, detail="Invalid table name.")

    tables = _get_all_tables(con)
    if table not in tables:
        raise HTTPException(status_code=404, detail=f"Table '{table}' not found.")

    limit = min(limit, _MAX_PREVIEW_ROWS)

    try:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        df = con.execute(f"SELECT * FROM {table} LIMIT {limit} OFFSET {offset}").df()

        columns = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in columns}
        rows = _df_to_rows(df)

        numeric_summary = {}
        for col in df.select_dtypes(include='number').columns:
            s = df[col].dropna()
            if len(s) > 0:
                numeric_summary[col] = {
                    "min":    _safe_val(float(s.min())),
                    "max":    _safe_val(float(s.max())),
                    "mean":   _safe_val(float(s.mean())),
                    "median": _safe_val(float(s.median())),
                    "std":    _safe_val(float(s.std())),
                    "nulls":  int(df[col].isna().sum()),
                }

        return DataLabPreviewResponse(
            table=table,
            columns=columns,
            dtypes=dtypes,
            row_count=row_count,
            rows=rows,
            numeric_summary=numeric_summary,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Preview failed for table %s: %s", table, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(exc)[:200]}")


# ---------------------------------------------------------------------------
# POST /api/datalab/execute
# ---------------------------------------------------------------------------
@router.post("/execute", response_model=DataLabOperationResponse)
async def execute_operation(req: DataLabOperationRequest):
    """Execute a structured DataLab operation on a table."""
    con = _get_con()

    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', req.table):
        raise HTTPException(status_code=400, detail="Invalid table name.")

    tables = _get_all_tables(con)
    if req.table not in tables:
        raise HTTPException(status_code=404, detail=f"Table '{req.table}' not found.")

    try:
        op = req.operation.lower()
        p = req.params

        if op == "filter":
            col = p.get("column", "")
            oper = p.get("operator", "=")
            val = p.get("value", "")
            safe_ops = {"=", "!=", "<", "<=", ">", ">=", "LIKE", "NOT LIKE", "IN", "IS NULL", "IS NOT NULL"}
            if oper.upper() not in safe_ops:
                raise HTTPException(status_code=400, detail=f"Unsupported operator: {oper}")
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                raise HTTPException(status_code=400, detail="Invalid column name.")
            if oper.upper() in ("IS NULL", "IS NOT NULL"):
                sql = f"SELECT * FROM {req.table} WHERE {col} {oper.upper()} LIMIT 2000"
            elif isinstance(val, str):
                escaped = val.replace("'", "''")
                sql = f"SELECT * FROM {req.table} WHERE {col} {oper} '{escaped}' LIMIT 2000"
            else:
                sql = f"SELECT * FROM {req.table} WHERE {col} {oper} {val} LIMIT 2000"
            df = con.execute(sql).df()

        elif op == "sort":
            col = p.get("column", "")
            asc = p.get("ascending", True)
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                raise HTTPException(status_code=400, detail="Invalid column name.")
            direction = "ASC" if asc else "DESC"
            df = con.execute(f"SELECT * FROM {req.table} ORDER BY {col} {direction} LIMIT 2000").df()

        elif op == "aggregate":
            group_by = p.get("group_by", "")
            metric = p.get("metric", "")
            agg_func = p.get("agg_func", "SUM").upper()
            safe_aggs = {"SUM", "AVG", "COUNT", "MIN", "MAX", "COUNT DISTINCT"}
            if agg_func not in safe_aggs:
                raise HTTPException(status_code=400, detail=f"Unsupported aggregation: {agg_func}")
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', group_by):
                raise HTTPException(status_code=400, detail="Invalid group_by column.")
            if agg_func == "COUNT DISTINCT":
                agg_expr = f"COUNT(DISTINCT {metric}) AS result"
            elif agg_func == "COUNT":
                agg_expr = "COUNT(*) AS result"
            else:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', metric):
                    raise HTTPException(status_code=400, detail="Invalid metric column.")
                agg_expr = f"{agg_func}({metric}) AS result"
            df = con.execute(
                f"SELECT {group_by}, {agg_expr} FROM {req.table} "
                f"GROUP BY {group_by} ORDER BY result DESC LIMIT 500"
            ).df()

        elif op == "sample":
            n = min(int(p.get("n", 100)), 1000)
            df = con.execute(f"SELECT * FROM {req.table} USING SAMPLE {n} ROWS LIMIT {n}").df()

        elif op == "describe":
            df = con.execute(f"SUMMARIZE {req.table}").df()

        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {op}")

        return DataLabOperationResponse(
            success=True,
            table=req.table,
            operation=op,
            columns=list(df.columns),
            rows=_df_to_rows(df),
            row_count=len(df),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DataLab operation failed: %s", exc, exc_info=True)
        return DataLabOperationResponse(
            success=False,
            table=req.table,
            operation=req.operation,
            error=str(exc)[:300],
        )


# ---------------------------------------------------------------------------
# POST /api/datalab/sql
# ---------------------------------------------------------------------------
@router.post("/sql", response_model=DataLabSqlResponse)
async def run_sql(req: DataLabSqlRequest):
    """Execute a user-provided SELECT SQL query. Only SELECT statements permitted."""
    con = _get_con()

    sql = req.sql.strip()
    first_word = sql.split()[0].upper() if sql else ""
    if first_word not in ("SELECT", "WITH", "SHOW", "DESCRIBE", "SUMMARIZE", "EXPLAIN"):
        raise HTTPException(
            status_code=400,
            detail="Only SELECT/WITH/SHOW/DESCRIBE/SUMMARIZE/EXPLAIN queries are allowed."
        )

    try:
        df = con.execute(sql).df()
        if len(df) > _MAX_SQL_ROWS:
            df = df.head(_MAX_SQL_ROWS)

        return DataLabSqlResponse(
            success=True,
            columns=list(df.columns),
            rows=_df_to_rows(df),
            row_count=len(df),
        )
    except Exception as exc:
        logger.warning("DataLab SQL failed: %s", exc)
        return DataLabSqlResponse(success=False, error=str(exc)[:300])


# ---------------------------------------------------------------------------
# GET /api/datalab/download/{table}
# ---------------------------------------------------------------------------
@router.get("/download/{table}")
async def download_table(table: str):
    """Download a table as a CSV file."""
    con = _get_con_for_table(table)

    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
        raise HTTPException(status_code=400, detail="Invalid table name.")

    tables = _get_all_tables(con)
    if table not in tables:
        raise HTTPException(status_code=404, detail=f"Table '{table}' not found.")

    try:
        df = con.execute(f"SELECT * FROM {table}").df()
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{table}.csv"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(exc)[:200]}")


# ---------------------------------------------------------------------------
# POST /api/datalab/transform  — Pandas/NumPy transform with SQL fallback + verifier
# ---------------------------------------------------------------------------
@router.post("/transform", response_model=DataLabTransformV2Response)
async def transform_data(req: DataLabTransformV2Request):
    """
    NL → Python (pandas/numpy) transformation.

    Workflow:
      1. AI identifies the target dataset from prompt (if table not specified)
      2. Generate pandas/numpy code (shown to user)
      3. Execute in sandboxed context → result_df
      4. If pandas fails, fall back to SQL with verifier:
         a. Generate DuckDB SQL
         b. Verify all column refs exist
         c. If columns missing → AI rewrites SQL using valid columns
         d. Execute and return
    """
    con = _get_con_for_table(req.table)

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", req.table):
        raise HTTPException(status_code=400, detail="Invalid table name.")

    if not sentinel_ns.is_initialized:
        raise HTTPException(status_code=400, detail="SENTINEL not initialized.")

    call_llm_fn = sentinel_ns._ns.get("call_llm")
    fast_model = sentinel_ns._ns.get("FAST_MODEL", "")
    if not call_llm_fn:
        return DataLabTransformV2Response(success=False, error="LLM not available in namespace.")

    # Base SQL for loading current state
    base_sql = req.current_sql.strip() if req.current_sql else f"SELECT * FROM {req.table}"

    try:
        df = con.execute(f"SELECT * FROM ({base_sql}) AS _base LIMIT 100000").df()
    except Exception as exc:
        return DataLabTransformV2Response(success=False, error=f"Cannot read current data: {exc}")

    # ── Phase 1: Try pandas/numpy transform ─────────────────────────────────
    code, result_df, pandas_error, pandas_verifier_notes = _generate_pandas_transform(
        df, req.prompt, call_llm_fn, fast_model,
        prior_step_codes=req.prior_step_codes or [],
    )

    if result_df is not None:
        display_df = result_df.head(500)
        return DataLabTransformV2Response(
            success=True,
            mode="pandas",
            code=code,
            sql="",
            columns=list(result_df.columns),
            rows=_df_to_rows(display_df),
            row_count=len(result_df),
            verifier_notes=pandas_verifier_notes,
        )

    # ── Phase 2: SQL fallback with verifier ──────────────────────────────────
    logger.info("Pandas transform failed (%s), trying SQL fallback", pandas_error)

    actual_columns = list(df.columns)
    schema_ctx_lines = [f"Total columns: {len(df.columns)} | Rows loaded: {len(df)}"]
    num_cols_list = df.select_dtypes(include="number").columns.tolist()
    cat_cols_list = [c for c in df.select_dtypes(include=["object", "category"]).columns
                     if df[c].nunique() <= 60]

    for col in df.columns:
        dtype = str(df[col].dtype)
        if col in cat_cols_list:
            uniq = df[col].dropna().unique()[:12].tolist()
            schema_ctx_lines.append(f"  {col} ({dtype}): unique={uniq}")
        elif col in num_cols_list:
            s = df[col].dropna()
            if len(s):
                schema_ctx_lines.append(
                    f"  {col} ({dtype}): min={s.min():.3g}, max={s.max():.3g}, "
                    f"mean={s.mean():.3g}, nulls={df[col].isna().sum()}"
                )
        else:
            schema_ctx_lines.append(f"  {col} ({dtype})")

    schema_ctx = "\n".join(schema_ctx_lines)
    sample_str = df.head(5).to_string(index=False)

    sql_prompt = f"""You are a DuckDB SQL expert. Convert the transformation request into a single DuckDB SELECT statement.

BASE QUERY (current data):
{base_sql}

AVAILABLE COLUMNS ONLY (use ONLY these — never invent column names):
{actual_columns}

SCHEMA:
{schema_ctx}

SAMPLE (first 5 rows):
{sample_str}

USER REQUEST: {req.prompt}

TRANSFORMATION RULES:
• ONLY use columns listed in AVAILABLE COLUMNS above
• If a column doesn't exist but can be COMPUTED, derive it (e.g. discount_pct = discount_amount/base_amount*100)
• Wrap as: SELECT *, <new_computed_columns> FROM ({base_sql}) AS _t
• KEEP ALL ORIGINAL ROWS — use SELECT *, not SELECT <subset>
• For per-group aggregates, use window functions: SUM(col) OVER(PARTITION BY group_col) AS total_per_group
• NEVER use GROUP BY unless the user explicitly says "aggregate", "summarize", or "group"
• One-hot encode: CASE WHEN col='val' THEN 1 ELSE 0 END AS col_val
• Normalize (0-1): (col - MIN(col) OVER()) / NULLIF(MAX(col) OVER() - MIN(col) OVER(), 0) AS col_norm
• Standardize (z): (col - AVG(col) OVER()) / NULLIF(STDDEV_POP(col) OVER(), 0) AS col_std
• Log transform: LN(GREATEST(col, 0) + 1) AS col_log

OUTPUT: Return ONLY the SQL SELECT statement — no markdown, no explanation.
SQL:"""

    try:
        raw_sql = call_llm_fn(sql_prompt, model=fast_model, temperature=0.0).strip()
    except Exception as exc:
        return DataLabTransformV2Response(
            success=False,
            code=code,
            error=f"Both pandas and SQL generation failed. Pandas: {pandas_error}. SQL LLM: {str(exc)[:200]}"
        )

    raw_sql = re.sub(r"^```[\w]*\s*", "", raw_sql).strip()
    raw_sql = re.sub(r"\s*```\s*$", "", raw_sql).strip()

    verifier_notes = ""

    # ── Verify column references ──────────────────────────────────────────
    all_valid, missing = _verify_sql_columns(raw_sql, actual_columns)
    if not all_valid:
        verifier_notes = f"⚠ Verifier found missing columns: {missing}. Attempting auto-fix..."
        logger.info("SQL verifier: missing columns %s — attempting LLM rewrite", missing)
        fixed = _try_fix_sql(raw_sql, missing, actual_columns, call_llm_fn, fast_model, base_sql)
        if fixed:
            raw_sql = fixed
            verifier_notes += " SQL rewritten successfully."
        else:
            verifier_notes += " Rewrite failed — executing original (may error)."

    try:
        result_df2 = con.execute(raw_sql).df()
        return DataLabTransformV2Response(
            success=True,
            mode="sql",
            code=f"# pandas approach failed: {pandas_error}\n# Falling back to SQL\n\n# SQL equivalent:\n# {raw_sql}",
            sql=raw_sql,
            columns=list(result_df2.columns),
            rows=_df_to_rows(result_df2.head(500)),
            row_count=len(result_df2),
            verifier_notes=verifier_notes,
        )
    except Exception as exc:
        return DataLabTransformV2Response(
            success=False,
            mode="sql",
            code=code,
            sql=raw_sql,
            error=f"SQL execution error: {str(exc)[:400]}",
            verifier_notes=verifier_notes,
        )


# ---------------------------------------------------------------------------
# GET /api/datalab/autoplot/{table}
# ---------------------------------------------------------------------------
def _select_corr_method(df: pd.DataFrame, c1: str, c2: str) -> str:
    """Intelligently choose correlation method based on data characteristics."""
    s1, s2 = df[c1].dropna(), df[c2].dropna()
    u1, u2 = s1.nunique(), s2.nunique()
    # Binary vs binary → Phi coefficient (computed as Pearson on 0/1)
    if u1 <= 2 and u2 <= 2:
        return "phi"
    # Binary vs continuous → Point Biserial (same formula as Pearson)
    if u1 <= 2 or u2 <= 2:
        return "point_biserial"
    # Small sample → Kendall tau (more robust, handles ties)
    n = min(len(s1), len(s2))
    if n < 30:
        return "kendall"
    # Normality test on sample ≤ 200 (Shapiro-Wilk)
    try:
        from scipy import stats as _stats
        samp = min(200, n)
        _, p1 = _stats.shapiro(s1.sample(samp, random_state=0))
        _, p2 = _stats.shapiro(s2.sample(samp, random_state=0))
        if p1 > 0.05 and p2 > 0.05:
            return "pearson"
    except Exception:
        pass
    # Default: Spearman (rank-based, robust to outliers & non-linearity)
    return "spearman"


def _compute_corr(df: pd.DataFrame, num_cols: list, method: str = "auto") -> pd.DataFrame:
    """Compute correlation matrix using the chosen or auto-selected method."""
    if method == "auto":
        # For the full matrix, choose the method that fits most column pairs
        methods = [_select_corr_method(df, c, num_cols[0]) for c in num_cols[1:]]
        counts = {}
        for m in methods:
            counts[m] = counts.get(m, 0) + 1
        method = max(counts, key=counts.get) if counts else "spearman"
    try:
        from scipy import stats as _stats
        n = len(num_cols)
        mat = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s1 = df[num_cols[i]].fillna(0)
                s2 = df[num_cols[j]].fillna(0)
                if method == "pearson":
                    r, _ = _stats.pearsonr(s1, s2)
                elif method == "kendall":
                    r, _ = _stats.kendalltau(s1, s2)
                elif method in ("phi", "point_biserial"):
                    r, _ = _stats.pearsonr(s1, s2)
                else:
                    r, _ = _stats.spearmanr(s1, s2)
                mat[i, j] = mat[j, i] = round(float(r), 4)
        return pd.DataFrame(mat, index=num_cols, columns=num_cols)
    except Exception:
        return df[num_cols].corr()


@router.get("/autoplot/{table}", response_model=DataLabAutoPlotResponse)
async def autoplot_table(table: str, sql: str = ""):
    """Auto-generate 15+ intelligent Plotly charts for a table."""
    con = _get_con_for_table(table)

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
        raise HTTPException(status_code=400, detail="Invalid table name.")

    base_sql = sql.strip() if sql.strip() else f"SELECT * FROM {table}"

    try:
        df = con.execute(f"SELECT * FROM ({base_sql}) AS _t LIMIT 5000").df()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ── Profile ──────────────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if 1 < df[c].nunique() <= 50]
    high_card_cats = [c for c in df.select_dtypes(include=["object", "category"]).columns
                      if df[c].nunique() > 50]
    date_cols: list = []
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.7:
                df[col] = parsed
                date_cols.append(col)
        except Exception:
            pass
    date_cols += list(df.select_dtypes(include=["datetime64"]).columns)
    date_cols = list(dict.fromkeys(date_cols))   # deduplicate

    binary_cols = [c for c in num_cols if df[c].dropna().nunique() <= 2]

    charts = []

    # ── 1. Distributions (histogram + KDE overlay) ──────────────────────────
    for col in num_cols[:4]:
        try:
            series = df[col].dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=series, nbinsx=35, name="Histogram",
                marker_color="#3B82F6", opacity=0.65,
                histnorm="probability density",
            ))
            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(series)
                xs  = np.linspace(series.min(), series.max(), 200)
                fig.add_trace(go.Scatter(
                    x=xs, y=kde(xs), mode="lines", name="KDE",
                    line=dict(color="#06B6D4", width=2),
                ))
            except Exception:
                pass
            _apply_dark(fig, f"Distribution + KDE — {col}")
            charts.append({"title": f"Distribution — {col}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 2. Intelligent correlation heatmap ──────────────────────────────────
    if len(num_cols) >= 2:
        try:
            cols_for_corr = num_cols[:12]
            method_used   = _select_corr_method(df, cols_for_corr[0], cols_for_corr[1]) if len(cols_for_corr) >= 2 else "pearson"
            corr = _compute_corr(df, cols_for_corr, method_used)
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu_r", zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                colorbar=dict(tickfont=dict(color="#94A3B8")),
            ))
            _apply_dark(fig, f"Correlation Heatmap ({method_used.capitalize()})", height=420)
            charts.append({"title": f"Correlation Heatmap ({method_used.capitalize()})", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 3. Scatter (top 2 numeric, coloured by category) ────────────────────
    if len(num_cols) >= 2:
        try:
            c1, c2   = num_cols[0], num_cols[1]
            color_col = cat_cols[0] if cat_cols else None
            sub  = df[[c1, c2] + ([color_col] if color_col else [])].dropna()
            samp = sub.sample(min(800, len(sub)), random_state=42)
            if color_col:
                vals    = samp[color_col].unique().tolist()
                col_map = {v: _CHART_COLORS[i % len(_CHART_COLORS)] for i, v in enumerate(vals)}
                traces  = [
                    go.Scatter(
                        x=samp.loc[samp[color_col] == v, c1],
                        y=samp.loc[samp[color_col] == v, c2],
                        mode="markers", name=str(v),
                        marker=dict(color=col_map[v], size=5, opacity=0.65),
                    )
                    for v in vals[:10]
                ]
                fig = go.Figure(traces)
            else:
                fig = go.Figure(go.Scatter(
                    x=samp[c1], y=samp[c2], mode="markers",
                    marker=dict(color="#3B82F6", size=5, opacity=0.65),
                ))
            _apply_dark(fig, f"Scatter — {c1} vs {c2}", height=400)
            fig.update_layout(xaxis_title=c1, yaxis_title=c2)
            charts.append({"title": f"Scatter — {c1} vs {c2}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 4. Box / Violin plots by category ────────────────────────────────────
    if num_cols and cat_cols:
        try:
            nc, cc    = num_cols[0], cat_cols[0]
            top_cats  = df[cc].value_counts().head(10).index.tolist()
            traces    = [
                go.Box(
                    y=df.loc[df[cc] == cat, nc].dropna(),
                    name=str(cat),
                    marker_color=_CHART_COLORS[i % len(_CHART_COLORS)],
                    boxmean="sd",
                )
                for i, cat in enumerate(top_cats)
            ]
            fig = go.Figure(traces)
            _apply_dark(fig, f"Box — {nc} by {cc}")
            charts.append({"title": f"Box — {nc} by {cc}", "html": _fig_html(fig)})
        except Exception:
            pass

    if num_cols and cat_cols:
        try:
            nc, cc   = num_cols[0], cat_cols[0]
            top_cats = df[cc].value_counts().head(8).index.tolist()
            traces   = [
                go.Violin(
                    y=df.loc[df[cc] == cat, nc].dropna(),
                    name=str(cat), box_visible=True,
                    line_color=_CHART_COLORS[i % len(_CHART_COLORS)],
                    meanline_visible=True,
                )
                for i, cat in enumerate(top_cats)
            ]
            fig = go.Figure(traces)
            _apply_dark(fig, f"Violin — {nc} by {cc}")
            charts.append({"title": f"Violin — {nc} by {cc}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 5. Value counts for categorical cols ─────────────────────────────────
    for col in cat_cols[:2]:
        try:
            vc  = df[col].value_counts().head(20)
            fig = go.Figure(go.Bar(
                x=vc.index.tolist(), y=vc.values.tolist(),
                marker_color=_CHART_COLORS[:len(vc)],
                text=vc.values.tolist(), textposition="outside",
            ))
            _apply_dark(fig, f"Value Counts — {col}")
            charts.append({"title": f"Value Counts — {col}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 6. Time-series (line + rolling mean) ─────────────────────────────────
    if date_cols and num_cols:
        try:
            dc, nc = date_cols[0], num_cols[0]
            ts = df[[dc, nc]].dropna().copy()
            ts[dc] = pd.to_datetime(ts[dc], errors="coerce")
            ts = ts.dropna().sort_values(dc)
            if len(ts) > 1:
                ts_r = ts.set_index(dc)[nc].resample("D").mean().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_r[dc], y=ts_r[nc], mode="lines", name=nc,
                    line=dict(color="#06B6D4", width=2),
                ))
                # Rolling 7-day average
                if len(ts_r) >= 7:
                    roll = ts_r[nc].rolling(7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=ts_r[dc], y=roll, mode="lines", name="7-day avg",
                        line=dict(color="#F59E0B", width=2, dash="dash"),
                    ))
                _apply_dark(fig, f"{nc} Over Time")
                fig.update_layout(xaxis_title=str(dc), yaxis_title=nc)
                charts.append({"title": f"{nc} Over Time", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 7. ECDF (Empirical Cumulative Distribution Function) ─────────────────
    for col in num_cols[:2]:
        try:
            series = df[col].dropna().sort_values()
            ecdf   = np.arange(1, len(series) + 1) / len(series)
            fig    = go.Figure(go.Scatter(
                x=series, y=ecdf, mode="lines", name="ECDF",
                line=dict(color="#8B5CF6", width=2),
            ))
            _apply_dark(fig, f"ECDF — {col}")
            fig.update_layout(xaxis_title=col, yaxis_title="Cumulative Probability")
            charts.append({"title": f"ECDF — {col}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 8. Q-Q plot (normality check) ────────────────────────────────────────
    if num_cols:
        try:
            from scipy import stats as _stats
            col    = num_cols[0]
            series = df[col].dropna().values
            (osm, osr), (slope, intercept, _) = _stats.probplot(series)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Quantiles",
                                     marker=dict(color="#3B82F6", size=4, opacity=0.7)))
            fig.add_trace(go.Scatter(x=[min(osm), max(osm)],
                                     y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
                                     mode="lines", name="Normal line",
                                     line=dict(color="#EF4444", dash="dash")))
            _apply_dark(fig, f"Q-Q Plot — {col} (normality check)")
            fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
            charts.append({"title": f"Q-Q Plot — {col}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 9. Missing values bar chart ───────────────────────────────────────────
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            pct = (missing / len(df) * 100).round(1)
            fig = go.Figure(go.Bar(
                x=missing.index.tolist(),
                y=pct.values.tolist(),
                marker_color=[
                    "#EF4444" if p >= 30 else "#F59E0B" if p >= 10 else "#3B82F6"
                    for p in pct.values
                ],
                text=[f"{p}%" for p in pct.values],
                textposition="outside",
            ))
            _apply_dark(fig, "Missing Values (%)")
            fig.update_layout(yaxis_title="Missing %", xaxis_title="Column")
            charts.append({"title": "Missing Values", "html": _fig_html(fig)})
    except Exception:
        pass

    # ── 10. Categorical cross-tabulation heatmap (Cramér's V) ────────────────
    if len(cat_cols) >= 2:
        try:
            from scipy.stats import chi2_contingency
            c1, c2 = cat_cols[0], cat_cols[1]
            ct     = pd.crosstab(df[c1].fillna("N/A"), df[c2].fillna("N/A"))
            fig    = go.Figure(go.Heatmap(
                z=ct.values,
                x=[str(x) for x in ct.columns],
                y=[str(y) for y in ct.index],
                colorscale="Blues",
                text=ct.values,
                texttemplate="%{text}",
                colorbar=dict(tickfont=dict(color="#94A3B8")),
            ))
            _apply_dark(fig, f"Cross-Tab — {c1} × {c2}", height=420)
            charts.append({"title": f"Cross-Tab — {c1} × {c2}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 11. Pair scatter matrix (top 4 numeric) ───────────────────────────────
    if len(num_cols) >= 3:
        try:
            import plotly.express as px
            cols_plot = num_cols[:4]
            color_col = cat_cols[0] if cat_cols else None
            samp      = df[cols_plot + ([color_col] if color_col else [])].dropna()
            samp      = samp.sample(min(500, len(samp)), random_state=42)
            fig = px.scatter_matrix(
                samp, dimensions=cols_plot,
                color=color_col if color_col else None,
                color_discrete_sequence=_CHART_COLORS,
                opacity=0.5,
            )
            fig.update_traces(marker=dict(size=3))
            _apply_dark(fig, "Pair Scatter Matrix", height=500)
            charts.append({"title": "Pair Scatter Matrix", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 12. IQR outlier visualisation ────────────────────────────────────────
    if num_cols:
        try:
            cols_show = num_cols[:5]
            outlier_data = []
            for col in cols_show:
                s   = df[col].dropna()
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                out = s[(s < lo) | (s > hi)]
                n_out = len(out)
                outlier_data.append({
                    "col": col,
                    "q1": round(float(q1), 3), "q3": round(float(q3), 3),
                    "median": round(float(s.median()), 3),
                    "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                    "n_outliers": n_out,
                    "pct_outliers": round(n_out / len(s) * 100, 1),
                })
            fig = go.Figure(go.Bar(
                x=[d["col"] for d in outlier_data],
                y=[d["pct_outliers"] for d in outlier_data],
                marker_color=[
                    "#EF4444" if d["pct_outliers"] >= 10 else "#F59E0B" if d["pct_outliers"] >= 3 else "#3B82F6"
                    for d in outlier_data
                ],
                text=[f"{d['pct_outliers']}%" for d in outlier_data],
                textposition="outside",
            ))
            _apply_dark(fig, "Outlier Rate by Column (IQR method)")
            fig.update_layout(yaxis_title="% Outliers", xaxis_title="Column")
            charts.append({"title": "Outlier Rate (IQR)", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 13. Rolling statistics for time series ────────────────────────────────
    if date_cols and num_cols and len(df) >= 14:
        try:
            dc, nc = date_cols[0], num_cols[0]
            ts = df[[dc, nc]].dropna().copy()
            ts[dc] = pd.to_datetime(ts[dc], errors="coerce")
            ts = ts.dropna().sort_values(dc).set_index(dc)[nc].resample("D").mean()
            if len(ts) >= 14:
                roll_m = ts.rolling(7, min_periods=1).mean()
                roll_s = ts.rolling(7, min_periods=1).std()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines",
                                         name="Daily", line=dict(color="#3B82F6", width=1, dash="dot"), opacity=0.5))
                fig.add_trace(go.Scatter(x=roll_m.index, y=roll_m.values, mode="lines",
                                         name="7d Mean", line=dict(color="#06B6D4", width=2)))
                fig.add_trace(go.Scatter(
                    x=pd.concat([roll_m + roll_s*2, (roll_m - roll_s*2).iloc[::-1]]).index,
                    y=pd.concat([roll_m + roll_s*2, (roll_m - roll_s*2).iloc[::-1]]).values,
                    fill="toself", fillcolor="rgba(6,182,212,0.1)", line=dict(color="rgba(0,0,0,0)"),
                    name="±2σ band", showlegend=True,
                ))
                _apply_dark(fig, f"{nc} — Rolling Mean ± 2σ")
                charts.append({"title": f"{nc} Rolling Stats", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 14. Treemap for top-2 categorical cols ────────────────────────────────
    if len(cat_cols) >= 2:
        try:
            import plotly.express as px
            c1, c2 = cat_cols[0], cat_cols[1]
            nc     = num_cols[0] if num_cols else None
            grp    = (df.groupby([c1, c2]).agg(
                _val=(nc if nc else c1, "count")
            ).reset_index() if nc else df.groupby([c1, c2]).size().reset_index(name="_val"))
            grp.columns = [c1, c2, "count"]
            fig = px.treemap(
                grp, path=[c1, c2], values="count",
                color="count", color_continuous_scale="Blues",
            )
            _apply_dark(fig, f"Treemap — {c1} / {c2}", height=420)
            charts.append({"title": f"Treemap — {c1}/{c2}", "html": _fig_html(fig)})
        except Exception:
            pass
    elif cat_cols:
        try:
            import plotly.express as px
            c1    = cat_cols[0]
            vc    = df[c1].value_counts().head(25).reset_index()
            vc.columns = [c1, "count"]
            fig   = px.treemap(vc, path=[c1], values="count",
                               color="count", color_continuous_scale="Blues")
            _apply_dark(fig, f"Treemap — {c1}", height=380)
            charts.append({"title": f"Treemap — {c1}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 15. Stacked bar (numeric totals by category) ──────────────────────────
    if cat_cols and len(num_cols) >= 2:
        try:
            cc      = cat_cols[0]
            cols_n  = num_cols[:4]
            grp     = df.groupby(cc)[cols_n].mean().reset_index()
            grp     = grp.sort_values(cols_n[0], ascending=False).head(10)
            fig     = go.Figure()
            for i, nc in enumerate(cols_n):
                fig.add_trace(go.Bar(
                    name=nc, x=grp[cc].astype(str).tolist(), y=grp[nc].round(2).tolist(),
                    marker_color=_CHART_COLORS[i % len(_CHART_COLORS)],
                ))
            fig.update_layout(barmode="group")
            _apply_dark(fig, f"Mean Metrics by {cc}", height=380)
            charts.append({"title": f"Metrics by {cc}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 16. Lag autocorrelation (time-series only) ────────────────────────────
    if date_cols and num_cols and len(df) >= 20:
        try:
            dc, nc = date_cols[0], num_cols[0]
            ts = df[[dc, nc]].dropna().copy()
            ts[dc] = pd.to_datetime(ts[dc], errors="coerce")
            ts = ts.dropna().sort_values(dc)[nc].values
            max_lag = min(30, len(ts) // 3)
            lags    = range(1, max_lag + 1)
            acf_vals = [
                float(np.corrcoef(ts[:-lag], ts[lag:])[0, 1])
                for lag in lags
            ]
            ci = 1.96 / np.sqrt(len(ts))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(lags), y=acf_vals,
                                  marker_color=["#EF4444" if abs(v) > ci else "#3B82F6" for v in acf_vals]))
            fig.add_hline(y=ci,  line=dict(color="#F59E0B", dash="dash"), annotation_text="+95% CI")
            fig.add_hline(y=-ci, line=dict(color="#F59E0B", dash="dash"), annotation_text="-95% CI")
            _apply_dark(fig, f"Lag Autocorrelation — {nc}")
            fig.update_layout(xaxis_title="Lag", yaxis_title="ACF")
            charts.append({"title": f"Autocorrelation — {nc}", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 17. Feature variance ranking ─────────────────────────────────────────
    if num_cols:
        try:
            from sklearn.preprocessing import StandardScaler
            X      = df[num_cols].fillna(0)
            X_s    = StandardScaler().fit_transform(X)
            variances = np.var(X_s, axis=0)
            feat_var  = sorted(zip(num_cols, variances), key=lambda x: -x[1])
            cols_v, vals_v = zip(*feat_var)
            fig = go.Figure(go.Bar(
                x=list(cols_v), y=[round(v, 4) for v in vals_v],
                marker_color=_CHART_COLORS[:len(cols_v)],
                text=[f"{v:.2f}" for v in vals_v], textposition="outside",
            ))
            _apply_dark(fig, "Feature Variance (standardised)")
            fig.update_layout(xaxis_title="Feature", yaxis_title="Variance (std-scaled)")
            charts.append({"title": "Feature Variance", "html": _fig_html(fig)})
        except Exception:
            pass

    # ── 18. PCA 2-D projection ────────────────────────────────────────────────
    if len(num_cols) >= 3:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            samp    = df[num_cols].fillna(0)
            samp    = samp.sample(min(1000, len(samp)), random_state=42)
            X_s     = StandardScaler().fit_transform(samp)
            pcs     = PCA(n_components=2).fit_transform(X_s)
            pca_df  = pd.DataFrame(pcs, columns=["PC1", "PC2"])
            color_col = cat_cols[0] if cat_cols else None
            if color_col:
                pca_df[color_col] = df[color_col].iloc[samp.index].values
            fig = go.Figure(go.Scatter(
                x=pca_df["PC1"], y=pca_df["PC2"], mode="markers",
                marker=dict(color="#8B5CF6", size=4, opacity=0.6),
            )) if not color_col else go.Figure([
                go.Scatter(
                    x=pca_df.loc[pca_df[color_col] == v, "PC1"],
                    y=pca_df.loc[pca_df[color_col] == v, "PC2"],
                    mode="markers", name=str(v),
                    marker=dict(color=_CHART_COLORS[i % len(_CHART_COLORS)], size=4, opacity=0.65),
                )
                for i, v in enumerate(pca_df[color_col].unique()[:10])
            ])
            _apply_dark(fig, "PCA — 2D Projection", height=420)
            fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
            charts.append({"title": "PCA 2D Projection", "html": _fig_html(fig)})
        except Exception:
            pass

    return DataLabAutoPlotResponse(charts=charts)


# ---------------------------------------------------------------------------
# POST /api/datalab/plot — NL → custom Plotly chart
# ---------------------------------------------------------------------------
@router.post("/plot", response_model=DataLabPlotResponse)
async def custom_plot(req: DataLabPlotRequest):
    """Convert a NL chart request into executable Plotly Python code and execute it."""
    con = _get_con_for_table(req.table)

    if not sentinel_ns.is_initialized:
        raise HTTPException(status_code=400, detail="SENTINEL not initialised.")

    call_llm_fn = sentinel_ns._ns.get("call_llm")
    fast_model = sentinel_ns._ns.get("FAST_MODEL", "")
    if not call_llm_fn:
        return DataLabPlotResponse(success=False, error="LLM not available.")

    base_sql = req.current_sql.strip() if req.current_sql else f"SELECT * FROM {req.table}"
    try:
        df = con.execute(f"SELECT * FROM ({base_sql}) AS _t LIMIT 5000").df()
    except Exception as exc:
        return DataLabPlotResponse(success=False, error=f"Cannot read data: {exc}")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if df[c].nunique() <= 50]
    schema_ctx = (
        f"Columns: {list(df.columns)}\n"
        f"Numeric columns: {num_cols}\n"
        f"Categorical columns: {cat_cols}\n"
        f"Total rows: {len(df)}\n"
        f"Sample:\n{df.head(4).to_string(index=False)}"
    )

    llm_prompt = f"""You are a Plotly Python expert. Generate a Python code snippet to create the requested chart.

DATA CONTEXT:
{schema_ctx}

USER REQUEST: {req.prompt}

RULES:
1. The dataframe is already loaded as `df`
2. Available: go (plotly.graph_objects), px (plotly.express), pd (pandas), np (numpy)
3. Apply dark theme: paper_bgcolor='#111827', plot_bgcolor='#111827', font=dict(color='#F1F5F9')
4. Use ONLY column names from the DATA CONTEXT above
5. Use color palette: {_CHART_COLORS}
6. End with fig.show() — it is intercepted to capture the HTML
7. Return ONLY executable Python — no imports, no markdown fences, no explanation

PYTHON CODE:"""

    try:
        code = call_llm_fn(llm_prompt, model=fast_model, temperature=0.1).strip()
    except Exception as exc:
        return DataLabPlotResponse(success=False, error=f"LLM error: {str(exc)[:200]}")

    code = re.sub(r"^```[\w]*\s*", "", code).strip()
    code = re.sub(r"\s*```\s*$", "", code).strip()

    import plotly.express as px

    captured: list = []

    def _cap_show(fig_self, *a, **kw):
        try:
            captured.append({"title": req.prompt[:70], "html": _fig_html(fig_self)})
        except Exception:
            pass

    orig_show = go.Figure.show
    go.Figure.show = _cap_show
    exec_ns = {
        "df": df, "go": go, "px": px, "pd": pd, "np": np,
        "_CHART_COLORS": _CHART_COLORS,
    }
    try:
        exec(compile(code, "<datalab_plot>", "exec"), exec_ns)
    except Exception as exc:
        go.Figure.show = orig_show
        return DataLabPlotResponse(
            success=False, code=code,
            error=f"Chart execution error: {str(exc)[:300]}"
        )
    finally:
        go.Figure.show = orig_show

    return DataLabPlotResponse(
        success=True if captured else False,
        charts=captured,
        code=code,
        error=None if captured else "No figure was generated (code ran but didn't call fig.show())",
    )


# ---------------------------------------------------------------------------
# POST /api/datalab/switch-version — swap active DuckDB connection
# ---------------------------------------------------------------------------
class SwitchVersionRequest(BaseModel):
    version: str  # 'original' | 'modified'

@router.post("/switch-version")
async def switch_version(req: SwitchVersionRequest):
    """
    Switch the global active DuckDB connection to either 'original' or 'modified'.
    Called when the user toggles the version in the TopBar or DataLab.
    """
    _get_con()  # ensure initialized
    result = await sentinel_ns.switch_active_version(req.version)
    return result


# ---------------------------------------------------------------------------
# POST /api/datalab/promote — copy modified dataset to intelligence
# ---------------------------------------------------------------------------


@router.post("/promote", response_model=DataLabPromoteResponse)
async def promote_dataset(req: DataLabPromoteRequest):
    """
    Promote a transform result into the shared DuckDB as a _modified table.
    Both original and modified tables live in the SAME connection — no
    separate .duckdb files, no connection swapping, no connection-closed errors.
    """
    con = _get_con()

    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', req.table):
        raise HTTPException(status_code=400, detail="Invalid table name.")

    tables = _get_all_tables(con)
    if req.table not in tables:
        raise HTTPException(status_code=404, detail=f"Table '{req.table}' not found.")

    # Strip existing _modified suffix to prevent double-suffixing
    base_table = re.sub(r'_modified$', '', req.table)
    version_name = req.version_name or f"{base_table}_modified"
    version_name = re.sub(r'[^a-zA-Z0-9_]', '_', version_name)[:80]
    if not re.match(r'^[a-zA-Z_]', version_name):
        version_name = f"t_{version_name}"

    try:
        # Load base data
        base_sql = req.current_sql.strip() if req.current_sql else f"SELECT * FROM {req.table}"
        df = con.execute(f"SELECT * FROM ({base_sql}) AS _base LIMIT 500000").df()

        # Replay prior pandas steps
        if req.prior_step_codes:
            df, replay_err = _replay_prior_steps(df, req.prior_step_codes)
            if replay_err:
                logger.warning("Promote replay error (continuing): %s", replay_err)

        # Drop old modified table if it already exists
        try:
            con.execute(f"DROP TABLE IF EXISTS {version_name}")
        except Exception:
            pass

        # Create modified table in the SAME shared connection
        con.execute(f"CREATE TABLE {version_name} AS SELECT * FROM df")
        row_count = con.execute(f"SELECT COUNT(*) FROM {version_name}").fetchone()[0]

        # Build schema for the new table
        mod_filename = "modified.csv"
        schema_lines = []
        try:
            cols_df = con.execute(f"DESCRIBE {version_name}").df()
            schema_lines.append(f"TABLE {version_name}:")
            for _, row in cols_df.iterrows():
                schema_lines.append(f"  {row['column_name']} {row['column_type']}")
        except Exception:
            pass
        new_schema = "\n".join(schema_lines)

        # Register as modified.csv — same con, no db_path
        if sentinel_ns.is_initialized:
            await sentinel_ns.update_data(
                new_con=con,
                new_schema=new_schema,
                date_min=None,
                date_max=None,
                filename=mod_filename,
                new_tables=[version_name],
                row_count=row_count,
                all_tables=[version_name],
            )

        # Store table name references for the version toggle
        sentinel_ns._ns["original_table"] = base_table
        sentinel_ns._ns["modified_table"] = version_name

        logger.info(
            "Promoted table '%s' → '%s' (%d rows) [same shared connection]",
            req.table, version_name, row_count,
        )

        return DataLabPromoteResponse(
            success=True,
            new_table=version_name,
            filename=mod_filename,
            row_count=row_count,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Promote failed: %s", exc, exc_info=True)
        return DataLabPromoteResponse(success=False, error=f"Promote failed: {str(exc)[:300]}")


# ---------------------------------------------------------------------------
# DELETE /api/datalab/tables/{table} — drop a single table
# ---------------------------------------------------------------------------
@router.delete("/tables/{table}", response_model=DataLabDropTableResponse)
async def drop_single_table(table: str):
    """
    Drop a single DuckDB table. Used for deleting individual versions
    (e.g. just the modified copy, or just the original).
    Also removes the table from any dataset registry entry.
    If the table belongs to an isolated DuckDB, uses that connection and
    delegates full cleanup to remove_dataset.
    """
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
        raise HTTPException(status_code=400, detail="Invalid table name.")

    try:
        # Find which dataset owns this table
        registry = sentinel_ns.get_dataset_registry()
        owner_filename = None
        owner_info = None
        for fn, info in registry.items():
            if table in info.get("tables", []):
                owner_filename = fn
                owner_info = info
                break

        if owner_info and owner_info.get("db_path"):
            # Table lives in an isolated DuckDB → delegate full cleanup
            # to remove_dataset which closes connection + deletes file
            await sentinel_ns.remove_dataset(owner_filename)
        else:
            # Table lives in the shared connection
            con = _get_con()
            tables = _get_all_tables(con)
            if table not in tables:
                raise HTTPException(status_code=404, detail=f"Table '{table}' not found.")

            con.execute(f"DROP TABLE IF EXISTS {table}")

            # Remove from registry entries
            for fn, info in registry.items():
                ds_tables = info.get("tables", [])
                if table in ds_tables:
                    ds_tables.remove(table)
                    if not ds_tables:
                        await sentinel_ns.remove_dataset(fn)
                    break

        logger.info("Dropped table '%s'", table)
        return DataLabDropTableResponse(success=True, table=table)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Drop table failed: %s", exc, exc_info=True)
        return DataLabDropTableResponse(success=False, table=table, error=f"Failed: {str(exc)[:200]}")

