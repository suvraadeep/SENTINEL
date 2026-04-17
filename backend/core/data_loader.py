"""
Data Loader — ingest user-uploaded CSV / Excel / SQLite / Parquet into DuckDB.

Handles files with 100 k+ rows using chunked streaming.
Supports multi-dataset mode: pass `existing_con` to add tables to an existing
DuckDB connection instead of creating a fresh one.

After ingestion, creates materialized summary tables, rebuilds the schema
string from ALL tables in the database, and detects the date range so the
SQL agents always query real data.
"""

from __future__ import annotations
import os
import io
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import duckdb
import numpy as np

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10_000          # rows per pandas chunk for CSV
MAX_SAMPLE_ROWS = 5          # rows shown in schema preview
MAX_CATEGORY_UNIQUE = 50     # columns with fewer unique values treated as categorical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first column that looks like a date/datetime."""
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(20)
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() / max(len(sample), 1) > 0.7:
                return col
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def _infer_col_profile(df: pd.DataFrame) -> Dict[str, str]:
    """Return {col: 'numeric' | 'categorical' | 'datetime' | 'text'}."""
    profile: Dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            profile[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            profile[col] = "numeric"
        elif df[col].nunique(dropna=True) <= MAX_CATEGORY_UNIQUE:
            profile[col] = "categorical"
        else:
            profile[col] = "text"
    return profile


def _build_schema_string(con: duckdb.DuckDBPyConnection, tables: List[str]) -> str:
    """Build the SCHEMA string (same format as dataset.py get_schema())."""
    parts = []
    for tbl in tables:
        try:
            cols = con.execute(f"DESCRIBE {tbl}").df()
            cols_str = ", ".join(
                f"{r['column_name']}:{r['column_type']}" for _, r in cols.iterrows()
            )
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            sample = con.execute(f"SELECT * FROM {tbl} LIMIT {MAX_SAMPLE_ROWS}").df()
            sample_str = sample.to_string(index=False)
            parts.append(
                f"TABLE {tbl} ({n:,} rows)\nCOLUMNS: {cols_str}\nSAMPLE:\n{sample_str}"
            )
        except Exception as exc:
            logger.warning("Could not describe table %s: %s", tbl, exc)
    return "\n\n".join(parts)


def _get_all_user_tables(con: duckdb.DuckDBPyConnection) -> List[str]:
    """Return all user-defined tables in the DuckDB connection (excludes system tables)."""
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


def _create_summary_tables(
    con: duckdb.DuckDBPyConnection,
    main_table: str,
    date_col: Optional[str],
    numeric_cols: List[str],
    category_cols: List[str],
) -> List[str]:
    """
    Auto-create materialized summary tables so agents can do fast aggregations.
    Uses table-prefixed names to avoid collisions in multi-dataset mode.
    Returns list of created table names.
    """
    created = []
    safe_prefix = re.sub(r"[^a-z0-9_]", "_", main_table.lower())

    # daily_summary — if there is a date column and at least one numeric
    if date_col and numeric_cols:
        try:
            agg_exprs = ", ".join(
                [f"SUM({c}) AS total_{c}, AVG({c}) AS avg_{c}"
                 for c in numeric_cols[:4]]
            )
            cat_exprs = ", ".join(category_cols[:3]) if category_cols else ""
            group_cols = f"{date_col}" + (f", {cat_exprs}" if cat_exprs else "")
            summary_name = f"{safe_prefix}_daily_summary"
            con.execute(f"DROP TABLE IF EXISTS {summary_name}")
            con.execute(f"""
                CREATE TABLE {summary_name} AS
                SELECT {group_cols}, COUNT(*) AS record_count, {agg_exprs}
                FROM {main_table}
                GROUP BY {group_cols}
                ORDER BY {date_col}
            """)
            created.append(summary_name)
        except Exception as exc:
            logger.warning("Could not create daily_summary for %s: %s", main_table, exc)

    # category_summary — if there are category columns and numeric cols
    if category_cols and numeric_cols:
        try:
            cat_col = category_cols[0]
            agg_exprs = ", ".join(
                [f"SUM({c}) AS total_{c}, AVG({c}) AS avg_{c}, COUNT(*) AS count_{c}"
                 for c in numeric_cols[:4]]
            )
            summary_name = f"{safe_prefix}_category_summary"
            con.execute(f"DROP TABLE IF EXISTS {summary_name}")
            con.execute(f"""
                CREATE TABLE {summary_name} AS
                SELECT {cat_col}, {agg_exprs}
                FROM {main_table}
                GROUP BY {cat_col}
            """)
            created.append(summary_name)
        except Exception as exc:
            logger.warning("Could not create category_summary for %s: %s", main_table, exc)

    return created


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------
def ingest_file(
    file_bytes: bytes,
    filename: str,
    db_path: str,
    existing_con: Optional[duckdb.DuckDBPyConnection] = None,
) -> Dict[str, Any]:
    """
    Read uploaded file → load into DuckDB → return metadata dict.

    Parameters
    ----------
    file_bytes   : raw file content
    filename     : original filename (used to derive table name)
    db_path      : path to DuckDB file (used when creating a new connection)
    existing_con : if provided, add tables to this connection instead of
                   opening a fresh one.  Enables multi-dataset mode.

    Returns
    -------
    {
      "con":          duckdb.DuckDBPyConnection,
      "schema":       str,           ← ALL tables in DB
      "tables":       List[str],     ← only newly created tables for this file
      "all_tables":   List[str],     ← every table in the DB after ingestion
      "date_min":     date | None,
      "date_max":     date | None,
      "row_count":    int,
      "columns":      {col: type_str},
      "primary_table": str,
      "date_col":     str | None,
      "numeric_cols": List[str],
      "category_cols": List[str],
    }
    """
    ext = Path(filename).suffix.lower()
    table_name = re.sub(r"[^a-z0-9_]", "_", Path(filename).stem.lower())[:50]
    if not table_name or table_name[0].isdigit():
        table_name = "uploaded_" + table_name

    logger.info("Ingesting file: %s  (ext=%s, size=%.1f KB)", filename, ext, len(file_bytes) / 1024)

    # Use existing connection or create a new one
    if existing_con is not None:
        con = existing_con
        logger.info("Reusing existing DuckDB connection (multi-dataset mode)")
    else:
        con = duckdb.connect(db_path)
        con.execute("PRAGMA threads=4")

    new_tables: List[str] = [table_name]
    total_rows = 0

    # ------------------------------------------------------------------
    # Load based on format
    # ------------------------------------------------------------------
    if ext == ".csv":
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        chunk_iter = pd.read_csv(
            io.BytesIO(file_bytes), chunksize=CHUNK_SIZE,
            low_memory=False, encoding="utf-8-sig",
        )
        first = True
        for chunk in chunk_iter:
            chunk.columns = [
                re.sub(r"[^a-z0-9_]", "_", c.lower().strip()) for c in chunk.columns
            ]
            if first:
                con.register("_chunk_df", chunk)
                con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _chunk_df")
                first = False
            else:
                con.register("_chunk_df", chunk)
                con.execute(f"INSERT INTO {table_name} SELECT * FROM _chunk_df")
            total_rows += len(chunk)
        # Clean up the temporary registered view so it doesn't appear as a table
        try:
            con.unregister("_chunk_df")
        except Exception:
            pass

    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        df.columns = [re.sub(r"[^a-z0-9_]", "_", c.lower().strip()) for c in df.columns]
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("_df", df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _df")
        total_rows = len(df)

    elif ext == ".parquet":
        df = pd.read_parquet(io.BytesIO(file_bytes))
        df.columns = [re.sub(r"[^a-z0-9_]", "_", c.lower().strip()) for c in df.columns]
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("_df", df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _df")
        total_rows = len(df)

    elif ext in (".db", ".sqlite", ".sqlite3"):
        import sqlite3
        src_con = sqlite3.connect(":memory:")
        src_con.deserialize(file_bytes)  # type: ignore[attr-defined]
        cursor = src_con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        sqlite_tables = [r[0] for r in cursor.fetchall()]
        new_tables = []
        for tbl in sqlite_tables:
            sdf = pd.read_sql_query(f"SELECT * FROM {tbl}", src_con)
            sdf.columns = [re.sub(r"[^a-z0-9_]", "_", c.lower().strip()) for c in sdf.columns]
            safe_tbl = re.sub(r"[^a-z0-9_]", "_", tbl.lower())
            con.execute(f"DROP TABLE IF EXISTS {safe_tbl}")
            con.register("_sdf", sdf)
            con.execute(f"CREATE TABLE {safe_tbl} AS SELECT * FROM _sdf")
            total_rows += len(sdf)
            new_tables.append(safe_tbl)
        table_name = new_tables[0] if new_tables else table_name
        src_con.close()

    else:
        raise ValueError(f"Unsupported file format: {ext}. Use CSV, Excel, Parquet, or SQLite.")

    # ------------------------------------------------------------------
    # Detect column profiles from the primary table
    # ------------------------------------------------------------------
    sample_df = con.execute(f"SELECT * FROM {table_name} LIMIT 1000").df()
    profile = _infer_col_profile(sample_df)

    date_col = _detect_date_col(sample_df)
    numeric_cols = [c for c, t in profile.items() if t == "numeric"]
    category_cols = [c for c, t in profile.items() if t == "categorical"]

    # Parse date column if needed
    date_min = date_max = None
    if date_col:
        try:
            date_row = con.execute(
                f"SELECT MIN(CAST({date_col} AS DATE)), MAX(CAST({date_col} AS DATE)) "
                f"FROM {table_name}"
            ).fetchone()
            date_min, date_max = date_row
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Create summary tables (prefixed per-file to avoid collisions)
    # ------------------------------------------------------------------
    summary_tables = _create_summary_tables(
        con, table_name, date_col, numeric_cols, category_cols
    )
    new_tables += summary_tables

    # ------------------------------------------------------------------
    # Get ALL tables in the DB (for full schema rebuild)
    # ------------------------------------------------------------------
    all_db_tables = _get_all_user_tables(con)

    # ------------------------------------------------------------------
    # Build schema string from ALL tables in the DB
    # ------------------------------------------------------------------
    schema = _build_schema_string(con, all_db_tables)

    logger.info(
        "Ingestion complete: %d rows, new_tables=%s, all_tables=%s, date_col=%s, range=%s→%s",
        total_rows, new_tables, all_db_tables, date_col, date_min, date_max,
    )

    return {
        "con":           con,
        "schema":        schema,
        "tables":        new_tables,        # only this file's tables
        "all_tables":    all_db_tables,     # every table now in DB
        "date_min":      date_min,
        "date_max":      date_max,
        "row_count":     total_rows,
        "columns":       profile,
        "date_col":      date_col,
        "numeric_cols":  numeric_cols,
        "category_cols": category_cols,
        "primary_table": table_name,
    }
