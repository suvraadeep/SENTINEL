"""
SQL-of-Thought (SoT) — Research-grade SQL generation pipeline.

Implements the 6-agent SoT architecture from the paper:
  "SQL-of-Thought: Multi-Agent SQL Generation with Chain-of-Thought Reasoning"

Forward Pipeline (attempt 1):
  1. Schema Linking Agent   — extract relevant tables/columns/joins from schema
  2. Subproblem Agent       — decompose into clause-level JSON subproblems
  3. Query Plan Agent       — CoT step-by-step execution plan (no SQL yet)
  4. SQL Agent              — generate SQL from the plan

Guided Correction Loop (on failure):
  5. Correction Plan Agent  — analyze error via error taxonomy + CoT
  6. Correction SQL Agent   — regenerate corrected SQL from the plan

Interface is unchanged:
  sql_builder(state)     → used by LangGraph as-is
  should_retry_sql(state) → routes to correction loop or viz_agent

Only this file changes — no modifications to route_intent.py, namespace.py,
sql_validator.py, or any other file.
"""
import re
import json
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Error Taxonomy (from Shen et al. via SoT paper §3.1)
# ─────────────────────────────────────────────────────────────────────────────
_ERROR_TAXONOMY = """
ERROR TAXONOMY — classify the error into one or more of these categories:

1. SCHEMA_MISMATCH
   - Wrong table name (table does not exist)
   - Wrong column name (column does not exist in that table)
   - Ambiguous column reference — must qualify with table alias

2. JOIN_INCONSISTENCY
   - Missing JOIN for a table that is referenced
   - Wrong join condition (mismatched key names or types)
   - Cartesian product (missing ON clause)
   - Self-join incorrectly aliased

3. AGGREGATION_MISUSE
   - Non-aggregated column in SELECT without GROUP BY
   - GROUP BY on wrong or missing column
   - HAVING used instead of WHERE (or vice versa)
   - Window function missing OVER() clause

4. CONDITION_ERROR
   - Wrong comparison operator or type mismatch
   - NULL handling: must use IS NULL, not = NULL
   - LIKE pattern missing % wildcard
   - Date/string type formatting issue

5. SUBQUERY_CTE_ERROR
   - CTE defined but not referenced afterward
   - Correlated subquery referencing wrong outer alias
   - Scalar subquery returning multiple rows
   - Missing column in CTE SELECT that is used later

6. ORDERING_LIMIT_ERROR
   - ORDER BY on a column not in SELECT (with DISTINCT)
   - LIMIT without ORDER BY when determinism is required

7. SYNTAX_ERROR
   - Missing parenthesis, extra comma, unmatched quotes
   - BETWEEN without AND
   - Missing keyword (e.g., ON, AS, FROM)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_sql(raw: str) -> str:
    """
    Strip markdown fences and extract exactly one SQL statement.

    CTE SAFETY: A query starting with WITH has this structure:
        WITH cte1 AS (...), cte2 AS (...) SELECT ... FROM cte1
    The final SELECT is PART of the WITH statement — do NOT truncate it.
    We only split at a second WITH (which would be a separate statement).

    Non-CTE queries: split at the second top-level SELECT (col 0).
    """
    sql = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip().strip("`").strip()

    if sql.upper().lstrip().startswith("WITH"):
        # CTE query — only truncate at a second top-level WITH
        tops = list(re.finditer(r"^WITH\b", sql, re.IGNORECASE | re.MULTILINE))
        if len(tops) > 1:
            sql = sql[: tops[1].start()].rstrip()
    else:
        # Regular query — truncate at second top-level SELECT
        tops = list(re.finditer(r"^SELECT\b", sql, re.IGNORECASE | re.MULTILINE))
        if len(tops) > 1:
            sql = sql[: tops[1].start()].rstrip()

    return sql.rstrip(";").strip()


def _parse_json(text: str, default: dict = None) -> dict:
    """Extract and parse the largest JSON block from LLM output."""
    if default is None:
        default = {}
    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    for m in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(m)
        except Exception:
            pass
    return default


def _llm(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Call LLM, never raises — returns empty string on error."""
    try:
        return call_llm(prompt, system=system, temperature=temperature) or ""
    except Exception as exc:
        print(f"  [SoT] LLM error: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# DB Ground-Truth Discovery — run once per sql_builder call
# ─────────────────────────────────────────────────────────────────────────────

def _get_real_tables() -> dict:
    """
    Query the live DuckDB for the real table names and their columns.
    Returns: {"table_name": ["col1", "col2", ...], ...}
    """
    try:
        tables_df = run_sql("SHOW TABLES")
        result = {}
        for tbl in tables_df.iloc[:, 0].tolist():
            try:
                cols_df = run_sql(f"DESCRIBE {tbl}")
                result[tbl] = cols_df.iloc[:, 0].tolist()
            except Exception:
                result[tbl] = []
        return result
    except Exception as e:
        print(f"  [SoT] DB discovery failed: {e}")
        return {}


def _build_constraint_block(real_tables: dict) -> str:
    """
    Build a hard constraint string injected into every agent prompt.
    The LLM sees EXACTLY which tables and columns exist — no guessing.
    """
    if not real_tables:
        return ""
    lines = ["HARD CONSTRAINT — use ONLY these tables and columns (no others exist):"]
    for tbl, cols in real_tables.items():
        lines.append(f"  {tbl}: {', '.join(cols[:25])}")  # cap at 25 cols
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Schema Linking Agent
# ─────────────────────────────────────────────────────────────────────────────

_SYS_SCHEMA_LINKER = """You are a Schema Linking Agent for SQL generation.

Given a natural language question and a database schema, identify ONLY the
components needed to answer the question.

Output EXACTLY this JSON structure (no other text):
{
  "tables": ["table1", "table2"],
  "columns": {"table1": ["col_a", "col_b"], "table2": ["col_c"]},
  "joins": [{"left": "table1.col", "right": "table2.col"}],
  "aggregations_needed": ["AVG(price)", "COUNT(*)"],
  "filters_needed": ["year_built >= 2020", "status = 'delivered'"],
  "group_by_needed": true,
  "window_functions_needed": false,
  "cte_needed": false,
  "self_join_needed": false
}

Be precise — only include tables/columns actually needed."""


def _agent_schema_linker(question: str, schema: str, real_tables: dict) -> dict:
    constraint = _build_constraint_block(real_tables)
    prompt = (
        f"{constraint}\n\n"
        f"DATABASE SCHEMA:\n{schema[:3000]}\n\n"
        f"QUESTION: {question}\n\n"
        f"Output the schema-linking JSON (only use tables listed in HARD CONSTRAINT):"
    )
    resp   = _llm(prompt, system=_SYS_SCHEMA_LINKER)
    result = _parse_json(resp, {"tables": [], "columns": {}, "joins": []})

    # ── Ground-truth validation: strip any hallucinated table ──────────────────
    if real_tables:
        verified = [t for t in result.get("tables", []) if t in real_tables]
        if not verified:
            # LLM produced entirely wrong tables — fall back to all real tables
            verified = list(real_tables.keys())
            print(f"  [SoT/1-SchemaLinker] ⚠ All suggested tables invalid, using full DB")
        result["tables"] = verified
        # Also filter columns
        result["columns"] = {
            t: [c for c in cols if c in real_tables.get(t, [])]
            for t, cols in result.get("columns", {}).items()
            if t in verified
        }

    print(f"  [SoT/1-SchemaLinker] verified_tables={result.get('tables', [])}")
    return result



# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Subproblem Agent
# ─────────────────────────────────────────────────────────────────────────────

_SYS_SUBPROBLEM = """You are a Subproblem Decomposition Agent for SQL generation.

Decompose the question into clause-level subproblems using the linked schema.
Use EXACT column and table names from the schema.

Output EXACTLY this JSON (null for unused clauses):
{
  "SELECT":    "exact columns or expressions to return",
  "FROM":      "primary table",
  "JOIN":      "join type and condition, or null",
  "WHERE":     "row-level filter predicates, or null",
  "GROUP_BY":  "columns to group by, or null",
  "HAVING":    "group-level filter, or null",
  "ORDER_BY":  "sort column and direction, or null",
  "LIMIT":     null,
  "DISTINCT":  false,
  "CTE":       "describe CTE subquery logic, or null",
  "WINDOW":    "window function expression, or null",
  "SUBQUERY":  "subquery logic description, or null"
}"""


def _agent_subproblem(question: str, schema_link: dict, schema: str, constraint: str) -> dict:
    prompt = (
        f"{constraint}\n\n"
        f"RELEVANT TABLES: {json.dumps(schema_link.get('tables', []))}\n"
        f"RELEVANT COLUMNS: {json.dumps(schema_link.get('columns', {}))}\n"
        f"JOINS NEEDED: {json.dumps(schema_link.get('joins', []))}\n"
        f"FILTERS NEEDED: {schema_link.get('filters_needed', [])}\n"
        f"AGGREGATIONS: {schema_link.get('aggregations_needed', [])}\n\n"
        f"SCHEMA EXCERPT:\n{schema[:1800]}\n\n"
        f"QUESTION: {question}\n\n"
        f"Output the clause decomposition JSON (only use tables in HARD CONSTRAINT):"
    )
    resp   = _llm(prompt, system=_SYS_SUBPROBLEM)
    result = _parse_json(resp, {})
    active = [k for k, v in result.items() if v and v is not False]
    print(f"  [SoT/2-Subproblem] active_clauses={active}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — Query Plan Agent (Chain-of-Thought)
# ─────────────────────────────────────────────────────────────────────────────

_SYS_QUERY_PLAN = """You are a Query Plan Agent for SQL generation.

Generate a numbered, step-by-step execution plan that maps the user's question
to a SQL query. Use Chain-of-Thought reasoning — explain WHY each decision is made.

Rules:
- Reason through every clause (FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY)
- Identify edge cases: NULLs, empty groups, correlated subqueries, division by zero
- Explain join conditions explicitly
- Describe aggregation logic and what window functions are needed
- DO NOT write SQL — only a natural language procedural plan
- Number each step: "Step 1:", "Step 2:", etc."""


def _agent_query_plan(question: str, schema_link: dict, subproblems: dict, schema: str, constraint: str) -> str:
    prompt = (
        f"{constraint}\n\n"
        f"SCHEMA (relevant excerpt):\n{schema[:1800]}\n\n"
        f"SCHEMA LINKING:\n{json.dumps(schema_link, indent=2)[:1000]}\n\n"
        f"CLAUSE SUBPROBLEMS:\n{json.dumps(subproblems, indent=2)[:700]}\n\n"
        f"QUESTION: {question}\n\n"
        f"Write the numbered query plan (NO SQL, only use tables in HARD CONSTRAINT):"
    )
    plan = _llm(prompt, system=_SYS_QUERY_PLAN)
    print(f"  [SoT/3-QueryPlan] {len(plan)} chars")
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — SQL Agent
# ─────────────────────────────────────────────────────────────────────────────

_SYS_SQL_AGENT = """You are an expert DuckDB SQL Agent.

You receive a step-by-step query plan and generate the final executable SQL.

DUCKDB SYNTAX REFERENCE:
- Percentile:  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
- Buckets:     WIDTH_BUCKET(col, min_val, max_val, n_buckets)
- Quantile:    NTILE(n) OVER (ORDER BY col)
- Rolling avg: AVG(x) OVER (PARTITION BY y ORDER BY z ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
- Std dev:     STDDEV(col)
- Cond agg:    COUNT(*) FILTER (WHERE cond)   OR   SUM(CASE WHEN x THEN 1 ELSE 0 END)
- Text match:  LOWER(col) LIKE '%keyword%'
- Date trunc:  DATE_TRUNC('month', col)
- Self join:   FROM table t1 JOIN table t2 ON t1.col = t2.col
- Safe divide: col / NULLIF(denominator, 0)

CTE SCOPE RULES (CRITICAL — most common failure mode):
- A CTE column is NOT automatically available in the final SELECT unless you
  SELECT FROM the CTE directly:
    CORRECT:   WITH cte AS (...) SELECT cte.col FROM cte
    WRONG:     WITH cte AS (...) SELECT col FROM other_table WHERE col = cte.col
- To filter by a CTE aggregate, join the CTE:
    WITH counts AS (...) SELECT ... FROM base JOIN counts ON base.key = counts.key WHERE counts.val > X
- To use a scalar from a CTE as a threshold, use a subquery:
    WHERE metric >= (SELECT threshold_col FROM scalar_cte)
- NEVER reference a CTE column directly in WHERE/HAVING from a different table

ABSOLUTE RULES:
1. Output ONLY the SQL — no explanations, no markdown fences, no semicolons
2. Every non-aggregated SELECT column MUST appear in GROUP BY
3. Use ONLY table/column names visible in the schema
4. Always include the final SELECT after all CTEs
5. Qualify ambiguous columns with table aliases"""


_SYS_SQL_QUICKFIX = """You are a DuckDB SQL fixer. You receive a SQL query that failed with an error.
Fix ONLY the error. Output ONLY the corrected SQL — no explanation, no markdown, no semicolons.

CTE SCOPE REMINDER:
- CTE columns are only accessible if you SELECT FROM the CTE (or JOIN it)
- To filter by a CTE scalar, use: WHERE col >= (SELECT scalar FROM cte)
- To use CTE aggregate alongside base table cols, JOIN the CTE:
  FROM base JOIN cte ON base.key = cte.key
- GROUP BY must include ALL non-aggregated SELECT columns"""


def _agent_sql(question: str, query_plan: str, subproblems: dict, schema: str, constraint: str) -> str:
    """
    Generate SQL from the query plan.
    Inner try-fix loop: run the SQL immediately in DuckDB.
    If it fails, do ONE fast retry with the exact error baked into the prompt.
    This catches the most common errors (CTE scope, GROUP BY) before they
    reach the LangGraph correction cycle.
    """
    prompt = (
        f"{constraint}\n\n"
        f"SCHEMA:\n{schema[:2000]}\n\n"
        f"CLAUSE BREAKDOWN:\n{json.dumps(subproblems, indent=2)[:600]}\n\n"
        f"QUERY PLAN:\n{query_plan}\n\n"
        f"QUESTION: {question}\n\n"
        f"Generate the DuckDB SQL query (no semicolon, no markdown).\n"
        f"CRITICAL: use ONLY tables listed in HARD CONSTRAINT above."
    )
    raw = _llm(prompt, system=_SYS_SQL_AGENT)
    sql = _strip_sql(raw)
    print(f"  [SoT/4-SQLAgent] {sql[:200]}...")

    # ── Inner self-validation: try to execute, fix immediately if fails ────────
    try:
        run_sql(sql)
        print(f"  [SoT/4-SQLAgent] ✓ inner validation passed")
    except Exception as inner_err:
        err_msg = str(inner_err)[:400]
        print(f"  [SoT/4-SQLAgent] ✗ inner error: {err_msg[:120]} — quick-fixing...")
        fix_prompt = (
            f"FAILED SQL:\n{sql}\n\n"
            f"EXECUTION ERROR:\n{err_msg}\n\n"
            f"SCHEMA:\n{schema[:1500]}\n\n"
            f"Fix this SQL. Output ONLY the corrected SQL:"
        )
        raw2 = _llm(fix_prompt, system=_SYS_SQL_QUICKFIX)
        sql2 = _strip_sql(raw2)
        if sql2:
            sql = sql2
            print(f"  [SoT/4-SQLAgent] quick-fixed: {sql[:200]}...")

    return sql


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — Correction Plan Agent (Error Taxonomy + CoT)
# ─────────────────────────────────────────────────────────────────────────────

_SYS_CORRECTION_PLAN = (
    "You are a Correction Plan Agent for SQL debugging.\n\n"
    "You receive a failed SQL, the execution error, and the database schema.\n"
    "Use the error taxonomy to classify the error and produce a CoT correction plan.\n\n"
    + _ERROR_TAXONOMY
    + "\nOutput a numbered correction plan (NO SQL). Be specific:\n"
    "1. Which error category applies\n"
    "2. What exactly is wrong in the failed SQL\n"
    "3. Step-by-step instructions to fix each issue"
)


def _agent_correction_plan(question: str, failed_sql: str, error: str, schema: str) -> str:
    prompt = (
        f"SCHEMA:\n{schema[:2000]}\n\n"
        f"QUESTION: {question}\n\n"
        f"FAILED SQL:\n{failed_sql}\n\n"
        f"EXECUTION ERROR:\n{error[:400]}\n\n"
        f"Classify the error and write a numbered correction plan (no SQL):"
    )
    plan = _llm(prompt, system=_SYS_CORRECTION_PLAN)
    print(f"  [SoT/5-CorrPlan] {len(plan)} chars")
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 — Correction SQL Agent
# ─────────────────────────────────────────────────────────────────────────────

_SYS_CORRECTION_SQL = """You are a Correction SQL Agent.

You receive a failed SQL, a correction plan, and the schema.
Generate a corrected SQL that exactly follows the correction plan.

Rules:
- Output ONLY the corrected SQL — no explanation, no markdown, no semicolons
- Fix EXACTLY the issues described in the correction plan
- Every non-aggregated SELECT column MUST appear in GROUP BY
- Use ONLY table/column names from the schema
- Safe divide: use NULLIF(denominator, 0)"""


def _agent_correction_sql(question: str, failed_sql: str, corr_plan: str, schema: str) -> str:
    prompt = (
        f"SCHEMA:\n{schema[:2000]}\n\n"
        f"QUESTION: {question}\n\n"
        f"FAILED SQL:\n{failed_sql}\n\n"
        f"CORRECTION PLAN:\n{corr_plan}\n\n"
        f"Generate the corrected DuckDB SQL:"
    )
    raw = _llm(prompt, system=_SYS_CORRECTION_SQL)
    sql = _strip_sql(raw)
    print(f"  [SoT/6-CorrSQL] {sql[:200]}...")
    return sql


# ─────────────────────────────────────────────────────────────────────────────
# Public Interface — same signatures as before (LangGraph compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def sql_builder(state: SentinelState) -> dict:
    """
    SQL-of-Thought pipeline (replaces the single-step sql_builder).

    Attempt 1 (validation_attempts == 0):  Forward pipeline
      Agent 1: Schema Linking  →  extract relevant tables/columns/joins
      Agent 2: Subproblem      →  clause-level JSON decomposition
      Agent 3: Query Plan      →  CoT procedural plan (no SQL)
      Agent 4: SQL Agent       →  generate SQL from plan

    Attempt 2+ (correction loop):  Guided Correction Loop
      Agent 5: Correction Plan →  classify error (taxonomy) + CoT fix plan
      Agent 6: Correction SQL  →  regenerate SQL from correction plan
    """
    question = state["query"]
    attempts = state["validation_attempts"]

    # Get schema — SCHEMA is the full schema string in namespace, fall back gracefully
    try:
        schema = SCHEMA
    except Exception:
        schema = state.get("linked_schema", "")

    print(f"\n[SoT] {'='*60}")
    print(f"[SoT] Attempt {attempts + 1} | {question[:100]}")

    # ── Discover real tables once — ground truth for all agents ──────────────
    real_tables = _get_real_tables()
    constraint  = _build_constraint_block(real_tables)
    print(f"[SoT] DB tables verified: {list(real_tables.keys())}")

    if attempts == 0:
        # ── Forward pipeline ─────────────────────────────────────────────────
        print("[SoT] ► Forward pipeline: SchemaLink → Subproblem → QueryPlan → SQL")
        schema_link = _agent_schema_linker(question, schema, real_tables)
        subproblems = _agent_subproblem(question, schema_link, schema, constraint)
        query_plan  = _agent_query_plan(question, schema_link, subproblems, schema, constraint)
        sql         = _agent_sql(question, query_plan, subproblems, schema, constraint)

    else:
        # ── Guided Correction Loop ───────────────────────────────────────────
        failed_sql = state.get("sql_query", "")
        error      = state.get("validation_error", "")
        print(f"[SoT] ► Correction loop #{attempts} | error: {error[:120]}")
        corr_plan = _agent_correction_plan(question, failed_sql, error, schema)
        sql       = _agent_correction_sql(question, failed_sql, corr_plan, schema)

    return {
        "sql_query":        sql,
        "sql_candidates":   json.dumps([sql]),
        "validation_error": "",
    }


def should_retry_sql(state: SentinelState) -> Literal["sql_builder", "viz_agent", END]:
    """
    Route after sql_validator:
    - If error and under retry cap  → correction loop (sql_builder)
    - If error and max retries hit  → END (show whatever we have)
    - If no error                   → viz_agent
    """
    effective_max = min(MAX_RETRIES, 3)
    if state["validation_error"] and state["validation_attempts"] < effective_max:
        print(f"[SoT/Router] Retry {state['validation_attempts']}/{effective_max} → correction loop")
        return "sql_builder"
    if state["validation_error"]:
        print(f"[SoT/Router] Max retries reached → END")
        return END
    return "viz_agent"