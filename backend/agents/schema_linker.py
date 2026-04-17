import json
from typing import Dict, List

def schema_linker(state: SentinelState) -> dict:
    """
    NOVEL: Schema linking before SQL generation (from DIN-SQL + CHESS papers).
    Maps NL query entities → exact schema elements, reducing hallucination by ~40%.
    Standard text-to-SQL skips this step — we implement it as a dedicated agent.
    """
    prompt = f"""You are a database schema expert performing schema linking.

TASK: Map every entity/concept in the user query to the EXACT schema element.

SCHEMA TABLES AND COLUMNS:
{state['schema_context'][:3000]}

USER QUERY: {state['query']}

Return a JSON with EXACT column names from schema:
{{
  "relevant_tables": ["table1", ...],
  "column_links": {{"query_phrase": "table.column", ...}},
  "required_joins": ["JOIN condition"],
  "filter_conditions": ["condition"],
  "aggregations": ["expression AS alias"],
  "groupby_cols": ["col"],
  "orderby_cols": ["col ASC/DESC"],
  "date_filter": "SQL date filter or empty string",
  "needs_subquery": true/false,
  "query_complexity": "simple/moderate/complex"
}}

Business rules to apply:
{l3_get_business_rules()}

Return ONLY valid JSON."""

    raw = call_llm(prompt, model=MAIN_MODEL, temperature=0.0)
    # Build a dynamic fallback: extract real table names from schema_context
    # instead of hardcoding "orders" (which would fail on non-e-commerce datasets).
    _fallback_tables: list = []
    try:
        _tables_df = run_sql("SHOW TABLES")
        _fallback_tables = _tables_df.iloc[:, 0].tolist() if not _tables_df.empty else []
        # Exclude utility tables from the fallback list
        _fallback_tables = [t for t in _fallback_tables
                             if not any(s in t.lower() for s in ("modified", "tmp", "temp"))]
    except Exception:
        pass
    if not _fallback_tables:
        # Last resort: parse table names from schema_context text
        import re as _re
        _fallback_tables = _re.findall(r'Table:\s*(\w+)', state.get('schema_context', ''))
    if not _fallback_tables:
        _fallback_tables = []

    linked = extract_json(raw, fallback={
        "relevant_tables": _fallback_tables,
        "column_links": {},
        "required_joins": [],
        "filter_conditions": [],
        "aggregations": [],
        "groupby_cols": [],
        "orderby_cols": [],
        "date_filter": "",
        "needs_subquery": False,
        "query_complexity": "simple"
    })

    print(f"[SchemaLinker] Tables: {linked.get('relevant_tables', [])} | "
          f"Complexity: {linked.get('query_complexity', '?')} | "
          f"Links: {len(linked.get('column_links', {}))}")
    return {"linked_schema": json.dumps(linked)}