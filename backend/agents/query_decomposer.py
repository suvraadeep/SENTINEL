import json
from typing import Dict, List

def query_decomposer(state: SentinelState) -> dict:
    """
    NOVEL: Multi-granularity query decomposition (MAC-SQL paper adapted).
    For complex queries, breaks into atomic sub-questions each solvable with one SQL.
    Uses linked_schema to guide decomposition — this combination is novel.
    """
    linked = extract_json(state["linked_schema"], {})
    complexity = linked.get("query_complexity", "simple")

    if complexity == "simple":
        print("[QueryDecomposer] Simple query — no decomposition needed")
        return {"sub_queries": json.dumps([state["query"]])}

    prompt = f"""You are an expert at decomposing complex analytical queries.

ORIGINAL QUERY: {state['query']}
SCHEMA PLAN: {state['linked_schema'][:1000]}
DATA DATE RANGE: {DATA_DATE_MIN} to {DATA_DATE_MAX}

Break this into 2-4 sequential atomic sub-questions that:
1. Each can be answered with a single SQL query
2. Are ordered so later ones may depend on results from earlier ones
3. Together fully answer the original query

Return ONLY a JSON array of strings:
["sub-question 1", "sub-question 2", ...]"""

    raw = call_llm(prompt, model=MAIN_MODEL, temperature=0.0)
    sub_queries = extract_json(raw, fallback=[state["query"]])
    if not isinstance(sub_queries, list):
        sub_queries = [state["query"]]

    print(f"[QueryDecomposer] Decomposed into {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  [{i}] {sq[:80]}")

    return {"sub_queries": json.dumps(sub_queries)}