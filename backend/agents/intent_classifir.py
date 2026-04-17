import json
from typing import Dict, List

def intent_classifier(state: SentinelState) -> dict:
    """Route query to correct agent pipeline. Load all 4 memory tiers in parallel."""

    query = state['query']
    q_low = query.lower()

    # ── Keyword pre-classification for prediction-type queries ────────────
    _PREDICTION_KEYWORDS = [
        "premium", "estimate", "predict", "regression", "ml model",
        "machine learning", "train a model", "feature importance",
        "impact of", "worth", "contribution", "effect of",
        "how much does", "how much would", "value of having",
        "combined premium", "combined effect",
    ]
    if any(kw in q_low for kw in _PREDICTION_KEYWORDS):
        intent = "prediction"
    else:
        # Intent classification via LLM
        prompt = f"""Classify this data analytics query into exactly one intent.
Query: {query}

Intents:
- sql_query  : ANY query that can be answered with SQL — retrieval, aggregation,
               comparison, segmentation, ranking, filtering, window functions,
               percentiles, GROUP BY, JOINs, CASE/WHEN, text matching (LIKE),
               market analysis, price comparison, distribution, trend by group,
               cohort analysis, rolling averages, CTEs, self-joins.
               DEFAULT — use this when in doubt.

- rca        : root cause analysis — WHY did a specific metric drop or spike?
               Explain a sudden change in revenue, DAU, conversion rate, etc.

- forecast   : time-series prediction — what will happen in the FUTURE?
               "predict next month", "project next quarter", trend extrapolation.

- anomaly    : detect statistical anomalies or outliers across the WHOLE dataset.
               Health check, "find unusual patterns", z-score across all rows.

- math       : TRUE mathematical computation that CANNOT be expressed as SQL —
               calculus (derivatives/integrals), CMGR formula, CLV calculation,
               price elasticity coefficient, statistical tests (t-test, chi-square,
               ANOVA), Gini coefficient, regression coefficients, optimization.
               DO NOT use for comparisons or aggregations answerable with SQL.

- prediction : ML-based prediction / estimation — estimate a value based on
               feature combinations, calculate premiums or impacts of features,
               build a regression or classification model, detect overpriced /
               underpriced items, analyze feature importance, "how much does X
               add to Y", "estimate the combined effect of A, B, C".

Reply with ONLY the intent name."""

        intent = call_llm(prompt, model=FAST_MODEL, temperature=0.0).strip().lower()
        if intent not in {"sql_query", "rca", "forecast", "anomaly", "math", "prediction"}:
            intent = "sql_query"

    # Load all memory tiers
    episodes   = l2_retrieve(state["query"], top_k=3)
    few_shots  = l4_retrieve(state["query"], top_k=2)
    l3_ctx     = l3_get_context(state["query"].split()[0])
    biz_rules  = l3_get_business_rules()

    mem_lines = [
        "═══ L2 EPISODIC MEMORY (past similar queries) ═══"
    ]
    for i, ep in enumerate(episodes, 1):
        mem_lines.append(f"[{i}] Q: {ep['question']}")
        mem_lines.append(f"    SQL: {ep.get('sql','')[:120]}...")
        mem_lines.append(f"    Summary: {ep.get('result_summary','')[:100]}")

    mem_lines.append("\n═══ L4 PROCEDURAL MEMORY (SQL templates) ═══")
    for fs in few_shots:
        mem_lines.append(f"  Type: {fs['problem_type']}")
        mem_lines.append(f"  SQL:  {fs.get('sql_template','')[:120]}")

    mem_lines.append(f"\n═══ L3 CAUSAL GRAPH CONTEXT ═══\n{l3_ctx}")
    mem_lines.append(f"\n═══ BUSINESS RULES ═══\n{biz_rules}")

    print(f"[IntentClassifier] '{state['query'][:60]}...' → intent={intent}")
    return {
        "intent":         intent,
        "schema_context": SCHEMA,
        "memory_context": "\n".join(mem_lines),
    }