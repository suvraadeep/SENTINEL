import json
import pandas as pd

def sql_response_writer(state: SentinelState) -> dict:
    sql_result_raw = state.get("sql_result_json", "")
    error          = state.get("validation_error", "")

    # ── 0-row or failed result ────────────────────────────────────────────────
    if not sql_result_raw:
        # Ask LLM to explain WHY there are no results
        explanation = call_llm(
            f"A SQL query returned 0 results or failed. "
            f"Explain possible reasons in 2-3 sentences with suggestions.\n\n"
            f"Query: {state['query']}\n"
            f"SQL: {state['sql_query'][:300]}\n"
            f"Error (if any): {error[:200]}",
            model=FAST_MODEL, temperature=0.1
        ) or (
            f"No results found. The query '{state['query'][:80]}' returned 0 rows — "
            f"the applied filters or conditions may be too restrictive for the current dataset."
        )
        return {"final_response": explanation}

    # ── Parse results ─────────────────────────────────────────────────────────
    try:
        df = pd.read_json(sql_result_raw, orient="records")
    except Exception:
        return {"final_response": "Results could not be parsed. The SQL executed but returned malformed data."}

    ci_info = extract_json(state.get("aqp_ci", "{}"), {})
    ci_str  = ""
    if ci_info:
        ci_parts = [f"{col}: 95% CI [{v['ci_lower']:.2f}, {v['ci_upper']:.2f}]"
                    for col, v in list(ci_info.items())[:3]]
        ci_str = "\n\nStatistical confidence (AQP):\n" + "\n".join(ci_parts)

    # ── LLM narrative ─────────────────────────────────────────────────────────
    response = call_llm(
        f"You are a data analyst. Give a 2-3 sentence business summary "
        f"with specific numbers and the top finding.\n\n"
        f"Question: {state['query'][:200]}\n"
        f"Result ({df.shape[0]} rows × {df.shape[1]} cols):\n"
        f"{df.head(6).to_string(index=False)[:1500]}"
        f"{ci_str}",
        model=FAST_MODEL, temperature=0.1
    )

    # ── Robust fallback if LLM unavailable ────────────────────────────────────
    if not response or len(response.strip()) < 15:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        parts = [f"**{len(df)} rows** returned."]
        for col in num_cols[:3]:
            s = df[col].dropna()
            if len(s):
                parts.append(
                    f"**{col}**: min={s.min():,.2f}, max={s.max():,.2f}, avg={s.mean():,.2f}"
                )
        if cat_cols:
            col  = cat_cols[0]
            top  = df[col].value_counts().head(3)
            parts.append("Top **{}**: ".format(col) + ", ".join(f"{k} ({v})" for k, v in top.items()))
        response = " ".join(parts)
        print(f"[ResponseWriter] LLM unavailable — using rule-based summary")

    chart_expls = state.get("chart_explanations", "")
    full_response = (
        f"{response}\n\n{'─'*50}\nChart Analysis:\n{chart_expls}"
        if chart_expls else response
    )

    print(f"\n[ResponseWriter]\n{full_response}")
    l2_store(state["query"], state["sql_query"], response, score=1.0)
    print(f"[L2] Stored. Total: {l2_collection.count()}")
    return {"final_response": full_response}