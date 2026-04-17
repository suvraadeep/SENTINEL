def route_intent(state: SentinelState) -> str:
    return {
        "rca":        "rca_agent",
        "forecast":   "forecast_agent",
        "anomaly":    "anomaly_agent",
        "math":       "math_agent",
        "prediction": "prediction_agent",
        "sql_query":  "schema_linker",
    }.get(state["intent"], "schema_linker")


builder = StateGraph(SentinelState)

builder.add_node("intent_classifier",  intent_classifier)
builder.add_node("schema_linker",       schema_linker)
builder.add_node("query_decomposer",    query_decomposer)
builder.add_node("sql_builder",         sql_builder)
builder.add_node("sql_validator",       sql_validator)
builder.add_node("viz_agent",           viz_agent)
builder.add_node("sql_response_writer", sql_response_writer)
builder.add_node("rca_agent",           rca_agent)
builder.add_node("forecast_agent",      forecast_agent)
builder.add_node("anomaly_agent",       anomaly_agent)
builder.add_node("math_agent",          math_agent)
builder.add_node("prediction_agent",    prediction_agent)

builder.set_entry_point("intent_classifier")

builder.add_conditional_edges(
    "intent_classifier", route_intent,
    {"schema_linker": "schema_linker", "rca_agent": "rca_agent",
     "forecast_agent":"forecast_agent","anomaly_agent":"anomaly_agent",
     "math_agent":    "math_agent",    "prediction_agent": "prediction_agent"}
)
builder.add_edge("schema_linker",    "query_decomposer")
builder.add_edge("query_decomposer", "sql_builder")
builder.add_edge("sql_builder",      "sql_validator")
builder.add_conditional_edges(
    "sql_validator", should_retry_sql,
    {"sql_builder":"sql_builder","viz_agent":"viz_agent", END:END}
)
builder.add_edge("viz_agent",           "sql_response_writer")
builder.add_edge("sql_response_writer", END)
builder.add_edge("rca_agent",         END)
builder.add_edge("forecast_agent",    END)
builder.add_edge("anomaly_agent",     END)
builder.add_edge("math_agent",        END)
builder.add_edge("prediction_agent",  END)

sentinel = builder.compile()


def ask(query: str) -> str:
    """
    Main entry point.
    1. Clears the figure queue (prevents leftover charts from prior calls)
    2. Runs the full SENTINEL pipeline
    3. Flushes all queued charts ONCE from main thread — zero duplicates
    """
    global _FIG_QUEUE
    _FIG_QUEUE.clear()   # ← clear stale queue before each new query

    print(f"\n{'═'*70}")
    print(f"QUERY: {query}")
    print(f"{'═'*70}\n")

    state  = empty_state(query)
    result = sentinel.invoke(state)

    n_shown = flush_charts()
    if n_shown > 0:
        print(f"\n[VizLayer] Displayed {n_shown} chart(s)")

    resp = result.get("final_response", "No response generated.")
    print(f"\n{'─'*70}")
    print(f"FINAL RESPONSE:\n{resp}")
    print(f"{'─'*70}")
    return resp


print("LangGraph compiled.")
print(f"Nodes: {list(sentinel.nodes.keys())}")