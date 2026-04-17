import re
from typing import TypedDict, List, Dict, Any, Literal, Optional, Tuple


class SentinelState(TypedDict):
    query:               str
    intent:              str
    linked_schema:       str
    sub_queries:         str
    memory_context:      str
    schema_context:      str
    sql_query:           str
    sql_candidates:      str
    sql_result_json:     str
    validation_attempts: int
    validation_error:    str
    aqp_ci:              str
    rca_result:          dict
    forecast_result:     dict
    anomaly_result:      dict
    math_result:         dict
    n_charts_requested:  int
    chart_explanations:  str
    final_response:      str
    error:               str


def empty_state(query: str) -> SentinelState:
    n_charts = 0
    m = re.search(r'\b(\d+)\s+chart', query, re.IGNORECASE)
    if m:
        n_charts = min(int(m.group(1)), 8)
    return SentinelState(
        query=query, intent="", linked_schema="", sub_queries="",
        memory_context="", schema_context=SCHEMA,
        sql_query="", sql_candidates="", sql_result_json="",
        validation_attempts=0, validation_error="", aqp_ci="",
        rca_result={}, forecast_result={}, anomaly_result={}, math_result={},
        n_charts_requested=n_charts,
        chart_explanations="",
        final_response="", error=""
    )
