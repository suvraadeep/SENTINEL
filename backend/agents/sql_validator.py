import json
from typing import Literal

def sql_validator(state: SentinelState) -> dict:
    """Execute SQL, validate result quality, apply AQP for large results."""
    sql      = state["sql_query"]
    attempts = state["validation_attempts"] + 1

    try:
        result_df = run_sql(sql)

        if result_df.empty:
            return {
                "validation_attempts": attempts,
                "validation_error":    "Query returned 0 rows. Check filters, date range, or column names.",
                "sql_result_json":     "",
            }

        # AQP: Compute confidence intervals on numeric columns
        numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
        ci_info = {}
        if len(result_df) >= 10 and numeric_cols:
            z = 1.96  # 95% CI
            for col in numeric_cols[:4]:
                se = result_df[col].sem()
                mu = result_df[col].mean()
                ci_info[col] = {
                    "mean":     round(float(mu), 4),
                    "ci_lower": round(float(mu - z * se), 4),
                    "ci_upper": round(float(mu + z * se), 4),
                }

        result_json = result_df.to_json(orient="records", date_format="iso")

        print(f"\n[SQLValidator] ✓ Attempt {attempts} | Shape: {result_df.shape}")
        print(f"  {'─'*60}")
        print(result_df.head(6).to_string(index=False))
        if ci_info:
            print(f"\n  [AQP] 95% CI on aggregates:")
            for col, ci in list(ci_info.items())[:3]:
                print(f"    {col}: {ci['ci_lower']:.2f} ↔ {ci['ci_upper']:.2f} (μ={ci['mean']:.2f})")

        return {
            "sql_result_json":    result_json,
            "validation_attempts": attempts,
            "validation_error":   "",
            "aqp_ci":             json.dumps(ci_info),
        }

    except Exception as e:
        # Catch ALL exceptions: duckdb.Error, ValueError, CatalogException, etc.
        err = str(e)[:400]
        print(f"[SQLValidator] ✗ Attempt {attempts} failed: {err}")
        return {
            "validation_attempts": attempts,
            "validation_error":    err,
            "sql_result_json":     "",
            "aqp_ci":              "",
        }


def should_retry_sql(state: SentinelState) -> Literal["sql_builder","viz_agent",END]:
    # MAX_RETRIES is 5 but that burns the Groq rate limit fast.
    # Use min(MAX_RETRIES, 3) so we burn at most 3 LLM calls on SQL.
    effective_max = min(MAX_RETRIES, 3)
    if state["validation_error"] and state["validation_attempts"] < effective_max:
        print(f"[Router] Retry {state['validation_attempts']}/{effective_max} → sql_builder")
        return "sql_builder"
    if state["validation_error"]:
        print(f"[Router] Max retries reached → END")
        return END
    return "viz_agent"