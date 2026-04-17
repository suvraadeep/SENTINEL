import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet


def _discover_forecast_schema():
    """
    Discover real table, date column, and numeric metric columns.
    Returns dict {table, date_col, metric_cols} or None if no tables found.
    """
    try:
        tables_df = run_sql("SHOW TABLES")
        tables = tables_df.iloc[:, 0].tolist() if not tables_df.empty else []
    except Exception:
        return None

    if not tables:
        return None

    skip_kw = ("modified", "tmp", "temp")
    real_tables = [t for t in tables if not any(s in t.lower() for s in skip_kw)]
    if not real_tables:
        real_tables = tables

    table = real_tables[0]
    for name in ["orders", "sales", "transactions", "events", "records", "data", "main"]:
        for t in real_tables:
            if name in t.lower():
                table = t
                break

    try:
        desc_df = run_sql(f"DESCRIBE \"{table}\"")
    except Exception:
        return {"table": table, "date_col": None, "metric_cols": []}

    cname_col = desc_df.columns[0]
    ctype_col = desc_df.columns[1] if len(desc_df.columns) > 1 else cname_col

    date_col    = None
    metric_cols = []

    for _, row in desc_df.iterrows():
        cn = str(row[cname_col])
        ct = str(row[ctype_col]).lower()
        cl = cn.lower()

        if ("date" in cl or "time" in cl or "period" in cl) and \
           ("date" in ct or "timestamp" in ct or "date" in cl or "time" in cl):
            if date_col is None:
                date_col = cn
            continue

        is_num = any(t in ct for t in ["int", "float", "double", "decimal",
                                        "numeric", "real", "bigint", "number"])
        if is_num:
            # Skip ID-like columns
            if cn.lower().endswith("_id") or cn.lower() == "id":
                continue
            # Prefer amount/revenue/count-like columns
            if any(k in cl for k in ["amount", "revenue", "value", "sales",
                                      "count", "total", "profit", "cost", "price", "fee"]):
                metric_cols.insert(0, cn)
            else:
                metric_cols.append(cn)

    return {
        "table":      table,
        "date_col":   date_col,
        "metric_cols": metric_cols[:4],  # cap at 4 metrics
    }


def forecast_agent(state: SentinelState) -> dict:
    """
    Dataset-agnostic Prophet forecasting.
    Auto-discovers the real table and date/metric columns — no hardcoded 'orders'.
    """
    print("[ForecastAgent] Building Prophet forecasts...")

    schema = _discover_forecast_schema()
    if schema is None:
        return {
            "forecast_result": {"error": "No tables found"},
            "final_response": "No dataset loaded. Please upload a dataset first.",
        }

    table       = schema["table"]
    date_col    = schema["date_col"]
    metric_cols = schema["metric_cols"]

    if not date_col:
        return {
            "forecast_result": {"error": "No date column found"},
            "final_response": (
                f"No date/time column found in table **{table}**. "
                "Forecasting requires a date column to produce a time series."
            ),
        }

    if not metric_cols:
        return {
            "forecast_result": {"error": "No numeric metric columns found"},
            "final_response": (
                f"No numeric metric columns found in table **{table}**. "
                "Cannot build a forecast without a measurable quantity."
            ),
        }

    print(f"  [ForecastAgent] table={table} | date={date_col} | "
          f"metrics={metric_cols[:2]}")

    # Build time-series aggregation query dynamically
    # Aggregate each metric by date
    agg_parts = ", ".join(
        f'SUM("{c}") AS {c.lower().replace(" ", "_")}' for c in metric_cols[:2]
    )
    try:
        df = run_sql(f"""
            SELECT "{date_col}", {agg_parts}
            FROM "{table}"
            WHERE "{date_col}" IS NOT NULL
            GROUP BY "{date_col}"
            ORDER BY "{date_col}"
        """)
    except Exception as exc:
        return {
            "forecast_result": {"error": str(exc)},
            "final_response": f"Could not build time series for forecasting: {exc}",
        }

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    results      = {}
    all_analysis = []

    # Determine forecast horizon from query
    query_lower = state["query"].lower()
    if "month" in query_lower:
        horizon, freq = 30, "D"
    elif "week" in query_lower:
        horizon, freq = 7, "D"
    elif "quarter" in query_lower:
        horizon, freq = 90, "D"
    elif "year" in query_lower:
        horizon, freq = 365, "D"
    else:
        horizon, freq = 7, "D"  # default: 7 days

    # Run Prophet for each metric (up to 2)
    col_names = [c.lower().replace(" ", "_") for c in metric_cols[:2]]
    metric_labels = [
        (label, col) for label, col in zip(metric_cols[:2], col_names)
        if col in df.columns
    ]

    for metric_label, col in metric_labels:
        prophet_df = (
            df.rename(columns={date_col: "ds", col: "y"})[["ds", "y"]]
            .dropna()
        )
        prophet_df = prophet_df[prophet_df["y"].notna()]

        if len(prophet_df) < 5:
            print(f"  Insufficient data for {metric_label} ({len(prophet_df)} rows)")
            continue

        try:
            # Adapt seasonality to data frequency/length
            n_pts = len(prophet_df)
            date_range_days = (prophet_df["ds"].max() - prophet_df["ds"].min()).days
            has_weekly  = date_range_days >= 14
            has_yearly  = date_range_days >= 180
            has_daily   = date_range_days >= 7 and n_pts >= 14

            m = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                yearly_seasonality=has_yearly,
                weekly_seasonality=has_weekly,
                daily_seasonality=has_daily,
                interval_width=0.90,
                n_changepoints=min(5, max(1, n_pts // 10)),
            )
            m.fit(prophet_df)

            future   = m.make_future_dataframe(periods=horizon, freq=freq)
            forecast = m.predict(future)

            last_actual = float(prophet_df["y"].iloc[-1])
            fc_end      = float(forecast["yhat"].iloc[-1])
            pct_change  = (fc_end - last_actual) / (abs(last_actual) + 1e-9) * 100
            ci_upper    = float(forecast["yhat_upper"].iloc[-1])
            ci_lower    = float(forecast["yhat_lower"].iloc[-1])

            print(f"\n  [{metric_label}] Current: {last_actual:,.2f} | "
                  f"{horizon}d forecast: {fc_end:,.2f} ({pct_change:+.1f}%)")
            print(f"  90% CI: [{ci_lower:,.2f}, {ci_upper:,.2f}]")

            try:
                delta_abs        = np.abs(m.params["delta"].mean(axis=0))
                cp_mask          = delta_abs > 0.01
                changepoints_str = [cp.strftime("%Y-%m-%d")
                                    for cp in m.changepoints[cp_mask]][:4]
            except Exception:
                changepoints_str = []

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prophet_df["ds"], y=prophet_df["y"],
                name="Actual", line=dict(color="#2196F3", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"],
                name="Forecast", line=dict(color="#FF9800", dash="dash", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
                y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
                fill="toself", fillcolor="rgba(255,152,0,0.15)",
                line=dict(color="rgba(255,152,0,0)"), name="90% CI",
            ))
            for cp_str in changepoints_str:
                try:
                    safe_vline(fig, cp_str, color="gray", dash="dot")
                except Exception:
                    pass
            try:
                safe_vline(fig, prophet_df["ds"].max().strftime("%Y-%m-%d"),
                           label="Forecast start", color="#4CAF50", dash="dash")
            except Exception:
                pass

            fig.update_layout(
                title=f"{metric_label} — {horizon}-Day Forecast (90% CI)",
                xaxis_title=date_col, yaxis_title=metric_label,
                template="sentinel", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            safe_show(fig, f"{metric_label} — {horizon}-Day Forecast")

            analysis = _analyze_chart(
                f"{metric_label} — {horizon}-Day Prophet Forecast",
                "line + confidence band", "ds", "yhat",
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                extra_context=(
                    f"Actual last value: {last_actual:,.2f}\n"
                    f"Forecast {horizon}d: {fc_end:,.2f} ({pct_change:+.1f}%)\n"
                    f"90% CI: [{ci_lower:,.2f}, {ci_upper:,.2f}]\n"
                    f"Changepoints: {changepoints_str}"
                ),
            )
            all_analysis.append(f"**{metric_label} Forecast**:\n{analysis}")

            results[col] = {
                "current":         round(last_actual, 4),
                f"forecast_{horizon}d": round(fc_end, 4),
                "pct_change":      round(pct_change, 1),
                "ci_upper":        round(ci_upper, 4),
                "ci_lower":        round(ci_lower, 4),
                "changepoints":    changepoints_str,
            }

        except Exception as exc:
            print(f"  Prophet failed for {metric_label}: {exc}")
            # Holt-Winters fallback
            try:
                sp = min(7, max(2, len(prophet_df) - 1))
                hw = ExponentialSmoothing(
                    prophet_df["y"], trend="add",
                    seasonal="add", seasonal_periods=sp,
                ).fit()
                fc_vals = hw.forecast(horizon)

                fig_hw = go.Figure()
                fig_hw.add_trace(go.Scatter(
                    x=prophet_df["ds"], y=prophet_df["y"],
                    name="Actual", line=dict(color="#2196F3"),
                ))
                future_dates = pd.date_range(
                    prophet_df["ds"].max(), periods=horizon + 1, freq=freq
                )[1:]
                fig_hw.add_trace(go.Scatter(
                    x=future_dates, y=fc_vals.values,
                    name="HW Forecast", line=dict(color="#FF9800", dash="dash"),
                ))
                fig_hw.update_layout(
                    title=f"{metric_label} — Holt-Winters Forecast ({horizon} days)",
                    template="sentinel", height=380,
                )
                safe_show(fig_hw, f"{metric_label} — Holt-Winters Forecast")

                results[col] = {
                    "current":         round(float(prophet_df["y"].iloc[-1]), 4),
                    f"forecast_{horizon}d": round(float(fc_vals.iloc[-1]), 4),
                    "method":          "Holt-Winters fallback",
                }
                print(f"  HW {horizon}d={fc_vals.iloc[-1]:,.2f}")
            except Exception as exc2:
                print(f"  Both Prophet and HW failed for {metric_label}: {exc2}")
                results[col] = {"error": str(exc2)}

    if not results:
        return {
            "forecast_result": {"error": "No metrics could be forecast"},
            "final_response": (
                f"Could not generate forecasts from table **{table}**. "
                "Ensure the dataset has a date column with at least 5 time periods "
                "and at least one numeric metric column."
            ),
        }

    narrative = call_llm(
        f"Summarize these {horizon}-day forecasts for a business audience in 2–3 bullet points. "
        f"Cite specific numbers (current value, forecast value, % change). "
        f"Flag the biggest risk or opportunity.\n"
        f"Dataset: {table} | Date column: {date_col}\n"
        f"{json.dumps(results, indent=2, default=str)}",
        model=FAST_MODEL, temperature=0.1,
    )

    chart_section = "\n\n".join(all_analysis)
    full_response = (
        f"{narrative}\n\n---\n\n**Chart Analysis:**\n{chart_section}"
        if chart_section else narrative
    )
    print(f"\n[ForecastAgent] Narrative:\n{narrative}")

    l2_store(state["query"], "/* Prophet forecast */",
             json.dumps({k: v.get(f"forecast_{horizon}d", 0)
                         for k, v in results.items() if isinstance(v, dict)}))

    return {"forecast_result": results, "final_response": full_response}
