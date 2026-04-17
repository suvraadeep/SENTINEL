import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

def anomaly_agent(state: SentinelState) -> dict:
    """
    Anomaly agent — dataset-agnostic guard added.
    For non-e-commerce datasets, users should use the dedicated Anomaly page.
    """
    print("[AnomalyAgent] Running anomaly scan...")

    # Guard: only run for e-commerce domain datasets
    current_domain = globals().get("_CURRENT_DATASET_TYPE", "generic")
    if current_domain != "ecommerce":
        try:
            tables_df  = run_sql("SHOW TABLES")
            real_table = next(
                (t for t in tables_df.iloc[:, 0].tolist()
                 if "modified" not in t.lower()),
                "unknown"
            )
        except Exception:
            real_table = "unknown"
        msg = (
            f"This inline anomaly agent is optimised for e-commerce datasets. "
            f"Your dataset (**{real_table}**, domain: *{current_domain}*) "
            f"is best analysed with the dedicated **Anomaly** page in the sidebar, "
            f"which runs full multi-method detection on any dataset."
        )
        return {"anomaly_result": {}, "final_response": msg}

    try:
        tables_df   = run_sql("SHOW TABLES")
        table_names = [t.lower() for t in tables_df.iloc[:, 0].tolist()]
        if "orders" not in table_names:
            return {"anomaly_result": {}, "final_response":
                    "The 'orders' table was not found. Please use the **Anomaly** page."}
    except Exception:
        pass

    df = run_sql("""
        SELECT order_date, category, city,
               SUM(final_amount) AS revenue,
               COUNT(*) AS order_count,
               AVG(rating) AS avg_rating,
               SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS return_rate
        FROM orders
        GROUP BY order_date, category, city
        ORDER BY order_date
    """)
    df["order_date"] = pd.to_datetime(df["order_date"])

    anomalies = []
    WINDOW, Z_THRESH = 5, 2.0
    Z_HIGH, Z_CRIT = 3.0, 4.0

    for (cat, city), group in df.groupby(["category","city"]):
        grp = group.sort_values("order_date").reset_index(drop=True)
        if len(grp) < WINDOW + 2:
            continue
        for metric in ["revenue","order_count"]:
            series = grp[metric].values
            dates  = grp["order_date"].values
            for i in range(WINDOW, len(series)):
                window   = series[i - WINDOW: i]
                mu, sigma = window.mean(), window.std() + 1e-9
                z         = (series[i] - mu) / sigma
                q1, q3    = np.percentile(window, 25), np.percentile(window, 75)
                iqr       = q3 - q1
                fl, fh    = q1 - 1.5*iqr, q3 + 1.5*iqr
                if abs(z) > Z_THRESH or series[i] < fl or series[i] > fh:
                    if abs(z) >= Z_CRIT:
                        sev_label = "CRITICAL"
                    elif abs(z) >= Z_HIGH:
                        sev_label = "HIGH"
                    else:
                        sev_label = "MEDIUM"
                    anomalies.append({
                        "date":     str(pd.Timestamp(dates[i]).date()),
                        "category": cat, "city": city, "metric": metric,
                        "value":    round(float(series[i]), 2),
                        "baseline": round(float(mu), 2),
                        "z_score":  round(float(z), 2),
                        "severity": sev_label,
                    })

    if not anomalies:
        msg = "No anomalies detected above threshold."
        print(f"  {msg}")
        return {"anomaly_result": {"count": 0}, "final_response": msg}

    anom_df      = pd.DataFrame(anomalies)
    anom_df["date"] = pd.to_datetime(anom_df["date"])
    critical     = anom_df[anom_df["severity"] == "CRITICAL"]
    high         = anom_df[anom_df["severity"].isin(["HIGH", "CRITICAL"])]
    all_analysis = []

    print(f"\n  Total: {len(anom_df)} | CRITICAL: {len(critical)} | HIGH: {len(high) - len(critical)}")
    print(anom_df.sort_values("z_score", key=abs, ascending=False)
              [["date","category","city","metric","z_score","severity"]]
              .head(8).to_string(index=False))

    plot_df = anom_df.copy()
    plot_df["abs_z"] = plot_df["z_score"].abs().clip(upper=10)
    fig1 = px.scatter(
        plot_df, x="date", y="abs_z", color="category",
        symbol="severity", size="abs_z", size_max=20,
        facet_col="metric",
        title="Anomalies — |Z-score| by Date & Category",
        template="plotly_white",
        hover_data=["city","value","baseline","z_score"]
    )
    safe_show(fig1, "Anomaly scatter — |Z-score| by date")

    analysis1 = _analyze_chart(
        "Anomalies — |Z-score| by Date & Category", "scatter (faceted)",
        "date", "abs_z", plot_df,
        extra_context=(
            f"Total anomalies: {len(anom_df)} | HIGH: {len(high)}\n"
            f"Worst: z={plot_df['z_score'].abs().max():.2f} "
            f"({plot_df.loc[plot_df['z_score'].abs().idxmax(),'category']} / "
            f"{plot_df.loc[plot_df['z_score'].abs().idxmax(),'city']})"
        )
    )
    print(f"\n  ── Chart Analysis: Anomaly Scatter ──\n  {analysis1}")
    all_analysis.append(f"**Anomaly Scatter**: {analysis1}")

    pivot = (anom_df.groupby(["category","metric"])["z_score"]
                    .apply(lambda x: x.abs().max())
                    .reset_index()
                    .pivot(index="category", columns="metric", values="z_score")
                    .fillna(0))
    fig2 = px.imshow(
        pivot, title="Worst |Z-score| Heatmap — Category × Metric",
        color_continuous_scale="RdYlGn_r", text_auto=".1f",
        template="plotly_white"
    )
    safe_show(fig2, "Severity heatmap — category × metric")

    pivot_long = pivot.reset_index().melt(id_vars="category",
                                          var_name="metric", value_name="max_z")
    analysis2 = _analyze_chart(
        "Worst |Z-score| Heatmap", "heatmap",
        "metric", "max_z", pivot_long,
        extra_context=f"Categories: {list(pivot.index)}"
    )
    print(f"\n  ── Chart Analysis: Severity Heatmap ──\n  {analysis2}")
    all_analysis.append(f"**Severity Heatmap**: {analysis2}")

    rev_daily = run_sql("""
        SELECT order_date, SUM(final_amount) AS revenue
        FROM orders WHERE status='delivered'
        GROUP BY order_date ORDER BY order_date
    """)
    rev_daily["order_date"] = pd.to_datetime(rev_daily["order_date"])

    rev_anom_daily = (
        anom_df[anom_df["metric"] == "revenue"]
        .groupby("date")["z_score"]
        .apply(lambda x: x.abs().max()).reset_index()
        .merge(rev_daily, left_on="date", right_on="order_date", how="inner")
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=rev_daily["order_date"], y=rev_daily["revenue"],
        name="Daily Revenue", line=dict(color="#2196F3", width=2)
    ))
    if not rev_anom_daily.empty:
        fig3.add_trace(go.Scatter(
            x=rev_anom_daily["date"], y=rev_anom_daily["revenue"],
            mode="markers", name="Anomaly Dates",
            marker=dict(color="red", size=14, symbol="x-thin-open",
                        line=dict(width=3))
        ))
    fig3.update_layout(
        title="Revenue Timeline with Anomaly Markers",
        template="plotly_white", height=400
    )
    safe_show(fig3, "Revenue timeline + anomaly markers")

    analysis3 = _analyze_chart(
        "Revenue Timeline with Anomaly Markers", "line + scatter overlay",
        "order_date", "revenue", rev_daily,
        extra_context=(
            f"Anomaly dates: {rev_anom_daily['date'].dt.strftime('%m-%d').tolist() if not rev_anom_daily.empty else []}\n"
            f"Revenue range: {rev_daily['revenue'].min():,.0f}–{rev_daily['revenue'].max():,.0f}"
        )
    )
    print(f"\n  ── Chart Analysis: Revenue Timeline ──\n  {analysis3}")
    all_analysis.append(f"**Revenue Timeline + Anomaly Markers**: {analysis3}")

    cat_anom_count = (anom_df.groupby("category")
                             .agg(anomaly_count=("z_score","count"),
                                  max_z=("z_score", lambda x: x.abs().max()))
                             .reset_index()
                             .sort_values("anomaly_count", ascending=False))
    fig4 = px.bar(cat_anom_count, x="category", y="anomaly_count",
                  color="max_z", color_continuous_scale="Reds",
                  title="Anomaly Count by Category (color = worst |Z-score|)",
                  template="plotly_white", text="anomaly_count")
    fig4.update_traces(textposition="outside")
    safe_show(fig4, "Anomaly count by category")

    analysis4 = _analyze_chart(
        "Anomaly Count by Category", "bar",
        "category", "anomaly_count", cat_anom_count,
        extra_context=f"Max z-score per category shown as color"
    )
    print(f"\n  ── Chart Analysis: Category Anomaly Count ──\n  {analysis4}")
    all_analysis.append(f"**Anomaly Count by Category**: {analysis4}")

    top5  = anom_df.sort_values("z_score", key=abs, ascending=False).head(5)
    alert = call_llm(
        f"Write a 3-4 sentence data alert for these anomalies. "
        f"State the severity and suggest one immediate action.\n{top5.to_string(index=False)}",
        model=FAST_MODEL, temperature=0.1
    )
    print(f"\nALERT:\n{alert}")

    chart_section = "\n\n".join(all_analysis)
    full_response = (
        f"{alert}\n\n"
        f"{'─'*50}\n"
        f"Chart-by-chart analysis (charts rendered above):\n{chart_section}"
    )

    l2_store(state["query"], "/* anomaly scan */",
             f"{len(anom_df)} anomalies, {len(critical)} CRITICAL, {len(high)-len(critical)} HIGH")
    return {"anomaly_result": {
                "count":          len(anom_df),
                "total_anomalies": len(anom_df),
                "critical_count": len(critical),
                "high_count":     len(high) - len(critical),
                "alert":          alert,
                "anomalies":      anom_df.head(20).to_dict("records"),
            },
            "final_response": full_response}