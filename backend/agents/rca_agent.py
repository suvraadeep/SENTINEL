import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

def rca_agent(state: SentinelState) -> dict:
    """
    RCA agent — dataset-agnostic guard added.
    For non-e-commerce datasets, users should use the dedicated RCA page.
    """
    print("[RCAAgent] Causal root cause analysis starting...")

    # Guard: only run if the dataset is e-commerce (has an orders table)
    # For all other domains, the dedicated RCA page provides better analysis
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
            f"This inline RCA agent is optimised for e-commerce datasets. "
            f"Your dataset (**{real_table}**, domain: *{current_domain}*) "
            f"is best analysed with the dedicated **RCA** page in the sidebar, "
            f"which runs a full dataset-agnostic causal analysis."
        )
        return {"rca_result": {}, "final_response": msg}

    # Also verify the orders table exists (failsafe)
    try:
        tables_df   = run_sql("SHOW TABLES")
        table_names = [t.lower() for t in tables_df.iloc[:, 0].tolist()]
        if "orders" not in table_names:
            return {"rca_result": {}, "final_response":
                    "The 'orders' table was not found. Please use the **RCA** page."}
    except Exception:
        pass

    df = run_sql("""
        SELECT order_date, category, city, platform, payment_method,
               SUM(final_amount) AS revenue,
               COUNT(*) AS order_count,
               AVG(rating) AS avg_rating,
               SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS return_rate,
               AVG(delivery_time_hrs) AS avg_delivery_hrs
        FROM orders WHERE status IN ('delivered','returned','cancelled')
        GROUP BY order_date, category, city, platform, payment_method
        ORDER BY order_date
    """)
    df["order_date"] = pd.to_datetime(df["order_date"])
 
    all_analysis = []          # ← collect chart analyses here
 
    # ── Revenue delta by category ────────────────────────────────────
    cat_totals = run_sql("""
        SELECT category,
               AVG(CASE WHEN order_date < DATE '2024-06-08'
                        THEN final_amount END) AS first_half,
               AVG(CASE WHEN order_date >= DATE '2024-06-08'
                        THEN final_amount END) AS second_half
        FROM orders WHERE status='delivered'
        GROUP BY category
    """)
    cat_totals["delta_pct"] = (
        (cat_totals["second_half"] - cat_totals["first_half"])
        / (cat_totals["first_half"] + 1e-9) * 100
    )
    cat_totals = cat_totals.sort_values("delta_pct")
    worst_cat   = cat_totals.iloc[0]["category"]
    worst_delta = cat_totals.iloc[0]["delta_pct"]
 
    print(f"\nCategory revenue change (2nd half vs 1st half of 15 days):")
    print(cat_totals[["category", "first_half", "second_half", "delta_pct"]]
          .to_string(index=False))
 
    # ── Granger causality ────────────────────────────────────────────
    cat_df   = df[df["category"] == worst_cat].sort_values("order_date")
    daily_cat = (
        cat_df.groupby("order_date")[
            ["revenue", "avg_rating", "return_rate", "avg_delivery_hrs"]
        ].mean().dropna()
    )
 
    granger_results = {}
    for col in ["avg_rating", "return_rate", "avg_delivery_hrs"]:
        if col not in daily_cat.columns:
            continue
        try:
            test_df = daily_cat[["revenue", col]].dropna()
            if len(test_df) < 8:
                continue
            res   = grangercausalitytests(test_df, maxlag=3, verbose=False)
            min_p = min(res[lag][0]["ssr_ftest"][1] for lag in res)
            granger_results[col] = round(min_p, 4)
        except Exception:
            granger_results[col] = 1.0
 
    print(f"\nGranger causality → {worst_cat} revenue:")
    for k, v in granger_results.items():
        print(f"  {k}: p={v} → "
              f"{'✓ significant' if v < 0.05 else '✗ not significant'}")
 
    # ── Correlations ─────────────────────────────────────────────────
    corr_cols      = ["revenue", "avg_rating", "return_rate", "avg_delivery_hrs"]
    corr_available = [c for c in corr_cols if c in daily_cat.columns]
    corr_matrix    = (
        daily_cat[corr_available].corr()["revenue"].drop("revenue").to_dict()
    )
    print(f"\nCorrelation with {worst_cat} revenue:")
    for k, v in sorted(corr_matrix.items(),
                        key=lambda x: abs(x[1]), reverse=True):
        print(f"  {k}: {v:.3f}")
 
    l3_ctx    = l3_get_context("revenue")
    narrative = call_llm(
        f"""You are a senior data scientist writing a root cause analysis.
 
WORST PERFORMING CATEGORY: {worst_cat} ({worst_delta:+.1f}% revenue change)
 
CATEGORY DELTA TABLE:
{cat_totals[['category','delta_pct']].to_string(index=False)}
 
GRANGER CAUSALITY:
{json.dumps(granger_results, indent=2)}
 
CORRELATIONS WITH REVENUE:
{json.dumps({k: round(v, 3) for k, v in corr_matrix.items()}, indent=2)}
 
CAUSAL GRAPH PRIORS:
{l3_ctx}
 
Write a 3-paragraph RCA:
1. What happened and scale of impact
2. Causal drivers (cite exact p-values and correlations)
3. Prioritized action items""",
        temperature=0.2,
    )
    print(f"\n{'=' * 60}\nRCA NARRATIVE:\n{narrative}\n{'=' * 60}")
 
    # ── Chart 1: Revenue trend by category ───────────────────────────
    rev_trend = (
        df.groupby(["order_date", "category"])["revenue"]
        .sum().reset_index()
    )
    fig1 = px.line(
        rev_trend, x="order_date", y="revenue", color="category",
        title="Daily Revenue by Category — RCA Overview",
        template="plotly_white"
    )
    cutoff = pd.Timestamp("2024-06-08")
    safe_vline(fig1, cutoff, label="Analysis split", color="red", dash="dash")
    safe_show(fig1, "Revenue trend by category")
 
    expl1 = _analyze_chart(
        fig_title     = "Daily Revenue by Category — RCA Overview",
        chart_type    = "line",
        x_col         = "order_date",
        y_col         = "revenue",
        df            = rev_trend,
        color_col     = "category",
        extra_context = (
            f"Worst category: {worst_cat} ({worst_delta:+.1f}%)\n"
            f"Analysis split date: 2024-06-08\n"
            f"Granger p-values: {granger_results}"
        ),
    )
    print(f"\n  ── Chart Analysis: Revenue Trend by Category ──\n  {expl1}")
    all_analysis.append(
        f"**Daily Revenue by Category — RCA Overview**:\n{expl1}"
    )
 
    # ── Chart 2: Category delta bar ───────────────────────────────────
    fig2 = px.bar(
        cat_totals.sort_values("delta_pct"), x="category", y="delta_pct",
        color="delta_pct", color_continuous_scale="RdYlGn",
        title="Revenue % Change: 2nd Half vs 1st Half (15 days)",
        template="plotly_white", text="delta_pct"
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(coloraxis_showscale=False)
    safe_show(fig2, "Category delta comparison")
 
    expl2 = _analyze_chart(
        fig_title     = "Revenue % Change: 2nd Half vs 1st Half",
        chart_type    = "bar",
        x_col         = "category",
        y_col         = "delta_pct",
        df            = cat_totals[["category", "delta_pct"]].copy(),
        extra_context = (
            f"Worst performer: {worst_cat} at {worst_delta:+.1f}%\n"
            f"Best performer: {cat_totals.iloc[-1]['category']} at "
            f"{cat_totals.iloc[-1]['delta_pct']:+.1f}%"
        ),
    )
    print(f"\n  ── Chart Analysis: Category Delta Comparison ──\n  {expl2}")
    all_analysis.append(
        f"**Revenue % Change: 2nd Half vs 1st Half**:\n{expl2}"
    )
 
    # ── Chart 3: Granger causality ────────────────────────────────────
    if granger_results:
        gran_df = pd.DataFrame([
            {"driver": k, "p_value": v, "significant": v < 0.05}
            for k, v in granger_results.items()
        ])
        fig3 = px.bar(
            gran_df, x="driver", y="p_value", color="significant",
            color_discrete_map={True: "#4CAF50", False: "#F44336"},
            title=f"Granger Causality p-values → {worst_cat} Revenue",
            template="plotly_white"
        )
        safe_hline(fig3, 0.05, label="p=0.05")
        safe_show(fig3, "Granger causality")
 
        expl3 = _analyze_chart(
            fig_title     = f"Granger Causality p-values → {worst_cat} Revenue",
            chart_type    = "bar",
            x_col         = "driver",
            y_col         = "p_value",
            df            = gran_df,
            extra_context = (
                f"Significance threshold: p < 0.05\n"
                f"Significant drivers: "
                f"{[k for k,v in granger_results.items() if v < 0.05]}"
            ),
        )
        print(f"\n  ── Chart Analysis: Granger Causality ──\n  {expl3}")
        all_analysis.append(
            f"**Granger Causality p-values → {worst_cat} Revenue**:\n{expl3}"
        )
 
    # ── Chart 4: Correlation heatmap ──────────────────────────────────
    if len(corr_available) >= 3:
        corr_full = daily_cat[corr_available].corr()
        fig4 = px.imshow(
            corr_full, title="Metric Correlation Matrix",
            color_continuous_scale="RdBu_r", text_auto=".2f",
            template="plotly_white"
        )
        safe_show(fig4, "Correlation heatmap")
 
        # Flatten for _analyze_chart
        corr_long = corr_full.reset_index().melt(
            id_vars="index", var_name="metric_b", value_name="correlation"
        ).rename(columns={"index": "metric_a"})
        expl4 = _analyze_chart(
            fig_title     = "Metric Correlation Matrix",
            chart_type    = "heatmap",
            x_col         = "metric_a",
            y_col         = "metric_b",
            z_col         = "correlation",
            df            = corr_long,
            extra_context = (
                f"Category: {worst_cat}\n"
                f"Revenue correlations: "
                f"{json.dumps({k: round(v,3) for k,v in corr_matrix.items()})}"
            ),
        )
        print(f"\n  ── Chart Analysis: Correlation Heatmap ──\n  {expl4}")
        all_analysis.append(f"**Metric Correlation Matrix**:\n{expl4}")
 
    # ── Build full response ───────────────────────────────────────────
    chart_section = "\n\n".join(all_analysis)
    full_response = (
        f"{narrative}\n\n"
        f"{'─' * 50}\n"
        f"📊 Chart Analysis:\n{chart_section}"
    )
 
    rca_result = {
        "worst_category": worst_cat,
        "delta_pct":      worst_delta,
        "granger":        granger_results,
        "correlations":   corr_matrix,
        "narrative":      narrative,
    }
    l2_store(state["query"], "/* RCA */",
             f"RCA: {worst_cat} revenue {worst_delta:+.1f}%")
 
    return {"rca_result": rca_result, "final_response": full_response}