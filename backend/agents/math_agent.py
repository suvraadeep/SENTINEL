import json
import re
import math
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Schema discovery ───────────────────────────────────────────────────────────
def _discover_schema_for_math():
    """
    Domain-aware schema discovery for math analysis.
    Reads the current domain from _CURRENT_DATASET_TYPE (set by memory.seed_for_dataset_type)
    and applies domain-specific column priority so each domain gets the most
    semantically correct columns for analysis.

    Returns dict with: table, date_col, amount_col, group_col, group_cols,
                       num_cols, id_col, all_columns, domain,
                       status_col, status_filter, customer_col
    """
    domain = globals().get("_CURRENT_DATASET_TYPE", "generic")

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

    # Domain-specific table priority
    domain_table_hints = {
        "ecommerce":   ["order", "sale", "transaction"],
        "real_estate": ["listing", "property", "house", "home"],
        "financial":   ["stock", "trade", "price", "ticker"],
        "hr":          ["employee", "staff", "headcount", "people"],
    }
    table = real_tables[0]
    for hint in domain_table_hints.get(domain, []):
        for t in real_tables:
            if hint in t.lower():
                table = t
                break

    # Fallback priority names
    if table == real_tables[0]:
        for name in ["orders", "sales", "transactions", "listings", "properties",
                     "records", "data", "main", "facts", "events"]:
            for t in real_tables:
                if name in t.lower():
                    table = t
                    break

    try:
        desc_df = run_sql(f"DESCRIBE \"{table}\"")
    except Exception:
        return {"table": table, "date_col": None, "amount_col": None,
                "group_col": None, "group_cols": [], "num_cols": [],
                "id_col": None, "all_columns": [], "domain": domain,
                "status_col": None, "status_filter": "", "customer_col": None}

    if desc_df.empty:
        return {"table": table, "date_col": None, "amount_col": None,
                "group_col": None, "group_cols": [], "num_cols": [],
                "id_col": None, "all_columns": [], "domain": domain,
                "status_col": None, "status_filter": "", "customer_col": None}

    cname_col = desc_df.columns[0]
    ctype_col = desc_df.columns[1] if len(desc_df.columns) > 1 else cname_col

    try:
        sample = run_sql(f"SELECT * FROM \"{table}\" LIMIT 300")
    except Exception:
        sample = pd.DataFrame()

    date_col     = None
    amount_col   = None
    id_col       = None
    customer_col = None
    status_col   = None
    group_cols   = []
    num_cols     = []

    # Domain-specific amount column priority keywords
    amount_kws = {
        "ecommerce":   ["final_amount", "total_amount", "amount", "revenue", "price", "value", "cost"],
        "real_estate": ["sale_price", "list_price", "price", "value", "amount"],
        "financial":   ["close", "adj_close", "price", "last", "value"],
        "hr":          ["salary", "compensation", "pay", "wage", "income", "amount"],
        "generic":     ["amount", "revenue", "price", "value", "cost", "fee", "total",
                        "final", "sales", "profit", "earning", "sqft", "area", "income"],
    }
    amt_kw_list = amount_kws.get(domain, amount_kws["generic"])

    for _, row in desc_df.iterrows():
        cn = str(row[cname_col])
        ct = str(row[ctype_col]).lower()
        cl = cn.lower()

        # ── Date/time ─────────────────────────────────────────────────────
        is_pure_year = ("year_built" in cl or "built_year" in cl or
                        (cl == "year" and domain != "financial"))
        if not is_pure_year and (
            "date" in cl or "time" in cl or "period" in cl or
            "date" in ct or "timestamp" in ct
        ):
            if date_col is None:
                date_col = cn
            continue

        # ── ID columns ────────────────────────────────────────────────────
        if (cl.endswith("_id") or cl == "id" or cl.startswith("id_") or
                (cl.endswith("id") and len(cl) > 2)):
            if id_col is None:
                id_col = cn
            # E-commerce: track customer ID separately
            if domain == "ecommerce" and any(k in cl for k in
                                              ["customer", "client", "buyer", "user"]):
                customer_col = cn
            continue

        is_num = any(t in ct for t in ["int", "float", "double", "decimal",
                                        "numeric", "real", "bigint", "number"])

        if is_num:
            if any(kw in cl for kw in amt_kw_list):
                if amount_col is None:
                    amount_col = cn
                else:
                    num_cols.append(cn)
            else:
                num_cols.append(cn)
        else:
            n_uniq = sample[cn].nunique() if cn in sample.columns else 999
            if n_uniq <= 30:
                group_cols.append(cn)
                # Track status col for e-commerce
                if domain == "ecommerce" and "status" in cl and status_col is None:
                    status_col = cn

    if amount_col is None and num_cols:
        amount_col = num_cols.pop(0)

    # Domain-specific group col priority
    group_col_priority = {
        "ecommerce":   ["category", "platform", "channel", "tier", "status", "type"],
        "real_estate": ["neighborhood", "property_type", "type", "zone", "district"],
        "financial":   ["ticker", "symbol", "sector", "type"],
        "hr":          ["department", "dept", "team", "division", "seniority"],
        "generic":     ["type", "category", "status", "class", "segment", "group"],
    }
    group_col = None
    for kw in group_col_priority.get(domain, group_col_priority["generic"]):
        for gc in group_cols:
            if kw in gc.lower():
                group_col = gc
                break
        if group_col:
            break
    if group_col is None and group_cols:
        group_col = group_cols[0]

    # E-commerce: build a WHERE filter for completed/delivered rows
    status_filter = ""
    if domain == "ecommerce" and status_col:
        try:
            vals = run_sql(f'SELECT DISTINCT "{status_col}" FROM "{table}" LIMIT 30')
            vals_list = vals[status_col].astype(str).tolist()
            delivered = [v for v in vals_list if any(
                k in v.lower() for k in ["delivered", "complete", "success", "paid",
                                          "fulfilled", "closed", "confirmed"]
            )]
            if delivered:
                vals_sql = ", ".join(f"'{v}'" for v in delivered[:5])
                status_filter = f'AND "{status_col}" IN ({vals_sql})'
        except Exception:
            pass

    return {
        "table":         table,
        "date_col":      date_col,
        "amount_col":    amount_col,
        "group_col":     group_col,
        "group_cols":    group_cols,
        "num_cols":      num_cols,
        "id_col":        id_col,
        "customer_col":  customer_col or id_col,
        "status_col":    status_col,
        "status_filter": status_filter,   # e.g. "AND status IN ('delivered')"
        "domain":        domain,
        "all_columns":   desc_df[cname_col].tolist(),
    }


def math_agent(state: SentinelState) -> dict:
    """
    Dataset-agnostic math/statistics agent.
    Auto-discovers the real table & columns — never hardcodes 'orders'.
    All SQL is built dynamically from the actual schema.
    """
    print("[MathAgent] Executing mathematical/statistical analysis...")

    query      = state["query"].lower()
    results    = {}
    all_analysis = []

    # ── Discover real schema ──────────────────────────────────────────────────
    schema = _discover_schema_for_math()
    if schema is None:
        return {
            "math_result": {"error": "No tables found"},
            "final_response": "No dataset is loaded. Please upload a dataset first.",
        }

    table          = schema["table"]
    date_col       = schema["date_col"]
    amount_col     = schema["amount_col"]
    group_col      = schema["group_col"]
    group_cols     = schema["group_cols"]
    num_cols       = schema["num_cols"]
    id_col         = schema["id_col"]
    customer_col   = schema.get("customer_col") or id_col
    status_filter  = schema.get("status_filter", "")   # e.g. AND status IN ('delivered')
    domain         = schema.get("domain", "generic")
    all_cols       = schema["all_columns"]

    print(f"  [MathAgent] domain={domain} | table={table} | date={date_col} | "
          f"amount={amount_col} | group={group_col} | status_filter={bool(status_filter)}")

    # ── 1. CMGR / Growth Rate ────────────────────────────────────────────────
    if any(k in query for k in ["growth", "cmgr", "rate", "trend", "acceleration"]):
        if date_col and amount_col:
            try:
                daily_rev = run_sql(f"""
                    SELECT "{date_col}", SUM("{amount_col}") AS metric_value
                    FROM "{table}"
                    WHERE "{amount_col}" IS NOT NULL {status_filter}
                    GROUP BY "{date_col}"
                    ORDER BY "{date_col}"
                """)
                daily_rev[date_col] = pd.to_datetime(daily_rev[date_col], errors="coerce")
                daily_rev = daily_rev.dropna(subset=[date_col]).sort_values(date_col)
                daily_rev["metric_smooth"] = (
                    daily_rev["metric_value"].rolling(3, center=True).mean()
                )

                n = len(daily_rev) - 1
                if n > 0 and float(daily_rev["metric_value"].iloc[0]) > 0:
                    cmgr = (float(daily_rev["metric_value"].iloc[-1]) /
                            float(daily_rev["metric_value"].iloc[0])) ** (1 / n) - 1
                    results["cmgr_period"] = round(cmgr * 100, 4)
                    results["cmgr_annualized"] = round(((1 + cmgr) ** 365 - 1) * 100, 2)

                rev_vals = daily_rev["metric_smooth"].dropna().values
                if len(rev_vals) > 2:
                    d1 = np.gradient(rev_vals)
                    d2 = np.gradient(d1)
                    results["trend"] = "accelerating" if d2[-3:].mean() > 0 else "decelerating"
                    results["avg_velocity"] = round(float(d1.mean()), 2)

                print(f"  CMGR: {results.get('cmgr_period', 0):+.4f}% | "
                      f"trend: {results.get('trend', '?')}")

                if len(rev_vals) > 3:
                    vel_series = np.gradient(daily_rev["metric_value"].values)
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=[
                            f"{amount_col} over {date_col} (smoothed)",
                            f"{amount_col} Velocity (1st derivative)",
                        ],
                    )
                    fig.add_trace(
                        go.Scatter(x=daily_rev[date_col], y=daily_rev["metric_value"],
                                   name=amount_col, line=dict(color="#2196F3")),
                        row=1, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(x=daily_rev[date_col], y=vel_series,
                                   name="Velocity", line=dict(color="#FF5722"),
                                   fill="tozeroy"),
                        row=2, col=1,
                    )
                    fig.update_layout(
                        title=f"Growth Analysis — {amount_col}",
                        template="sentinel", height=500,
                    )
                    safe_show(fig, f"Growth Analysis — {amount_col}")

                    vel_df = pd.DataFrame({
                        date_col:    daily_rev[date_col].values,
                        "metric":    daily_rev["metric_value"].values,
                        "velocity":  vel_series,
                    })
                    expl = _analyze_chart(
                        fig_title=f"Growth Analysis — {amount_col}",
                        chart_type="line", x_col=date_col, y_col="metric", df=vel_df,
                        extra_context=(
                            f"CMGR: {results.get('cmgr_period', 0):+.4f}%\n"
                            f"Annualized CMGR: {results.get('cmgr_annualized', 0):+.2f}%\n"
                            f"Trend: {results.get('trend', '?')}"
                        ),
                    )
                    all_analysis.append(f"**Growth Analysis ({amount_col})**:\n{expl}")

            except Exception as e:
                print(f"  [MathAgent] Growth branch error: {e}")
                results["growth_error"] = str(e)
        else:
            # No date/amount — basic descriptive stats
            metric = amount_col or (num_cols[0] if num_cols else None)
            if metric:
                try:
                    stats_df = run_sql(
                        f'SELECT MIN("{metric}") AS min_val, MAX("{metric}") AS max_val, '
                        f'AVG("{metric}") AS avg_val, STDDEV("{metric}") AS std_val '
                        f'FROM "{table}"'
                    )
                    results["metric_stats"] = stats_df.to_dict("records")[0]
                    results["note"] = f"No date column detected; descriptive stats for {metric}"
                except Exception as e:
                    results["fallback_error"] = str(e)

    # ── 2. Statistical t-test / hypothesis ──────────────────────────────────
    if any(k in query for k in ["t-test", "significant", "hypothesis",
                                  "t test", "compare groups", "difference between",
                                  "ab test", "a/b"]):
        if group_col and amount_col:
            try:
                groups_df = run_sql(f"""
                    SELECT DISTINCT "{group_col}"
                    FROM "{table}"
                    WHERE "{group_col}" IS NOT NULL
                    LIMIT 10
                """)
                groups = groups_df[group_col].tolist()

                if len(groups) >= 2:
                    g1, g2 = str(groups[0]), str(groups[1])

                    def _safe_str(v):
                        return str(v).replace("'", "''")

                    grp1_df = run_sql(f"""
                        SELECT "{amount_col}" FROM "{table}"
                        WHERE "{group_col}" = '{_safe_str(g1)}'
                          AND "{amount_col}" IS NOT NULL
                        LIMIT 5000
                    """)
                    grp2_df = run_sql(f"""
                        SELECT "{amount_col}" FROM "{table}"
                        WHERE "{group_col}" = '{_safe_str(g2)}'
                          AND "{amount_col}" IS NOT NULL
                        LIMIT 5000
                    """)

                    if len(grp1_df) > 5 and len(grp2_df) > 5:
                        t_stat, p_val = ttest_ind(
                            grp1_df[amount_col].dropna(),
                            grp2_df[amount_col].dropna(),
                            equal_var=False,
                        )
                        results["ttest"] = {
                            "group_1":     g1,
                            "group_2":     g2,
                            "t_statistic": round(float(t_stat), 4),
                            "p_value":     round(float(p_val), 6),
                            "significant": bool(p_val < 0.05),
                            "conclusion": (
                                f"'{g1}' significantly higher"
                                if (p_val < 0.05 and t_stat > 0)
                                else f"'{g2}' significantly higher"
                                if (p_val < 0.05 and t_stat < 0)
                                else "No significant difference detected"
                            ),
                        }
                        print(f"  t-test '{g1}' vs '{g2}': t={t_stat:.4f}, p={p_val:.6f}")

            except Exception as e:
                print(f"  [MathAgent] t-test branch error: {e}")
                results["ttest_error"] = str(e)
        else:
            print(f"  [MathAgent] t-test: need group_col and amount_col "
                  f"(got group={group_col}, amount={amount_col})")

    # ── 3. Gini / Pareto ────────────────────────────────────────────────────
    if any(k in query for k in ["gini", "pareto", "inequality",
                                  "concentration", "80-20", "80/20"]):
        metric = amount_col or (num_cols[0] if num_cols else None)
        entity = id_col or group_col

        if metric:
            try:
                if entity:
                    dist_df = run_sql(f"""
                        SELECT "{entity}", SUM("{metric}") AS total_metric
                        FROM "{table}"
                        WHERE "{metric}" IS NOT NULL
                        GROUP BY "{entity}"
                        ORDER BY total_metric
                    """)
                    rev_sorted = np.sort(dist_df["total_metric"].dropna().values)
                else:
                    dist_df = run_sql(f"""
                        SELECT "{metric}" FROM "{table}"
                        WHERE "{metric}" IS NOT NULL
                        LIMIT 10000
                    """)
                    rev_sorted = np.sort(dist_df[metric].dropna().values)

                n = len(rev_sorted)
                if n > 1:
                    cumrev = np.cumsum(rev_sorted)
                    gini   = (n + 1 - 2 * np.sum(cumrev) / cumrev[-1]) / n
                    results["gini_coefficient"] = round(float(gini), 4)

                    total       = rev_sorted.sum()
                    cumsum_desc = np.cumsum(rev_sorted[::-1])
                    top_pct     = np.searchsorted(cumsum_desc, 0.80 * total) / n
                    results["pareto_top_pct"]       = round(top_pct * 100, 1)
                    results["pareto_interpretation"] = (
                        f"Top {top_pct*100:.1f}% entities account for 80% of {metric}"
                    )

                    print(f"  Gini: {gini:.4f} | "
                          f"Pareto: top {top_pct*100:.1f}% = 80%")

                    cum_ent = np.arange(1, n + 1) / n
                    cum_m   = cumrev / cumrev[-1]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cum_ent, y=cum_m,
                        name=f"Lorenz Curve (Gini={gini:.3f})",
                        fill="tozeroy", line=dict(color="#9C27B0"),
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], name="Perfect Equality",
                        line=dict(dash="dash", color="gray"),
                    ))
                    fig.update_layout(
                        title=f"Lorenz Curve — {metric} Concentration",
                        xaxis_title="Cumulative % Entities",
                        yaxis_title=f"Cumulative % {metric}",
                        template="sentinel",
                    )
                    safe_show(fig, f"Lorenz Curve — {metric}")

                    lorenz_df = pd.DataFrame({"cum_entities": cum_ent, "cum_metric": cum_m})
                    expl = _analyze_chart(
                        fig_title=f"Lorenz Curve — {metric}",
                        chart_type="scatter",
                        x_col="cum_entities", y_col="cum_metric", df=lorenz_df,
                        extra_context=(
                            f"Gini: {gini:.4f} | "
                            f"Top {top_pct*100:.1f}% = 80% of {metric} | "
                            f"Entities analyzed: {n}"
                        ),
                    )
                    all_analysis.append(f"**Lorenz Curve ({metric})**:\n{expl}")

            except Exception as e:
                print(f"  [MathAgent] Gini branch error: {e}")
                results["gini_error"] = str(e)

    # ── 4. Simpson's Paradox / Price / Elasticity ───────────────────────────
    if any(k in query for k in ["elastic", "discount", "sensitivity", "price effect",
                                  "price impact", "simpson", "paradox",
                                  "contradict", "aggregate", "subgroup"]):

        # ── Simpson's Paradox detection ───────────────────────────────────
        if any(k in query for k in ["simpson", "paradox", "contradict", "aggregate"]):
            if group_col and amount_col:
                try:
                    overall_df = run_sql(f"""
                        SELECT AVG("{amount_col}") AS overall_mean,
                               MEDIAN("{amount_col}") AS overall_median,
                               COUNT(*) AS n
                        FROM "{table}"
                        WHERE "{amount_col}" IS NOT NULL
                    """)
                    group_df = run_sql(f"""
                        SELECT "{group_col}",
                               AVG("{amount_col}")    AS group_mean,
                               MEDIAN("{amount_col}") AS group_median,
                               COUNT(*)               AS n
                        FROM "{table}"
                        WHERE "{amount_col}" IS NOT NULL AND "{group_col}" IS NOT NULL
                        GROUP BY "{group_col}"
                        ORDER BY group_mean DESC
                        LIMIT 20
                    """)

                    overall_mean = float(overall_df["overall_mean"].iloc[0])
                    overall_med  = float(overall_df["overall_median"].iloc[0])
                    results["overall_mean"]   = round(overall_mean, 4)
                    results["overall_median"] = round(overall_med, 4)
                    results["group_stats"]    = group_df.to_dict("records")

                    group_meds  = group_df["group_median"].tolist()
                    n_above     = sum(1 for m in group_meds if m > overall_med)
                    paradox     = (n_above == len(group_meds) or n_above == 0)
                    results["simpsons_paradox_detected"] = paradox
                    results["simpsons_explanation"] = (
                        f"PARADOX DETECTED: all {len(group_meds)} groups have "
                        f"{'higher' if n_above == len(group_meds) else 'lower'} "
                        f"median than overall (composition bias / confounding)"
                        if paradox else
                        f"No clear paradox: {n_above}/{len(group_meds)} groups above overall median"
                    )

                    print(f"  Simpson's paradox: {results['simpsons_paradox_detected']}")

                    fig_s = go.Figure()
                    fig_s.add_trace(go.Bar(
                        x=group_df[group_col].astype(str),
                        y=group_df["group_mean"],
                        name=f"Group Mean ({group_col})",
                        marker_color="#3B82F6",
                    ))
                    fig_s.add_hline(
                        y=overall_mean, line_dash="dash", line_color="#EF4444",
                        annotation_text=f"Overall Mean: {overall_mean:.2f}",
                    )
                    fig_s.update_layout(
                        title=f"Simpson's Paradox Check — {amount_col} by {group_col}",
                        xaxis_title=group_col, yaxis_title=f"Mean {amount_col}",
                        template="sentinel",
                    )
                    safe_show(fig_s, "Simpson's Paradox Check")

                    simp_df = group_df[[group_col, "group_mean"]].rename(
                        columns={"group_mean": "mean_value"}
                    )
                    expl = _analyze_chart(
                        fig_title="Simpson's Paradox Check",
                        chart_type="bar",
                        x_col=group_col, y_col="mean_value", df=simp_df,
                        extra_context=(
                            f"Overall mean: {overall_mean:.4f}\n"
                            f"Overall median: {overall_med:.4f}\n"
                            f"Groups: {group_df[group_col].astype(str).tolist()}\n"
                            f"Group means: {[round(m, 2) for m in group_df['group_mean'].tolist()]}\n"
                            f"Paradox detected: {results['simpsons_paradox_detected']}\n"
                            f"{results['simpsons_explanation']}"
                        ),
                    )
                    all_analysis.append(f"**Simpson's Paradox Analysis**:\n{expl}")

                except Exception as e:
                    print(f"  [MathAgent] Simpson's branch error: {e}")
                    results["simpsons_error"] = str(e)

        # ── Discount / price elasticity ───────────────────────────────────
        elif any(k in query for k in ["elastic", "discount", "sensitivity"]):
            # Find discount and base price columns
            discount_col    = None
            base_price_col  = None
            for c in all_cols:
                cl = c.lower()
                if any(k in cl for k in ["discount", "rebate", "reduction", "markdown"]):
                    discount_col = c
                    break
            for c in all_cols:
                cl = c.lower()
                if any(k in cl for k in ["base", "list", "original", "market"]) and \
                   any(k in cl for k in ["price", "amount", "cost", "value"]):
                    base_price_col = c
                    break

            if discount_col and base_price_col and amount_col:
                try:
                    elast_df = run_sql(f"""
                        SELECT
                            ROUND(CAST("{discount_col}" AS DOUBLE) /
                                  NULLIF(CAST("{base_price_col}" AS DOUBLE), 0) * 100, 0)
                                AS discount_pct_bucket,
                            AVG("{amount_col}") AS avg_metric,
                            COUNT(*)            AS count
                        FROM "{table}"
                        WHERE "{base_price_col}" > 0
                          AND "{discount_col}" IS NOT NULL
                        GROUP BY 1 HAVING COUNT(*) > 10
                        ORDER BY 1
                    """)
                    if len(elast_df) > 3:
                        from scipy.stats import linregress as _lr
                        slope, _, r, p, _ = _lr(
                            elast_df["discount_pct_bucket"], elast_df["count"]
                        )
                        results["elasticity"] = {
                            "slope":       round(float(slope), 4),
                            "r_squared":   round(float(r ** 2), 4),
                            "p_value":     round(float(p), 6),
                            "interpretation": (
                                f"Each 1% discount → {slope:.1f} additional units "
                                f"(R²={r**2:.3f})"
                            ),
                        }
                        fig_el = px.scatter(
                            elast_df, x="discount_pct_bucket", y="count",
                            trendline="ols",
                            title=f"Discount % vs Volume — {table}",
                            template="sentinel",
                        )
                        safe_show(fig_el, "Discount Elasticity")

                        expl = _analyze_chart(
                            fig_title="Discount Elasticity", chart_type="scatter",
                            x_col="discount_pct_bucket", y_col="count", df=elast_df,
                            extra_context=(
                                f"Slope: {slope:.4f} | R²: {r**2:.3f} | p: {p:.6f}"
                            ),
                        )
                        all_analysis.append(f"**Discount Elasticity**:\n{expl}")
                except Exception as e:
                    print(f"  [MathAgent] Elasticity branch error: {e}")
                    results["elasticity_error"] = str(e)
            elif amount_col and group_col:
                # Fallback: price distribution by group
                try:
                    price_stats = run_sql(f"""
                        SELECT "{group_col}",
                               AVG("{amount_col}")    AS avg_value,
                               MEDIAN("{amount_col}") AS median_value,
                               STDDEV("{amount_col}") AS std_value,
                               COUNT(*)               AS n
                        FROM "{table}"
                        WHERE "{amount_col}" IS NOT NULL AND "{group_col}" IS NOT NULL
                        GROUP BY "{group_col}"
                        ORDER BY avg_value DESC
                        LIMIT 20
                    """)
                    results["price_by_group"] = price_stats.to_dict("records")

                    fig_p = px.bar(
                        price_stats, x=group_col, y="avg_value",
                        title=f"{amount_col} by {group_col}",
                        template="sentinel",
                    )
                    safe_show(fig_p, f"{amount_col} by {group_col}")

                    expl = _analyze_chart(
                        fig_title=f"{amount_col} Distribution by {group_col}",
                        chart_type="bar",
                        x_col=group_col, y_col="avg_value", df=price_stats,
                        extra_context=(
                            f"Average {amount_col} distribution across {group_col} groups"
                        ),
                    )
                    all_analysis.append(
                        f"**{amount_col} Analysis by {group_col}**:\n{expl}"
                    )
                except Exception as e:
                    print(f"  [MathAgent] Price stats branch error: {e}")

    # ── 5. CLV / Lifetime value ──────────────────────────────────────────────
    if any(k in query for k in ["clv", "lifetime value", "ltv",
                                  "customer value", "customer lifetime"]):
        # Find customer-specific ID column
        cust_col = id_col
        for c in all_cols:
            cl = c.lower()
            if any(k in cl for k in ["customer", "client", "user", "member", "buyer"]):
                cust_col = c
                break

        if cust_col and amount_col and date_col:
            try:
                clv_df = run_sql(f"""
                    SELECT
                        "{cust_col}",
                        COUNT(*)              AS purchase_count,
                        SUM("{amount_col}")   AS total_value,
                        AVG("{amount_col}")   AS avg_order_value,
                        DATEDIFF('day',
                            MIN("{date_col}")::DATE,
                            MAX("{date_col}")::DATE
                        ) AS tenure_days
                    FROM "{table}"
                    WHERE "{amount_col}" IS NOT NULL {status_filter}
                    GROUP BY "{cust_col}"
                """)
                clv_df["tenure_days"] = (
                    pd.to_numeric(clv_df["tenure_days"], errors="coerce").fillna(7)
                )
                clv_df["purchase_freq"] = (
                    clv_df["purchase_count"] /
                    (clv_df["tenure_days"].clip(lower=1) / 7)
                )
                clv_df["estimated_clv"] = (
                    clv_df["avg_order_value"] * clv_df["purchase_freq"] * 52
                )

                results["clv_summary"] = {
                    "avg_clv":     round(float(clv_df["estimated_clv"].mean()), 2),
                    "median_clv":  round(float(clv_df["estimated_clv"].median()), 2),
                    "top_10pct":   round(float(clv_df["estimated_clv"].quantile(0.9)), 2),
                    "n_customers": int(len(clv_df)),
                }

                if group_col:
                    try:
                        cust_group = run_sql(f"""
                            SELECT "{cust_col}", "{group_col}"
                            FROM "{table}"
                            WHERE "{cust_col}" IS NOT NULL AND "{group_col}" IS NOT NULL
                            GROUP BY "{cust_col}", "{group_col}"
                        """)
                        clv_merged  = clv_df.merge(cust_group, on=cust_col, how="left")
                        clv_by_grp  = (
                            clv_merged.groupby(group_col)["estimated_clv"]
                            .mean().reset_index()
                        )
                        clv_by_grp.columns = [group_col, "estimated_clv"]

                        fig = px.bar(
                            clv_by_grp, x=group_col, y="estimated_clv",
                            title=f"Estimated CLV by {group_col}",
                            template="sentinel", text_auto=True,
                        )
                        safe_show(fig, f"CLV by {group_col}")

                        expl = _analyze_chart(
                            fig_title=f"CLV by {group_col}", chart_type="bar",
                            x_col=group_col, y_col="estimated_clv", df=clv_by_grp,
                            extra_context=f"CLV summary: {results['clv_summary']}",
                        )
                        all_analysis.append(
                            f"**Customer Lifetime Value by {group_col}**:\n{expl}"
                        )
                    except Exception as e:
                        print(f"  [MathAgent] CLV group chart error: {e}")

            except Exception as e:
                print(f"  [MathAgent] CLV branch error: {e}")
                results["clv_error"] = str(e)

        elif amount_col:
            # Simplified: top entities by total spend
            entity = id_col or group_col
            if entity:
                try:
                    top_ents = run_sql(f"""
                        SELECT "{entity}",
                               SUM("{amount_col}") AS total_value,
                               COUNT(*)            AS n_transactions
                        FROM "{table}"
                        WHERE "{amount_col}" IS NOT NULL
                        GROUP BY "{entity}"
                        ORDER BY total_value DESC
                        LIMIT 20
                    """)
                    results["top_entities_by_value"] = top_ents.to_dict("records")
                    all_analysis.append(
                        f"**Top entities by {amount_col}** (top 20 shown)."
                    )
                except Exception as e:
                    print(f"  [MathAgent] CLV fallback error: {e}")

    # ── Fallback: general descriptive stats ──────────────────────────────────
    if not results and not all_analysis:
        metrics = ([amount_col] if amount_col else []) + num_cols[:4]
        metrics  = [m for m in metrics if m]
        if metrics:
            try:
                sel = ", ".join(
                    f'AVG("{m}") AS avg_{m}, MEDIAN("{m}") AS med_{m}'
                    for m in metrics[:4]
                )
                stats_df = run_sql(f'SELECT {sel} FROM "{table}"')
                results["descriptive_stats"] = stats_df.to_dict("records")[0]
            except Exception as e:
                results["note"] = f"Could not compute descriptive stats: {e}"

    # ── Narrative synthesis ────────────────────────────────────────────────────
    prompt = (
        f"Summarize these mathematical/statistical findings for a business analyst. "
        f"Be specific with numbers. Use GitHub-flavored Markdown with 3–5 bullet points.\n"
        f"Query: {state['query']}\n"
        f"Dataset: {table} (date={date_col}, amount={amount_col}, group={group_col})\n"
        f"Results: {json.dumps(results, indent=2, default=str)}"
    )
    narrative = call_llm(prompt, model=FAST_MODEL, temperature=0.1)
    print(f"\n[MathAgent] Narrative:\n{narrative}")

    chart_section = "\n\n".join(all_analysis)
    full_response = (
        f"{narrative}\n\n---\n\n**Chart Analysis:**\n{chart_section}"
        if chart_section else narrative
    )

    l2_store(state["query"], "/* math computation */",
             str(results)[:300], score=1.0)
    return {"math_result": results, "final_response": full_response}
