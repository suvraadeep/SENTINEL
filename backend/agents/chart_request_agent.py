"""
Chart Request Agent — Extra Plotly chart generator.

When a user explicitly requests a specific chart type (heatmap, scatter, bar,
pie, histogram, box, violin, treemap, line, bubble) this helper detects the
request, fetches or reuses data, and generates the chart in Plotly.

Called from backend/api/query.py as a post-processing step.
"""
import json
import re
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Chart type keywords ─────────────────────────────────────────────────────
_CHART_KEYWORDS = {
    "heatmap":   ["heatmap", "heat map", "heat-map"],
    "scatter":   ["scatter plot", "scatter chart", "scatter graph", "scatterplot",
                  "scatter for", "show me a scatter", "show a scatter"],
    "bar":       ["bar chart", "bar graph", "bar plot", "barchart", "grouped bar",
                  "comparison bar", "highlighted bar"],
    "pie":       ["pie chart", "pie graph", "donut chart", "doughnut"],
    "histogram": ["histogram", "freq distribution"],
    "box":       ["box plot", "box chart", "boxplot", "box-plot"],
    "violin":    ["violin plot", "violin chart"],
    "treemap":   ["treemap", "tree map"],
    "line":      ["line chart", "line graph", "multi-line", "time series chart"],
    "bubble":    ["bubble chart", "bubble plot"],
}

_DARK = dict(
    paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(color="#F1F5F9", family="Inter, sans-serif", size=12),
    legend=dict(bgcolor="#1E293B", bordercolor="#334155",
                font=dict(color="#94A3B8")),
    xaxis=dict(gridcolor="#1E293B", linecolor="#334155",
               tickfont=dict(color="#94A3B8")),
    yaxis=dict(gridcolor="#1E293B", linecolor="#334155",
               tickfont=dict(color="#94A3B8")),
    colorway=["#3B82F6","#06B6D4","#8B5CF6","#10B981",
              "#F59E0B","#EF4444","#EC4899","#14B8A6"],
)


def detect_chart_request(query: str) -> Optional[str]:
    """Return the first explicit chart type keyword found in query, or None."""
    q = query.lower()
    for chart_type, keywords in _CHART_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return chart_type
    return None


def _html(fig) -> str:
    import plotly.io as pio
    return fig.to_html(
        include_plotlyjs="cdn", full_html=False,
        config={"responsive": True, "displayModeBar": True},
    )


def _layout(fig, title: str = "") -> None:
    fig.update_layout(**_DARK)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color="#F1F5F9", size=15)))


def _mentioned(query: str, df: pd.DataFrame) -> List[str]:
    """Return column names explicitly mentioned in the query."""
    q = query.lower()
    return [c for c in df.columns if c.lower() in q or
            c.lower().replace("_", " ") in q]


def _num(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _cat(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ── Generators ──────────────────────────────────────────────────────────────

def _make_heatmap(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    nums = _num(df)
    cats = _cat(df)

    # Case 1: The data has two low-cardinality columns + one value column
    # e.g. stories × garage × avg_price (already aggregated from SQL)
    low_card_nums = [c for c in nums if df[c].nunique() <= 20]
    high_card_nums = [c for c in nums if df[c].nunique() > 20]

    # Prefer mentioned cols
    m_nums = [c for c in mcols if c in nums]
    m_cats = [c for c in mcols if c in cats]
    m_low  = [c for c in mcols if c in low_card_nums]

    # Strategy A: cat × cat → numeric pivot (e.g. property_type × segment)
    if len(m_cats) >= 2 and (m_nums or high_card_nums):
        val = (m_nums or high_card_nums)[0]
        try:
            pivot = df.groupby(m_cats[:2])[val].mean().unstack()
            z = pivot.values.astype(float)
            fig = go.Figure(go.Heatmap(
                z=z, x=pivot.columns.astype(str).tolist(),
                y=pivot.index.astype(str).tolist(),
                colorscale=[[0,"#0A0E1A"],[0.5,"#3B82F6"],[1,"#06B6D4"]],
                text=[[f"{v:,.1f}" if not np.isnan(v) else "" for v in r] for r in z],
                texttemplate="%{text}",
                colorbar=dict(tickfont=dict(color="#94A3B8")),
            ))
            title = f"{val.replace('_',' ').title()} — {m_cats[0]} × {m_cats[1]}"
            _layout(fig, title)
            return title, _html(fig)
        except Exception as e:
            logger.debug("Heatmap A failed: %s", e)

    # Strategy B: two low-cardinality numeric cols (e.g. stories × garage) → pivot on numeric value
    if len(low_card_nums) >= 2 and high_card_nums:
        r_col, c_col = low_card_nums[0], low_card_nums[1]
        val = high_card_nums[0]
        # If df already looks aggregated (few rows), use directly
        if len(df) <= 200:
            try:
                pivot = df.pivot_table(index=r_col, columns=c_col, values=val, aggfunc="mean")
                z = pivot.values.astype(float)
                fig = go.Figure(go.Heatmap(
                    z=z, x=pivot.columns.astype(str).tolist(),
                    y=pivot.index.astype(str).tolist(),
                    colorscale="Blues",
                    text=[[f"{v:,.1f}" if not np.isnan(v) else "" for v in r] for r in z],
                    texttemplate="%{text}",
                    colorbar=dict(tickfont=dict(color="#94A3B8")),
                ))
                title = f"{val.replace('_',' ').title()} by {r_col} × {c_col}"
                _layout(fig, title)
                return title, _html(fig)
            except Exception as e:
                logger.debug("Heatmap B failed: %s", e)

    # Strategy C: correlation matrix of numeric cols
    if len(nums) >= 2:
        cols = (m_nums or nums)[:8]
        corr = df[cols].corr()
        z = corr.values
        fig = go.Figure(go.Heatmap(
            z=z, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0,
            text=[[f"{v:.2f}" for v in r] for r in z],
            texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color="#94A3B8")),
        ))
        title = "Correlation Heatmap — " + ", ".join(cols)
        _layout(fig, title)
        return title, _html(fig)

    return "", ""


def _make_scatter(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    nums = _num(df)
    cats = _cat(df)
    if len(nums) < 2:
        return "", ""

    # Pick x and y — prefer mentioned cols
    m_nums = [c for c in mcols if c in nums]
    x_col = m_nums[0] if len(m_nums) >= 1 else nums[0]
    y_col = next((c for c in m_nums if c != x_col), None) or \
            next((c for c in nums if c != x_col), None)
    if not y_col:
        return "", ""

    color_col = next((c for c in mcols if c in cats), cats[0] if cats else None)

    samp = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
    samp = samp.sample(min(500, len(samp)), random_state=42) if len(samp) > 500 else samp

    try:
        fig = px.scatter(samp, x=x_col, y=y_col, color=color_col,
                         trendline="ols" if not color_col else None,
                         template="plotly_dark",
                         color_discrete_sequence=["#3B82F6"])
    except Exception:
        fig = px.scatter(samp, x=x_col, y=y_col, template="plotly_dark",
                         color_discrete_sequence=["#3B82F6"])
    title = f"{y_col.replace('_',' ').title()} vs {x_col.replace('_',' ').title()}"
    _layout(fig, title)
    return title, _html(fig)


def _make_bar(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    nums = _num(df)
    cats = _cat(df)

    m_cats = [c for c in mcols if c in cats]
    m_nums = [c for c in mcols if c in nums]

    x_col = (m_cats or cats[:1] or [None])[0]
    y_col = (m_nums or nums[:1] or [None])[0]

    # Check for multi-series (grouped bar) — e.g. segment × multiple metrics
    if x_col and len(nums) >= 2:
        val_cols = m_nums if len(m_nums) >= 2 else nums[:4]
        if x_col in df.columns and all(c in df.columns for c in val_cols):
            try:
                melted = df[[x_col] + val_cols].groupby(x_col).mean().reset_index()
                melted = pd.melt(melted, id_vars=[x_col], value_vars=val_cols,
                                 var_name="metric", value_name="value")
                fig = px.bar(melted, x=x_col, y="value", color="metric", barmode="group",
                             template="plotly_dark")
                title = f"Grouped Metrics by {x_col.replace('_',' ').title()}"
                _layout(fig, title)
                return title, _html(fig)
            except Exception:
                pass

    if not x_col or not y_col:
        return "", ""

    plot_df = df.groupby(x_col)[y_col].mean().reset_index() \
                .sort_values(y_col, ascending=False).head(25)
    fig = px.bar(plot_df, x=x_col, y=y_col, color=y_col,
                 color_continuous_scale="Blues", template="plotly_dark")
    title = f"{y_col.replace('_',' ').title()} by {x_col.replace('_',' ').title()}"
    _layout(fig, title)
    return title, _html(fig)


def _make_line(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    nums  = _num(df)
    cats  = _cat(df)
    # Try to find a date/time x-axis
    date_col = next(
        (c for c in df.columns if any(t in c.lower()
         for t in ["date","month","year","week","time","period"])), None
    )
    x_col = date_col or (mcols[0] if mcols and mcols[0] in nums + cats else
                         (nums[0] if nums else None))
    y_cols = [c for c in mcols if c in nums and c != x_col] or \
             [c for c in nums if c != x_col][:3]
    if not x_col or not y_cols:
        return "", ""

    color_col = next((c for c in cats if c != x_col), None)
    y_col = y_cols[0]

    try:
        plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
        fig = px.line(plot_df, x=x_col, y=y_col, color=color_col,
                      template="plotly_dark", markers=True)
    except Exception:
        return "", ""

    title = f"{y_col.replace('_',' ').title()} over {x_col.replace('_',' ').title()}"
    _layout(fig, title)
    return title, _html(fig)


def _make_bubble(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    nums = _num(df)
    cats = _cat(df)
    if len(nums) < 2:
        return "", ""

    m_nums = [c for c in mcols if c in nums]
    x_col  = m_nums[0] if m_nums else nums[0]
    y_col  = m_nums[1] if len(m_nums) >= 2 else (nums[1] if len(nums) > 1 else None)
    s_col  = m_nums[2] if len(m_nums) >= 3 else (nums[2] if len(nums) > 2 else None)
    c_col  = next((c for c in mcols if c in cats), cats[0] if cats else None)
    if not y_col:
        return "", ""

    samp = df.dropna(subset=[x_col, y_col])
    fig = px.scatter(samp, x=x_col, y=y_col, size=s_col, color=c_col,
                     template="plotly_dark", size_max=40,
                     color_discrete_sequence=["#3B82F6","#06B6D4","#8B5CF6","#10B981"])
    title = f"Bubble: {y_col.replace('_',' ').title()} vs {x_col.replace('_',' ').title()}"
    _layout(fig, title)
    return title, _html(fig)


def _make_pie(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    cats = _cat(df)
    nums = _num(df)
    label_col = next((c for c in mcols if c in cats), cats[0] if cats else None)
    val_col   = next((c for c in mcols if c in nums), nums[0] if nums else None)
    if not label_col:
        return "", ""
    if val_col:
        plot_df = df.groupby(label_col)[val_col].sum().reset_index().head(12)
        fig = px.pie(plot_df, names=label_col, values=val_col)
    else:
        vc = df[label_col].value_counts().head(12).reset_index()
        vc.columns = [label_col, "count"]
        fig = px.pie(vc, names=label_col, values="count")
    title = f"Distribution of {label_col.replace('_',' ').title()}"
    _layout(fig, title)
    fig.update_traces(textfont_color="#F1F5F9")
    return title, _html(fig)


def _make_box(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    nums = _num(df)
    cats = _cat(df)
    y_col = next((c for c in mcols if c in nums), nums[0] if nums else None)
    x_col = next((c for c in mcols if c in cats), cats[0] if cats else None)
    if not y_col:
        return "", ""
    fig = px.box(df, x=x_col, y=y_col, color=x_col, template="plotly_dark",
                 color_discrete_sequence=["#3B82F6","#06B6D4","#8B5CF6",
                                          "#10B981","#F59E0B","#EF4444"])
    title = f"{y_col.replace('_',' ').title()} Box Plot" + \
            (f" by {x_col.replace('_',' ').title()}" if x_col else "")
    _layout(fig, title)
    return title, _html(fig)


def _make_histogram(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    nums = _num(df)
    col  = next((c for c in mcols if c in nums), nums[0] if nums else None)
    if not col:
        return "", ""
    fig = px.histogram(df, x=col, nbins=40, template="plotly_dark",
                       color_discrete_sequence=["#3B82F6"])
    title = f"{col.replace('_',' ').title()} Distribution"
    _layout(fig, title)
    return title, _html(fig)


def _make_violin(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    nums = _num(df)
    cats = _cat(df)
    y_col = next((c for c in mcols if c in nums), nums[0] if nums else None)
    x_col = next((c for c in mcols if c in cats), None)
    if not y_col:
        return "", ""
    fig = px.violin(df, x=x_col, y=y_col, color=x_col, box=True,
                    points="outliers", template="plotly_dark")
    title = f"{y_col.replace('_',' ').title()} Violin" + \
            (f" by {x_col.replace('_',' ').title()}" if x_col else "")
    _layout(fig, title)
    return title, _html(fig)


def _make_treemap(df: pd.DataFrame, mcols: List[str], query: str) -> Tuple[str, str]:
    cats = _cat(df)
    nums = _num(df)
    if not cats or not nums:
        return "", ""
    path  = [c for c in mcols if c in cats] or cats[:2]
    value = next((c for c in mcols if c in nums), nums[0])
    plot_df = df.groupby(path)[value].sum().reset_index()
    fig = px.treemap(plot_df, path=path, values=value,
                     color=value, color_continuous_scale="Blues",
                     template="plotly_dark")
    _layout(fig, f"{value.replace('_',' ').title()} Treemap")
    return f"{value.replace('_',' ').title()} Treemap", _html(fig)


_GENERATORS = {
    "heatmap":   _make_heatmap,
    "scatter":   _make_scatter,
    "bar":       _make_bar,
    "pie":       _make_pie,
    "histogram": _make_histogram,
    "box":       _make_box,
    "violin":    _make_violin,
    "treemap":   _make_treemap,
    "line":      _make_line,
    "bubble":    _make_bubble,
}


# ── Public API ───────────────────────────────────────────────────────────────

def generate_requested_charts(
    query: str,
    sql_result_json: Optional[str],
    run_sql_fn,
    schema: str,
    call_llm_fn,
    fast_model: str,
) -> List[Tuple[str, str]]:
    """
    Main entry point called from query.py as a post-processing step.
    Returns list of (title, html_string) for each chart generated.
    """
    chart_type = detect_chart_request(query)
    if chart_type is None:
        return []

    logger.info("[ChartReq] Detected chart type: %s", chart_type)

    # ── Step 1: Get available table names for LLM SQL prompt ─────────────────
    available_tables: List[str] = []
    if run_sql_fn:
        try:
            tables_df = run_sql_fn("SHOW TABLES")
            available_tables = tables_df.iloc[:, 0].tolist()
            logger.info("[ChartReq] Tables discovered: %s", available_tables)
        except Exception as e:
            logger.warning("[ChartReq] SHOW TABLES failed: %s", e)

    # ── Step 2: Try to use the existing SQL result (fast path) ──────────────
    df = None
    if sql_result_json:
        try:
            rows = json.loads(sql_result_json)
            if isinstance(rows, list) and len(rows) >= 2:
                candidate = pd.DataFrame(rows)
                if len(candidate.columns) >= 2:
                    df = candidate
                    logger.info("[ChartReq] Using existing SQL result (%d rows)", len(df))
        except Exception as e:
            logger.debug("[ChartReq] sql_result_json parse failed: %s", e)

    # ── Step 3: Ask LLM to generate the right SQL if we have no data ─────────
    if (df is None or df.empty or len(df.columns) < 2) and run_sql_fn and call_llm_fn:
        tables_str = ", ".join(available_tables) if available_tables else "see schema"
        sql_prompt = (
            f"Given this database schema:\n{schema[:3000]}\n\n"
            f"Available tables: {tables_str}\n\n"
            f"Write ONE SQL SELECT query to fetch the data needed to draw a "
            f"'{chart_type}' chart for this request:\n\"{query}\"\n\n"
            f"Rules:\n"
            f"- Use ONLY tables listed above. Do NOT invent table names.\n"
            f"- Return 2-4 columns most relevant to the chart\n"
            f"- For heatmaps: return row_dim, col_dim, value\n"
            f"- For scatter: return x_col, y_col (optionally a color category)\n"
            f"- For bar: return category_col, value_col\n"
            f"- LIMIT to 2000 rows max\n"
            f"- Return ONLY the SQL with NO explanation and NO markdown fences"
        )
        try:
            sql = call_llm_fn(sql_prompt, model=fast_model, temperature=0.0)
            # Strip any markdown fences
            sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip().strip("` \n")
            if sql.upper().startswith("SELECT"):
                logger.info("[ChartReq] Running LLM SQL: %s", sql[:120])
                df = run_sql_fn(sql)
                logger.info("[ChartReq] LLM SQL returned %d rows", len(df) if df is not None else 0)
            else:
                logger.warning("[ChartReq] LLM did not return a SELECT: %s", sql[:100])
        except Exception as exc:
            logger.warning("[ChartReq] LLM SQL execution failed: %s", exc)

    if df is None or df.empty:
        logger.warning("[ChartReq] No data available — cannot generate %s chart", chart_type)
        return []

    # ── Step 4: Generate the chart ────────────────────────────────────────────
    # Coerce obvious numeric strings
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    mcols = _mentioned(query, df)
    logger.info("[ChartReq] Mentioned cols: %s | DataFrame cols: %s",
                mcols, df.columns.tolist())

    gen_fn = _GENERATORS.get(chart_type)
    if gen_fn is None:
        logger.warning("[ChartReq] No generator for chart type: %s", chart_type)
        return []

    try:
        title, html = gen_fn(df, mcols, query)
        if html:
            logger.info("[ChartReq] Generated %s chart: '%s'", chart_type, title)
            return [(title or f"Requested {chart_type.title()}", html)]
        else:
            logger.warning("[ChartReq] Generator returned empty html for %s", chart_type)
    except Exception as exc:
        logger.warning("[ChartReq] Generator %s crashed: %s", chart_type, exc, exc_info=True)

    return []
