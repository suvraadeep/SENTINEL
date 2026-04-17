import json
import re
from typing import Optional, Dict, List, Set
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def render_chart(spec: dict, df: pd.DataFrame,
                 rendered_sigs: set, honor_explicit: bool = False) -> Optional[str]:
    """
    Render one chart. Returns title on success, None on skip/fail.

    Dedup logic:
    - honor_explicit=True  (user named chart types): dedup on (chart_type, x, y)
    - honor_explicit=False (auto-pick):              dedup on (x, y) only
    """
    ct    = spec.get("chart", "bar")
    x     = spec.get("x")
    y     = spec.get("y")
    color = spec.get("color") or None
    z     = spec.get("z") or None
    title = spec.get("title", f"{ct} chart")
    names = spec.get("names") or x
    vals  = spec.get("values") or y

    sig = (ct, str(x), str(y)) if honor_explicit else (str(x), str(y))
    if sig in rendered_sigs:
        print(f"  ↩ Skipped duplicate: {ct}({x},{y})")
        return None
    rendered_sigs.add(sig)

    # Column validation
    cols_needed = [c for c in [x, y, color, z, names, vals] if c is not None]
    missing     = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"  ✗ Missing columns {missing} for {ct}")
        return None
    if df.empty or (y and y in df.columns and df[y].isna().all()):
        print(f"  ✗ No data for {ct}")
        return None

    try:
        fig = None

        if ct == "bar":
            fig = px.bar(df.head(25), x=x, y=y, color=color, title=title,
                         template="plotly_white", text_auto=".2s")
        elif ct == "grouped_bar":
            fig = px.bar(df.head(25), x=x, y=y, color=color, title=title,
                         barmode="group", template="plotly_white")
        elif ct == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title,
                          template="plotly_white", markers=True)
        elif ct == "area":
            fig = px.area(df, x=x, y=y, color=color, title=title,
                          template="plotly_white")
        elif ct == "scatter":
            use_trendline = len(df) > 5 and not df[x].isna().any()
            fig = px.scatter(df, x=x, y=y, color=color, title=title,
                             template="plotly_white",
                             trendline="ols" if use_trendline else None)
        elif ct == "pie":
            top_df = df.nlargest(8, vals) if (vals and vals in df.columns) else df.head(8)
            fig = px.pie(top_df, names=names, values=vals, title=title)
        elif ct == "heatmap":
            if z and x and y and all(c in df.columns for c in [x, y, z]):
                pivot = df.pivot_table(values=z, index=y, columns=x,
                                       aggfunc="sum").fillna(0)
                fig = px.imshow(pivot, title=title, template="plotly_white",
                                color_continuous_scale="Blues", text_auto=".2s")
            else:
                print(f"  ✗ Heatmap needs x, y, z columns")
                return None
        elif ct == "histogram":
            fig = px.histogram(df, x=x, color=color, title=title,
                               template="plotly_white", nbins=25)
        elif ct == "box":
            fig = px.box(df, x=x, y=y, color=color, title=title,
                         template="plotly_white", points="outliers")
        elif ct == "treemap":
            path_cols  = spec.get("path") or ([x] if x else [])
            valid_path = [c for c in path_cols if c in df.columns]
            if not valid_path:
                valid_path = [x] if x and x in df.columns else []
            v_col = vals if (vals and vals in df.columns) else y
            if valid_path and v_col and v_col in df.columns:
                fig = px.treemap(df, path=valid_path, values=v_col, title=title)
            else:
                print(f"  ✗ Treemap missing path/values")
                return None
        elif ct == "funnel":
            fig = px.funnel(df.head(12), y=x, x=y, title=title)
        elif ct == "violin":
            fig = px.violin(df, x=x, y=y, color=color, title=title,
                            template="plotly_white", box=True)
        elif ct == "bubble":
            size_col = spec.get("size") or z
            if size_col and size_col in df.columns:
                fig = px.scatter(df, x=x, y=y, size=size_col, color=color,
                                 title=title, template="plotly_white")
            else:
                fig = px.scatter(df, x=x, y=y, color=color, title=title,
                                 template="plotly_white")
        else:
            fig = px.bar(df.head(25), x=x, y=y, title=title, template="plotly_white")

        if fig is None:
            return None

        safe_show(fig, title)
        return title

    except Exception as e:
        print(f"  ✗ {ct} error: {e}")
        return None


def _decide_chart_count(df: pd.DataFrame, n_requested: int) -> int:
    if n_requested > 0:
        return n_requested
    num = len(df.select_dtypes(include="number").columns)
    cat = len(df.select_dtypes(include=["object","category"]).columns)
    dat = sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
    # Always render at least 1 chart when we have any meaningful columns
    if num == 0 and cat == 0:
        return 0   # truly empty — nothing to plot
    if num == 0:
        return 1   # categorical only — pie/bar
    if num == 1:
        return 1   # single numeric — histogram or bar
    if dat >= 1 and num >= 1:
        return 2   # date + numeric — line + bar
    if num >= 2 and cat >= 2:
        return 3
    if num >= 4:
        return 4
    return 2


def viz_agent(state: SentinelState) -> dict:
    if not state["sql_result_json"]:
        print("[VizAgent] No data to visualize.")
        return {}

    df = pd.read_json(state["sql_result_json"], orient="records")
    if df.empty:
        return {}

    # Parse datetime columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.8:
                    df[col] = parsed
            except Exception:
                pass

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols    = [c for c in df.columns
                    if pd.api.types.is_datetime64_any_dtype(df[c])]

    n_requested  = state.get("n_charts_requested", 0)
    n_target     = _decide_chart_count(df, n_requested)
    honor_explicit = n_requested > 0   # user named specific types → dedup by (type,x,y)

    if n_target == 0:
        print("[VizAgent] Data has no plottable columns.")
        return {}

    print(f"\n[VizAgent] Target: {n_target} chart(s) "
          f"({'user-requested' if honor_explicit else 'auto'})")

    col_profile = {
        "numeric": numeric_cols, "categorical": cat_cols, "datetime": date_cols,
        "n_rows": len(df),
        "n_unique_cat": {c: int(df[c].nunique()) for c in cat_cols[:5]},
    }

    explicit_note = ""
    if honor_explicit:
        explicit_note = (
            f"\nThe user EXPLICITLY requested these chart types — you MUST include them:\n"
            f"Query: {state['query'][:300]}\n"
            f"Generate specs matching the user's requested types in the order mentioned."
        )

    prompt = f"""You are a data visualization expert. Recommend exactly {n_target} chart spec(s).

DATA PROFILE:
{json.dumps(col_profile, indent=2, default=str)}

FIRST 3 ROWS:
{df.head(3).to_dict('records')}

QUERY CONTEXT: {state['query'][:250]}
{explicit_note}

CHART TYPES AVAILABLE:
bar, grouped_bar, line, area, scatter, pie, heatmap, histogram, box, treemap, funnel, violin, bubble

RULES:
- Each spec uses columns that EXIST in the data profile above
- For pie: use 'names' (categorical) and 'values' (numeric)
- For heatmap: must include x, y (both categorical), z (numeric)
- For treemap: include 'path' (list of categorical cols) and 'values' (numeric)
- For scatter/bubble: x and y must both be numeric
- Titles must be specific and descriptive (not generic like "Revenue Chart")

Return ONLY a JSON array of exactly {n_target} spec(s):
[
  {{"chart":"bar","x":"col","y":"col","color":"col_or_null","title":"specific title"}},
  ...
]"""

    raw   = call_llm(prompt, model=FAST_MODEL, temperature=0.0)
    specs = extract_json(raw, fallback=[])

    if not specs or not isinstance(specs, list):
        # Intelligent fallback
        specs = []
        if date_cols and numeric_cols:
            specs.append({"chart":"line","x":date_cols[0],"y":numeric_cols[0],
                          "title":f"{numeric_cols[0]} over time"})
        if cat_cols and numeric_cols and len(specs) < n_target:
            specs.append({"chart":"bar","x":cat_cols[0],"y":numeric_cols[0],
                          "title":f"{numeric_cols[0]} by {cat_cols[0]}"})
        if len(numeric_cols) >= 2 and len(specs) < n_target:
            specs.append({"chart":"scatter","x":numeric_cols[0],"y":numeric_cols[1],
                          "title":f"{numeric_cols[0]} vs {numeric_cols[1]}"})

    rendered_sigs = set()
    explanations  = []
    n_rendered    = 0

    for spec in specs[:max(n_target, 8)]:
        rendered_title = render_chart(spec, df.copy(), rendered_sigs, honor_explicit)
        if rendered_title:
            n_rendered += 1
            x_c = spec.get("x", "")
            y_c = spec.get("y", "")

            expl = _analyze_chart(
                  fig_title     = spec.get("title", ""),
                  chart_type    = spec.get("chart", "bar"),
                  x_col         = spec.get("x", ""),
                  y_col         = spec.get("y", ""),
                  df            = df.copy(),
                  color_col     = spec.get("color"),
                  z_col         = spec.get("z"),
                  extra_context = f"Query: {state['query'][:200]}",
              )
            print(f"\n  Analysis [{n_rendered}] — {rendered_title}:")
            print(f"  {expl}\n")
            explanations.append(f"**{rendered_title}**: {expl}")

    print(f"[VizAgent] Done: {n_rendered} chart(s) queued")

    chart_expls = "\n\n".join(explanations)
    return {"chart_explanations": chart_expls}