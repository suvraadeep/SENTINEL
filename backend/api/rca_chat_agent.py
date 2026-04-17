"""
RCA Chat Agent — LangGraph ReAct agent for the Root Cause Analysis dashboard.

Tools:
  1. run_sql(sql)                           — SQL SELECT against DuckDB
  2. traverse_feature(feature)              — BFS upstream causal chain → markdown
  3. compare_feature_periods(feature)       — first vs second half means + t-test
  4. generate_chart(chart_type, cols, title)— Plotly chart injected into dashboard
  5. explain_cause(feature)                 — synthesize Spearman ρ, partial corr,
                                             Granger p, MI into a narrative

Graph: START → agent_node ↔ tool_node → END (max 4 iterations)

System prompt: strict Markdown, never invent stats, LaTeX for formulas.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def _tool_run_sql(sql: str, con, table: str) -> str:
    """Execute SQL against DuckDB; return markdown table (50-row cap)."""
    if con is None:
        return "**Error:** No database connection available."
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        return "**Error:** Only SELECT queries are permitted."
    try:
        df = con.execute(sql).df()
        if df.empty:
            return "_No rows returned._"
        preview = df.head(50)
        cols   = list(preview.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        rows   = ["| " + " | ".join(str(preview[c].iloc[i]) for c in cols) + " |"
                  for i in range(len(preview))]
        note = f"\n\n_Showing {len(preview)} of {len(df)} rows._" if len(df) > 50 else ""
        return "\n".join([header, sep] + rows) + note
    except Exception as exc:
        return f"**SQL Error:** {exc}"


def _tool_traverse_feature(feature: str, context: Dict) -> str:
    """
    BFS upstream of feature in the causal graph from RCA context.
    Returns a markdown causal chain.
    """
    graph_data = context.get("graph", {})
    nodes_raw  = graph_data.get("nodes", [])
    edges_raw  = graph_data.get("edges", [])

    if not nodes_raw or not edges_raw:
        return (
            f"**No causal graph available.** "
            f"Run RCA first to build the causal graph for `{feature}`."
        )

    # Build adjacency (edge direction: source → target, so upstream = reverse)
    upstream: Dict[str, List[str]] = {}
    edge_info: Dict[str, Dict] = {}
    for e in edges_raw:
        src = e.get("source") or e.get("from", "")
        tgt = e.get("target") or e.get("to", "")
        if not src or not tgt:
            continue
        upstream.setdefault(tgt, []).append(src)
        edge_info[f"{src}->{tgt}"] = e

    # BFS
    visited: List[str] = []
    queue   = [feature]
    seen    = {feature}
    while queue:
        node = queue.pop(0)
        visited.append(node)
        for parent in upstream.get(node, []):
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)

    if len(visited) <= 1:
        return f"**`{feature}`** has no identified upstream causes in the causal graph."

    # Format as chain
    lines = [f"## Causal Chain Upstream of `{feature}`", ""]
    for i, node in enumerate(visited):
        prefix = "🎯 **Target:**" if i == 0 else f"  {'└' if i == len(visited)-1 else '├'}─"
        edge_k = f"{visited[i-1]}->{node}" if i > 0 else None
        edge_meta = ""
        if edge_k and edge_k in edge_info:
            e = edge_info[edge_k]
            rho = e.get("rho") or e.get("weight")
            direction = e.get("direction", "")
            if rho is not None:
                edge_meta = f" _(ρ={round(float(rho), 3)}{', ' + direction if direction else ''})_"
        lines.append(f"{prefix} `{node}`{edge_meta}")

    lines += [
        "",
        f"**Root causes identified:** {len(visited) - 1}",
        "_Follow edges upstream (→) to find the original driver._",
    ]
    return "\n".join(lines)


def _tool_compare_feature_periods(feature: str, con, table: str) -> str:
    """Compare first vs second half of a feature with t-test."""
    if con is None or not table:
        return "**Error:** No connection or table specified."
    try:
        from scipy.stats import ttest_ind, ks_2samp

        df = con.execute(f'SELECT "{feature}" FROM "{table}"').df()
        series = df[feature].dropna()
        if len(series) < 10:
            return f"**Error:** Only {len(series)} rows — insufficient for comparison."

        half   = len(series) // 2
        first  = series.iloc[:half].values
        second = series.iloc[half:].values

        t_stat, t_p   = ttest_ind(first, second, equal_var=False)
        ks_stat, ks_p = ks_2samp(first, second)
        shift = second.mean() - first.mean()
        pct   = (shift / abs(first.mean()) * 100) if first.mean() != 0 else float("inf")

        sig = "⚠️ **Statistically significant**" if t_p < 0.05 else "Not statistically significant"
        dist = "⚠️ **Distribution changed**" if ks_p < 0.05 else "Distribution stable"

        lines = [
            f"## Period Comparison — `{feature}`",
            "",
            f"| Period | n | Mean | Std | Min | Max |",
            f"| --- | --- | --- | --- | --- | --- |",
            f"| First half | {half} | `{first.mean():.4f}` | `{first.std():.4f}` | `{first.min():.4f}` | `{first.max():.4f}` |",
            f"| Second half | {len(series)-half} | `{second.mean():.4f}` | `{second.std():.4f}` | `{second.min():.4f}` | `{second.max():.4f}` |",
            "",
            f"**Welch t-test:** t=`{t_stat:.4f}`, p=`{t_p:.4f}` — {sig}",
            f"**KS test:** stat=`{ks_stat:.4f}`, p=`{ks_p:.4f}` — {dist}",
            f"**Mean shift:** `{shift:+.4f}` ({pct:+.1f}%)",
            "",
            ("**Interpretation:** The feature shows a significant change between periods, "
             "suggesting a regime shift or intervention effect."
             if t_p < 0.05 else
             "**Interpretation:** No significant change detected between periods."),
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"**Error comparing periods:** {exc}"


def _tool_generate_chart(chart_type: str, columns: list, title: str, con, table: str) -> Dict[str, Any]:
    """Generate a Plotly chart and return {title, html}."""
    if con is None or not table:
        return {"title": title, "html": "<p>No connection available.</p>"}
    try:
        import plotly.graph_objects as go
        import plotly.express as px

        df = con.execute(f'SELECT * FROM "{table}" LIMIT 10000').df()
        num_cols = df.select_dtypes("number").columns.tolist()
        ctype = chart_type.lower()

        if ctype == "scatter" and len(columns) >= 2:
            x_col = columns[0] if columns[0] in df.columns else num_cols[0]
            y_col = columns[1] if columns[1] in df.columns else num_cols[min(1, len(num_cols)-1)]
            fig = px.scatter(df, x=x_col, y=y_col, title=title,
                             trendline="ols", template="plotly_dark",
                             color_discrete_sequence=["#3B82F6"])
        elif ctype == "line" and columns:
            valid = [c for c in columns if c in df.columns and c in num_cols]
            if not valid:
                valid = num_cols[:2]
            fig = go.Figure()
            colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
            for i, c in enumerate(valid[:4]):
                fig.add_trace(go.Scatter(y=df[c], mode="lines", name=c,
                                         line=dict(color=colors[i % len(colors)])))
            fig.update_layout(title=title, template="plotly_dark")
        elif ctype == "bar" and columns:
            col = columns[0] if columns[0] in df.columns else df.columns[0]
            vc  = df[col].value_counts().head(20)
            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=title,
                         template="plotly_dark", color_discrete_sequence=["#3B82F6"])
        elif ctype == "heatmap":
            valid = [c for c in (columns or num_cols) if c in df.columns and c in num_cols][:8]
            corr  = df[valid].corr().round(3)
            fig = px.imshow(corr, title=title, template="plotly_dark",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            text_auto=True)
        elif ctype == "histogram" and columns:
            col = columns[0] if columns[0] in df.columns else num_cols[0]
            fig = px.histogram(df, x=col, title=title, template="plotly_dark",
                               color_discrete_sequence=["#3B82F6"], nbins=40)
        elif ctype == "box" and columns:
            valid = [c for c in columns if c in df.columns and c in num_cols]
            if not valid:
                valid = num_cols[:4]
            fig = go.Figure()
            for c in valid:
                fig.add_trace(go.Box(y=df[c], name=c, marker_color="#3B82F6"))
            fig.update_layout(title=title, template="plotly_dark")
        else:
            # Fallback: scatter matrix
            valid = [c for c in (columns or num_cols) if c in df.columns and c in num_cols][:4]
            if len(valid) >= 2:
                fig = px.scatter_matrix(df[valid], title=title, template="plotly_dark")
            else:
                fig = px.histogram(df, x=num_cols[0] if num_cols else df.columns[0],
                                   title=title, template="plotly_dark",
                                   color_discrete_sequence=["#3B82F6"])

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font_color   ="#E2E8F0",
            margin=dict(l=40, r=20, t=40, b=40),
            height=340,
        )
        html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
        return {"title": title, "html": html, "agent_generated": True}
    except Exception as exc:
        return {"title": title, "html": f"<p>Chart error: {exc}</p>"}


def _tool_explain_cause(feature: str, context: Dict) -> str:
    """Synthesize all statistical evidence for a feature being a root cause."""
    statistics = context.get("statistics", {})
    root_causes = context.get("root_causes", [])

    # Find this feature in root causes
    rc_entry = next((r for r in root_causes if r.get("column") == feature), None)

    spearman_data = (statistics.get("spearman") or {}).get(feature, {})
    partial_data  = (statistics.get("partial_correlations") or {}).get(feature, {})
    granger_data  = (statistics.get("granger") or {}).get(feature, {})
    mi_data       = statistics.get("mutual_information", {})
    mi_val        = (mi_data.get(feature) if isinstance(mi_data, dict) else None)
    dist_corr     = (statistics.get("distance_correlation") or {}).get(feature)

    lines = [f"## Root Cause Evidence for `{feature}`", ""]

    # Summary from root_causes list
    if rc_entry:
        influence = rc_entry.get("influence_score", rc_entry.get("influence", "N/A"))
        rank      = rc_entry.get("rank", "N/A")
        lines += [
            f"**Rank:** #{rank}  |  **Influence score:** `{round(float(influence), 4) if isinstance(influence, (int, float)) else influence}`",
            "",
        ]

    # Statistical evidence table
    lines += ["### Statistical Evidence", "", "| Method | Metric | Value | Interpretation |", "| --- | --- | --- | --- |"]

    if spearman_data:
        rho   = spearman_data.get("rho", "N/A")
        p_val = spearman_data.get("p_value", "N/A")
        sig   = "✅ Significant" if isinstance(p_val, float) and p_val < 0.05 else "❌ Not significant"
        lines.append(f"| Spearman ρ | Rank correlation | `{rho}` (p=`{p_val}`) | {sig} |")

    if partial_data:
        p_rho = partial_data.get("partial_rho", partial_data.get("rho", "N/A"))
        lines.append(f"| Partial Corr | After controlling others | `{p_rho}` | Controls for confounders |")

    if mi_val is not None:
        lines.append(f"| Mutual Information | Non-linear dependence | `{round(float(mi_val), 4)}` | Higher = stronger link |")

    if dist_corr is not None:
        lines.append(f"| Distance Corr | Catches non-linear/non-monotonic | `{round(float(dist_corr), 4)}` | 0=independent, 1=perfect |")

    if granger_data:
        best_p = min(
            (v.get("p_value", 1.0) for v in granger_data.values() if isinstance(v, dict)),
            default=None
        )
        if best_p is not None:
            lines.append(f"| Granger Causality | Temporal precedence | best p=`{round(best_p, 4)}` | {'✅ Granger-causes target' if best_p < 0.05 else '❌ No temporal precedence'} |")

    lines += [
        "",
        "### Interpretation",
        "",
    ]

    evidences = []
    if spearman_data:
        rho = spearman_data.get("rho", 0)
        if isinstance(rho, (int, float)) and abs(rho) > 0.3:
            direction = "positively" if rho > 0 else "negatively"
            evidences.append(
                f"`{feature}` is {direction} correlated with the target "
                f"($\\rho = {rho}$), explaining {round(rho**2*100, 1)}% of rank variance."
            )
    if mi_val is not None and isinstance(mi_val, (int, float)) and mi_val > 0.1:
        evidences.append(f"Mutual information of `{round(mi_val, 3)}` bits indicates non-trivial shared information.")
    if granger_data:
        best_p = min(
            (v.get("p_value", 1.0) for v in granger_data.values() if isinstance(v, dict)),
            default=1.0
        )
        if best_p < 0.05:
            evidences.append(
                f"Granger causality (p={round(best_p, 4)}) suggests `{feature}` **temporally precedes** "
                f"changes in the target — consistent with a causal role."
            )

    if not evidences:
        lines.append(
            "_Insufficient statistical data in context. Run RCA first or use `run_sql` "
            "to query the raw data for manual investigation._"
        )
    else:
        for e in evidences:
            lines.append(f"- {e}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph / ReAct agent entry point
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SENTINEL's Root Cause Analysis Expert — a world-class statistician and causal inference researcher embedded inside the RCA dashboard.

## Rules (non-negotiable)
1. **Always respond in GitHub-flavored Markdown.** Use headers, tables, bold text, code blocks.
2. **Never invent statistics.** Only cite numbers returned by tool calls or explicitly in the context. If you don't know, call a tool.
3. **LaTeX:** Use `$...$` for inline formulas, e.g. $\\rho_{xy} = \\frac{\\text{cov}(R_x, R_y)}{\\sigma_{R_x}\\sigma_{R_y}}$.
4. **Max 3 tool calls per response.** Use the most impactful tools first.
5. **If you generate a chart:** end your response with "I've added **[chart name]** to your dashboard above."
6. **Cite tool names** when referencing their outputs (e.g., "According to `explain_cause`…").
7. **Distinguish correlation from causation** — use precise language.

## Causal reasoning framework
- **Spearman ρ** measures monotonic association — necessary but not sufficient for causation.
- **Granger causality**: if X Granger-causes Y (p < 0.05), X's past values predict Y's future values — evidence of temporal precedence.
- **Partial correlation**: controls for all other features — isolates the direct relationship.
- **Distance correlation**: detects non-linear, non-monotonic dependence; dcor > 0.3 is noteworthy.
- **Mutual information**: information-theoretic dependence — captures any statistical relationship.
- **PC algorithm**: constraint-based causal skeleton — edges represent conditional independence violations.

## Available Tools
- `run_sql` — execute SQL SELECT against the live dataset
- `traverse_feature` — BFS upstream causal chain from a feature
- `compare_feature_periods` — compare first vs second half, t-test + KS
- `generate_chart` — create scatter, line, bar, heatmap, histogram, box charts
- `explain_cause` — synthesize all statistical evidence for a root cause feature
"""


def run_rca_agent(
    message:      str,
    context:      Dict[str, Any],
    table:        str,
    chat_history: List[Dict],
    con,
    call_llm,
) -> Dict[str, Any]:
    """
    Run the RCA chat agent.
    Returns: {response: str, charts: list, tool_calls_made: list}
    """
    generated_charts: List[Dict] = []
    tool_calls_made:  List[str]  = []

    try:
        return _run_langgraph_agent(
            message, context, table, chat_history, con, call_llm,
            generated_charts, tool_calls_made
        )
    except ImportError:
        logger.info("LangGraph not available; using manual ReAct loop.")
    except Exception as exc:
        logger.warning("LangGraph RCA agent error: %s", exc)

    return _run_manual_react(
        message, context, table, chat_history, con, call_llm,
        generated_charts, tool_calls_made
    )


def _dispatch_tool(name: str, args: Dict, con, table: str, context: Dict) -> Any:
    if name == "run_sql":
        return _tool_run_sql(args.get("sql", ""), con, table)
    elif name == "traverse_feature":
        return _tool_traverse_feature(args.get("feature", ""), context)
    elif name == "compare_feature_periods":
        return _tool_compare_feature_periods(args.get("feature", ""), con, table)
    elif name == "generate_chart":
        return _tool_generate_chart(
            args.get("chart_type", "scatter"),
            args.get("columns", []),
            args.get("title", "Chart"),
            con, table
        )
    elif name == "explain_cause":
        return _tool_explain_cause(args.get("feature", ""), context)
    else:
        return f"**Error:** Unknown tool '{name}'."


def _run_langgraph_agent(
    message, context, table, chat_history, con, call_llm,
    generated_charts, tool_calls_made
) -> Dict[str, Any]:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.tools import tool as lc_tool
    from typing import TypedDict, Annotated, Sequence
    import operator

    @lc_tool
    def run_sql(sql: str) -> str:
        """Execute a SQL SELECT query against the dataset."""
        tool_calls_made.append("run_sql")
        return _tool_run_sql(sql, con, table)

    @lc_tool
    def traverse_feature(feature: str) -> str:
        """BFS upstream causal chain from a feature in the causal graph."""
        tool_calls_made.append("traverse_feature")
        return _tool_traverse_feature(feature, context)

    @lc_tool
    def compare_feature_periods(feature: str) -> str:
        """Compare first vs second half of a feature with t-test and KS test."""
        tool_calls_made.append("compare_feature_periods")
        return _tool_compare_feature_periods(feature, con, table)

    @lc_tool
    def generate_chart(chart_type: str, columns: list, title: str) -> str:
        """Generate a chart. chart_type: scatter|line|bar|heatmap|histogram|box"""
        tool_calls_made.append("generate_chart")
        result = _tool_generate_chart(chart_type, columns, title, con, table)
        generated_charts.append(result)
        return f"Chart '{title}' generated and added to dashboard."

    @lc_tool
    def explain_cause(feature: str) -> str:
        """Synthesize all statistical evidence (Spearman, partial corr, Granger, MI) for a feature."""
        tool_calls_made.append("explain_cause")
        return _tool_explain_cause(feature, context)

    tools = [run_sql, traverse_feature, compare_feature_periods, generate_chart, explain_cause]

    class AgentState(TypedDict):
        messages: Annotated[Sequence, operator.add]
        iterations: int

    from langchain_anthropic import ChatAnthropic
    import os

    try:
        from backend.core.namespace import sentinel_ns
        api_key = sentinel_ns._ns.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
        max_tokens=2048,
        temperature=0.1,
    ).bind_tools(tools)

    def agent_node(state: AgentState):
        if state["iterations"] >= 4:
            return {"messages": [], "iterations": state["iterations"]}
        response = llm.invoke(state["messages"])
        return {"messages": [response], "iterations": state["iterations"] + 1}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if state["iterations"] >= 4:
            return END
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    app = graph.compile()

    init_messages = [SystemMessage(content=SYSTEM_PROMPT + _context_to_str(context))]
    for m in (chat_history or [])[-6:]:
        role = m.get("role", "user")
        text = m.get("text", m.get("content", ""))
        if role == "user":
            init_messages.append(HumanMessage(content=text))
        else:
            init_messages.append(AIMessage(content=text))
    init_messages.append(HumanMessage(content=message))

    result = app.invoke({"messages": init_messages, "iterations": 0})
    final  = result["messages"][-1]
    response_text = final.content if hasattr(final, "content") else str(final)
    if isinstance(response_text, list):
        response_text = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in response_text
        )

    return {
        "response":        response_text,
        "charts":          generated_charts,
        "tool_calls_made": tool_calls_made,
    }


def _run_manual_react(
    message, context, table, chat_history, con, call_llm,
    generated_charts, tool_calls_made
) -> Dict[str, Any]:
    import re as _re

    if call_llm is None:
        return {
            "response": "**Error:** LLM not configured. Please set your API key.",
            "charts":   [],
            "tool_calls_made": [],
        }

    tools_desc = """
Available tools (call by writing JSON in a ```tool``` block):
- run_sql: {"tool": "run_sql", "sql": "SELECT ..."}
- traverse_feature: {"tool": "traverse_feature", "feature": "col_name"}
- compare_feature_periods: {"tool": "compare_feature_periods", "feature": "col_name"}
- generate_chart: {"tool": "generate_chart", "chart_type": "scatter|line|bar|heatmap|histogram|box", "columns": ["col1","col2"], "title": "Title"}
- explain_cause: {"tool": "explain_cause", "feature": "col_name"}

Call at most 3 tools. Provide tool calls in ```tool``` blocks. After results, give your Markdown analysis.
"""

    history_str = ""
    for m in (chat_history or [])[-4:]:
        role = m.get("role", "user")
        text = m.get("text", m.get("content", ""))
        history_str += f"\n**{role.upper()}:** {text}"

    conversation = (
        SYSTEM_PROMPT
        + _context_to_str(context)
        + tools_desc
        + history_str
        + f"\n\n**USER:** {message}\n\nThink step by step and use tools as needed."
    )

    tool_call_count = 0
    for _ in range(4):
        raw = call_llm(conversation, temperature=0.1)
        tool_blocks = _re.findall(r'```tool\s*\n(.*?)\n```', raw, _re.DOTALL)
        if not tool_blocks or tool_call_count >= 3:
            final = _re.sub(r'```tool\s*\n.*?\n```', '', raw, flags=_re.DOTALL).strip()
            return {"response": final, "charts": generated_charts, "tool_calls_made": tool_calls_made}

        tool_results = []
        for block in tool_blocks[:3]:
            if tool_call_count >= 3:
                break
            try:
                spec = json.loads(block.strip())
            except json.JSONDecodeError:
                tool_results.append("**Tool parse error:** Invalid JSON.")
                continue

            name = spec.pop("tool", "")
            tool_calls_made.append(name)
            tool_call_count += 1
            result = _dispatch_tool(name, spec, con, table, context)

            if isinstance(result, dict) and result.get("agent_generated"):
                generated_charts.append(result)
                tool_results.append(f"Chart '{result['title']}' generated and added to dashboard.")
            else:
                tool_results.append(f"**Tool `{name}` result:**\n{result}")

        results_str = "\n\n".join(tool_results)
        conversation += f"\n\n{raw}\n\n**Tool Results:**\n{results_str}\n\nNow provide your final Markdown analysis."

    final_raw = call_llm(conversation + "\n\nProvide your final Markdown answer.", temperature=0.1)
    return {"response": final_raw, "charts": generated_charts, "tool_calls_made": tool_calls_made}


def _context_to_str(context: Dict) -> str:
    if not context:
        return ""
    import json as _json
    target = context.get("target_col", "unknown")
    table  = context.get("table", "unknown")
    causes = context.get("root_causes", [])[:5]
    stats  = context.get("statistics", {})

    stat_summary = {}
    sp = stats.get("spearman", {})
    if sp:
        top3 = sorted(sp.items(), key=lambda kv: abs(kv[1].get("rho", 0)), reverse=True)[:3]
        stat_summary["top_spearman"] = {k: v.get("rho") for k, v in top3}

    return (
        f"\n\n---\n**ACTIVE RCA CONTEXT:**\n"
        f"- Table: `{table}`\n"
        f"- Target column: `{target}`\n"
        f"- Top root causes: {_json.dumps(causes)}\n"
        f"- Statistical summary: {_json.dumps(stat_summary)}\n---\n"
    )
