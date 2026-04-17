"""
Anomaly Chat Agent — LangGraph ReAct agent for the Anomaly Detection dashboard.

Tools:
  1. run_sql(sql)                   — execute SQL against DuckDB, return markdown table
  2. compute_stat(column, stat)     — descriptive statistics for a column
  3. explain_anomaly(row_index, column) — deep row-level explanation
  4. generate_chart(chart_type, columns, title) — creates a Plotly chart, returns HTML
  5. compare_distributions(column)  — KS test first-half vs second-half

Graph: START → agent_node ↔ tool_node → END (max 4 iterations)

System prompt enforces:
  • Strict GitHub-flavored Markdown
  • Never invent statistics — only cite tool outputs or provided context
  • LaTeX via $...$ inline
  • Max 3 tool calls per response
  • If chart generated: tell user "I've added [name] to your dashboard above"
"""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations (pure Python, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _tool_run_sql(sql: str, con, table: str) -> str:
    """Execute SQL against DuckDB; return markdown table (50-row cap)."""
    if con is None:
        return "**Error:** No database connection available."
    # Safety: only allow SELECT statements
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        return "**Error:** Only SELECT queries are permitted."
    try:
        df = con.execute(sql).df()
        if df.empty:
            return "_No rows returned._"
        preview = df.head(50)
        # Build markdown table
        cols = list(preview.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        rows   = []
        for _, row in preview.iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        note = f"\n\n_Showing {len(preview)} of {len(df)} rows._" if len(df) > 50 else ""
        return "\n".join([header, sep] + rows) + note
    except Exception as exc:
        return f"**SQL Error:** {exc}"


def _tool_compute_stat(column: str, stat: str, con, table: str) -> str:
    """Compute a statistic for a column."""
    if con is None or not table:
        return "**Error:** No connection or table specified."
    try:
        stat_lower = stat.lower()
        if stat_lower == "describe":
            df = con.execute(f'SELECT "{column}" FROM "{table}"').df()
            desc = df[column].describe().round(4)
            lines = ["| Stat | Value |", "| --- | --- |"]
            for k, v in desc.items():
                lines.append(f"| {k} | {v} |")
            return "\n".join(lines)
        elif stat_lower in ("mean", "avg"):
            val = con.execute(f'SELECT AVG("{column}") FROM "{table}"').fetchone()[0]
            return f"**Mean of `{column}`:** {round(val, 4) if val is not None else 'N/A'}"
        elif stat_lower == "std":
            val = con.execute(f'SELECT STDDEV("{column}") FROM "{table}"').fetchone()[0]
            return f"**Std dev of `{column}`:** {round(val, 4) if val is not None else 'N/A'}"
        elif stat_lower in ("min", "max"):
            val = con.execute(f'SELECT {stat_lower.upper()}("{column}") FROM "{table}"').fetchone()[0]
            return f"**{stat_lower.title()} of `{column}`:** {val}"
        elif stat_lower.startswith("p"):
            # e.g. p95 → 95th percentile
            try:
                pct = float(stat_lower[1:]) / 100.0
            except ValueError:
                return f"**Error:** Unrecognised stat '{stat}'."
            val = con.execute(
                f'SELECT PERCENTILE_CONT({pct}) WITHIN GROUP (ORDER BY "{column}") FROM "{table}"'
            ).fetchone()[0]
            return f"**P{int(pct*100)} of `{column}`:** {round(val, 4) if val is not None else 'N/A'}"
        elif stat_lower == "correlation":
            # correlation with all other numeric cols
            df = con.execute(f'SELECT * FROM "{table}" LIMIT 5000').df()
            num_cols = df.select_dtypes("number").columns.tolist()
            if column not in num_cols:
                return f"**Error:** `{column}` is not numeric."
            corr = df[num_cols].corr()[column].drop(column).round(4).sort_values(key=abs, ascending=False)
            lines = [f"**Correlations with `{column}`:**", "| Feature | ρ |", "| --- | --- |"]
            for feat, r in corr.items():
                lines.append(f"| {feat} | {r} |")
            return "\n".join(lines)
        else:
            return f"**Error:** Unknown stat '{stat}'. Supported: mean, std, min, max, pNN (percentile), describe, correlation."
    except Exception as exc:
        return f"**Error computing stat:** {exc}"


def _tool_explain_anomaly(row_index: int, column: str, con, table: str) -> str:
    """Explain a specific anomaly row in context of its neighbours."""
    if con is None or not table:
        return "**Error:** No connection or table specified."
    try:
        df = con.execute(f'SELECT * FROM "{table}" LIMIT 50000').df()
        if column not in df.columns:
            return f"**Error:** Column `{column}` not found in table `{table}`."
        if row_index < 0 or row_index >= len(df):
            return f"**Error:** Row index {row_index} out of range (table has {len(df)} rows)."

        series = df[column].dropna()
        value  = df.loc[row_index, column]

        # Rolling baseline (window=20)
        WINDOW = min(20, max(5, len(series) // 10))
        rm = series.rolling(WINDOW, min_periods=2).mean()
        rs = series.rolling(WINDOW, min_periods=2).std().fillna(1e-9)

        baseline = rm.iloc[row_index] if row_index < len(rm) else series.mean()
        std_val  = rs.iloc[row_index] if row_index < len(rs) else series.std()
        z_score  = abs((value - baseline) / max(std_val, 1e-9))

        # IQR
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr    = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        severity = (
            "**CRITICAL** (z > 4)" if z_score > 4
            else "**HIGH** (z 3–4)"   if z_score > 3
            else "**MEDIUM** (z 2–3)" if z_score > 2
            else "**LOW** (z ≤ 2)"
        )

        # Neighbours
        start = max(0, row_index - 3)
        end   = min(len(df), row_index + 4)
        neighbours = df[column].iloc[start:end].round(4).tolist()

        lines = [
            f"## Anomaly Explanation — Row {row_index}, `{column}`",
            "",
            f"| Property | Value |",
            f"| --- | --- |",
            f"| Observed value | `{round(float(value), 4)}` |",
            f"| Rolling baseline (w={WINDOW}) | `{round(float(baseline), 4)}` |",
            f"| Rolling std | `{round(float(std_val), 4)}` |",
            f"| Z-score | `{round(z_score, 2)}` |",
            f"| Severity | {severity} |",
            f"| IQR fence | `[{round(float(lower_fence),4)}, {round(float(upper_fence),4)}]` |",
            f"| Outside IQR fence | `{'Yes' if value < lower_fence or value > upper_fence else 'No'}` |",
            "",
            f"**Neighbourhood (rows {start}–{end-1}):** `{neighbours}`",
            "",
            "**Interpretation:** "
            + (f"The value deviates {round(z_score,2)}σ from its local baseline. "
               + ("This is likely a systemic event." if z_score > 4
                  else "This may be a transient spike — review adjacent rows for corroborating signals.")),
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"**Error explaining anomaly:** {exc}"


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

        if ctype == "histogram" and columns:
            col = columns[0] if columns[0] in df.columns else num_cols[0]
            fig = px.histogram(df, x=col, title=title,
                               color_discrete_sequence=["#3B82F6"],
                               template="plotly_dark")
        elif ctype == "scatter" and len(columns) >= 2:
            x_col = columns[0] if columns[0] in df.columns else num_cols[0]
            y_col = columns[1] if columns[1] in df.columns else num_cols[min(1, len(num_cols)-1)]
            fig = px.scatter(df, x=x_col, y=y_col, title=title,
                             template="plotly_dark",
                             color_discrete_sequence=["#3B82F6"])
        elif ctype in ("line", "timeseries") and columns:
            col = columns[0] if columns[0] in df.columns else num_cols[0]
            fig = px.line(df, y=col, title=title,
                          template="plotly_dark",
                          color_discrete_sequence=["#3B82F6"])
        elif ctype == "box" and columns:
            valid = [c for c in columns if c in df.columns and c in num_cols]
            if not valid:
                valid = num_cols[:4]
            fig = go.Figure()
            for c in valid:
                fig.add_trace(go.Box(y=df[c], name=c, marker_color="#3B82F6"))
            fig.update_layout(title=title, template="plotly_dark")
        elif ctype == "heatmap":
            valid = [c for c in (columns or num_cols) if c in df.columns and c in num_cols][:8]
            corr  = df[valid].corr().round(3)
            fig = px.imshow(corr, title=title, template="plotly_dark",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        else:
            # Fallback: bar chart of value counts for first column
            col = columns[0] if columns and columns[0] in df.columns else (num_cols[0] if num_cols else df.columns[0])
            vc  = df[col].value_counts().head(20)
            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=title,
                         template="plotly_dark",
                         color_discrete_sequence=["#3B82F6"])
            fig.update_xaxes(title=col)
            fig.update_yaxes(title="Count")

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


def _tool_compare_distributions(column: str, con, table: str) -> str:
    """KS test comparing first vs second half of a column."""
    if con is None or not table:
        return "**Error:** No connection or table specified."
    try:
        from scipy.stats import ks_2samp, ttest_ind

        df = con.execute(f'SELECT "{column}" FROM "{table}"').df()
        series = df[column].dropna()
        if len(series) < 10:
            return f"**Error:** Not enough data in `{column}` (only {len(series)} rows)."

        half   = len(series) // 2
        first  = series.iloc[:half].values
        second = series.iloc[half:].values

        ks_stat, ks_p = ks_2samp(first, second)
        t_stat,  t_p  = ttest_ind(first, second, equal_var=False)

        mean1, mean2 = first.mean(), second.mean()
        std1,  std2  = first.std(),  second.std()
        shift  = mean2 - mean1
        pct    = (shift / abs(mean1) * 100) if mean1 != 0 else float("inf")

        distribution_changed = ks_p < 0.05

        lines = [
            f"## Distribution Comparison — `{column}`",
            "",
            f"| Metric | First Half (n={half}) | Second Half (n={len(series)-half}) |",
            "| --- | --- | --- |",
            f"| Mean | `{round(mean1,4)}` | `{round(mean2,4)}` |",
            f"| Std | `{round(std1,4)}` | `{round(std2,4)}` |",
            "",
            f"**KS Test:** statistic=`{round(ks_stat,4)}`, p=`{round(ks_p,4)}` "
            + ("— ⚠️ **Distribution shift detected** (p < 0.05)" if distribution_changed
               else "— distributions are statistically similar"),
            "",
            f"**Welch t-test:** t=`{round(t_stat,4)}`, p=`{round(t_p,4)}` "
            + ("— ⚠️ **Means significantly differ**" if t_p < 0.05 else "— means are not significantly different"),
            "",
            f"**Mean shift:** `{round(shift,4)}` ({round(pct,1)}%)",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"**Error comparing distributions:** {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SENTINEL's Anomaly Analysis Expert — a world-class data scientist embedded inside the SENTINEL anomaly detection dashboard.

## Rules (non-negotiable)
1. **Always respond in GitHub-flavored Markdown.** Use headers, tables, bold text, code blocks, lists.
2. **Never invent statistics.** Only cite numbers returned by tool calls or explicitly provided in the context. If you don't know, call a tool.
3. **LaTeX:** Use `$...$` for inline formulas (e.g. $z = \\frac{x - \\mu}{\\sigma}$) and `$$...$$` for block formulas.
4. **Max 3 tool calls per response.** Prioritize the most impactful tools.
5. **If you generate a chart:** end your response with "I've added **[chart name]** to your dashboard above."
6. **Cite tool outputs:** When referencing numbers, include the tool that produced them.
7. **Be concise but precise.** Avoid filler text. Get to the insight.

## What you know
- You have access to the anomaly detection results (provided in context).
- You can query the raw data, compute statistics, explain individual anomalies, generate charts, and compare distributions.
- Z-score formula: $z = \\frac{x - \\bar{x}_{\\text{rolling}}}{{\\sigma}_{\\text{rolling}}}$
- Ensemble score: 40% statistical + 35% ML + 25% time-series

## Available Tools
- `run_sql` — execute SQL SELECT against the live dataset
- `compute_stat` — get mean/std/min/max/percentile/describe/correlation for a column
- `explain_anomaly` — deep explanation of a specific row and column
- `generate_chart` — create histogram, scatter, line, box, or heatmap
- `compare_distributions` — KS test + t-test comparing first vs second half of a column
"""

def run_anomaly_agent(
    message:      str,
    context:      Dict[str, Any],
    table:        str,
    chat_history: List[Dict],
    con,
    call_llm,
) -> Dict[str, Any]:
    """
    Run the anomaly chat agent.
    Tries LangGraph first; falls back to manual ReAct loop with the call_llm function.
    Returns: {response: str, charts: list, tool_calls_made: list}
    """
    generated_charts: List[Dict] = []
    tool_calls_made:  List[str]  = []

    # ── Try LangGraph ─────────────────────────────────────────────────────────
    try:
        return _run_langgraph_agent(
            message, context, table, chat_history, con, call_llm,
            generated_charts, tool_calls_made
        )
    except ImportError:
        logger.info("LangGraph not available; using manual ReAct loop.")
    except Exception as exc:
        logger.warning("LangGraph agent error: %s", exc)

    # ── Manual ReAct fallback ─────────────────────────────────────────────────
    return _run_manual_react(
        message, context, table, chat_history, con, call_llm,
        generated_charts, tool_calls_made
    )


def _dispatch_tool(name: str, args: Dict, con, table: str) -> Any:
    """Dispatch a tool call and return its output."""
    if name == "run_sql":
        return _tool_run_sql(args.get("sql", ""), con, table)
    elif name == "compute_stat":
        return _tool_compute_stat(args.get("column", ""), args.get("stat", "describe"), con, table)
    elif name == "explain_anomaly":
        return _tool_explain_anomaly(int(args.get("row_index", 0)), args.get("column", ""), con, table)
    elif name == "generate_chart":
        result = _tool_generate_chart(
            args.get("chart_type", "histogram"),
            args.get("columns", []),
            args.get("title", "Chart"),
            con, table
        )
        return result  # dict
    elif name == "compare_distributions":
        return _tool_compare_distributions(args.get("column", ""), con, table)
    else:
        return f"**Error:** Unknown tool '{name}'."


def _run_langgraph_agent(
    message, context, table, chat_history, con, call_llm,
    generated_charts, tool_calls_made
) -> Dict[str, Any]:
    """LangGraph-based ReAct agent execution."""
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool as lc_tool
    from typing import TypedDict

    # Define tools as LangChain tool objects
    @lc_tool
    def run_sql(sql: str) -> str:
        """Execute a SQL SELECT query against the dataset. Returns markdown table (50 rows max)."""
        tool_calls_made.append("run_sql")
        return _tool_run_sql(sql, con, table)

    @lc_tool
    def compute_stat(column: str, stat: str) -> str:
        """Compute a statistic for a column. stat options: mean, std, min, max, pNN (percentile), describe, correlation."""
        tool_calls_made.append("compute_stat")
        return _tool_compute_stat(column, stat, con, table)

    @lc_tool
    def explain_anomaly(row_index: int, column: str) -> str:
        """Explain a specific anomaly at the given row_index and column."""
        tool_calls_made.append("explain_anomaly")
        return _tool_explain_anomaly(row_index, column, con, table)

    @lc_tool
    def generate_chart(chart_type: str, columns: list, title: str) -> str:
        """Generate a chart. chart_type: histogram|scatter|line|box|heatmap. Returns confirmation."""
        tool_calls_made.append("generate_chart")
        result = _tool_generate_chart(chart_type, columns, title, con, table)
        generated_charts.append(result)
        return f"Chart '{title}' generated successfully and added to dashboard."

    @lc_tool
    def compare_distributions(column: str) -> str:
        """Compare first-half vs second-half distribution of a column using KS test."""
        tool_calls_made.append("compare_distributions")
        return _tool_compare_distributions(column, con, table)

    tools = [run_sql, compute_stat, explain_anomaly, generate_chart, compare_distributions]

    # Build LangGraph
    class AgentState(TypedDict):
        messages: Annotated[Sequence, operator.add]
        iterations: int

    # Use call_llm as a callable LLM wrapper
    # Wrap it to behave like a LangChain chat model
    from langchain_anthropic import ChatAnthropic
    import os

    api_key = None
    ns_api_key = None
    try:
        from backend.core.namespace import sentinel_ns
        ns_api_key = sentinel_ns._ns.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:
        ns_api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=ns_api_key,
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

    # Build initial messages
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

    # Extract final text response
    final = result["messages"][-1]
    response_text = final.content if hasattr(final, "content") else str(final)
    if isinstance(response_text, list):
        response_text = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in response_text
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
    """Manual ReAct loop using call_llm (string in, string out)."""
    import re as _re

    if call_llm is None:
        return {
            "response": "**Error:** LLM not configured. Please set your API key.",
            "charts":   [],
            "tool_calls_made": [],
        }

    ctx_str = _context_to_str(context)

    # History
    history_str = ""
    for m in (chat_history or [])[-4:]:
        role = m.get("role", "user")
        text = m.get("text", m.get("content", ""))
        history_str += f"\n**{role.upper()}:** {text}"

    # Available tools description
    tools_desc = """
Available tools (call by writing JSON in a ```tool``` block):
- run_sql: {"tool": "run_sql", "sql": "SELECT ..."}
- compute_stat: {"tool": "compute_stat", "column": "col_name", "stat": "describe|mean|std|pNN|correlation"}
- explain_anomaly: {"tool": "explain_anomaly", "row_index": 0, "column": "col_name"}
- generate_chart: {"tool": "generate_chart", "chart_type": "histogram|scatter|line|box|heatmap", "columns": ["col1"], "title": "Title"}
- compare_distributions: {"tool": "compare_distributions", "column": "col_name"}

Call at most 3 tools. After each tool call I will provide the result. Then write your final answer.
"""

    system = (
        SYSTEM_PROMPT
        + ctx_str
        + tools_desc
        + history_str
        + f"\n\n**USER:** {message}\n\nThink step by step and use tools as needed."
    )

    tool_call_count = 0
    conversation    = system

    for _ in range(4):
        raw = call_llm(conversation, temperature=0.1)

        # Parse tool calls from ```tool ... ``` blocks
        tool_blocks = _re.findall(r'```tool\s*\n(.*?)\n```', raw, _re.DOTALL)
        if not tool_blocks or tool_call_count >= 3:
            # Final answer — strip any remaining tool blocks
            final = _re.sub(r'```tool\s*\n.*?\n```', '', raw, flags=_re.DOTALL).strip()
            return {
                "response":        final,
                "charts":          generated_charts,
                "tool_calls_made": tool_calls_made,
            }

        tool_results = []
        for block in tool_blocks[:3]:
            if tool_call_count >= 3:
                break
            try:
                spec = json.loads(block.strip())
            except json.JSONDecodeError:
                tool_results.append("**Tool parse error:** Invalid JSON in tool block.")
                continue

            name = spec.pop("tool", "")
            tool_calls_made.append(name)
            tool_call_count += 1

            result = _dispatch_tool(name, spec, con, table)

            if isinstance(result, dict) and result.get("agent_generated"):
                generated_charts.append(result)
                tool_results.append(f"Chart '{result['title']}' generated and added to dashboard.")
            else:
                tool_results.append(f"**Tool `{name}` result:**\n{result}")

        results_str = "\n\n".join(tool_results)
        conversation += f"\n\n{raw}\n\n**Tool Results:**\n{results_str}\n\nNow provide your final analysis in Markdown."

    # Fallback final answer
    final_raw = call_llm(conversation + "\n\nProvide your final Markdown answer now.", temperature=0.1)
    return {
        "response":        final_raw,
        "charts":          generated_charts,
        "tool_calls_made": tool_calls_made,
    }


def _context_to_str(context: Dict) -> str:
    if not context:
        return ""
    import json as _json
    stats   = context.get("stats", {})
    profile = context.get("profile", {})
    table   = context.get("table", "unknown")
    top_a   = context.get("anomalies", [])[:5]
    return (
        f"\n\n---\n**ACTIVE ANOMALY SCAN CONTEXT:**\n"
        f"- Table: `{table}`\n"
        f"- Stats: {_json.dumps(stats)}\n"
        f"- Data profile: {_json.dumps(profile)}\n"
        f"- Top 5 anomalies (by ensemble score): {_json.dumps(top_a)}\n---\n"
    )
