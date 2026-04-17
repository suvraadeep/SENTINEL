import uuid
import json
from typing import List, Tuple, Optional
import plotly.graph_objects as go

_chart_counter = 0
_FIG_QUEUE: List[Tuple[str, str]] = []


def safe_show(fig: go.Figure, title_suffix: str = "") -> Optional[str]:
    global _chart_counter, _FIG_QUEUE
    _chart_counter += 1
    try:
        fig.update_layout(margin=dict(l=55, r=55, t=65, b=55), font=dict(size=12))
        # Store HTML string directly no file, no IFrame, no 404
        html_str = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,        # just the <div>, not a full page
            config={"responsive": True}
        )
        _FIG_QUEUE.append((html_str, title_suffix))
        print(f"  ✓ Queued [{_chart_counter}]: {title_suffix}")
        return title_suffix
    except Exception as e:
        print(f"  ✗ Queue failed ({title_suffix}): {e}")
        return None


def flush_charts() -> int:
    global _FIG_QUEUE
    n = 0
    if not _FIG_QUEUE:
        return 0
    print(f"\n{'─'*60}")
    print(f"📊 Rendering {len(_FIG_QUEUE)} chart(s)")
    print(f"{'─'*60}")
    for html_str, title in _FIG_QUEUE:
        try:
            if title:
                display(HTML(
                    f"<p style='font-size:13px;font-weight:600;"
                    f"margin:12px 0 2px;color:#333'>📊 {title}</p>"
                ))
            # Embed chart HTML inline — works in Kaggle, Colab, JupyterLab
            display(HTML(
                f"<div style='width:100%;min-height:480px'>{html_str}</div>"
            ))
            n += 1
        except Exception as e:
            print(f"  ✗ Display failed ({title}): {e}")
    _FIG_QUEUE.clear()
    return n


def safe_vline(fig: go.Figure, x_val, label: str = "",
               color: str = "red", dash: str = "dash") -> None:
    x_str = (x_val.strftime("%Y-%m-%d %H:%M:%S")
             if hasattr(x_val, "strftime") else str(x_val))
    fig.add_shape(
        type="line", x0=x_str, x1=x_str, y0=0, y1=1,
        yref="paper", xref="x",
        line=dict(color=color, dash=dash, width=1.5)
    )
    if label:
        fig.add_annotation(
            x=x_str, y=1.02, yref="paper", xref="x",
            text=label, showarrow=False,
            font=dict(color=color, size=11),
            bgcolor="white", bordercolor=color, borderwidth=1, borderpad=3
        )


def safe_hline(fig: go.Figure, y_val: float, label: str = "",
               color: str = "gray", dash: str = "dot") -> None:
    fig.add_shape(
        type="line", x0=0, x1=1, y0=y_val, y1=y_val,
        xref="paper", yref="y",
        line=dict(color=color, dash=dash, width=1.2)
    )
    if label:
        fig.add_annotation(
            x=1.01, y=y_val, xref="paper", yref="y",
            text=label, showarrow=False,
            font=dict(color=color, size=10), xanchor="left"
        )


def extract_json(text: str, fallback=None):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in [r'\{[^{}]*\}', r'\[.*?\]', r'\{.*\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                continue
    return fallback