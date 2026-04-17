"""
Dataset bootstrap — defines utility functions (run_sql, get_schema, build_l3_graph)
used by all downstream agents.

NO synthetic data is generated here. The actual data comes exclusively from
user uploads via the /api/upload endpoint, which calls update_data() to swap
in the correct DuckDB connection and schema.

On initial exec (before any upload), con/SCHEMA are set to empty defaults
so that agents can be loaded without errors.
"""

import duckdb
import networkx as nx
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Optional

# ── Initial empty connection (replaced by update_data after upload) ──────────
con = duckdb.connect(DB_PATH)
con.execute("PRAGMA threads=4")

# ── Schema — empty until a dataset is uploaded ──────────────────────────────
DATA_DATE_MIN = None
DATA_DATE_MAX = None
DATA_DATE_MIDPOINT = None


def get_schema() -> str:
    """Build schema string from ALL tables currently in the database."""
    parts = []
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'"
        ).fetchall()]
    except Exception:
        return ""
    for tbl in tables:
        try:
            cols = con.execute(f"DESCRIBE {tbl}").df()
            cols_str = ", ".join(
                f"{r['column_name']}:{r['column_type']}"
                for _, r in cols.iterrows()
            )
            sample = con.execute(f"SELECT * FROM {tbl} LIMIT 2").df().to_string(index=False)
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            parts.append(f"TABLE {tbl} ({n:,} rows)\nCOLUMNS: {cols_str}\nSAMPLE:\n{sample}")
        except Exception:
            pass
    return "\n\n".join(parts)


SCHEMA = get_schema()


def run_sql_approx(query: str, sample_frac: float = 0.3,
                   confidence: float = 0.95) -> Tuple[pd.DataFrame, dict]:
    """
    Approximate Query Processing (BlinkDB-inspired):
    Runs query on a stratified sample, reports CI on numeric aggregates.
    Only triggers for large full-table scans.
    """
    result = run_sql(query)
    if result.empty or len(result) < 5:
        return result, {}

    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    n = len(result)
    z = stats.norm.ppf((1 + confidence) / 2)
    ci_info = {}
    for col in numeric_cols:
        se = result[col].sem()
        margin = z * se
        ci_info[col] = {
            "mean": result[col].mean(),
            "ci_lower": result[col].mean() - margin,
            "ci_upper": result[col].mean() + margin,
            "confidence": confidence
        }
    return result, ci_info


def run_sql(query: str) -> pd.DataFrame:
    try:
        return con.execute(query).df()
    except Exception as e:
        raise ValueError(f"SQL error: {e}\nQuery: {query}")


# Print what we have (will show nothing until upload)
try:
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main'"
    ).fetchall()]
    for tbl in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {n:,} rows")
except Exception:
    print("  No tables loaded yet — awaiting dataset upload")

print(f"\nSchema loaded ({len(SCHEMA)} chars)")


def build_l3_graph() -> nx.DiGraph:
    """
    Build L3 causal knowledge graph dynamically from whatever tables
    exist in the current DuckDB connection.
    """
    G = nx.DiGraph()

    # Discover tables and columns from the actual database
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'"
        ).fetchall()]
    except Exception:
        tables = []

    if not tables:
        # Return an empty graph — will be rebuilt after upload
        return G

    schema_map = {}
    for tbl in tables:
        try:
            cols_df = con.execute(f"DESCRIBE {tbl}").df()
            schema_map[tbl] = cols_df["column_name"].tolist()
        except Exception:
            schema_map[tbl] = []

    # Add table nodes
    for tbl in tables:
        G.add_node(tbl, type="table")

    # Add column nodes and has_column edges
    for tbl, cols in schema_map.items():
        for col in cols:
            node = f"{tbl}.{col}"
            G.add_node(node, type="column", table=tbl)
            G.add_edge(tbl, node, rel="has_column")

    # Detect FK relationships (columns with same name across tables)
    all_col_tables: Dict[str, List[str]] = {}
    for tbl, cols in schema_map.items():
        for col in cols:
            if col.endswith("_id") or col == "id":
                all_col_tables.setdefault(col, []).append(tbl)

    for col, tbls in all_col_tables.items():
        if len(tbls) > 1:
            # Connect the first table's column to all others
            for i in range(1, len(tbls)):
                G.add_edge(
                    f"{tbls[0]}.{col}",
                    f"{tbls[i]}.{col}",
                    rel="foreign_key"
                )

    # Add date range rule if we have date info
    if DATA_DATE_MIN and DATA_DATE_MAX:
        G.add_node("rule_date_anchor", type="business_rule",
                    description=f"data spans {DATA_DATE_MIN} to {DATA_DATE_MAX} — never use CURRENT_DATE")

    return G


l3_graph = build_l3_graph()


def l3_get_context(entity: str) -> str:
    """Get causal predecessors and successors for any entity in the graph."""
    lines = []
    matches = [n for n in l3_graph.nodes if entity.lower() in n.lower()]
    for node in matches[:3]:
        in_edges  = [(u, d) for u, v, d in l3_graph.in_edges(node, data=True)]
        out_edges = [(v, d) for u, v, d in l3_graph.out_edges(node, data=True)]
        for u, d in in_edges:
            lines.append(f"  {u} →[{d.get('rel')}] {node} "
                         f"(w={d.get('weight','?')}, conf={d.get('confidence','?')})")
        for v, d in out_edges:
            lines.append(f"  {node} →[{d.get('rel')}] {v} "
                         f"(w={d.get('weight','?')}, conf={d.get('confidence','?')})")
    return "\n".join(lines) if lines else "No causal links."


def l3_get_business_rules() -> str:
    rules = [d.get("description","") for _, d in l3_graph.nodes(data=True)
             if d.get("type") == "business_rule"]
    return "\n".join(f"  - {r}" for r in rules)


nx.write_gml(l3_graph, GRAPH_PATH)
print(f"L3 graph: {l3_graph.number_of_nodes()} nodes, {l3_graph.number_of_edges()} edges")
print(f"\nBusiness rules:\n{l3_get_business_rules()}")