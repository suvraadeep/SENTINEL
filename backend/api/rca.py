"""
Causal RCA Engine v2 — production-grade, dataset-agnostic.

Architecture:
  Layer 1 — Statistical : Bulk Spearman (scipy matrix API) + MI (3k cap) + Partial corr
                          + Distance Correlation (dcor, graceful fallback)
  Layer 2 — Temporal    : Granger top-2 only (≥30 rows required) + vectorized xcorr lag
                          + CUSUM change points
  Layer 3 — Causal Graph: PC Algorithm simplified (n_cols ≤ 8), else correlation DAG
  Layer 4 — Traversal   : BFS + PageRank importance + anomaly column boost

All base layers run in parallel via ThreadPoolExecutor.
Per-method results in response so frontend can toggle between Statistical / Temporal / Graph / Ensemble.
Each chart includes a deep explanation (why / what / how_to_read).

POST /api/rca/analyze   — full RCA pipeline
POST /api/rca/chat      — agentic LangGraph Q&A
POST /api/rca/traverse  — dynamic single-feature upstream traversal
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rca", tags=["rca"])

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class RCARequest(BaseModel):
    table:           str
    target_col:      Optional[str]           = None
    p_threshold:     float                   = 0.05
    top_k:           int                     = 8
    anomaly_context: Optional[Dict[str, Any]] = None

class RCAResponse(BaseModel):
    table:        str
    target_col:   str
    explanation:  str
    root_causes:  List[Dict[str, Any]]
    graph:        Dict[str, Any]
    charts:       List[Dict[str, Any]]        # includes explanation per chart
    statistics:   Dict[str, Any]
    profile:      Dict[str, Any]
    change_points: List[Dict[str, Any]]
    methods:      Dict[str, Any]              # per-method view for frontend toggle

class RCAChatRequest(BaseModel):
    message:      str
    context:      Optional[Dict[str, Any]] = None
    table:        Optional[str]            = None
    chat_history: List[Dict]               = []

class RCAChatResponse(BaseModel):
    response:        str
    traversal_path:  List[str]            = []
    charts:          List[Dict[str, Any]] = []
    tool_calls_made: List[str]            = []

class RCATraverseRequest(BaseModel):
    table:      str
    feature:    str
    target_col: Optional[str] = None

class RCATraverseResponse(BaseModel):
    feature:      str
    causal_chain: List[Dict[str, Any]]
    explanation:  str


# ─────────────────────────────────────────────────────────────────────────────
# Chart explanation registry
# ─────────────────────────────────────────────────────────────────────────────

CHART_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "causal_graph": {
        "why": "The Metric Dependency Graph externalizes the hidden statistical structure of your dataset as a navigable network. Without this visualization, multi-hop causal chains (A→B→target) are invisible in tabular statistics alone.",
        "what": "Nodes = numeric features. Red node = target metric. Yellow/amber edges = Granger-causal (driver changes BEFORE target — temporal precedence confirmed). Blue edges = strong correlation to target. Grey dashed = inter-feature correlation. Node size ∝ degree centrality.",
        "how_to_read": "Follow yellow edges upstream from the red node to find root causes with the highest confidence. Blue edges are candidates worth investigating. Features connected only by grey dashed edges are likely confounders rather than direct causes.",
    },
    "root_cause_ranking": {
        "why": "A ranked bar chart transforms multi-dimensional statistical evidence (Spearman ρ, partial correlation, MI, Granger p-value) into a single interpretable influence score, enabling fast triage.",
        "what": "Each bar = one feature. Bar length = composite influence score (40% Spearman + 30% partial corr + 30% MI, normalized). Amber = Granger-causal. Red = also anomalous. Percentage label = contribution estimate.",
        "how_to_read": "The top-3 bars are your primary investigation targets. Amber bars warrant immediate root cause investigation (temporal causation confirmed). A feature with high Spearman but near-zero partial correlation is a confounder, not a cause.",
    },
    "metrics_timeline": {
        "why": "Overlaying the target metric with its top drivers on a shared time axis reveals the temporal signature of causal relationships — drivers that move before the target are candidates for causation.",
        "what": "Solid thick line = target metric. Dashed thin lines = top-3 driver features. Vertical red dashed lines = detected change points (abrupt mean shifts ≥ 1.5σ).",
        "how_to_read": "Look for driver peaks or valleys that precede target peaks or valleys by 1–5 periods. Change-point lines mark where a permanent regime shift may have occurred — correlate them with known events (deployments, data source changes, seasonality).",
    },
    "lag_correlation": {
        "why": "Standard correlation is symmetric — it cannot distinguish cause from effect. Lag correlation measures whether feature X at time t predicts target Y at time t+k, providing directional evidence of temporal precedence.",
        "what": "Each bar = one feature. Bar height = Pearson correlation at optimal lag. Amber = driver leads target (lag > 0, temporal precedence). Blue = concurrent correlation (lag = 0). Text label shows the optimal lag value.",
        "how_to_read": "Tall amber bars (lag > 0) are the strongest causal candidates — they change before the target does. Very high concurrent blue bars suggest instantaneous co-movement, which could be reverse causation or a common driver.",
    },
    "spearman_vs_partial": {
        "why": "Comparing Spearman (total) vs. partial correlation (direct, confounders removed) reveals whether a feature's apparent influence is genuine or entirely mediated through other variables.",
        "what": "Grouped bars: blue = total Spearman ρ (includes indirect paths), purple = partial correlation ρ (direct effect only, all other variables controlled). Features where partial ≫ Spearman have an amplified direct effect. Features where partial ≈ 0 but Spearman is high are confounders.",
        "how_to_read": "A bar pair where purple ≈ 0 and blue is large → the feature is a confounder, not a cause. Where purple > blue → the feature has a direct causal effect amplified by the correlation structure.",
    },
    "mutual_information": {
        "why": "Mutual information captures ALL types of statistical dependence (linear, non-linear, non-monotonic), unlike correlation which only detects linear relationships. A feature with high MI and low Spearman ρ has a non-linear relationship with the target.",
        "what": "Horizontal bars sorted by MI score. Color gradient from dark (low MI) to cyan (high MI). MI is in nats (base-e) — values above 0.3 are generally considered strong.",
        "how_to_read": "Features with high MI but low Spearman ρ have non-linear relationships worth investigating with scatter plots or decision tree analysis. Features near the top of both MI and Spearman rankings are the most robustly influential drivers.",
    },
    "correlation_heatmap": {
        "why": "The full correlation matrix reveals the collinearity structure of your features — critical for understanding which root causes are independent vs. confounded by shared drivers.",
        "what": "Symmetric matrix with Pearson correlation coefficients. Red = strong positive, blue = strong negative, white = uncorrelated. Values on each cell for precision reading.",
        "how_to_read": "Clusters of strongly correlated features (red blocks on the diagonal) may be measuring the same underlying phenomenon. If two root cause candidates are highly correlated with each other, one may be mediating the other — check the causal graph for the path.",
    },
    "distance_correlation": {
        "why": "Distance correlation (dCor) detects ALL forms of dependence including non-linear, non-monotonic, and interaction effects that Spearman and Pearson completely miss. A value of 0 implies statistical independence.",
        "what": "Horizontal bar chart of distance correlation coefficients between each feature and the target metric. Range [0, 1]; higher = more dependent. Unlike linear correlation, dCor=0 ↔ independence (no false negatives for non-linear patterns).",
        "how_to_read": "Features ranked highly by dCor but not by Spearman have non-linear or interaction effects. These are candidates for polynomial regression, tree-based models, or kernel methods in downstream modelling.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CausalRCAEngine v2
# ─────────────────────────────────────────────────────────────────────────────

class CausalRCAEngine:
    """Dataset-agnostic causal RCA engine (v2)."""

    _COLORS = ["#3B82F6","#06B6D4","#8B5CF6","#10B981",
               "#F59E0B","#EF4444","#EC4899","#14B8A6"]
    _LAYOUT = dict(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#F1F5F9", family="Inter, sans-serif", size=11),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155",
                    font=dict(color="#94A3B8")),
        margin=dict(l=44, r=24, t=44, b=32),
    )

    def __init__(self, df: pd.DataFrame, table_name: str = "table",
                 target_col: Optional[str] = None, p_threshold: float = 0.05,
                 top_k: int = 8):
        self.table_name  = table_name
        self.p_threshold = p_threshold
        self.raw_df      = df.copy()
        self.profile     = self._profile(df)
        self.num_cols    = self.profile["num_cols"]
        self.cat_cols    = self.profile["cat_cols"]
        self.date_col    = self.profile["date_col"]

        if target_col and target_col in self.num_cols:
            self.target_col = target_col
        else:
            self.target_col = self.profile["target_col"] or (self.num_cols[0] if self.num_cols else None)

        self.top_k = min(top_k, max(0, len(self.num_cols) - 1))

        self.df = df.copy()
        if self.date_col:
            self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

        self._spearman_cache: Optional[Dict] = None
        self._mi_cache:       Optional[Dict] = None
        self._dist_corr_cache: Optional[Dict] = None
        self._graph = None

    # ── Profiling ─────────────────────────────────────────────────────────────

    @staticmethod
    def _profile(df: pd.DataFrame) -> Dict[str, Any]:
        date_col = None
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c; break
            if any(kw in c.lower() for kw in ("date","time","created","updated","ts","month","year")):
                try:
                    tmp = pd.to_datetime(df[c], errors="coerce")
                    if tmp.notna().sum() > len(df) * 0.5:
                        df[c] = tmp; date_col = c; break
                except Exception:
                    pass

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.select_dtypes(include="object").columns
                    if df[c].nunique() < 50]
        _target_re = re.compile(
            r"(revenue|amount|sales|price|value|total|income|profit|cost|fee|return|close|adj)",
            re.IGNORECASE,
        )
        target_col = next((c for c in num_cols if _target_re.search(c)),
                          num_cols[0] if num_cols else None)
        return {
            "date_col":   date_col,
            "num_cols":   num_cols,
            "cat_cols":   cat_cols,
            "target_col": target_col,
            "row_count":  len(df),
            "col_count":  len(df.columns),
        }

    # ── Layer 1a: Bulk Spearman (scipy matrix API — single C call) ─────────────

    def _spearman_all(self) -> Dict[str, Dict]:
        if self._spearman_cache is not None:
            return self._spearman_cache
        from scipy.stats import spearmanr

        if not self.target_col or not self.num_cols:
            return {}

        work = self.df[self.num_cols].dropna()
        if len(work) > 10_000:
            work = work.sample(10_000, random_state=42)
        if len(work) < 6:
            return {}

        try:
            rho_mat, p_mat = spearmanr(work.values, axis=0)

            # Handle 2-column (scalar) case
            if not hasattr(rho_mat, "__getitem__"):
                rho_mat = np.array([[1.0, rho_mat], [rho_mat, 1.0]])
                p_mat   = np.array([[0.0, p_mat],   [p_mat,   0.0]])

            target_i = self.num_cols.index(self.target_col)
            results  = {}
            for j, col in enumerate(self.num_cols):
                if col == self.target_col:
                    continue
                results[col] = {
                    "rho":     round(float(rho_mat[target_i, j]), 4),
                    "p_value": round(float(p_mat[target_i, j]),   4),
                }
            self._spearman_cache = results
            return results
        except Exception as exc:
            logger.debug("Bulk Spearman failed: %s", exc)
            return {}

    # ── Layer 1b: Partial correlation (precision matrix) ─────────────────────

    def _partial_correlations(self, cols: List[str]) -> Dict[str, float]:
        if not cols or not self.target_col:
            return {}
        work_cols = [self.target_col] + [c for c in cols if c in self.num_cols]
        work_df   = self.df[work_cols].dropna()
        if len(work_df) < len(work_cols) + 5:
            return {}
        try:
            corr_mat  = np.corrcoef(work_df.values.T)
            precision = np.linalg.pinv(corr_mat)
            target_i  = 0
            results   = {}
            for j, col in enumerate(work_cols[1:], 1):
                denom = np.sqrt(max(precision[target_i, target_i] *
                                    precision[j, j], 1e-12))
                pc = -precision[target_i, j] / denom
                results[col] = round(float(np.clip(pc, -1, 1)), 4)
            return results
        except Exception as exc:
            logger.debug("Partial corr failed: %s", exc)
            return {}

    # ── Layer 1c: Mutual information (3 000 row cap) ──────────────────────────

    def _mutual_information(self, cols: List[str]) -> Dict[str, float]:
        if not cols or not self.target_col:
            return {}
        if self._mi_cache is not None:
            return self._mi_cache
        try:
            from sklearn.feature_selection import mutual_info_regression
            valid_cols = [c for c in cols if c in self.num_cols]
            work_df    = self.df[[self.target_col] + valid_cols].dropna()
            if len(work_df) > 3_000:
                work_df = work_df.sample(3_000, random_state=42)
            if len(work_df) < 10:
                return {}
            X      = work_df[valid_cols].values
            y      = work_df[self.target_col].values
            scores = mutual_info_regression(X, y, random_state=42)
            self._mi_cache = {c: round(float(s), 4) for c, s in zip(valid_cols, scores)}
            return self._mi_cache
        except Exception as exc:
            logger.debug("MI failed: %s", exc)
            return {}

    # ── Layer 1d: Distance Correlation (dcor, graceful fallback) ─────────────

    def _distance_correlation(self, cols: List[str]) -> Dict[str, float]:
        """
        Detects non-linear AND non-monotonic dependence.
        dcor library required; silently returns {} if not installed.
        Capped at 2 000 rows to stay fast.
        """
        if self._dist_corr_cache is not None:
            return self._dist_corr_cache
        if not self.target_col:
            return {}
        try:
            import dcor
            target_v = self.df[self.target_col].fillna(0).values
            n        = len(target_v)
            if n > 2_000:
                idx      = np.random.RandomState(42).choice(n, 2_000, replace=False)
                target_v = target_v[idx]
            else:
                idx = None

            results = {}
            for col in cols[:self.top_k]:
                if col == self.target_col or col not in self.num_cols:
                    continue
                x = self.df[col].fillna(0).values
                if idx is not None:
                    x = x[idx]
                dc = dcor.distance_correlation(x, target_v)
                results[col] = round(float(dc), 4)
            self._dist_corr_cache = results
            return results
        except ImportError:
            return {}
        except Exception as exc:
            logger.debug("Distance corr failed: %s", exc)
            return {}

    # ── Layer 1e: Feature selection (combined ranking) ────────────────────────

    def _select_top_k_features(self) -> List[str]:
        other_cols = [c for c in self.num_cols if c != self.target_col]
        if not other_cols:
            return []
        spearman = self._spearman_all()
        mi       = self._mutual_information(other_cols)
        mi_max   = max(mi.values(), default=0) + 1e-9
        scores: Dict[str, float] = {}
        for col in other_cols:
            sp   = abs(spearman.get(col, {}).get("rho", 0))
            mi_n = mi.get(col, 0) / mi_max
            scores[col] = 0.5 * sp + 0.5 * mi_n
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:self.top_k]]

    # ── Layer 2a: Granger (top-2, max_lag=2, skip if < 30 rows) ─────────────

    def _granger_selective(self, top_k_cols: List[str]) -> Dict[str, float]:
        if not self.date_col or not self.target_col or len(self.df) < 30:
            return {}
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            return {}

        results: Dict[str, float] = {}
        target_s = self.df[self.target_col].ffill().fillna(0)

        for col in top_k_cols[:2]:          # hard limit: top-2 only
            try:
                col_s   = self.df[col].ffill().fillna(0)
                test_df = pd.DataFrame({"target": target_s.values,
                                        "driver": col_s.values}).dropna()
                if len(test_df) < 20:
                    continue
                max_lag = min(2, len(test_df) // 6)
                gc      = grangercausalitytests(
                    test_df[["target", "driver"]], maxlag=max_lag, verbose=False
                )
                min_p = min(gc[lag][0]["ssr_ftest"][1] for lag in gc)
                results[col] = round(float(min_p), 4)
            except Exception:
                continue
        return results

    # ── Layer 2b: Vectorized lag correlation ─────────────────────────────────

    def _lag_correlation(self, col: str, max_lag: int = 5) -> Dict[str, Any]:
        from scipy.stats import pearsonr
        n = len(self.df)
        if n < max_lag * 2 + 4:
            return {"optimal_lag": 0, "correlation": 0.0, "p_value": 1.0,
                    "temporal_precedence": False}
        target_v = self.df[self.target_col].fillna(0).values
        driver_v = self.df[col].fillna(0).values
        best = {"optimal_lag": 0, "correlation": 0.0, "p_value": 1.0}
        for lag in range(0, min(max_lag + 1, n // 4)):
            try:
                t_sl = target_v[lag:] if lag > 0 else target_v
                d_sl = driver_v[:n - lag] if lag > 0 else driver_v
                r, p = pearsonr(t_sl, d_sl)
                if abs(r) > abs(best["correlation"]):
                    best = {"optimal_lag": lag,
                            "correlation": round(float(r), 4),
                            "p_value":     round(float(p), 4)}
            except Exception:
                pass
        best["temporal_precedence"] = best["optimal_lag"] > 0
        return best

    # ── Layer 2c: CUSUM change points ─────────────────────────────────────────

    def _detect_change_points(self, col: str) -> List[Dict[str, Any]]:
        series = self.df[col].dropna().values
        if len(series) < 12:
            return []
        w   = max(4, len(series) // 8)
        cps: List[Dict] = []
        for i in range(w, len(series) - w):
            left  = series[max(0, i - w): i]
            right = series[i: min(len(series), i + w)]
            mu_l, mu_r = left.mean(), right.mean()
            sigma_l    = left.std() + 1e-9
            shift = abs(mu_r - mu_l) / sigma_l
            if shift > 1.5:
                cps.append({"index": i, "shift_sigma": round(float(shift), 3),
                             "direction": "up" if mu_r > mu_l else "down",
                             "left_mean":  round(float(mu_l), 4),
                             "right_mean": round(float(mu_r), 4)})
        if not cps:
            return []
        # Merge nearby
        merged: List[Dict] = [cps[0]]
        for cp in cps[1:]:
            if cp["index"] - merged[-1]["index"] > w:
                merged.append(cp)
            elif cp["shift_sigma"] > merged[-1]["shift_sigma"]:
                merged[-1] = cp
        for cp in merged:
            if self.date_col and cp["index"] < len(self.df):
                cp["at"] = str(self.df[self.date_col].iloc[cp["index"]])
            else:
                cp["at"] = f"row {cp['index']}"
        return sorted(merged, key=lambda x: x["shift_sigma"], reverse=True)[:3]

    # ── Layer 3: Causal Graph (PC algorithm or correlation DAG) ───────────────

    def _pc_algorithm_simplified(self, cols: List[str],
                                  partial_corr: Dict) -> Optional[Any]:
        """
        Simplified PC skeleton algorithm (constraint-based causal discovery).
        Only runs for n_cols ≤ 8 to remain O(n²).
        Returns nx.DiGraph or None if not applicable.
        """
        if len(cols) > 8 or len(cols) < 2:
            return None
        try:
            import networkx as nx
            from scipy.stats import pearsonr

            work_cols = [self.target_col] + [c for c in cols if c in self.num_cols]
            work_df   = self.df[work_cols].dropna()
            if len(work_df) < 20:
                return None

            # Step 1: Start with complete undirected graph
            G_und = nx.Graph()
            G_und.add_nodes_from(work_cols)
            for i, c1 in enumerate(work_cols):
                for c2 in work_cols[i+1:]:
                    G_und.add_edge(c1, c2)

            # Step 2: Remove edges failing conditional independence
            edges_to_remove = []
            for c1, c2 in list(G_und.edges()):
                sep = [c for c in work_cols if c not in (c1, c2)][:2]
                # Partial correlation conditioning on sep
                cond_df = work_df[[c1, c2] + sep].dropna()
                if len(cond_df) < 15:
                    continue
                if sep:
                    # Regress out sep from both c1 and c2
                    from numpy.linalg import lstsq
                    S = cond_df[sep].values
                    S_aug = np.column_stack([S, np.ones(len(S))])
                    r1 = cond_df[c1].values - S_aug @ lstsq(S_aug, cond_df[c1].values, rcond=None)[0]
                    r2 = cond_df[c2].values - S_aug @ lstsq(S_aug, cond_df[c2].values, rcond=None)[0]
                else:
                    r1 = cond_df[c1].values
                    r2 = cond_df[c2].values
                try:
                    _, p = pearsonr(r1, r2)
                    if p > self.p_threshold * 2:
                        edges_to_remove.append((c1, c2))
                except Exception:
                    pass

            G_und.remove_edges_from(edges_to_remove)

            # Step 3: Orient v-structures (colliders: A-C-B where A-B not adjacent)
            G_dir = nx.DiGraph()
            G_dir.add_nodes_from(work_cols)
            for c1, c2 in G_und.edges():
                G_dir.add_edge(c1, c2)
                G_dir.add_edge(c2, c1)

            for c1, c2 in list(G_und.edges()):
                for c3 in set(G_und.neighbors(c1)) & set(G_und.neighbors(c2)):
                    if not G_und.has_edge(c1, c2):
                        # V-structure: c1→c3←c2
                        if G_dir.has_edge(c3, c1): G_dir.remove_edge(c3, c1)
                        if G_dir.has_edge(c3, c2): G_dir.remove_edge(c3, c2)

            return G_dir
        except Exception as exc:
            logger.debug("PC algorithm failed: %s", exc)
            return None

    def build_causal_graph(self, top_k_cols: List[str], spearman: Dict,
                           granger: Dict, partial_corr: Dict,
                           mi: Dict, use_pc: bool = False) -> Any:
        import networkx as nx

        pc_graph = None
        if use_pc:
            pc_graph = self._pc_algorithm_simplified(top_k_cols, partial_corr)

        G = nx.DiGraph()
        for col in top_k_cols + ([self.target_col] if self.target_col else []):
            G.add_node(col, is_target=(col == self.target_col),
                       spearman_rho=spearman.get(col, {}).get("rho", 0),
                       mi_score=mi.get(col, 0))

        mi_max = max(mi.values(), default=1e-9) + 1e-9

        for col in top_k_cols:
            sp_rho = spearman.get(col, {}).get("rho", 0)
            sp_p   = spearman.get(col, {}).get("p_value", 1.0)
            gc_p   = granger.get(col)
            pc     = partial_corr.get(col, 0)
            mi_s   = mi.get(col, 0)

            if sp_p > 0.1 and (gc_p is None or gc_p > 0.1):
                continue

            w = (abs(sp_rho) * 0.4 + abs(pc) * 0.3 + (mi_s / mi_max) * 0.3)
            is_causal = gc_p is not None and gc_p < self.p_threshold

            # If PC graph available, use its directionality
            if pc_graph and pc_graph.has_edge(col, self.target_col):
                is_causal = True
            elif pc_graph and pc_graph.has_edge(self.target_col, col):
                # target→col direction — skip (reverse)
                continue

            G.add_edge(col, self.target_col,
                       weight=round(float(w), 4),
                       spearman_rho=round(float(sp_rho), 4),
                       spearman_p=round(float(sp_p), 4),
                       granger_p=gc_p,
                       partial_corr=round(float(pc), 4),
                       mi_score=round(float(mi_s), 4),
                       edge_type="granger" if is_causal else "correlation",
                       is_causal=is_causal,
                       used_pc=use_pc and pc_graph is not None)

        # Inter-feature edges
        from scipy.stats import spearmanr as _sr
        for i, c1 in enumerate(top_k_cols):
            for c2 in top_k_cols[i+1:]:
                valid = self.df[[c1, c2]].dropna()
                if len(valid) < 6:
                    continue
                try:
                    rho, p = _sr(valid[c1], valid[c2])
                    if abs(rho) > 0.5 and p < 0.05:
                        G.add_edge(c1, c2, weight=round(abs(float(rho)), 4),
                                   spearman_rho=round(float(rho), 4),
                                   spearman_p=round(float(p), 4),
                                   edge_type="inter_feature", is_causal=False)
                except Exception:
                    pass

        self._graph = G
        return G

    def serialize_graph(self) -> Dict[str, Any]:
        if self._graph is None:
            return {"nodes": [], "edges": []}
        import networkx as nx
        G = self._graph

        try:
            k_val = 2.5 / max(len(G.nodes()) ** 0.5, 1)
            pos = nx.spring_layout(G, k=k_val, iterations=80, seed=42)
        except Exception:
            pos = {n: (i * 0.3, 0.5) for i, n in enumerate(G.nodes())}

        # PageRank for node importance
        try:
            pr = nx.pagerank(G, alpha=0.85)
        except Exception:
            pr = {n: 0.1 for n in G.nodes()}

        nodes = []
        for n, attr in G.nodes(data=True):
            x, y = pos.get(n, (0, 0))
            nodes.append({
                "id":          n, "label": n,
                "x":           round(float(x), 4),
                "y":           round(float(y), 4),
                "is_target":   bool(attr.get("is_target", False)),
                "spearman":    float(attr.get("spearman_rho", 0)),
                "mi_score":    float(attr.get("mi_score", 0)),
                "pagerank":    round(float(pr.get(n, 0.1)), 4),
                "degree":      G.degree(n),
                "in_degree":   G.in_degree(n),
                "out_degree":  G.out_degree(n),
            })

        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source":       u, "target": v,
                "weight":       data.get("weight", 0),
                "spearman_rho": data.get("spearman_rho", 0),
                "granger_p":    data.get("granger_p"),
                "partial_corr": data.get("partial_corr"),
                "mi_score":     data.get("mi_score", 0),
                "edge_type":    data.get("edge_type", "correlation"),
                "is_causal":    bool(data.get("is_causal", False)),
                "used_pc":      bool(data.get("used_pc", False)),
            })

        return {"nodes": nodes, "edges": edges}

    # ── Layer 4: Traversal (BFS + PageRank boost) ─────────────────────────────

    def traverse_root_causes(self, anomaly_cols: Optional[List[str]] = None) -> List[Dict]:
        if self._graph is None:
            return []
        G = self._graph

        try:
            import networkx as nx
            pr = nx.pagerank(G, alpha=0.85)
        except Exception:
            pr = {n: 0.1 for n in G.nodes()}

        root_causes: List[Dict] = []
        visited: set = set()
        queue = [self.target_col]

        while queue:
            node = queue.pop(0)
            if node not in G:
                continue
            for pred in G.predecessors(node):
                if pred in visited:
                    continue
                visited.add(pred)
                queue.append(pred)

                ed     = G[pred][node]
                sp_rho = abs(ed.get("spearman_rho", 0))
                pc     = abs(ed.get("partial_corr") or 0)
                mi_s   = ed.get("mi_score", 0)
                mi_max = max((G[u][v].get("mi_score", 0)
                              for u, v in G.edges()), default=1e-9) + 1e-9

                influence = sp_rho * 0.4 + pc * 0.3 + (mi_s / mi_max) * 0.3
                gc_p      = ed.get("granger_p")
                influence += 0.20 if (gc_p is not None and gc_p < self.p_threshold) else 0
                influence += 0.10 * pr.get(pred, 0.1)   # PageRank bonus
                if anomaly_cols and pred in anomaly_cols:
                    influence += 0.15

                influence = round(min(float(influence), 1.0), 4)

                root_causes.append({
                    "name":            pred,
                    "target":          node,
                    "depth":           0,
                    "influence_score": influence,
                    "contribution":    round(influence * 100, 1),
                    "spearman_rho":    ed.get("spearman_rho", 0),
                    "p_value":         ed.get("spearman_p", 1.0),
                    "granger_p":       gc_p,
                    "partial_corr":    ed.get("partial_corr"),
                    "mi_score":        mi_s,
                    "pagerank":        round(float(pr.get(pred, 0.1)), 4),
                    "edge_type":       ed.get("edge_type", "correlation"),
                    "is_causal":       bool(ed.get("is_causal", False)),
                    "is_anomalous":    bool(anomaly_cols and pred in anomaly_cols),
                })

        import networkx as nx
        if self.target_col in G:
            try:
                rev = G.reverse()
                lengths = nx.single_source_shortest_path_length(rev, self.target_col)
                for rc in root_causes:
                    rc["depth"] = lengths.get(rc["name"], 1)
            except Exception:
                pass

        root_causes.sort(key=lambda x: x["influence_score"], reverse=True)
        return root_causes

    # ── Visualization ─────────────────────────────────────────────────────────

    def generate_charts(self, root_causes: List[Dict], top_k_cols: List[str],
                        spearman: Dict, granger: Dict, mi: Dict,
                        change_points: List[Dict],
                        dist_corr: Optional[Dict] = None) -> List[Dict[str, Any]]:
        charts: List[Dict[str, Any]] = []

        def _html(fig, title: str, expl_key: str) -> Dict[str, Any]:
            fig.update_layout(**self._LAYOUT,
                              title=dict(text=title, font=dict(size=13)))
            return {
                "title":      title,
                "chart_type": expl_key,
                "html":  fig.to_html(include_plotlyjs="cdn", full_html=False,
                                     config={"responsive": True}),
                "explanation": CHART_EXPLANATIONS.get(expl_key, {}),
            }

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            # ── 1. Causal graph ───────────────────────────────────────────────
            if self._graph is not None:
                G          = self._graph
                graph_data = self.serialize_graph()
                nodes      = graph_data["nodes"]
                edges      = graph_data["edges"]
                node_map   = {n["id"]: n for n in nodes}

                def _etrace(edge_list, color, dash, name):
                    xs, ys = [], []
                    for e in edge_list:
                        n0, n1 = node_map.get(e["source"]), node_map.get(e["target"])
                        if n0 and n1:
                            xs += [n0["x"], n1["x"], None]
                            ys += [n0["y"], n1["y"], None]
                    return go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                      line=dict(color=color, width=2, dash=dash),
                                      opacity=0.7, hoverinfo="skip")

                causal_e = [e for e in edges if e["is_causal"]]
                corr_e   = [e for e in edges if not e["is_causal"]
                            and e["edge_type"] != "inter_feature"]
                inter_e  = [e for e in edges if e["edge_type"] == "inter_feature"]

                node_clr = ["#EF4444" if n["is_target"] else
                             "#F59E0B" if any(rc["name"] == n["id"] and rc["is_causal"]
                                             for rc in root_causes[:3]) else
                             "#3B82F6" for n in nodes]
                node_sz  = [28 if n["is_target"] else
                             14 + int(n["pagerank"] * 60) for n in nodes]

                hover = [f"<b>{n['label']}</b><br>ρ={n['spearman']:.3f}"
                         f"<br>MI={n['mi_score']:.3f}<br>PageRank={n['pagerank']:.3f}"
                         for n in nodes]

                fig_g = go.Figure()
                fig_g.add_trace(_etrace(inter_e,  "#475569", "dot",   "Correlated (inter-feature)"))
                fig_g.add_trace(_etrace(corr_e,   "#3B82F6", "solid", "Correlation → target"))
                fig_g.add_trace(_etrace(causal_e, "#F59E0B", "solid", "Granger causal → target"))
                fig_g.add_trace(go.Scatter(
                    x=[n["x"] for n in nodes], y=[n["y"] for n in nodes],
                    mode="markers+text",
                    marker=dict(size=node_sz, color=node_clr,
                                line=dict(width=1.5, color="#1E293B")),
                    text=[n["label"] for n in nodes], textposition="top center",
                    textfont=dict(size=9, color="#F1F5F9"),
                    hovertext=hover, hoverinfo="text", name="Features",
                ))
                fig_g.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    showlegend=True, height=420,
                )
                charts.append(_html(fig_g, "Metric Dependency Graph", "causal_graph"))

            # ── 2. Root cause ranking ─────────────────────────────────────────
            if root_causes:
                rc_df = pd.DataFrame(root_causes[:10])
                clrs  = ["#EF4444" if r.get("is_causal") else
                          "#F59E0B" if r.get("is_anomalous") else "#3B82F6"
                          for _, r in rc_df.iterrows()]
                fig_rc = go.Figure(go.Bar(
                    y=rc_df["name"], x=rc_df["influence_score"],
                    orientation="h", marker_color=clrs,
                    text=[f"{s:.0%}" for s in rc_df["influence_score"]],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b><br>Influence: %{x:.3f}<br>"
                        "Spearman ρ: %{customdata[0]:.3f}<br>"
                        "p-value: %{customdata[1]:.4f}<extra></extra>"
                    ),
                    customdata=rc_df[["spearman_rho","p_value"]].values,
                ))
                fig_rc.update_layout(yaxis=dict(autorange="reversed"),
                                     xaxis=dict(range=[0, 1.15]),
                                     height=max(280, len(rc_df) * 34 + 60))
                charts.append(_html(fig_rc, "Root Cause Ranking — Influence Score",
                                    "root_cause_ranking"))

            # ── 3. Metrics timeline ───────────────────────────────────────────
            if self.date_col and self.target_col:
                plot_cols = [self.target_col] + top_k_cols[:3]
                fig_ts    = go.Figure()
                for i, col in enumerate(plot_cols):
                    fig_ts.add_trace(go.Scatter(
                        x=self.df[self.date_col], y=self.df[col],
                        mode="lines", name=col,
                        line=dict(color=self._COLORS[i % len(self._COLORS)],
                                  width=2.5 if col == self.target_col else 1.5,
                                  dash="solid" if col == self.target_col else "dot"),
                    ))
                for cp in change_points[:2]:
                    idx = cp.get("index", 0)
                    if idx < len(self.df):
                        x_val = self.df[self.date_col].iloc[idx]
                        fig_ts.add_vline(
                            x=x_val, line_dash="dash", line_color="#EF4444",
                            annotation_text=f"Shift {cp['direction']} {cp['shift_sigma']:.1f}σ",
                            annotation_font_color="#EF4444",
                        )
                fig_ts.update_layout(height=320)
                charts.append(_html(fig_ts,
                    f"Metrics Timeline — {self.target_col} + Top Drivers",
                    "metrics_timeline"))

            # ── 4. Lag cross-correlation ──────────────────────────────────────
            if top_k_cols and self.target_col and len(self.df) >= 20:
                lag_recs = []
                for col in top_k_cols[:5]:
                    li = self._lag_correlation(col)
                    lag_recs.append({"feature": col, "best_lag": li["optimal_lag"],
                                     "correlation": li["correlation"],
                                     "temporal": li["temporal_precedence"]})
                lag_df = pd.DataFrame(lag_recs)
                fig_lag = go.Figure(go.Bar(
                    x=lag_df["feature"], y=lag_df["correlation"],
                    marker_color=["#F59E0B" if r["temporal"] else "#3B82F6"
                                  for _, r in lag_df.iterrows()],
                    text=[f"lag={r['best_lag']}" for _, r in lag_df.iterrows()],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>r=%{y:.3f}<br>lag=%{text}<extra></extra>",
                ))
                fig_lag.add_hline(y=0, line_color="#475569", line_width=1)
                fig_lag.update_layout(height=280, yaxis=dict(range=[-1.1, 1.1]))
                charts.append(_html(fig_lag,
                    "Lag Correlation (amber = driver leads target)",
                    "lag_correlation"))

            # ── 5. Spearman vs partial correlation ───────────────────────────
            if root_causes and len(root_causes) >= 2:
                rc_df2 = pd.DataFrame(root_causes[:8])
                rc_df2["pc_safe"] = rc_df2["partial_corr"].apply(
                    lambda x: float(x) if x is not None else 0.0
                )
                fig_pc = go.Figure()
                fig_pc.add_trace(go.Bar(name="Spearman ρ (total)",
                    x=rc_df2["name"], y=rc_df2["spearman_rho"],
                    marker_color="#3B82F6", opacity=0.85))
                fig_pc.add_trace(go.Bar(name="Partial corr (direct)",
                    x=rc_df2["name"], y=rc_df2["pc_safe"],
                    marker_color="#8B5CF6", opacity=0.85))
                fig_pc.update_layout(barmode="group", height=300,
                                     yaxis=dict(range=[-1.1, 1.1]))
                charts.append(_html(fig_pc,
                    "Spearman vs Partial Correlation (total vs direct effect)",
                    "spearman_vs_partial"))

            # ── 6. Mutual information ranking ─────────────────────────────────
            if mi:
                mi_df = (pd.DataFrame([{"feature": k, "mi": v}
                                        for k, v in mi.items()])
                         .sort_values("mi", ascending=True).tail(10))
                fig_mi = go.Figure(go.Bar(
                    y=mi_df["feature"], x=mi_df["mi"], orientation="h",
                    marker=dict(color=mi_df["mi"],
                                colorscale=[[0,"#1E293B"],[0.5,"#3B82F6"],[1,"#06B6D4"]],
                                showscale=False),
                    text=[f"{v:.3f}" for v in mi_df["mi"]], textposition="outside",
                ))
                fig_mi.update_layout(height=max(260, len(mi_df) * 30 + 60))
                charts.append(_html(fig_mi,
                    "Mutual Information — Non-linear Feature Relevance",
                    "mutual_information"))

            # ── 7. Correlation heatmap ────────────────────────────────────────
            heat_cols = ([self.target_col] + top_k_cols)[:8]
            heat_df   = self.df[heat_cols].dropna()
            if len(heat_df) >= 4:
                corr_mat = heat_df.corr()
                fig_hm   = go.Figure(go.Heatmap(
                    z=corr_mat.values,
                    x=corr_mat.columns.tolist(),
                    y=corr_mat.index.tolist(),
                    colorscale="RdBu_r", zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr_mat.values],
                    texttemplate="%{text}", showscale=True,
                ))
                fig_hm.update_layout(height=380)
                charts.append(_html(fig_hm,
                    "Correlation Heatmap — Target + Top Drivers",
                    "correlation_heatmap"))

            # ── 8. Distance correlation (if available) ────────────────────────
            if dist_corr and len(dist_corr) >= 2:
                dc_df = (pd.DataFrame([{"feature": k, "dcor": v}
                                        for k, v in dist_corr.items()])
                         .sort_values("dcor", ascending=True).tail(10))
                fig_dc = go.Figure(go.Bar(
                    y=dc_df["feature"], x=dc_df["dcor"], orientation="h",
                    marker=dict(color=dc_df["dcor"],
                                colorscale=[[0,"#1E293B"],[0.5,"#8B5CF6"],[1,"#EC4899"]],
                                showscale=False),
                    text=[f"{v:.3f}" for v in dc_df["dcor"]], textposition="outside",
                ))
                fig_dc.update_layout(height=max(260, len(dc_df) * 30 + 60))
                charts.append(_html(fig_dc,
                    "Distance Correlation — Full Dependence (linear + non-linear)",
                    "distance_correlation"))

        except Exception as exc:
            logger.warning("RCA chart generation failed: %s", exc, exc_info=True)

        return charts

    # ── LLM explanation ───────────────────────────────────────────────────────

    def llm_explanation(self, root_causes: List[Dict], spearman: Dict,
                        granger: Dict, partial_corr: Dict,
                        change_points: List[Dict], call_llm) -> str:
        if not root_causes:
            return f"No statistically significant root causes found for {self.target_col}."

        top3    = root_causes[:3]
        primary = top3[0]

        sp_lines = "\n".join(
            f"  {c}: ρ={d['rho']:+.3f} (p={d['p_value']:.4f})"
            for c, d in sorted(spearman.items(),
                               key=lambda x: abs(x[1]["rho"]), reverse=True)[:6]
        )
        gc_lines = "\n".join(
            f"  {c}: p={p:.4f} → {'SIGNIFICANT' if p < self.p_threshold else 'not significant'}"
            for c, p in granger.items()
        ) or "  None computed (requires time-ordered data ≥ 30 rows)"
        pc_lines = "\n".join(
            f"  {c}: partial_ρ={v:+.3f}"
            for c, v in sorted(partial_corr.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:5]
        ) or "  None available"
        cp_lines = "\n".join(
            f"  At {cp['at']}: {cp['direction']} shift of {cp['shift_sigma']:.1f}σ "
            f"(mean: {cp['left_mean']:.3f} → {cp['right_mean']:.3f})"
            for cp in change_points[:2]
        ) or "  No significant change points detected"
        rc_lines = "\n".join(
            f"  {i+1}. {rc['name']} — influence={rc['influence_score']:.3f}, "
            f"ρ={rc['spearman_rho']:+.3f}, "
            f"{'Granger p=' + str(rc['granger_p']) if rc['granger_p'] else 'no Granger'}, "
            f"{'CAUSAL' if rc['is_causal'] else 'correlated'}"
            for i, rc in enumerate(top3)
        )

        if call_llm is None:
            return (
                f"Root cause analysis of **{self.target_col}** (table: {self.table_name}):\n\n"
                f"Primary driver: **{primary['name']}** "
                f"(influence={primary['influence_score']:.2f}, ρ={primary['spearman_rho']:+.3f})\n\n"
                f"Top causes:\n{rc_lines}\n\n"
                f"Configure an LLM provider for AI-generated narrative explanations."
            )

        prompt = f"""You are a senior data scientist writing a root cause analysis report.

DATASET: {self.table_name}
TARGET METRIC: {self.target_col}
ROWS ANALYZED: {len(self.df):,}

RANKED ROOT CAUSES (by influence score):
{rc_lines}

SPEARMAN CORRELATIONS (monotonic):
{sp_lines}

PARTIAL CORRELATIONS (direct effect, confounders removed):
{pc_lines}

GRANGER CAUSALITY (temporal precedence):
{gc_lines}

CHANGE POINTS (abrupt metric shifts):
{cp_lines}

INSTRUCTIONS:
- Write exactly 3 paragraphs
- Paragraph 1: What changed — which metric, magnitude, time window
- Paragraph 2: Why — cite PRIMARY cause with exact statistics (ρ, p-value), explain temporal precedence if Granger significant, secondary drivers briefly
- Paragraph 3: Recommended actions — 3 specific, actionable, prioritised steps
- NEVER invent statistics; only use numbers from the evidence above
- Be concise (each paragraph ≤ 5 sentences)
"""
        try:
            return call_llm(prompt, temperature=0.15)
        except Exception as exc:
            logger.warning("LLM explanation failed: %s", exc)
            return (
                f"**{self.target_col}** — Primary driver: **{primary['name']}** "
                f"(influence={primary['influence_score']:.2f}, ρ={primary['spearman_rho']:+.3f})."
            )

    def answer_question(self, question: str, context: Dict[str, Any],
                        call_llm) -> Tuple[str, List[str]]:
        import json
        graph_data     = self.serialize_graph()
        rc_context     = context.get("root_causes", [])[:5]
        stats          = context.get("statistics", {})
        graph_nodes    = [n["id"] for n in graph_data["nodes"]]
        mentioned      = [n for n in graph_nodes if n.lower() in question.lower()]
        traversal_path = mentioned[:3]

        if self._graph and mentioned:
            preds = list(self._graph.predecessors(mentioned[0]))
            traversal_path = preds[:3] + mentioned[:1]

        if call_llm is None:
            return ("LLM not configured.", traversal_path)

        prompt = f"""You are an RCA assistant with access to a causal graph. Respond in strict Markdown.

CAUSAL GRAPH NODES:
{json.dumps(graph_data['nodes'][:10], indent=2)}

TOP ROOT CAUSES:
{json.dumps(rc_context, indent=2)}

STATISTICS (Spearman + Granger only):
{json.dumps({k: v for k, v in stats.items() if k in ('spearman','granger')}, indent=2)}

TRAVERSAL PATH: {traversal_path}

USER QUESTION: {question}

Instructions: Answer in Markdown. Cite specific statistics. Never invent numbers.
If the user asks about a causal chain, describe it step-by-step with evidence.
For any formula use $...$ LaTeX.
"""
        try:
            answer = call_llm(prompt, temperature=0.1)
        except Exception as exc:
            answer = f"Error: {exc}"
        return answer, traversal_path

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self, call_llm=None, anomaly_context: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.target_col or not self.num_cols:
            return {
                "table": self.table_name, "target_col": self.target_col or "",
                "explanation": "No numeric columns found.",
                "root_causes": [], "graph": {"nodes":[],"edges":[]},
                "charts": [], "statistics": {}, "profile": self.profile,
                "change_points": [], "methods": {},
            }

        other_cols = [c for c in self.num_cols if c != self.target_col]

        # ── Parallel: Spearman + MI ──────────────────────────────────────────
        spearman = mi = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_sp = pool.submit(self._spearman_all)
            fut_mi = pool.submit(self._mutual_information, other_cols)
            try:
                spearman = fut_sp.result(timeout=20)
            except (_Timeout, Exception) as exc:
                logger.warning("Spearman timeout: %s", exc)
            try:
                mi = fut_mi.result(timeout=20)
            except (_Timeout, Exception) as exc:
                logger.warning("MI timeout: %s", exc)

        top_k_cols   = self._select_top_k_features()
        partial_corr = self._partial_correlations(top_k_cols)
        dist_corr    = self._distance_correlation(top_k_cols)

        # ── Sequential: Granger (fast: top-2 only) ───────────────────────────
        granger      = self._granger_selective(top_k_cols)
        lag_analysis = {c: self._lag_correlation(c) for c in top_k_cols[:4]}
        change_pts   = self._detect_change_points(self.target_col)

        # ── Causal graph (PC algorithm if n_cols ≤ 8) ────────────────────────
        use_pc = len(top_k_cols) <= 7
        self.build_causal_graph(top_k_cols, spearman, granger, partial_corr, mi,
                                use_pc=use_pc)

        # ── Traversal ─────────────────────────────────────────────────────────
        anomaly_cols = list(anomaly_context.get("anomalous_columns", [])) \
            if anomaly_context else []
        root_causes = self.traverse_root_causes(anomaly_cols)

        # ── Charts ────────────────────────────────────────────────────────────
        charts = self.generate_charts(
            root_causes, top_k_cols, spearman, granger, mi, change_pts, dist_corr
        )

        # ── Explanation ───────────────────────────────────────────────────────
        explanation = self.llm_explanation(
            root_causes, spearman, granger, partial_corr, change_pts, call_llm
        )

        # ── Per-method views for frontend toggle ──────────────────────────────
        # statistical: all root causes ranked by spearman + MI (no is_causal filter)
        stat_root_causes = sorted(
            root_causes,
            key=lambda x: abs(x.get("spearman_rho", 0)) + x.get("mi_score", 0),
            reverse=True,
        )

        # temporal: Granger-causal OR lag-precedent features; fallback = all causes
        temporal_root_causes = [
            rc for rc in root_causes
            if rc.get("is_causal", False)
            or lag_analysis.get(rc["name"], {}).get("temporal_precedence", False)
        ] or root_causes

        methods = {
            "statistical": {
                "root_causes": stat_root_causes,
                "key_stats": {
                    "top_spearman": sorted(
                        [(k, v["rho"]) for k, v in spearman.items()],
                        key=lambda x: abs(x[1]), reverse=True
                    )[:5],
                    "top_mi": sorted(mi.items(), key=lambda x: x[1], reverse=True)[:5],
                    "top_dcor": sorted(
                        dist_corr.items(), key=lambda x: x[1], reverse=True
                    )[:5] if dist_corr else [],
                },
            },
            "temporal": {
                "root_causes": temporal_root_causes,
                "granger":     granger,
                "lag_analysis": lag_analysis,
                "change_points": change_pts,
                "has_temporal_data": bool(self.date_col and len(self.df) >= 30),
            },
            "graph": {
                "root_causes": root_causes,
                "graph":       self.serialize_graph(),
                "pc_used":     use_pc,
            },
            "ensemble": {
                "root_causes": root_causes,
            },
        }

        return {
            "table":        self.table_name,
            "target_col":   self.target_col,
            "explanation":  explanation,
            "root_causes":  root_causes,
            "graph":        self.serialize_graph(),
            "charts":       charts,
            "statistics": {
                "spearman":          spearman,
                "partial_corr":      partial_corr,
                "mutual_information": mi,
                "distance_corr":     dist_corr,
                "granger":           granger,
                "lag_analysis":      lag_analysis,
                "top_k_cols":        top_k_cols,
                "pc_algorithm_used": use_pc,
            },
            "profile":       self.profile,
            "change_points": change_pts,
            "methods":       methods,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_rca(req: RCARequest):
    from backend.core.namespace import sentinel_ns
    if not sentinel_ns.is_initialized:
        raise HTTPException(status_code=503, detail="SENTINEL not initialized")

    con = sentinel_ns._ns.get("con")
    if con is None:
        raise HTTPException(status_code=503, detail="No database connection")

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", req.table):
        raise HTTPException(status_code=400, detail="Invalid table name")

    try:
        df = con.execute(f'SELECT * FROM "{req.table}"').df()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read table: {exc}")

    if df.empty or len(df) < 4:
        raise HTTPException(status_code=400, detail="Table has insufficient data")

    engine = CausalRCAEngine(df, table_name=req.table, target_col=req.target_col,
                             p_threshold=req.p_threshold, top_k=req.top_k)
    if not engine.num_cols:
        raise HTTPException(status_code=400, detail="No numeric columns found in table")

    call_llm = sentinel_ns._ns.get("call_llm")
    return engine.run(call_llm=call_llm, anomaly_context=req.anomaly_context)


@router.post("/chat", response_model=RCAChatResponse)
async def rca_chat(req: RCAChatRequest):
    from backend.core.namespace import sentinel_ns
    if not sentinel_ns.is_initialized:
        raise HTTPException(status_code=503, detail="SENTINEL not initialized")

    call_llm = sentinel_ns._ns.get("call_llm")
    con      = sentinel_ns._ns.get("con")

    # Try LangGraph agent first
    try:
        from backend.api.rca_chat_agent import run_rca_agent
        result = run_rca_agent(
            message      = req.message,
            context      = req.context or {},
            table        = req.table or (req.context or {}).get("table", ""),
            chat_history = req.chat_history,
            con          = con,
            call_llm     = call_llm,
        )
        return RCAChatResponse(
            response        = result["response"],
            traversal_path  = result.get("traversal_path", []),
            charts          = result.get("charts", []),
            tool_calls_made = result.get("tool_calls_made", []),
        )
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("RCA LangGraph agent failed, falling back: %s", exc)

    # Fallback: rebuild engine + direct LLM
    if call_llm is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    ctx     = req.context or {}
    table   = req.table or ctx.get("table")
    engine  = None
    if table and con and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
        try:
            df         = con.execute(f'SELECT * FROM "{table}"').df()
            target_col = ctx.get("target_col") or ctx.get("profile", {}).get("target_col")
            engine     = CausalRCAEngine(df, table_name=table, target_col=target_col)
            engine._spearman_cache = ctx.get("statistics", {}).get("spearman")
            engine._mi_cache       = ctx.get("statistics", {}).get("mutual_information")
            top_k  = ctx.get("statistics", {}).get("top_k_cols", [])
            sp     = ctx.get("statistics", {}).get("spearman", {})
            pc     = ctx.get("statistics", {}).get("partial_corr", {})
            mi_d   = ctx.get("statistics", {}).get("mutual_information", {})
            if top_k and sp:
                engine.build_causal_graph(top_k, sp, {}, pc, mi_d)
        except Exception as exc:
            logger.debug("Chat engine rebuild failed: %s", exc)

    if engine:
        answer, path = engine.answer_question(req.message, ctx, call_llm)
    else:
        import json

        # Build history string
        history_str = ""
        for m in (req.chat_history or [])[-6:]:
            role = m.get("role", "user")
            text = m.get("text", m.get("content", ""))
            history_str += f"\n**{role.upper()}:** {text}"

        rc_summary = json.dumps(ctx.get("root_causes", [])[:5], indent=2)
        answer = call_llm(
            f"You are an RCA assistant. Respond in strict Markdown. "
            f"Never invent statistics.\n\nCONTEXT:\n{rc_summary}"
            f"{history_str}\n\n**USER:** {req.message}",
            temperature=0.1,
        )
        path = []

    return RCAChatResponse(response=answer, traversal_path=path)


@router.post("/traverse", response_model=RCATraverseResponse)
async def traverse_feature(req: RCATraverseRequest):
    from backend.core.namespace import sentinel_ns
    if not sentinel_ns.is_initialized:
        raise HTTPException(status_code=503, detail="SENTINEL not initialized")

    con = sentinel_ns._ns.get("con")
    if con is None:
        raise HTTPException(status_code=503, detail="No database connection")

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", req.table):
        raise HTTPException(status_code=400, detail="Invalid table name")

    try:
        df = con.execute(f'SELECT * FROM "{req.table}"').df()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read table: {exc}")

    engine = CausalRCAEngine(df, table_name=req.table, target_col=req.target_col)
    if not engine.num_cols:
        raise HTTPException(status_code=400, detail="No numeric columns")

    top_k    = engine._select_top_k_features()
    spearman = engine._spearman_all()
    granger  = engine._granger_selective(top_k)
    pc       = engine._partial_correlations(top_k)
    mi       = engine._mutual_information(top_k)
    engine.build_causal_graph(top_k, spearman, granger, pc, mi)

    orig_target       = engine.target_col
    engine.target_col = req.feature
    causal_chain      = engine.traverse_root_causes()
    engine.target_col = orig_target

    call_llm    = sentinel_ns._ns.get("call_llm")
    explanation = ""
    if call_llm and causal_chain:
        import json
        explanation = call_llm(
            f"Explain in 2 sentences why **{req.feature}** behaves as it does, "
            f"based on upstream causes:\n{json.dumps(causal_chain[:3], indent=2)}\n\n"
            f"Respond in Markdown. Cite specific statistics only.",
            temperature=0.1,
        )

    return RCATraverseResponse(
        feature=req.feature,
        causal_chain=causal_chain[:8],
        explanation=explanation,
    )
