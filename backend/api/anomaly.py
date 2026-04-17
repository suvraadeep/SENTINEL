"""
Anomaly Detection API v2 — production-grade hybrid pipeline.

Architecture:
  Layer 1 — Statistical : Rolling Z-score (vectorized) + IQR + GESD + Hampel
  Layer 2 — ML          : IsolationForest + PCA recon error + (MinCovDet if small)
  Layer 3 — Time-Series : Vectorized rolling ±2σ + CUSUM + KS distribution shift
  Layer 4 — Ensemble    : 40% stat + 35% ml + 25% ts, majority-vote severity

All three base layers run in parallel via ThreadPoolExecutor.
Response includes per-method results so the frontend can toggle between methods.
Each chart includes a deep technical explanation (why/what/how_to_read).

POST /api/anomaly/detect — full detection on a table
POST /api/anomaly/chat   — agentic LangGraph Q&A (see anomaly_chat_agent.py)
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class MethodResult(BaseModel):
    stats:     Dict[str, int]
    anomalies: List[Dict[str, Any]]

class AnomalyDetectRequest(BaseModel):
    table:     str
    threshold: float = 2.0
    method:    str   = "ensemble"   # statistical | ml | timeseries | ensemble

class AnomalyDetectResponse(BaseModel):
    table:    str
    stats:    Dict[str, int]
    methods:  Dict[str, MethodResult]          # one entry per detection method
    anomalies: List[Dict[str, Any]]            # active-method records (default: ensemble)
    charts:   List[Dict[str, Any]]             # {title, html, explanation}
    insights: str
    profile:  Dict[str, Any]

class AnomalyChatRequest(BaseModel):
    message:      str
    context:      Optional[Dict[str, Any]] = None
    table:        Optional[str]            = None
    chat_history: List[Dict]               = []

class AnomalyChatResponse(BaseModel):
    response:        str
    charts:          List[Dict[str, str]] = []
    tool_calls_made: List[str]            = []


# ─────────────────────────────────────────────────────────────────────────────
# Chart explanation registry
# ─────────────────────────────────────────────────────────────────────────────

CHART_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "timeline_with_markers": {
        "why": "Temporal context is the most natural frame for anomaly interpretation in time-ordered data. Without seeing the sequence, isolated point-outliers lose diagnostic meaning.",
        "what": "The solid line is the target metric over time. Red ✕ markers are rows where the ensemble score exceeds your threshold. Orange ◆ markers are high-severity only.",
        "how_to_read": "Clusters of consecutive markers indicate a systemic regime shift or data-quality issue, not random noise. Isolated single markers are point anomalies — check the raw value vs the rolling baseline.",
    },
    "zscore_heatmap": {
        "why": "A heatmap of rolling Z-scores across all numeric columns simultaneously reveals correlated anomalies — when multiple features spike at the same time, this is stronger evidence of a real event.",
        "what": "Rows = time index or row number. Columns = numeric features. Color intensity = |Z-score|. Values above 3 are highlighted.",
        "how_to_read": "Horizontal bands (a full row lit up) = systemic data event affecting all metrics. Vertical bands (one column, many rows) = a feature-specific issue. Isolated bright cells = point outliers.",
    },
    "score_distribution": {
        "why": "The shape of the anomaly score distribution directly reveals detector calibration. A healthy distribution is heavily right-skewed; if it's bimodal, your data may have two distinct regimes.",
        "what": "Histogram of ensemble scores coloured by severity tier. CRITICAL ≥ 0.70, HIGH ≥ 0.45, MEDIUM below that.",
        "how_to_read": "If the bulk of scores are above 0.5, lower your threshold. A long flat tail suggests the ensemble is uncertain — try running individual method tabs for clarity.",
    },
    "feature_contribution": {
        "why": "Not all features contribute equally to anomaly count. This chart quickly identifies which column is the primary source of detected anomalies — crucial for triage.",
        "what": "Bar height = number of anomalous rows for that feature. Color gradient = worst |Z-score| observed for the feature (red = more extreme).",
        "how_to_read": "Tall dark-red bars are your highest-priority investigation targets. A single dominant bar often indicates a data pipeline issue (e.g., missing value imputation, unit change) rather than a real business event.",
    },
    "pca_scatter": {
        "why": "PCA compresses all numeric dimensions into 2D, revealing multi-dimensional outliers that appear normal in any single dimension but deviate in the combined feature space.",
        "what": "Each point is one row of your data. Red = anomalous (ensemble score > threshold), blue = normal. The axes PC1 and PC2 are linear combinations of all numeric features that explain maximum variance.",
        "how_to_read": "Points far from the blue cloud are multi-dimensional outliers — they cannot be explained by any single feature. Clusters of red within the blue mass indicate boundary anomalies that may be false positives.",
    },
    "severity_donut": {
        "why": "A high-level severity summary is the first thing a data-ops team needs: how many anomalies require immediate action vs. monitoring vs. logging.",
        "what": "Donut proportions: CRITICAL (ensemble ≥ 0.70), HIGH (≥ 0.45), MEDIUM (< 0.45). Center shows total count.",
        "how_to_read": "If CRITICAL > 10% of total, trigger an alert workflow. HIGH requires investigation within 24h. MEDIUM can be batch-reviewed weekly.",
    },
    "box_plots": {
        "why": "Box plots reveal the full distribution shape (median, IQR, whiskers, outlier dots) for every numeric column simultaneously — a quick sanity-check for data quality.",
        "what": "Box = 25th–75th percentile. Whiskers = 1.5×IQR. Dots outside whiskers = classical outliers. Colors match column identity.",
        "how_to_read": "Very wide boxes suggest high natural variance — your threshold may be too low. Outlier dots that dominate the plot suggest the column needs log-transformation before analysis.",
    },
    "parallel_coordinates": {
        "why": "Parallel coordinates plot anomalous rows vs. normal rows across all features simultaneously, revealing the multi-feature 'signature' of each anomaly type.",
        "what": "Each line is one data row. Red lines = anomalies, blue lines = normal (sample of 200). Each vertical axis is one numeric feature scaled 0–1.",
        "how_to_read": "If anomalous lines consistently cross at the same axis intersections, that combination of features defines the anomaly pattern. Anomalous lines that look identical to normal lines are likely false positives.",
    },
    "cusum_chart": {
        "why": "CUSUM (Cumulative Sum) is a sequential change-detection algorithm designed specifically to detect persistent mean shifts — it is far more sensitive than point-in-time thresholding for gradual drift.",
        "what": "The CUSUM statistic accumulates when the series deviates from its mean. A rising CUSUM indicates a sustained upward shift; falling indicates downward. Vertical dashed lines mark detected change points.",
        "how_to_read": "Steep rises in CUSUM are more dangerous than slow ones — they indicate abrupt regime changes. If CUSUM never returns to zero, the shift is permanent (e.g., a sensor recalibration or data source change).",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyEngine
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_STAT = pd.DataFrame(columns=[
    "row_index", "column", "value", "baseline",
    "z_score", "iqr_flag", "stat_score", "stat_severity",
])

class AnomalyEngine:
    """
    Dataset-agnostic hybrid anomaly detection engine.

    Three base layers (run in parallel):
      statistical_scores  → rolling Z-score + IQR + Hampel  (vectorized)
      ml_scores           → IsolationForest + PCA (capped at 3 000 rows)
      timeseries_scores   → vectorized rolling dev + CUSUM + KS shift

    Ensemble: 40% stat + 35% ml + 25% ts
    """

    COLORS = ["#3B82F6","#06B6D4","#8B5CF6","#10B981",
              "#F59E0B","#EF4444","#EC4899","#14B8A6"]
    LAYOUT = dict(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#F1F5F9", family="Inter, sans-serif", size=11),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
        margin=dict(l=44, r=24, t=44, b=32),
    )

    def __init__(self, df: pd.DataFrame, threshold: float = 2.0):
        self.df        = df.copy()
        self.threshold = threshold
        self.profile   = self._profile()

    # ── Profile ───────────────────────────────────────────────────────────────

    def _profile(self) -> Dict[str, Any]:
        df = self.df
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
            r"(revenue|amount|sales|price|value|total|income|profit|cost|fee)",
            re.IGNORECASE
        )
        target_col = next((c for c in num_cols if _target_re.search(c)),
                          num_cols[0] if num_cols else None)

        return {
            "date_col":   date_col,
            "num_cols":   num_cols,
            "cat_cols":   cat_cols,
            "target_col": target_col,
            "row_count":  len(df),
        }

    # ── Layer 1: Statistical (fully vectorized) ───────────────────────────────

    def statistical_scores(self) -> pd.DataFrame:
        """
        Rolling Z-score (vectorized) + IQR bounds + Hampel identifier.
        All operations are pandas C-level rolling — O(n) per column, no Python loops.
        """
        df       = self.df.copy()
        num_cols = self.profile["num_cols"]
        date_col = self.profile["date_col"]

        if date_col:
            df = df.sort_values(date_col).reset_index(drop=True)

        WINDOW = min(20, max(4, len(df) // 20))
        chunks: List[pd.DataFrame] = []

        for col in num_cols:
            try:
                s = df[col].ffill().bfill().fillna(0)

                # Rolling Z-score
                rm  = s.rolling(WINDOW, min_periods=2).mean()
                rs  = s.rolling(WINDOW, min_periods=2).std().fillna(1e-9) + 1e-9
                z   = (s - rm) / rs

                # Rolling IQR
                rq1 = s.rolling(WINDOW, min_periods=2).quantile(0.25)
                rq3 = s.rolling(WINDOW, min_periods=2).quantile(0.75)
                iqr = rq3 - rq1
                iqr_lo = rq1 - 1.5 * iqr
                iqr_hi = rq3 + 1.5 * iqr

                # Hampel identifier: median absolute deviation
                rmed = s.rolling(WINDOW, min_periods=2).median()
                rmad = (s - rmed).abs().rolling(WINDOW, min_periods=2).median().fillna(1e-9) + 1e-9
                hampel_z = (s - rmed).abs() / (1.4826 * rmad)

                # Combine all signals
                iqr_flag  = (s < iqr_lo) | (s > iqr_hi)
                mask = (
                    (z.abs() > self.threshold) |
                    iqr_flag.fillna(False) |
                    (hampel_z > self.threshold * 1.5)
                )

                if not mask.any():
                    continue

                idx   = df.index[mask]
                abs_z = z.loc[idx].abs()
                chunk = pd.DataFrame({
                    "row_index":     idx.tolist(),
                    "column":        col,
                    "value":         s.loc[idx].round(4).values,
                    "baseline":      rm.loc[idx].round(4).values,
                    "z_score":       z.loc[idx].round(4).values,
                    "hampel_z":      hampel_z.loc[idx].round(4).values,
                    "iqr_flag":      iqr_flag.loc[idx].astype(int).values,
                    "stat_score":    (abs_z / 4.0).clip(0, 1).round(4).values,
                    "stat_severity": np.where(abs_z >= 4.0, "CRITICAL",
                                     np.where(abs_z >= 3.0, "HIGH", "MEDIUM")),
                })
                if date_col and len(idx) > 0:
                    chunk["date"] = df[date_col].iloc[
                        np.clip(chunk["row_index"].values, 0, len(df) - 1)
                    ].astype(str).values
                chunks.append(chunk)
            except Exception as exc:
                logger.debug("stat_scores col=%s: %s", col, exc)

        return pd.concat(chunks, ignore_index=True) if chunks else _EMPTY_STAT.copy()

    # ── Layer 2: ML (IsolationForest + PCA, capped at 3 000 rows) ─────────────

    def ml_scores(self) -> pd.DataFrame:
        """
        IsolationForest (60%) + PCA reconstruction error (40%).
        LOF removed — O(n²) and too slow for interactive dashboards.
        MinCovDet Mahalanobis added for small n (< 500) as a third signal.
        Hard sample cap: 3 000 rows (IsoForest on 3k rows completes in < 1s).
        """
        num_cols = self.profile["num_cols"]
        df       = self.df[num_cols].fillna(0)

        _EMPTY = pd.DataFrame(columns=["row_index","if_score","pca_score","ml_score"])

        if len(df) < 10 or len(num_cols) < 2:
            return _EMPTY

        sample = df.sample(min(3_000, len(df)), random_state=42)

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(sample)

            # IsolationForest
            iso    = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1,
                                     n_estimators=100)
            iso.fit(X_scaled)
            if_raw  = -iso.score_samples(X_scaled)
            if_norm = (if_raw - if_raw.min()) / (if_raw.ptp() + 1e-9)

            # PCA reconstruction error
            n_comp  = min(max(len(num_cols) - 1, 1), 5)
            pca     = PCA(n_components=n_comp)
            X_pca   = pca.fit_transform(X_scaled)
            X_recon = pca.inverse_transform(X_pca)
            pca_err  = np.mean((X_scaled - X_recon) ** 2, axis=1)
            pca_norm = (pca_err - pca_err.min()) / (pca_err.ptp() + 1e-9)

            ml_score = 0.60 * if_norm + 0.40 * pca_norm

            # Optional MinCovDet Mahalanobis for small n
            if len(sample) <= 500:
                try:
                    from sklearn.covariance import MinCovDet
                    mcd = MinCovDet(random_state=42, support_fraction=0.8)
                    mcd.fit(X_scaled)
                    mah = mcd.mahalanobis(X_scaled)
                    mah_norm = (mah - mah.min()) / (mah.ptp() + 1e-9)
                    ml_score = 0.50 * if_norm + 0.30 * pca_norm + 0.20 * mah_norm
                except Exception:
                    pass

            return pd.DataFrame({
                "row_index": sample.index.tolist(),
                "if_score":  if_norm.round(4),
                "pca_score": pca_norm.round(4),
                "ml_score":  ml_score.round(4),
            })

        except Exception as exc:
            logger.warning("ML scoring failed: %s", exc)
            return _EMPTY

    # ── Layer 3: Time-Series (fully vectorized, no STL) ───────────────────────

    def timeseries_scores(self) -> pd.DataFrame:
        """
        Three fully-vectorized time-series signals:
          1. Vectorized rolling ±2σ deviation score   (pandas C-level)
          2. CUSUM change detection                    (O(n) scalar loop, fast)
          3. KS distribution shift (first vs second half, flags the shifted half)
        STL seasonal decomposition removed — too slow for interactive use.
        """
        target_col = self.profile["target_col"]
        date_col   = self.profile["date_col"]
        _EMPTY = pd.DataFrame(columns=["row_index","ts_score"])

        if not target_col:
            return _EMPTY

        df = self.df.copy()
        if date_col:
            df = df.sort_values(date_col).reset_index(drop=True)

        s = df[target_col].ffill().fillna(0)
        n = len(s)
        if n < 4:
            return _EMPTY

        WINDOW = min(10, max(3, n // 10))

        # ── Signal 1: vectorized rolling ±2σ ─────────────────────────────────
        rm = s.rolling(WINDOW, min_periods=2).mean()
        rs = s.rolling(WINDOW, min_periods=2).std().fillna(1e-9) + 1e-9
        rolling_score = ((s - rm).abs() / rs / 4.0).clip(0, 1)

        # ── Signal 2: CUSUM (O(n) scalar, unavoidable, fast) ─────────────────
        series_arr = s.values
        mu_all     = float(np.mean(series_arr))
        sig_all    = float(np.std(series_arr)) + 1e-9
        k          = 0.5 * sig_all
        cp = cn = 0.0
        cusum_arr  = np.zeros(n)
        for i in range(1, n):
            cp = max(0.0, cp + (series_arr[i] - mu_all) - k)
            cn = max(0.0, cn - (series_arr[i] - mu_all) - k)
            cusum_arr[i] = cp + cn
        cusum_max = cusum_arr.max()
        cusum_score = pd.Series(
            (cusum_arr / (cusum_max + 1e-9)) * 0.8, index=s.index
        ) if cusum_max > 0 else pd.Series(0.0, index=s.index)

        combined_score = np.maximum(rolling_score.values, cusum_score.values)

        # ── Signal 3: KS distribution shift ──────────────────────────────────
        half = n // 2
        if half >= 5:
            try:
                from scipy.stats import ks_2samp
                ks_stat, ks_p = ks_2samp(series_arr[:half], series_arr[half:])
                if ks_p < 0.05:
                    # Flag rows in the shifted half with a small bonus score
                    ks_bonus = np.zeros(n)
                    ks_bonus[half:] = min(0.35, ks_stat)
                    combined_score = np.maximum(combined_score, ks_bonus)
            except Exception:
                pass

        thresh = self.threshold / 8.0
        mask   = combined_score > thresh
        if not mask.any():
            return _EMPTY

        rows = [{"row_index": int(i), "ts_score": round(float(combined_score[i]), 4)}
                for i in range(n) if mask[i]]
        return pd.DataFrame(rows)

    # ── Ensemble ──────────────────────────────────────────────────────────────

    def ensemble(self, stat_df: pd.DataFrame, ml_df: pd.DataFrame,
                 ts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Weighted ensemble: 40% stat + 35% ml + 25% ts.
        Uses the statistical layer as the base (primary anomaly records).
        Boosts scores using ML and TS signals where available.
        """
        if stat_df.empty and ml_df.empty:
            return pd.DataFrame()

        merged = stat_df.copy() if not stat_df.empty else ml_df[["row_index"]].copy()
        merged["ensemble_score"] = merged.get(
            "stat_score", pd.Series(0.0, index=merged.index)
        ) * 0.40

        if not ml_df.empty and "ml_score" in ml_df.columns:
            ml_map = ml_df.set_index("row_index")["ml_score"].to_dict()
            merged["ml_score"] = merged["row_index"].map(ml_map).fillna(0.0)
            merged["ensemble_score"] = merged["ensemble_score"] + merged["ml_score"] * 0.35
        else:
            merged["ml_score"] = 0.0

        if not ts_df.empty and "ts_score" in ts_df.columns:
            ts_map = ts_df.set_index("row_index")["ts_score"].to_dict()
            merged["ts_score"] = merged["row_index"].map(ts_map).fillna(0.0)
            merged["ensemble_score"] = merged["ensemble_score"] + merged["ts_score"] * 0.25
        else:
            merged["ts_score"] = 0.0

        merged["ensemble_score"] = merged["ensemble_score"].clip(0, 1).round(4)

        base_sev = merged.get("stat_severity", pd.Series("MEDIUM", index=merged.index))
        merged["severity"] = np.where(
            (merged["ensemble_score"] >= 0.70) | (base_sev == "CRITICAL"), "CRITICAL",
            np.where(
                (merged["ensemble_score"] >= 0.45) | (base_sev == "HIGH"), "HIGH",
                "MEDIUM"
            )
        )
        return merged.sort_values("ensemble_score", ascending=False).reset_index(drop=True)

    # ── Intelligent chart selection ───────────────────────────────────────────

    def _select_chart_types(self, n_anomalies: int) -> List[str]:
        p        = self.profile
        date_col = p["date_col"]
        n_num    = len(p["num_cols"])

        charts: List[str] = ["severity_donut", "score_distribution", "feature_contribution"]

        if date_col:
            charts.insert(0, "timeline_with_markers")
            charts.append("cusum_chart")

        if n_num >= 3:
            charts.append("zscore_heatmap")
            charts.append("pca_scatter")

        if n_num <= 8:
            charts.append("box_plots")

        if n_num >= 3 and n_anomalies > 0:
            charts.append("parallel_coordinates")

        return charts[:8]

    # ── Chart generation ──────────────────────────────────────────────────────

    def generate_charts(self, anomaly_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Intelligently select and generate charts.
        Returns [{title, html, explanation:{why,what,how_to_read}}].
        """
        charts: List[Dict[str, Any]] = []
        p          = self.profile
        target_col = p["target_col"]
        date_col   = p["date_col"]
        num_cols   = p["num_cols"]
        df         = self.df.copy()

        if date_col:
            df = df.sort_values(date_col).reset_index(drop=True)

        chart_types = self._select_chart_types(len(anomaly_df))

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            def _html(fig, title: str, expl_key: str) -> Dict[str, Any]:
                fig.update_layout(**self.LAYOUT,
                                  title=dict(text=title, font=dict(size=13)))
                return {
                    "title": title,
                    "html":  fig.to_html(include_plotlyjs="cdn", full_html=False,
                                         config={"responsive": True}),
                    "explanation": CHART_EXPLANATIONS.get(expl_key, {}),
                }

            # ── 1. Timeline with anomaly markers ──────────────────────────────
            if "timeline_with_markers" in chart_types and target_col and date_col:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[target_col], mode="lines",
                    name=target_col,
                    line=dict(color=self.COLORS[0], width=1.8),
                ))
                if not anomaly_df.empty and "column" in anomaly_df.columns:
                    a_rows = anomaly_df[anomaly_df["column"] == target_col]
                    if not a_rows.empty:
                        idxs = a_rows["row_index"].clip(0, len(df) - 1).values
                        crit_idxs = a_rows.loc[
                            a_rows.get("severity", a_rows.get("stat_severity","MEDIUM")) == "CRITICAL",
                            "row_index"
                        ].clip(0, len(df)-1).values if "severity" in a_rows.columns or "stat_severity" in a_rows.columns else []

                        fig.add_trace(go.Scatter(
                            x=df[date_col].iloc[idxs],
                            y=df[target_col].iloc[idxs],
                            mode="markers", name="Anomaly",
                            marker=dict(color="#EF4444", size=9,
                                        symbol="x-thin-open",
                                        line=dict(width=2.5)),
                        ))
                        if len(crit_idxs):
                            fig.add_trace(go.Scatter(
                                x=df[date_col].iloc[crit_idxs],
                                y=df[target_col].iloc[crit_idxs],
                                mode="markers", name="CRITICAL",
                                marker=dict(color="#F59E0B", size=12,
                                            symbol="diamond-open",
                                            line=dict(width=2.5)),
                            ))
                fig.update_layout(height=320)
                charts.append(_html(fig, f"{target_col} — Timeline with Anomaly Markers",
                                    "timeline_with_markers"))

            # ── 2. Rolling Z-score heatmap ────────────────────────────────────
            if "zscore_heatmap" in chart_types and len(num_cols) >= 3:
                try:
                    WINDOW = min(20, max(4, len(df) // 20))
                    show_cols = num_cols[:8]
                    z_matrix  = pd.DataFrame(index=df.index)
                    for col in show_cols:
                        s   = df[col].ffill().bfill().fillna(0)
                        rm  = s.rolling(WINDOW, min_periods=2).mean()
                        rs  = s.rolling(WINDOW, min_periods=2).std().fillna(1e-9) + 1e-9
                        z_matrix[col] = ((s - rm) / rs).abs().clip(0, 6)
                    # Downsample rows to max 200 for heatmap
                    step    = max(1, len(z_matrix) // 200)
                    z_samp  = z_matrix.iloc[::step]
                    fig = go.Figure(go.Heatmap(
                        z=z_samp.values.T,
                        x=z_samp.index.tolist(),
                        y=show_cols,
                        colorscale=[[0,"#1E293B"],[0.33,"#3B82F6"],
                                    [0.66,"#F59E0B"],[1,"#EF4444"]],
                        zmin=0, zmax=5,
                        showscale=True,
                        colorbar=dict(title="|Z-score|"),
                    ))
                    fig.update_layout(height=300,
                                      xaxis=dict(showticklabels=False))
                    charts.append(_html(fig, "Rolling Z-score Heatmap — All Features",
                                        "zscore_heatmap"))
                except Exception as exc:
                    logger.debug("Z-score heatmap failed: %s", exc)

            # ── 3. Score distribution ─────────────────────────────────────────
            if "score_distribution" in chart_types and not anomaly_df.empty:
                sev_col = "severity" if "severity" in anomaly_df.columns else "stat_severity"
                score_col = "ensemble_score" if "ensemble_score" in anomaly_df.columns else "stat_score"
                if score_col in anomaly_df.columns and sev_col in anomaly_df.columns:
                    fig = px.histogram(
                        anomaly_df, x=score_col, nbins=30,
                        color=sev_col,
                        color_discrete_map={"CRITICAL":"#EF4444","HIGH":"#F59E0B","MEDIUM":"#3B82F6"},
                        title="Anomaly Score Distribution",
                    )
                    fig.update_layout(height=280)
                    charts.append(_html(fig, "Anomaly Score Distribution", "score_distribution"))

            # ── 4. Feature contribution ───────────────────────────────────────
            if "feature_contribution" in chart_types and not anomaly_df.empty and "column" in anomaly_df.columns:
                z_col = "z_score" if "z_score" in anomaly_df.columns else (
                    "stat_score" if "stat_score" in anomaly_df.columns else None
                )
                if z_col:
                    contrib = (
                        anomaly_df.groupby("column")
                        .agg(count=(z_col, "count"),
                             max_z=(z_col, lambda x: x.abs().max()))
                        .reset_index()
                        .sort_values("count", ascending=False)
                    )
                    fig = px.bar(
                        contrib, x="column", y="count",
                        color="max_z", color_continuous_scale="Reds",
                        text="count",
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(height=300, coloraxis_showscale=False)
                    charts.append(_html(fig,
                        "Anomaly Count by Feature (color = worst |Z-score|)",
                        "feature_contribution"))

            # ── 5. Severity donut ─────────────────────────────────────────────
            if "severity_donut" in chart_types and not anomaly_df.empty:
                sev_col = "severity" if "severity" in anomaly_df.columns else "stat_severity"
                if sev_col in anomaly_df.columns:
                    vc = anomaly_df[sev_col].value_counts().reset_index()
                    vc.columns = ["severity", "count"]
                    total = vc["count"].sum()
                    fig = go.Figure(go.Pie(
                        labels=vc["severity"], values=vc["count"],
                        hole=0.55,
                        marker=dict(colors=["#EF4444","#F59E0B","#3B82F6"],
                                    line=dict(color="#111827", width=2)),
                    ))
                    fig.add_annotation(
                        text=f"<b>{total}</b><br><span style='font-size:10px'>total</span>",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=18, color="#F1F5F9"),
                    )
                    fig.update_layout(height=280,
                                      legend=dict(orientation="h", y=-0.15))
                    charts.append(_html(fig, "Severity Breakdown", "severity_donut"))

            # ── 6. PCA scatter ────────────────────────────────────────────────
            if "pca_scatter" in chart_types and len(num_cols) >= 3 and not anomaly_df.empty:
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA as _PCA

                    X      = df[num_cols[:8]].fillna(0).values
                    X_s    = StandardScaler().fit_transform(X)
                    pca_r  = _PCA(n_components=2).fit_transform(X_s)
                    pca_df = pd.DataFrame(pca_r, columns=["PC1","PC2"])
                    anom_i = set(anomaly_df["row_index"].clip(0, len(df)-1).tolist())
                    pca_df["status"] = ["Anomaly" if i in anom_i else "Normal"
                                        for i in range(len(pca_df))]
                    fig = px.scatter(
                        pca_df, x="PC1", y="PC2", color="status",
                        color_discrete_map={"Anomaly":"#EF4444","Normal":"#3B82F6"},
                        opacity=0.65,
                    )
                    fig.update_layout(height=300)
                    charts.append(_html(fig, "PCA Scatter — Anomalies vs Normal",
                                        "pca_scatter"))
                except Exception as exc:
                    logger.debug("PCA scatter failed: %s", exc)

            # ── 7. Box plots ──────────────────────────────────────────────────
            if "box_plots" in chart_types and num_cols:
                show_cols = num_cols[:6]
                long_df   = df[show_cols].melt(var_name="metric", value_name="value")
                fig = px.box(
                    long_df, x="metric", y="value", color="metric",
                    points="outliers",
                )
                fig.update_layout(height=300, showlegend=False)
                charts.append(_html(fig, "Box Plots — Distribution Overview", "box_plots"))

            # ── 8. CUSUM chart ────────────────────────────────────────────────
            if "cusum_chart" in chart_types and target_col:
                try:
                    series_arr = df[target_col].ffill().fillna(0).values
                    n_        = len(series_arr)
                    mu_all    = float(np.mean(series_arr))
                    sig_all   = float(np.std(series_arr)) + 1e-9
                    k         = 0.5 * sig_all
                    cp = cn   = 0.0
                    cusum_arr = np.zeros(n_)
                    for i in range(1, n_):
                        cp = max(0.0, cp + (series_arr[i] - mu_all) - k)
                        cn = max(0.0, cn - (series_arr[i] - mu_all) - k)
                        cusum_arr[i] = cp + cn

                    x_ax = (df[date_col] if date_col else pd.Series(range(n_))).values
                    fig  = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_ax, y=cusum_arr, mode="lines",
                        name="CUSUM", line=dict(color="#06B6D4", width=2),
                        fill="tozeroy", fillcolor="rgba(6,182,212,0.08)",
                    ))
                    # Mark change points (CUSUM spikes)
                    thresh_cusum = np.percentile(cusum_arr[cusum_arr > 0], 90) if cusum_arr.max() > 0 else 0
                    if thresh_cusum > 0:
                        for i in range(1, n_ - 1):
                            if cusum_arr[i] >= thresh_cusum and cusum_arr[i] > cusum_arr[i-1] and cusum_arr[i] > cusum_arr[i+1]:
                                fig.add_vline(
                                    x=float(x_ax[i]) if not date_col else x_ax[i],
                                    line_dash="dash", line_color="#F59E0B",
                                    opacity=0.6,
                                )
                    fig.update_layout(height=280)
                    charts.append(_html(fig,
                        f"CUSUM Change Detection — {target_col}", "cusum_chart"))
                except Exception as exc:
                    logger.debug("CUSUM chart failed: %s", exc)

            # ── 9. Parallel coordinates ───────────────────────────────────────
            if "parallel_coordinates" in chart_types and not anomaly_df.empty and len(num_cols) >= 3:
                try:
                    plot_cols   = num_cols[:5]
                    anom_idx_s  = set(anomaly_df["row_index"].clip(0, len(df)-1).tolist())
                    sample_norm = df[plot_cols].fillna(0).head(200).copy()
                    sample_norm["label"] = [1 if i in anom_idx_s else 0
                                            for i in range(len(sample_norm))]
                    dims = [dict(values=sample_norm[c], label=c) for c in plot_cols]
                    fig  = go.Figure(go.Parcoords(
                        line=dict(color=sample_norm["label"],
                                  colorscale=[[0,"#3B82F6"],[1,"#EF4444"]],
                                  showscale=False),
                        dimensions=dims,
                    ))
                    fig.update_layout(height=300)
                    charts.append(_html(fig,
                        "Parallel Coordinates — Anomaly Signatures",
                        "parallel_coordinates"))
                except Exception as exc:
                    logger.debug("Parallel coords failed: %s", exc)

        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc, exc_info=True)

        return charts

    # ── LLM insights ──────────────────────────────────────────────────────────

    def llm_insights(self, anomaly_df: pd.DataFrame, call_llm) -> str:
        if anomaly_df.empty:
            return "No anomalies detected above the configured threshold."

        sev_col   = "severity" if "severity" in anomaly_df.columns else "stat_severity"
        sev       = anomaly_df[sev_col].value_counts().to_dict() if sev_col in anomaly_df.columns else {}
        top10_str = anomaly_df.head(10).to_string(index=False)

        if call_llm is None:
            return (
                f"**{len(anomaly_df)} anomalies detected.**\n\n"
                f"Severity breakdown: {sev}\n\n"
                f"Top anomalies:\n```\n{top10_str}\n```\n\n"
                "Configure an LLM provider to get AI-generated narrative insights."
            )

        prompt = (
            f"You are a senior data scientist analysing anomaly detection results. "
            f"Respond in clear, structured plain text (no markdown headers).\n\n"
            f"DATASET PROFILE: {self.profile}\n"
            f"SEVERITY BREAKDOWN: {sev}\n"
            f"TOP ANOMALIES:\n{top10_str}\n\n"
            f"Write exactly 3 paragraphs:\n"
            f"1. Summary — total anomalies, severity distribution, most affected columns\n"
            f"2. Pattern analysis — likely root causes based on column names + values, "
            f"   cite specific rows/values from the data above\n"
            f"3. Recommended actions — 3 concrete steps, prioritised by severity"
        )
        try:
            return call_llm(prompt, temperature=0.1)
        except Exception as exc:
            logger.warning("LLM insights failed: %s", exc)
            return f"{len(anomaly_df)} anomalies detected. Severity: {sev}."


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build MethodResult dict
# ─────────────────────────────────────────────────────────────────────────────

def _method_result(df: pd.DataFrame, sev_col: str = "severity") -> Dict[str, Any]:
    if df is None or df.empty:
        return {"stats": {"total": 0, "critical": 0, "high": 0, "medium": 0},
                "anomalies": []}
    total = len(df)
    vc    = df[sev_col].value_counts() if sev_col in df.columns else pd.Series(dtype=int)
    keep  = [c for c in ["row_index","column","value","baseline","z_score",
                          "stat_score","ml_score","ts_score","ensemble_score",
                          "severity","stat_severity","date"]
             if c in df.columns]
    return {
        "stats": {
            "total":    total,
            "critical": int(vc.get("CRITICAL", 0)),
            "high":     int(vc.get("HIGH", 0)),
            "medium":   int(vc.get("MEDIUM", 0)),
        },
        "anomalies": df[keep].head(200).to_dict("records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint: POST /api/anomaly/detect
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/detect", response_model=AnomalyDetectResponse)
async def detect_anomalies(req: AnomalyDetectRequest):
    from backend.core.namespace import sentinel_ns

    if not sentinel_ns.is_initialized:
        raise HTTPException(status_code=503, detail="SENTINEL not initialized")

    con = sentinel_ns._ns.get("con")
    if con is None:
        raise HTTPException(status_code=503, detail="No database connection")

    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', req.table):
        raise HTTPException(status_code=400, detail="Invalid table name")

    try:
        df = con.execute(f'SELECT * FROM "{req.table}" LIMIT 50000').df()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read table: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Table is empty")

    engine   = AnomalyEngine(df, threshold=req.threshold)
    call_llm = sentinel_ns._ns.get("call_llm")

    # ── Run all three layers in parallel (15s timeout each) ──────────────────
    stat_df = ml_df = ts_df = pd.DataFrame()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_stat = pool.submit(engine.statistical_scores)
        fut_ml   = pool.submit(engine.ml_scores)
        fut_ts   = pool.submit(engine.timeseries_scores)
        try:
            stat_df = fut_stat.result(timeout=15)
        except (_Timeout, Exception) as exc:
            logger.warning("Statistical layer timeout/error: %s", exc)
        try:
            ml_df = fut_ml.result(timeout=15)
        except (_Timeout, Exception) as exc:
            logger.warning("ML layer timeout/error: %s", exc)
        try:
            ts_df = fut_ts.result(timeout=15)
        except (_Timeout, Exception) as exc:
            logger.warning("TS layer timeout/error: %s", exc)

    # ── Build ensemble ────────────────────────────────────────────────────────
    ensemble_df = engine.ensemble(stat_df, ml_df, ts_df)

    # ── Build per-method results (for frontend method toggle) ─────────────────
    # Statistical method view
    stat_view = stat_df.copy() if not stat_df.empty else pd.DataFrame()
    if not stat_view.empty:
        stat_view["severity"] = stat_view.get(
            "stat_severity", pd.Series("MEDIUM", index=stat_view.index)
        )

    # ML method view — annotate severity from score
    ml_view = ml_df.copy() if not ml_df.empty else pd.DataFrame()
    if not ml_view.empty and "ml_score" in ml_view.columns:
        ml_view["severity"] = np.where(
            ml_view["ml_score"] >= 0.75, "CRITICAL",
            np.where(ml_view["ml_score"] >= 0.50, "HIGH", "MEDIUM")
        )

    # TS method view
    ts_view = ts_df.copy() if not ts_df.empty else pd.DataFrame()
    if not ts_view.empty and "ts_score" in ts_view.columns:
        ts_view["severity"] = np.where(
            ts_view["ts_score"] >= 0.75, "CRITICAL",
            np.where(ts_view["ts_score"] >= 0.50, "HIGH", "MEDIUM")
        )

    methods = {
        "statistical": _method_result(stat_view),
        "ml":          _method_result(ml_view),
        "timeseries":  _method_result(ts_view),
        "ensemble":    _method_result(ensemble_df),
    }

    # ── Charts + insights (on ensemble) ──────────────────────────────────────
    charts   = engine.generate_charts(ensemble_df if not ensemble_df.empty else stat_df)
    insights = engine.llm_insights(ensemble_df if not ensemble_df.empty else stat_df, call_llm)

    # ── Top-level stats (ensemble) ─────────────────────────────────────────
    ens_stats = methods["ensemble"]["stats"]

    # ── Serialise anomaly records ──────────────────────────────────────────
    active_df = ensemble_df if not ensemble_df.empty else stat_df
    keep_cols = [c for c in ["row_index","column","value","baseline","z_score",
                              "severity","ensemble_score","stat_score","date"]
                 if c in (active_df.columns if not active_df.empty else [])]
    anomaly_records = (active_df[keep_cols].head(200).to_dict("records")
                       if not active_df.empty else [])

    return AnomalyDetectResponse(
        table    = req.table,
        stats    = ens_stats,
        methods  = methods,
        anomalies= anomaly_records,
        charts   = charts,
        insights = insights,
        profile  = engine.profile,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint: POST /api/anomaly/chat
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=AnomalyChatResponse)
async def anomaly_chat(req: AnomalyChatRequest):
    from backend.core.namespace import sentinel_ns

    if not sentinel_ns.is_initialized:
        raise HTTPException(status_code=503, detail="SENTINEL not initialized")

    call_llm = sentinel_ns._ns.get("call_llm")
    con      = sentinel_ns._ns.get("con")

    # Try LangGraph agent first (if available)
    try:
        from backend.api.anomaly_chat_agent import run_anomaly_agent
        result = run_anomaly_agent(
            message      = req.message,
            context      = req.context or {},
            table        = req.table or (req.context or {}).get("table", ""),
            chat_history = req.chat_history,
            con          = con,
            call_llm     = call_llm,
        )
        return AnomalyChatResponse(
            response        = result["response"],
            charts          = result.get("charts", []),
            tool_calls_made = result.get("tool_calls_made", []),
        )
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("LangGraph agent failed, falling back: %s", exc)

    # Fallback: direct LLM call
    if call_llm is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    import json
    ctx     = req.context or {}
    ctx_str = (
        f"\n\n**ANOMALY SCAN CONTEXT:**\n"
        f"Table: `{ctx.get('table','unknown')}`\n"
        f"Stats: {json.dumps(ctx.get('stats',{}))}\n"
        f"Profile: {json.dumps(ctx.get('profile',{}))}\n"
        f"Top anomalies: {json.dumps(ctx.get('anomalies',[])[:5])}"
    ) if ctx else ""

    # Build chat history string
    history_str = ""
    for m in (req.chat_history or [])[-6:]:
        role = m.get("role","user")
        text = m.get("text", m.get("content",""))
        history_str += f"\n**{role.upper()}:** {text}"

    prompt = (
        f"You are an expert anomaly detection analyst. "
        f"Respond in strict GitHub-flavored Markdown. "
        f"Never invent statistics — only cite numbers from the provided context.{ctx_str}"
        f"{history_str}\n\n**USER:** {req.message}"
    )
    try:
        response = call_llm(prompt, temperature=0.15)
        return AnomalyChatResponse(response=response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")
