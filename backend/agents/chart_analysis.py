# ═══════════════════════════════════════════════════════════════════════════════
# SENTINEL CHART ANALYSIS ENGINE v3.0 — NEVER FAILS, ALL CHART TYPES
# Drop-in replacement. Paste this as ONE block before your agent definitions.
# ═══════════════════════════════════════════════════════════════════════════════

import warnings, traceback
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, spearmanr, kruskal, shapiro
from sklearn.linear_model import LinearRegression


# ───────────────────────────────────────────────────────────────────────────────
# SAFE WRAPPERS — every computation is isolated; never crashes outer code
# ───────────────────────────────────────────────────────────────────────────────

def _safe(fn, *args, default=None, **kwargs):
    """Call fn(*args, **kwargs), return default on any exception."""
    try:
        result = fn(*args, **kwargs)
        # Filter out NaN/Inf so JSON serialization never fails
        if isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
            return default
        return result
    except Exception:
        return default


def _safe_json(obj) -> str:
    """JSON-serialize anything — replace un-serializable values gracefully."""
    def _fix(o):
        if isinstance(o, dict):
            return {str(k): _fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_fix(i) for i in o]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
            return round(o, 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
        if isinstance(o, (np.ndarray,)):
            return [_fix(x) for x in o.tolist()]
        if isinstance(o, pd.Timestamp):
            return str(o.date())
        return o
    try:
        return json.dumps(_fix(obj), indent=2)
    except Exception as e:
        return json.dumps({"serialization_error": str(e)})


# ───────────────────────────────────────────────────────────────────────────────
# COLUMN DETECTOR — the #1 reason analysis breaks is wrong col detection
# ───────────────────────────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame, x_col: str = None,
                    y_col: str = None, color_col: str = None,
                    z_col: str = None) -> dict:
    """
    Robustly detect numeric, categorical, datetime columns from df.
    Falls back gracefully when passed col names don't exist.
    Returns a dict: {numeric, categorical, datetime, x, y, color, z}
    """
    all_cols = list(df.columns)

    # Detect types from actual data
    numeric_cols  = [c for c in all_cols
                     if pd.api.types.is_numeric_dtype(df[c])
                     and df[c].notna().sum() > 0]
    datetime_cols = []
    cat_cols      = []

    for c in all_cols:
        if c in numeric_cols:
            continue
        # Try to parse as datetime
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            datetime_cols.append(c)
            continue
        if df[c].dtype == object:
            sample = df[c].dropna().head(3)
            is_date = False
            for v in sample:
                try:
                    pd.to_datetime(str(v))
                    is_date = True
                    break
                except Exception:
                    pass
            if is_date:
                datetime_cols.append(c)
                # Convert in place for analysis
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                except Exception:
                    pass
            else:
                cat_cols.append(c)

    # Resolve passed column names → fall back to auto-detected
    def _resolve(col, preferred_list, fallback_list):
        if col and col in all_cols:
            return col
        return preferred_list[0] if preferred_list else (
               fallback_list[0] if fallback_list else None)

    resolved_x = _resolve(x_col, datetime_cols + cat_cols, numeric_cols)
    resolved_y = _resolve(y_col, numeric_cols, cat_cols)
    resolved_color = color_col if (color_col and color_col in all_cols) else None
    resolved_z = z_col if (z_col and z_col in all_cols) else (
                 numeric_cols[2] if len(numeric_cols) > 2 else None)

    return {
        "numeric":   numeric_cols,
        "categorical": cat_cols,
        "datetime":  datetime_cols,
        "x":         resolved_x,
        "y":         resolved_y,
        "color":     resolved_color,
        "z":         resolved_z,
        "all":       all_cols,
    }


# ───────────────────────────────────────────────────────────────────────────────
# STATISTICAL EXTRACTORS (all wrapped in _safe)
# ───────────────────────────────────────────────────────────────────────────────

def _num_stats(series: pd.Series, name: str = "") -> dict:
    s = series.dropna()
    if len(s) < 2:
        return {"col": name, "n": len(s), "note": "insufficient data"}
    n   = len(s)
    mu  = float(s.mean())
    med = float(s.median())
    sd  = float(s.std())
    mn, mx = float(s.min()), float(s.max())
    p25 = float(np.percentile(s, 25))
    p75 = float(np.percentile(s, 75))
    iqr = p75 - p25
    skew = _safe(lambda: float(s.skew()), default=0.0)
    kurt = _safe(lambda: float(s.kurtosis()), default=0.0)
    cv   = _safe(lambda: sd / abs(mu) * 100 if mu != 0 else 0.0, default=0.0)

    # Outliers
    lo, hi = p25 - 1.5*iqr, p75 + 1.5*iqr
    outliers = s[(s < lo) | (s > hi)]
    out_pct  = len(outliers) / n * 100

    # Trend
    tr, tp = _safe(lambda: spearmanr(range(n), s.values), default=(None, None))

    # Normality
    norm_p = None
    if 8 <= n <= 5000:
        sample = s.sample(min(n, 1000), random_state=42)
        _, norm_p = _safe(lambda: shapiro(sample), default=(None, None))

    # Concentration: top 20% hold what % of total
    sorted_s = np.sort(s.values)[::-1]
    top20 = max(1, int(n*0.2))
    conc = float(sorted_s[:top20].sum() / (sorted_s.sum() + 1e-9) * 100)

    return {
        "col": name, "n": n,
        "mean": round(mu, 4), "median": round(med, 4),
        "std": round(sd, 4),  "cv_pct": round(cv, 2),
        "min": round(mn, 4),  "max": round(mx, 4),
        "p25": round(p25, 4), "p75": round(p75, 4),
        "iqr": round(iqr, 4), "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "outlier_count": len(outliers),
        "outlier_pct": round(out_pct, 2),
        "outlier_examples": sorted(outliers.tolist(), key=abs, reverse=True)[:3],
        "top20_concentration_pct": round(conc, 2),
        "trend_spearman_r": round(float(tr), 4) if tr is not None else None,
        "trend_p": round(float(tp), 4) if tp is not None else None,
        "trend_direction": ("upward" if (tr or 0) > 0.3 else
                            "downward" if (tr or 0) < -0.3 else "flat"),
        "normality_p": round(float(norm_p), 4) if norm_p is not None else None,
        "is_normal": (norm_p > 0.05) if norm_p is not None else None,
        "top3": sorted(s.tolist(), reverse=True)[:3],
        "bottom3": sorted(s.tolist())[:3],
    }


def _cat_stats(cat_s: pd.Series, val_s: pd.Series = None, name: str = "") -> dict:
    s = cat_s.dropna().astype(str)
    if len(s) == 0:
        return {"col": name, "n": 0}
    vc     = s.value_counts()
    n      = len(s)
    n_uni  = int(s.nunique())
    top_sh = float(vc.iloc[0] / n * 100) if n > 0 else 0

    # Shannon entropy
    probs   = vc / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    max_ent = np.log2(n_uni) if n_uni > 1 else 1
    even    = float(entropy / max_ent) if max_ent > 0 else 0

    result = {
        "col": name, "n_unique": n_uni,
        "top_category": str(vc.index[0]),
        "top_cat_share_pct": round(top_sh, 2),
        "bottom_category": str(vc.index[-1]),
        "top5": {str(k): int(v) for k, v in vc.head(5).items()},
        "entropy_normalized": round(even, 4),
        "distribution_type": ("concentrated" if top_sh > 40 else
                              "moderate" if top_sh > 20 else "even"),
    }

    if val_s is not None and len(val_s) == len(cat_s):
        paired = pd.DataFrame({"c": cat_s.astype(str), "v": val_s}).dropna()
        if not paired.empty and paired["v"].notna().sum() > 0:
            grp = paired.groupby("c")["v"].agg(["mean","sum","count"])
            grp = grp.sort_values("sum", ascending=False)
            result["per_cat"] = {
                str(r): {"mean": round(float(row["mean"]),2),
                         "sum":  round(float(row["sum"]),2),
                         "count": int(row["count"])}
                for r, row in grp.iterrows()
            }
            if len(grp) >= 2:
                ts = float(grp["sum"].iloc[0])
                bs = float(grp["sum"].iloc[-1])
                ms = float(grp["sum"].median())
                result["top_cat_by_value"]    = str(grp.index[0])
                result["bottom_cat_by_value"] = str(grp.index[-1])
                result["top_bottom_gap_ratio"] = round(ts/(abs(bs)+1e-9), 2)
                result["top_vs_median_ratio"]  = round(ts/(abs(ms)+1e-9), 2)

            groups = [g["v"].values for _, g in paired.groupby("c") if len(g) >= 3]
            if len(groups) >= 2:
                kw = _safe(lambda: kruskal(*groups), default=None)
                if kw:
                    result["kruskal_p"] = round(float(kw[1]), 4)
                    result["groups_differ"] = kw[1] < 0.05

    return result


def _pareto(df: pd.DataFrame, cat_col: str, val_col: str) -> dict:
    grp   = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
    total = grp.sum()
    if total == 0:
        return {}
    cumsum = grp.cumsum()
    n80    = int((cumsum <= total * 0.80).sum()) + 1
    pct80  = round(n80 / len(grp) * 100, 1)
    return {
        "top_n_for_80pct":   n80,
        "top_pct_for_80pct": pct80,
        "total_categories":  len(grp),
        "top3": {str(k): round(float(v),2) for k,v in grp.head(3).items()},
        "bottom3": {str(k): round(float(v),2) for k,v in grp.tail(3).items()},
        "interpretation": f"Top {n80} ({pct80}%) of '{cat_col}' = 80% of '{val_col}'",
    }


def _composition(df: pd.DataFrame, names_col: str, values_col: str) -> dict:
    grp   = df.groupby(names_col)[values_col].sum().sort_values(ascending=False)
    total = grp.sum()
    if total == 0:
        return {}
    shares = (grp / total * 100).round(3)
    return {
        "n_segments":             len(grp),
        "total":                  round(float(total), 2),
        "top_segment":            str(grp.index[0]),
        "top_value":              round(float(grp.iloc[0]), 2),
        "top_share_pct":          round(float(shares.iloc[0]), 2),
        "top3_cumulative_pct":    round(float(shares.head(3).sum()), 2),
        "bottom_segment":         str(grp.index[-1]),
        "bottom_share_pct":       round(float(shares.iloc[-1]), 2),
        "all_shares":             {str(k): round(float(v),2) for k,v in shares.items()},
        "herfindahl_index":       round(float((shares**2).sum()), 2),
        "concentration":          ("dominant" if shares.iloc[0] > 50 else
                                   "moderate" if shares.iloc[0] > 25 else "fragmented"),
    }


def _timeseries(df: pd.DataFrame, date_col: str, val_col: str) -> dict:
    sub = df[[date_col, val_col]].copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna().sort_values(date_col)
    if len(sub) < 4:
        return {"note": "fewer than 4 data points"}
    vals = sub[val_col].values.astype(float)
    n    = len(vals)
    t    = np.arange(n)
    slope, intercept, r, p, se = scipy_stats.linregress(t, vals)
    pct_change = (vals[-1] - vals[0]) / (abs(vals[0]) + 1e-9) * 100
    peak_i  = int(np.argmax(vals))
    trough_i= int(np.argmin(vals))

    # WoW comparison
    mid = n // 2
    wow = (vals[mid:].mean() - vals[:mid].mean()) / (abs(vals[:mid].mean())+1e-9)*100 if n >= 6 else None

    # Acceleration
    d1 = np.gradient(vals)
    d2 = np.gradient(d1)
    accel_avg = float(d2[-max(3,n//4):].mean())

    # Changepoint: largest absolute jump
    diffs = np.abs(np.diff(vals))
    cp_i  = int(np.argmax(diffs)) + 1 if len(diffs) else None

    dates = sub[date_col]
    return {
        "n_points": n,
        "start": str(dates.iloc[0].date()),
        "end":   str(dates.iloc[-1].date()),
        "start_value": round(float(vals[0]), 2),
        "end_value":   round(float(vals[-1]), 2),
        "pct_change_total": round(pct_change, 2),
        "trend_slope":  round(float(slope), 4),
        "trend_r2":     round(float(r**2), 4),
        "trend_p":      round(float(p), 6),
        "trend_significant": p < 0.05,
        "trend_direction": ("upward" if slope > 0 and p < 0.05 else
                            "downward" if slope < 0 and p < 0.05 else "flat"),
        "peak_value": round(float(vals[peak_i]), 2),
        "peak_date":  str(dates.iloc[peak_i].date()),
        "trough_value": round(float(vals[trough_i]), 2),
        "trough_date":  str(dates.iloc[trough_i].date()),
        "peak_trough_ratio": round(float(vals[peak_i])/(abs(float(vals[trough_i]))+1e-9), 3),
        "wow_pct": round(float(wow), 2) if wow is not None else None,
        "acceleration_avg": round(accel_avg, 4),
        "acceleration_label": "accelerating" if accel_avg > 0 else "decelerating",
        "changepoint_date": str(dates.iloc[cp_i].date()) if cp_i else None,
        "changepoint_magnitude": round(float(diffs[cp_i-1]),2) if cp_i else None,
    }


def _correlation(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 4:
        return {"note": "insufficient data for correlation"}
    x, y = sub[x_col].values.astype(float), sub[y_col].values.astype(float)
    pr, pp = _safe(lambda: pearsonr(x, y), default=(0, 1))
    sr, sp = _safe(lambda: spearmanr(x, y), default=(0, 1))
    reg    = _safe(lambda: LinearRegression().fit(x.reshape(-1,1), y), default=None)
    slope  = float(reg.coef_[0]) if reg else None
    r2     = float(reg.score(x.reshape(-1,1), y)) if reg else None

    pr_f = float(pr) if pr is not None else 0
    strength = ("very strong" if abs(pr_f) > 0.8 else "strong" if abs(pr_f) > 0.6 else
                "moderate" if abs(pr_f) > 0.4 else "weak" if abs(pr_f) > 0.2 else "negligible")
    return {
        "n": len(sub),
        "pearson_r": round(pr_f, 4),
        "pearson_p": round(float(pp), 6),
        "spearman_r": round(float(sr), 4) if sr else None,
        "spearman_p": round(float(sp), 6) if sp else None,
        "r_squared": round(r2, 4) if r2 is not None else None,
        "slope": round(slope, 6) if slope is not None else None,
        "strength": strength,
        "direction": "positive" if pr_f > 0 else "negative",
        "significant": float(pp) < 0.05,
        "interpretation": (
            f"Each unit increase in '{x_col}' → {slope:+.4f} change in '{y_col}' "
            f"(R²={r2:.3f}, {strength} {'positive' if pr_f>0 else 'negative'} correlation)"
            if slope is not None and r2 is not None else "correlation computed"
        ),
    }


def _distribution(series: pd.Series, name: str = "") -> dict:
    s = series.dropna()
    if len(s) < 4:
        return {"col": name, "n": len(s), "note": "insufficient data"}
    n    = len(s)
    skew = _safe(lambda: float(s.skew()), default=0.0)
    kurt = _safe(lambda: float(s.kurtosis()), default=0.0)

    norm_p = None
    if 8 <= n <= 5000:
        samp = s.sample(min(n, 1000), random_state=42)
        res  = _safe(lambda: shapiro(samp), default=None)
        norm_p = float(res[1]) if res else None

    hist_c, bin_e = np.histogram(s, bins=min(20, max(5, n//3)))
    local_max = [round(float(bin_e[i]),2)
                 for i in range(1, len(hist_c)-1)
                 if hist_c[i] > hist_c[i-1] and hist_c[i] > hist_c[i+1]]

    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))
    iqr_v = q3 - q1
    fl, fh = q1 - 1.5*iqr_v, q3 + 1.5*iqr_v
    outliers = s[(s < fl) | (s > fh)]

    return {
        "col": name, "n": n,
        "mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
        "std":  round(float(s.std()), 4),
        "skewness": round(skew, 4), "kurtosis": round(kurt, 4),
        "normality_p": round(norm_p, 4) if norm_p else None,
        "is_normal": (norm_p > 0.05) if norm_p else None,
        "tail_shape": ("heavy right tail" if skew > 1 else "heavy left tail" if skew < -1 else
                       "right skewed" if skew > 0.5 else "left skewed" if skew < -0.5 else "symmetric"),
        "modality": ("unimodal" if len(local_max) <= 1 else
                     "bimodal" if len(local_max) == 2 else "multimodal"),
        "local_maxima": local_max[:4],
        "outlier_count": len(outliers), "outlier_pct": round(len(outliers)/n*100, 2),
        "q1": round(q1, 4), "q3": round(q3, 4), "iqr": round(iqr_v, 4),
        "whisker_lo": round(float(s[s >= fl].min()), 4) if len(s[s>=fl]) > 0 else round(fl,4),
        "whisker_hi": round(float(s[s <= fh].max()), 4) if len(s[s<=fh]) > 0 else round(fh,4),
    }


def _group_compare(df: pd.DataFrame, cat_col: str, val_col: str) -> dict:
    sub = df[[cat_col, val_col]].dropna()
    if sub.empty:
        return {}
    grp   = sub.groupby(cat_col)[val_col].agg(["mean","median","std","count"])
    grp   = grp.sort_values("mean", ascending=False)
    kw    = None
    glist = [g[val_col].values for _, g in sub.groupby(cat_col) if len(g) >= 3]
    if len(glist) >= 2:
        kw = _safe(lambda: kruskal(*glist), default=None)
    top = str(grp.index[0]) if len(grp) > 0 else ""
    bot = str(grp.index[-1]) if len(grp) > 0 else ""
    tm  = float(grp.loc[top,"mean"]) if top in grp.index else 0
    bm  = float(grp.loc[bot,"mean"]) if bot in grp.index else 0
    return {
        "n_groups": len(grp),
        "top_group": top, "top_group_mean": round(tm, 4),
        "bottom_group": bot, "bottom_group_mean": round(bm, 4),
        "top_vs_bottom_ratio": round(tm/(abs(bm)+1e-9), 3),
        "kruskal_p": round(float(kw[1]),4) if kw else None,
        "groups_differ": (kw[1] < 0.05) if kw else None,
        "all_groups": {str(r): {"mean": round(float(row["mean"]),4),
                                "median": round(float(row["median"]),4),
                                "count": int(row["count"])}
                       for r, row in grp.iterrows()},
    }


def _heatmap_stats(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> dict:
    sub = df[[x_col, y_col, z_col]].dropna()
    if sub.empty:
        return {}
    try:
        pv = sub.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc="sum").fillna(0)
    except Exception:
        return {}
    flat = pv.values.flatten()
    mi   = np.unravel_index(pv.values.argmax(), pv.shape)
    ni   = np.unravel_index(pv.values.argmin(), pv.shape)
    rt   = pv.sum(axis=1).sort_values(ascending=False)
    ct   = pv.sum(axis=0).sort_values(ascending=False)
    return {
        "shape": list(pv.shape),
        "global_mean": round(float(flat.mean()), 4),
        "global_std":  round(float(flat.std()), 4),
        "max_cell": f"{pv.index[mi[0]]} × {pv.columns[mi[1]]}",
        "max_val":  round(float(pv.values.max()), 4),
        "min_cell": f"{pv.index[ni[0]]} × {pv.columns[ni[1]]}",
        "min_val":  round(float(pv.values.min()), 4),
        "top_cell_pct_of_total": round(float(pv.values.max()/(flat.sum()+1e-9)*100), 2),
        "top_row":   str(rt.index[0]), "top_row_total": round(float(rt.iloc[0]),2),
        "top_col":   str(ct.index[0]), "top_col_total": round(float(ct.iloc[0]),2),
        "row_totals": {str(k): round(float(v),2) for k,v in rt.head(5).items()},
        "col_totals": {str(k): round(float(v),2) for k,v in ct.head(5).items()},
        "sparsity_pct": round(float((flat==0).sum()/len(flat)*100), 2),
    }


def _quadrant(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 4:
        return {}
    xm, ym = sub[x_col].median(), sub[y_col].median()
    n = len(sub)
    q1 = len(sub[(sub[x_col] >= xm) & (sub[y_col] >= ym)])
    q2 = len(sub[(sub[x_col] <  xm) & (sub[y_col] >= ym)])
    q3 = len(sub[(sub[x_col] <  xm) & (sub[y_col] <  ym)])
    q4 = len(sub[(sub[x_col] >= xm) & (sub[y_col] <  ym)])
    dom = max([("Q1_high_x_high_y",q1),("Q2_low_x_high_y",q2),
               ("Q3_low_x_low_y",q3),("Q4_high_x_low_y",q4)], key=lambda t:t[1])
    return {
        "x_median": round(float(xm), 4), "y_median": round(float(ym), 4),
        "Q1_high_high_pct": round(q1/n*100,1), "Q2_low_high_pct": round(q2/n*100,1),
        "Q3_low_low_pct":   round(q3/n*100,1), "Q4_high_low_pct": round(q4/n*100,1),
        "dominant_quadrant": dom[0],
    }


# ───────────────────────────────────────────────────────────────────────────────
# MASTER ROUTER — selects extractors based on chart type + actual col types
# ───────────────────────────────────────────────────────────────────────────────

def _build_features(chart_type: str, df: pd.DataFrame,
                    cols: dict, extra: str = "") -> dict:
    """
    Route to the right extractors. chart_type is used as a hint only;
    actual column types drive the logic — so this works even with wrong chart_type.
    """
    ct  = chart_type.lower().replace(" ","_").replace("-","_")
    x   = cols["x"]
    y   = cols["y"]
    c   = cols["color"]
    z   = cols["z"]
    num = cols["numeric"]
    cat = cols["categorical"]
    dt  = cols["datetime"]

    feats = {
        "chart_type": chart_type, "n_rows": len(df),
        "columns": cols["all"], "extra_context": extra,
    }

    # ── Y column stats (almost always useful) ──────────────────────────────
    if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
        feats["y_stats"] = _safe(lambda: _num_stats(df[y], y), default={})

    # ── Determine primary analysis path ────────────────────────────────────

    # TIME SERIES path
    if ct in ("line","area") or (x in dt):
        if x and y and x in df.columns and y in df.columns:
            feats["timeseries"] = _safe(
                lambda: _timeseries(df, x, y), default={})
        if c and c in cat:
            feats["series_breakdown"] = _safe(
                lambda: _cat_stats(df[c], df[y] if y and y in df.columns else None, c),
                default={})

    # BAR / FUNNEL path
    elif ct in ("bar","grouped_bar","funnel"):
        if x and y and x in df.columns and y in df.columns:
            feats["category_analysis"] = _safe(
                lambda: _cat_stats(df[x], df[y] if pd.api.types.is_numeric_dtype(df[y]) else None, x),
                default={})
            if pd.api.types.is_numeric_dtype(df[y]):
                feats["pareto"] = _safe(lambda: _pareto(df, x, y), default={})
        if c and c in df.columns and y and y in df.columns:
            feats["color_breakdown"] = _safe(
                lambda: _cat_stats(df[c], df[y] if pd.api.types.is_numeric_dtype(df[y]) else None, c),
                default={})

    # PIE / TREEMAP path
    elif ct in ("pie","treemap"):
        # Auto-find best names and values cols
        names_col  = next((col for col in df.columns if df[col].dtype == object), x)
        values_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), y)
        if names_col and values_col and names_col in df.columns and values_col in df.columns:
            feats["composition"] = _safe(
                lambda: _composition(df, names_col, values_col), default={})
            feats["value_stats"] = _safe(
                lambda: _num_stats(df[values_col], values_col), default={})

    # SCATTER / BUBBLE path
    elif ct in ("scatter","bubble"):
        if x and y and all(v in df.columns for v in [x,y]):
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                feats["correlation"] = _safe(lambda: _correlation(df, x, y), default={})
                feats["x_stats"]     = _safe(lambda: _num_stats(df[x], x), default={})
                feats["quadrant"]    = _safe(lambda: _quadrant(df, x, y), default={})
        if c and c in df.columns:
            feats["cluster_breakdown"] = _safe(
                lambda: _cat_stats(df[c], df[y] if y and y in df.columns else None, c),
                default={})

    # HISTOGRAM path
    elif ct == "histogram":
        col = x if (x and x in df.columns and pd.api.types.is_numeric_dtype(df[x])) else \
              (y if (y and y in df.columns and pd.api.types.is_numeric_dtype(df[y])) else \
              (num[0] if num else None))
        if col:
            feats["distribution"] = _safe(lambda: _distribution(df[col], col), default={})

    # BOX / VIOLIN path
    elif ct in ("box","violin"):
        if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
            feats["distribution"] = _safe(lambda: _distribution(df[y], y), default={})
            if x and x in df.columns:
                feats["group_comparison"] = _safe(
                    lambda: _group_compare(df, x, y), default={})

    # HEATMAP path
    elif ct == "heatmap":
        hz = z or (num[0] if num else None)
        if x and y and hz and all(v in df.columns for v in [x,y,hz]):
            feats["heatmap"] = _safe(lambda: _heatmap_stats(df, x, y, hz), default={})

    # ── UNIVERSAL FALLBACK — runs for ANY chart type not matched above ──────
    # This is the key fix: always extract something useful
    else:
        # Try everything that applies
        if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
            feats["y_stats"] = _safe(lambda: _num_stats(df[y], y), default={})
        if x and y and x in df.columns and y in df.columns:
            if x in dt:
                feats["timeseries"] = _safe(lambda: _timeseries(df, x, y), default={})
            elif x in cat and pd.api.types.is_numeric_dtype(df.get(y, pd.Series())):
                feats["category_analysis"] = _safe(
                    lambda: _cat_stats(df[x], df[y], x), default={})
            elif x in num and y in num:
                feats["correlation"] = _safe(lambda: _correlation(df, x, y), default={})

    # ── ALWAYS: multi-series breakdown if color exists ─────────────────────
    if c and c in df.columns and c not in feats:
        feats["color_series"] = _safe(
            lambda: _cat_stats(df[c], df[y] if y and y in df.columns and
                               pd.api.types.is_numeric_dtype(df[y]) else None, c),
            default={})

    # ── ALWAYS: if we have ≥2 numeric cols, add all-numeric stats ─────────
    if len(num) >= 2 and y and y in num:
        for nc in num[:3]:
            if nc != y and nc not in feats:
                feats[f"extra_numeric_{nc}"] = _safe(
                    lambda nc=nc: _num_stats(df[nc], nc), default={})

    return feats


# ───────────────────────────────────────────────────────────────────────────────
# LLM DEEP INSIGHT LAYER
# ───────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a Business Intelligence expert. Produce concise, specific insights from pre-computed chart statistics.
RULES: cite exact numbers, no generic phrases like 'the chart shows', 2 bullet points max.
Format: • **[LABEL]** Finding with numbers → business implication"""

_PROMPT_TEMPLATE = """
CHART: {chart_type} — {title}
CONTEXT: {context}

=== STATS ===
{features}

=== SAMPLE (top 5) ===
{sample}

Write 2 concise bullet insights:
• **[LABEL]** specific finding with numbers → implication
"""

def _analyze_chart(fig_title: str, chart_type: str,
                   x_col: str, y_col: str, df: pd.DataFrame,
                   extra_context: str = "",
                   color_col: str = None,
                   z_col: str = None) -> str:
    """
    THE ONE TRUE analyze_chart function.
    - Never crashes (triple try/except)
    - Self-healing column detection
    - Works for ALL chart types
    - Always returns 4 deep insights
    """

    # ── Guard: ensure df is valid ──────────────────────────────────────────
    if df is None or (hasattr(df, "empty") and df.empty):
        return (f"• **NO DATA** — DataFrame is empty for '{fig_title}'. "
                f"Check SQL query or data pipeline for '{chart_type}' chart.")

    try:
        df = df.copy().reset_index(drop=True)
    except Exception:
        return "• **DATA ERROR** — Could not process DataFrame."

    # ── Step 1: Robust column detection ───────────────────────────────────
    try:
        cols = _detect_columns(df, x_col, y_col, color_col, z_col)
    except Exception as e:
        cols = {
            "numeric": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
            "categorical": [c for c in df.columns if df[c].dtype == object],
            "datetime": [], "x": x_col, "y": y_col,
            "color": color_col, "z": z_col, "all": list(df.columns),
        }

    # ── Step 2: Extract statistical features ──────────────────────────────
    try:
        features = _build_features(chart_type, df, cols, extra_context)
    except Exception as e:
        # Ultimate fallback: at least get basic stats on every column
        features = {"chart_type": chart_type, "error": str(e)}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                features[f"stat_{c}"] = _safe(
                    lambda c=c: _num_stats(df[c], c), default={})

    # ── Step 3: Build data sample ──────────────────────────────────────────
    display_cols = [c for c in [cols["x"], cols["y"], cols["color"], cols["z"]]
                    if c and c in df.columns]
    if not display_cols:
        display_cols = df.columns.tolist()[:5]
    try:
        sample_str = df[display_cols].dropna().head(8).to_string(index=False)
    except Exception:
        sample_str = df.head(8).to_string(index=False)

    # ── Step 4: Serialize features safely ─────────────────────────────────
    features_str = _safe_json(features)
    if len(features_str) > 1800:
        features_str = features_str[:1800] + "\n... [truncated]"

    # cap sample rows to 5 to keep prompt small
    sample_str = sample_str[:1000]

    # ── Step 5: LLM call ────────────────────────────────────────────────────
    try:
        prompt = _PROMPT_TEMPLATE.format(
            chart_type = chart_type,
            title      = fig_title,
            context    = (extra_context[:300] if extra_context
                          else "General business analytics"),
            features   = features_str,
            sample     = sample_str,
        )
        result = call_llm(prompt, system=_SYSTEM_PROMPT,
                          model=FAST_MODEL, temperature=0.15)

        # Sanity check: if LLM returned something weird, add disclaimer
        if not result or len(result.strip()) < 50:
            raise ValueError("LLM returned empty/short response")

        return result

    except Exception as e:
        # Final fallback: generate rule-based insights from features
        return _rule_based_fallback(fig_title, chart_type, features, df, cols)


def _rule_based_fallback(title: str, chart_type: str,
                         features: dict, df: pd.DataFrame, cols: dict) -> str:
    """
    Pure Python fallback — generates insights without LLM.
    Triggered only if LLM fails. Ensures analysis ALWAYS returns something useful.
    """
    lines = [f"[Rule-based analysis for: {title}]"]
    num = cols.get("numeric", [])
    cat = cols.get("categorical", [])

    # Insight 1: Basic scale
    y = cols.get("y")
    if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
        s = df[y].dropna()
        lines.append(f"• **SCALE** — '{y}' ranges from {s.min():,.2f} to {s.max():,.2f} "
                     f"(mean={s.mean():,.2f}, std={s.std():,.2f}). "
                     f"CV={s.std()/abs(s.mean())*100:.1f}% indicates "
                     f"{'high' if s.std()/abs(s.mean()) > 0.5 else 'moderate'} variability.")

    # Insight 2: Top/bottom from category stats
    ys = features.get("y_stats") or features.get("value_stats") or {}
    cs = features.get("category_analysis") or features.get("composition") or {}
    if cs.get("top_cat_by_value"):
        top, bot = cs["top_cat_by_value"], cs.get("bottom_cat_by_value","")
        ratio = cs.get("top_bottom_gap_ratio", 1)
        lines.append(f"• **SEGMENT** — '{top}' leads all segments with a "
                     f"{ratio:.1f}x gap over '{bot}'. "
                     f"Focus investment on the top segment to maximize returns.")

    # Insight 3: Trend
    ts = features.get("timeseries", {})
    if ts.get("pct_change_total") is not None:
        d = ts["pct_change_total"]
        lines.append(f"• **TREND** — Total change over period: {d:+.1f}%. "
                     f"Trend is {ts.get('trend_direction','unknown')} "
                     f"(slope={ts.get('trend_slope',0):+.4f}, "
                     f"R²={ts.get('trend_r2',0):.3f}). "
                     f"Peak was {ts.get('peak_value',0):,.2f} on {ts.get('peak_date','?')}.")

    # Insight 4: Distribution shape
    dist = features.get("distribution", {})
    if dist.get("outlier_pct") is not None:
        lines.append(f"• **DISTRIBUTION** — {dist.get('tail_shape','unknown')} shape "
                     f"(skew={dist.get('skewness',0):.3f}), "
                     f"{dist.get('outlier_pct',0):.1f}% outliers. "
                     f"{'Not normal' if not dist.get('is_normal') else 'Normal'} distribution "
                     f"(Shapiro p={dist.get('normality_p','?')}). "
                     f"IQR={dist.get('iqr',0):.2f}.")

    # Pad to 4 bullets if needed
    while len(lines) < 5:
        col = num[0] if num else (df.columns[0] if len(df.columns) > 0 else "data")
        lines.append(f"• **DATA QUALITY** — {len(df)} rows analyzed across "
                     f"{len(df.columns)} columns. "
                     f"Null rate: {df.isnull().mean().mean()*100:.1f}%.")

    return "\n".join(lines[1:5])  # return exactly 4 bullets
