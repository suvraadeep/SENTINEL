"""
SENTINEL Prediction Agent — multi-model ML prediction.

Auto-detects the task type and selects the best model:
  • Regression    — Ridge, Random Forest, Gradient Boosting, XGBoost (if installed)
  • Classification — LogisticRegression, RandomForestClassifier, GBClassifier
  • Time-series  — defers to forecast_agent if prophet available; falls back to ARIMA

Feature importance via permutation importance (SHAP-like without requiring SHAP).
Handles overprice / undervalue detection, premium estimation, and general prediction.
"""

import json
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, f1_score, roc_auc_score, classification_report,
)
from sklearn.inspection import permutation_importance

# Try XGBoost (optional)
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ── Constants ────────────────────────────────────────────────────────────────
_EXCL_PATTERNS = frozenset(["_id", "timestamp", "date", "time", "url",
                              "description", "name", "address", "email"])
_ID_LIKE = lambda c: any(p in c.lower() for p in _EXCL_PATTERNS)


# ── Schema / table discovery ─────────────────────────────────────────────────
def _discover_main_table(query_lower: str) -> str:
    try:
        tables_df = run_sql("SHOW TABLES")
        available = tables_df.iloc[:, 0].tolist()
    except Exception:
        available = []

    priority = ["sales", "transactions", "listings", "properties",
                "records", "data", "facts", "main", "events", "orders"]
    for p in priority:
        for t in available:
            if p in t.lower() and "modified" not in t.lower():
                return t

    for t in available:
        if "summary" not in t.lower() and "view" not in t.lower() and "modified" not in t.lower():
            return t

    if available:
        return available[0]
    raise ValueError("No tables found in database. Please upload a dataset first.")


def _load_data(table: str, limit: int = 8000) -> pd.DataFrame:
    try:
        return run_sql(f"SELECT * FROM {table} LIMIT {limit}")
    except Exception as e:
        raise ValueError(f"Cannot load table '{table}': {e}")


def _encode_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Encode categorical columns.
    - Low cardinality (≤ 20 unique values): LabelEncoder (ordinal).
    - High cardinality (> 20): drop the column — it would cause overfitting
      and LabelEncoder assigns arbitrary ordinal values that mislead tree models.
    """
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        n_uniq = df[col].nunique()
        if n_uniq > 50:
            # Too many categories — drop to prevent overfitting / noise
            df.drop(columns=[col], inplace=True, errors="ignore")
            continue
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("__NA__"))
        except Exception:
            df.drop(columns=[col], inplace=True, errors="ignore")
    return df


def _pick_target(df: pd.DataFrame, query: str) -> str:
    q = query.lower()
    num_cols = [c for c in df.select_dtypes(include="number").columns if not _ID_LIKE(c)]

    for col in num_cols:
        if col.lower() in q:
            return col

    for hint in ["final_amount", "revenue", "price", "amount", "value",
                 "sales", "total", "cost", "fee", "score", "rating"]:
        for col in num_cols:
            if hint in col.lower():
                return col

    return num_cols[0] if num_cols else None


def _detect_task_type(df: pd.DataFrame, target_col: str, query_lower: str) -> str:
    """
    Classify as 'regression', 'classification', or 'timeseries'.
    """
    # Explicit time-series keywords
    if any(k in query_lower for k in ["forecast", "next month", "next week",
                                        "next quarter", "future", "predict over time"]):
        return "timeseries"

    # Check target column cardinality
    n_unique = df[target_col].nunique()
    n_rows   = len(df)

    # Binary / low-cardinality → classification
    if n_unique <= 10 or (n_unique / n_rows < 0.02 and n_unique <= 50):
        if any(k in query_lower for k in ["classify", "class", "category",
                                           "churn", "convert", "succeed", "fail",
                                           "yes/no", "true/false", "binary"]):
            return "classification"
        if n_unique == 2:
            return "classification"

    return "regression"


def _select_model(task: str, n_rows: int, query_lower: str):
    """Return (model_obj, model_name) based on task and dataset size.

    Adaptive regularization prevents overfitting:
      - Ridge with cross-validated alpha (RidgeCV) — default for regression
      - Random Forest: max_depth and min_samples_leaf scale with dataset size
      - XGBoost: subsample + colsample_bytree for large datasets only
      - All trees capped so bias–variance is balanced for the actual dataset size
    """
    is_xgb_request = any(k in query_lower for k in ["xgb", "xgboost", "gbm"])
    is_forest      = any(k in query_lower for k in ["forest", "random forest", "tree"])
    is_gradient    = any(k in query_lower for k in ["gradient", "boost"])

    # Adaptive tree depth: deeper only when there's enough data
    # Heuristic: depth = floor(log2(n_rows / 50)) clamped to [3, 8]
    adaptive_depth    = max(3, min(8, int(np.log2(max(n_rows / 50, 1)) + 1)))
    # min_samples_leaf: at least 1% of data or 5, whichever is larger
    min_leaf          = max(5, n_rows // 100)

    if task == "regression":
        # XGBoost: only for explicit request or very large datasets
        if _HAS_XGB and (is_xgb_request or n_rows > 10_000):
            return (
                xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=min(adaptive_depth, 6),
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_weight=max(1, min_leaf // 5),
                    random_state=42, verbosity=0,
                ),
                "XGBoost",
            )

        if is_forest or is_gradient or n_rows > 3_000:
            return (
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=adaptive_depth,
                    min_samples_leaf=min_leaf,
                    max_features="sqrt",
                    max_samples=min(1.0, 5_000 / max(n_rows, 1)) if n_rows > 5_000 else None,
                    random_state=42,
                    n_jobs=-1,
                ),
                "Random Forest",
            )

        # Default: cross-validated Ridge — automatically selects alpha, no overfitting
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0]
        return (RidgeCV(alphas=alphas, cv=min(5, max(2, n_rows // 50))), "Ridge (CV)")

    else:  # classification
        if _HAS_XGB and (is_xgb_request or n_rows > 10_000):
            return (
                xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=min(adaptive_depth, 6),
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42, verbosity=0,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
                "XGBoost Classifier",
            )

        if n_rows > 3_000:
            return (
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=adaptive_depth,
                    min_samples_leaf=min_leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                "Random Forest Classifier",
            )

        return (
            LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs",
                random_state=42, class_weight="balanced",
            ),
            "Logistic Regression",
        )



def _get_feature_importance(model, X, y, feature_names, task, n_repeats=10) -> pd.DataFrame:
    """
    Compute permutation feature importance (works for any model).
    Falls back to model.coef_/feature_importances_ for speed.
    """
    importances = None
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
    except Exception:
        pass

    if importances is None:
        try:
            scoring = "r2" if task == "regression" else "accuracy"
            perm = permutation_importance(
                model, X, y, n_repeats=5, random_state=42, scoring=scoring
            )
            importances = perm.importances_mean
        except Exception:
            importances = np.ones(len(feature_names))

    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    total = df["importance"].abs().sum()
    df["importance_pct"] = (df["importance"].abs() / (total + 1e-9)) * 100
    return df.head(15)


# ── Chart helpers ─────────────────────────────────────────────────────────────
_COLORS = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#EF4444",
           "#06B6D4", "#EC4899", "#14B8A6", "#F97316", "#6366F1"]

def _bar_chart(title: str, x_vals, y_vals, color_by_sign=True) -> go.Figure:
    colors = (
        ["#10B981" if v >= 0 else "#EF4444" for v in y_vals]
        if color_by_sign else _COLORS[:len(x_vals)]
    )
    fig = go.Figure(go.Bar(
        x=list(x_vals), y=list(y_vals),
        marker_color=colors,
        text=[f"{v:+,.3g}" if color_by_sign else f"{v:.1f}%" for v in y_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, template="sentinel", height=430,
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#F1F5F9"),
        xaxis=dict(tickfont=dict(color="#94A3B8")),
        yaxis=dict(tickfont=dict(color="#94A3B8")),
    )
    return fig


# ── Scenario prediction — extract conditions from query and predict ──────────
def _compute_scenario_prediction(model, scaler, df_raw, df_enc, available_feats,
                                  target_col, query_lower):
    """
    Parse the user query for numeric/categorical conditions and compute a
    concrete model prediction for that scenario.

    Returns (predicted_value, description_str, breakdown_list) or (None, '', []).
    breakdown_list: [{"feature": ..., "value": ..., "delta": ..., "base": ...}]
    """
    import re as _re

    # Build a median-based baseline feature vector
    median_row = pd.DataFrame(df_enc[available_feats]).median().values.reshape(1, -1)
    baseline_pred = float(model.predict(scaler.transform(median_row))[0])

    # Parse conditions from query: "4+ bedrooms", "2+ garages", ">= 3 baths"
    # Also handles: "newly built", "is new", boolean-like flags
    condition_patterns = [
        # "4+ bedrooms" / "4 or more bedrooms"
        _re.compile(r'(\d+)\+?\s*(bedroom|bed|bath|garage|stor|room|floor|car)',  _re.I),
        # ">= 3 baths" / "> 2000 sqft"
        _re.compile(r'[>≥]=?\s*(\d+[\.,]?\d*)\s*(sqft|sq_ft|square|area|lot|price|age)', _re.I),
        # "being newly built" / "is new" / "new construction"
        _re.compile(r'(new(?:ly)?\s*(?:built|construction)|is\s+new|brand\s+new)', _re.I),
    ]

    conditions_found = []
    scenario_row = median_row.copy()

    for pat in condition_patterns:
        for m in pat.finditer(query_lower):
            groups = m.groups()
            if len(groups) >= 2 and groups[0].replace('.', '').replace(',', '').isdigit():
                value = float(groups[0].replace(',', ''))
                keyword = groups[1].lower()
                # Find matching feature column
                for i, feat in enumerate(available_feats):
                    feat_lower = feat.lower()
                    if keyword[:4] in feat_lower:
                        scenario_row[0, i] = value
                        conditions_found.append((feat, value))
                        break
            elif len(groups) >= 1:
                # Boolean "newly built" → set matching binary column to 1
                for i, feat in enumerate(available_feats):
                    feat_lower = feat.lower()
                    if any(k in feat_lower for k in ["new", "built", "year_built", "yearbuilt"]):
                        # For year_built, set to recent year; for binary, set to 1
                        if "year" in feat_lower:
                            scenario_row[0, i] = df_raw[feat].max() if feat in df_raw.columns else 2024
                        else:
                            scenario_row[0, i] = 1
                        conditions_found.append((feat, float(scenario_row[0, i])))
                        break

    # If no conditions found, use percentile-75 for top features as the scenario
    if not conditions_found:
        # Use top-3 feature importances set to 75th percentile
        if hasattr(model, 'feature_importances_'):
            imp_order = np.argsort(model.feature_importances_)[::-1]
        elif hasattr(model, 'coef_'):
            imp_order = np.argsort(np.abs(model.coef_.flatten()))[::-1]
        else:
            imp_order = list(range(len(available_feats)))

        top_feats = imp_order[:min(3, len(imp_order))]
        for idx in top_feats:
            feat = available_feats[idx]
            p75 = float(df_enc[feat].quantile(0.75))
            scenario_row[0, idx] = p75
            conditions_found.append((feat, round(p75, 2)))

    # Predict the scenario
    scenario_pred = float(model.predict(scaler.transform(scenario_row))[0])

    # Build breakdown: what each condition contributed vs baseline
    breakdown = []
    running = baseline_pred
    for feat, val in conditions_found:
        # Predict with just this one feature changed from baseline
        test_row = median_row.copy()
        feat_idx = available_feats.index(feat)
        test_row[0, feat_idx] = val
        single_pred = float(model.predict(scaler.transform(test_row))[0])
        delta = single_pred - baseline_pred
        breakdown.append({
            "feature": feat,
            "value": round(val, 2),
            "delta": round(delta, 2),
            "base": round(baseline_pred, 2),
        })
        running += delta

    # Build description
    cond_parts = [f"{feat}={val:.0f}" if val == int(val) else f"{feat}={val:.2f}"
                  for feat, val in conditions_found]
    desc = f"For scenario with {', '.join(cond_parts)}" if cond_parts else "For median-feature scenario"

    return scenario_pred, desc, breakdown


# ── Regression analysis ───────────────────────────────────────────────────────
def _run_regression(df, df_enc, available_feats, target_col, query_lower,
                    model, model_name, results, all_analysis):
    X = df_enc[available_feats].values
    y = df_enc[target_col].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split for honest metrics
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model.fit(X_tr, y_tr)

    y_pred_all = model.predict(X_scaled)
    y_pred_te  = model.predict(X_te)

    mae   = mean_absolute_error(y_te, y_pred_te)
    r2    = r2_score(y_te, y_pred_te)
    r2_tr = r2_score(y_tr, model.predict(X_tr))
    mape  = mean_absolute_percentage_error(y_te, y_pred_te + 1e-9) * 100

    try:
        # RidgeCV already performed internal CV — use its best_score_ if available
        if hasattr(model, "best_score_"):
            cv_r2 = float(model.best_score_)
        else:
            n_cv = min(5, max(2, len(y) // 20))
            cv_r2 = cross_val_score(
                model.__class__(**model.get_params()),
                X_scaled, y, cv=n_cv, scoring="r2",
            ).mean()
    except Exception:
        cv_r2 = r2

    results.update({
        "model_name":    model_name,
        "model_r2":      round(float(r2), 4),
        "model_r2_train":round(float(r2_tr), 4),
        "model_cv_r2":   round(float(cv_r2), 4),
        "model_mae":     round(float(mae), 4),
        "model_mape_pct":round(float(mape), 2),
        "n_samples":     int(len(y)),
    })

    print(f"  {model_name}: R²(test)={r2:.4f} | CV-R²={cv_r2:.4f} | MAE={mae:,.4f}")

    # ── Feature importance chart
    imp_df = _get_feature_importance(model, X_scaled, y, available_feats, "regression")
    results["feature_importance"] = {
        r["feature"]: round(float(r["importance_pct"]), 2)
        for _, r in imp_df.iterrows()
    }

    fig1 = _bar_chart(
        f"Feature Importance — {target_col} ({model_name}, R²={r2:.3f})",
        imp_df["feature"], imp_df["importance_pct"], color_by_sign=False,
    )
    safe_show(fig1, f"Feature Importance — {target_col}")

    expl1 = _analyze_chart(
        f"Feature Importance — {target_col}", "bar",
        "feature", "importance_pct", imp_df,
        extra_context=(
            f"Model: {model_name} | Target: {target_col}\n"
            f"R²={r2:.3f} | CV-R²={cv_r2:.3f} | MAE={mae:,.2f} | MAPE={mape:.1f}%\n"
            f"Top feature: {imp_df.iloc[0]['feature']} ({imp_df.iloc[0]['importance_pct']:.1f}% importance)"
        )
    )
    all_analysis.append(f"**Feature Importance**:\n{expl1}")

    # ── Actual vs Predicted scatter
    samp_n = min(500, len(y))
    idx    = np.random.RandomState(42).choice(len(y), samp_n, replace=False)
    fig2   = go.Figure()
    fig2.add_trace(go.Scatter(
        x=y_pred_all[idx], y=y[idx], mode="markers",
        marker=dict(color="#3B82F6", size=4, opacity=0.5),
        name="Data points",
    ))
    lo, hi = float(y.min()), float(y.max())
    fig2.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#10B981", dash="dash", width=2), name="Perfect fit",
    ))
    fig2.update_layout(
        title=f"Actual vs Predicted — {target_col}",
        xaxis_title="Predicted", yaxis_title="Actual",
        template="sentinel", height=420,
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#F1F5F9"),
    )
    safe_show(fig2, "Actual vs Predicted")

    # ── Over/under valuation (if asked)
    is_overpricing = any(k in query_lower for k in [
        "overpric", "underpric", "overvalu", "undervalu",
        "above market", "below market", "expensive", "cheap", "bargain",
    ])
    if is_overpricing:
        residual = y - y_pred_all
        pct_diff = (residual / (np.abs(y_pred_all) + 1e-9)) * 100

        over_count  = int((pct_diff >  10).sum())
        under_count = int((pct_diff < -10).sum())
        results["overvalued_count"]  = over_count
        results["undervalued_count"] = under_count
        results["avg_overval_pct"]   = round(float(pct_diff[pct_diff > 0].mean()), 2)

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=pct_diff, nbinsx=40, marker_color="#3B82F6", opacity=0.8,
        ))
        for xval, col, lbl in [(0, "#EF4444", "Fair"), (10, "#F59E0B", "+10%"), (-10, "#10B981", "-10%")]:
            fig3.add_vline(x=xval, line_color=col, line_dash="dash",
                           annotation_text=lbl,
                           annotation_font_color="#F1F5F9")
        fig3.update_layout(
            title=f"Valuation Distribution — {target_col}",
            xaxis_title="% vs Model Prediction", yaxis_title="Count",
            template="sentinel", height=380,
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#F1F5F9"),
        )
        safe_show(fig3, "Valuation Distribution")
        all_analysis.append(
            f"**Valuation**: {over_count} records >10% above model, "
            f"{under_count} records >10% below model."
        )

    # ── Per-1σ impact (premium estimation)
    is_premium = any(k in query_lower for k in [
        "premium", "estimate", "impact of", "value of", "how much", "worth",
        "contribution", "effect of", "combined",
    ])
    if is_premium:
        baseline_pred = float(model.predict(scaler.transform(
            pd.DataFrame(df_enc[available_feats]).median().values.reshape(1, -1)
        ))[0])
        results["baseline_prediction"] = round(baseline_pred, 4)

        deltas = {}
        for i, feat in enumerate(available_feats):
            col_std = float(df_enc[feat].std()) if feat in df_enc.columns else 0
            if col_std < 1e-9:
                continue
            hi_X          = scaler.transform(pd.DataFrame(df_enc[available_feats]).median().values.reshape(1, -1))
            hi_X[0, i]   += col_std / scaler.scale_[i]
            deltas[feat]  = round(float(model.predict(hi_X)[0]) - baseline_pred, 4)

        top_d = dict(sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12])
        results["per_1std_impact"] = top_d
        delta_df = pd.DataFrame([{"feature": k, "delta": v} for k, v in top_d.items()])
        fig4 = _bar_chart(
            f"1-σ Feature Impact on {target_col} (baseline={baseline_pred:,.2f})",
            delta_df["feature"], delta_df["delta"],
        )
        safe_show(fig4, f"Feature Impact — {target_col}")
        all_analysis.append(
            f"**Baseline {target_col}**: {baseline_pred:,.2f}. "
            f"Top driver: +1σ in '{delta_df.iloc[0]['feature']}' → "
            f"{delta_df.iloc[0]['delta']:+,.2f}"
        )

    # ── Concrete scenario prediction ─────────────────────────────────────
    # Always compute a scenario prediction so the user gets a real number
    scenario_pred, scenario_desc, scenario_breakdown = _compute_scenario_prediction(
        model, scaler, df, df_enc, available_feats, target_col, query_lower,
    )
    if scenario_pred is not None:
        results["scenario_prediction"] = round(scenario_pred, 4)
        results["scenario_description"] = scenario_desc
        results["scenario_breakdown"] = scenario_breakdown
        all_analysis.append(
            f"**🎯 Prediction Result**: {scenario_desc} → **{scenario_pred:,.2f}** {target_col}"
        )

        # Waterfall chart: base → feature adjustments → final
        if scenario_breakdown:
            wf_labels = ["Baseline"] + [b["feature"] for b in scenario_breakdown] + ["PREDICTION"]
            wf_values = [scenario_breakdown[0].get("base", 0)] if scenario_breakdown else [0]
            base_val = scenario_breakdown[0].get("base", 0) if scenario_breakdown else 0
            for b in scenario_breakdown:
                wf_values.append(b.get("delta", 0))
            wf_values.append(scenario_pred)

            wf_measure = ["absolute"] + ["relative"] * len(scenario_breakdown) + ["total"]
            wf_text = [f"{base_val:,.1f}"] + [f"{b.get('delta', 0):+,.1f}" for b in scenario_breakdown] + [f"{scenario_pred:,.1f}"]

            fig_wf = go.Figure(go.Waterfall(
                name="Prediction Breakdown",
                orientation="v",
                measure=wf_measure,
                x=wf_labels,
                y=wf_values,
                text=wf_text,
                textposition="outside",
                connector_line_color="#64748B",
                increasing_marker_color="#10B981",
                decreasing_marker_color="#EF4444",
                totals_marker_color="#3B82F6",
            ))
            fig_wf.update_layout(
                title=f"Prediction Breakdown — {target_col}",
                template="sentinel", height=430,
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font=dict(color="#F1F5F9"),
                xaxis=dict(tickfont=dict(color="#94A3B8")),
                yaxis=dict(tickfont=dict(color="#94A3B8"), title=target_col),
                showlegend=False,
            )
            safe_show(fig_wf, "Prediction Breakdown")
            all_analysis.append(
                f"**Prediction Breakdown**: Base ({base_val:,.1f}) "
                + " ".join(f"→ {b['feature']} ({b['delta']:+,.1f})" for b in scenario_breakdown)
                + f" = **{scenario_pred:,.1f}**"
            )


# ── Classification analysis ───────────────────────────────────────────────────
def _run_classification(df_enc, available_feats, target_col, query_lower,
                        model, model_name, results, all_analysis):
    le_tgt = LabelEncoder()
    y_raw  = df_enc[target_col]
    y      = le_tgt.fit_transform(y_raw.astype(str))
    X      = df_enc[available_feats].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_tr, y_tr)

    y_pred   = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    f1       = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    auc = None
    if len(le_tgt.classes_) == 2 and hasattr(model, "predict_proba"):
        try:
            auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        except Exception:
            pass

    results.update({
        "model_name": model_name,
        "task":       "classification",
        "accuracy":   round(float(accuracy), 4),
        "f1_score":   round(float(f1), 4),
        "auc_roc":    round(float(auc), 4) if auc else None,
        "n_classes":  len(le_tgt.classes_),
        "class_names": list(le_tgt.classes_[:20]),
        "n_samples":  int(len(y)),
    })

    print(f"  {model_name}: Accuracy={accuracy:.4f} | F1={f1:.4f}" +
          (f" | AUC={auc:.4f}" if auc else ""))

    # Feature importance
    imp_df = _get_feature_importance(model, X_scaled, y, available_feats, "classification")
    results["feature_importance"] = {
        r["feature"]: round(float(r["importance_pct"]), 2)
        for _, r in imp_df.iterrows()
    }

    fig1 = _bar_chart(
        f"Feature Importance — {target_col} ({model_name}, Acc={accuracy:.3f})",
        imp_df["feature"], imp_df["importance_pct"], color_by_sign=False,
    )
    safe_show(fig1, f"Feature Importance — {target_col}")

    expl1 = _analyze_chart(
        f"Feature Importance — {target_col}", "bar",
        "feature", "importance_pct", imp_df,
        extra_context=(
            f"Model: {model_name} | Task: Classification | Target: {target_col}\n"
            f"Accuracy={accuracy:.3f} | F1={f1:.3f}"
            + (f" | AUC={auc:.3f}" if auc else "")
        )
    )
    all_analysis.append(f"**Feature Importance**:\n{expl1}")

    # Class distribution
    class_counts = pd.Series(y_raw.astype(str)).value_counts().head(15)
    fig2 = go.Figure(go.Bar(
        x=class_counts.index.tolist(),
        y=class_counts.values.tolist(),
        marker_color=_COLORS[:len(class_counts)],
    ))
    fig2.update_layout(
        title=f"Class Distribution — {target_col}",
        xaxis_title="Class", yaxis_title="Count",
        template="sentinel", height=350,
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#F1F5F9"),
    )
    safe_show(fig2, f"Class Distribution — {target_col}")


# ── Main Agent ────────────────────────────────────────────────────────────────
def prediction_agent(state: SentinelState) -> dict:
    """
    Multi-model, task-aware ML prediction agent.
    Handles: regression, classification, over/under-valuation, premium estimation.
    """
    print("[PredictionAgent] Starting analysis...")

    query     = state["query"]
    q_low     = query.lower()
    results   = {}
    all_analysis: list = []

    # ── Discover table and load data ─────────────────────────────────────────
    table = _discover_main_table(q_low)
    print(f"  [PredictionAgent] Table: {table}")

    try:
        df = _load_data(table)
    except Exception as exc:
        return {"math_result": {"error": str(exc)},
                "final_response": f"Could not load data for prediction: {exc}"}

    if df is None or df.empty:
        return {"math_result": {"error": "Empty dataset"},
                "final_response": "The dataset is empty — cannot run prediction."}

    print(f"  Loaded {len(df)} rows × {len(df.columns)} cols from '{table}'")

    # ── Feature engineering ───────────────────────────────────────────────────
    num_cols = [c for c in df.select_dtypes(include="number").columns if not _ID_LIKE(c)]
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if not _ID_LIKE(c)]

    target_col = _pick_target(df, query)
    if target_col is None:
        return {"math_result": {"error": "No numeric target column found"},
                "final_response": "No numeric column found to use as a prediction target."}

    feature_cols    = [c for c in num_cols + cat_cols if c != target_col]
    df_enc          = _encode_features(df.copy(), [c for c in cat_cols if c != target_col])
    available_feats = [c for c in feature_cols if c in df_enc.columns]

    sub = df_enc[available_feats + [target_col]].dropna()
    if len(sub) < 30:
        return {"math_result": {"error": f"Too few rows ({len(sub)}) for ML"},
                "final_response": f"Only {len(sub)} complete rows — need ≥30."}

    # ── Detect task type and select model ────────────────────────────────────
    task             = _detect_task_type(sub, target_col, q_low)
    model, model_name = _select_model(task, len(sub), q_low)

    print(f"  Task={task} | Model={model_name} | Target={target_col}")

    results.update({
        "table": table, "target": target_col,
        "task":  task,  "model_name": model_name,
    })

    # ── Run task-specific analysis ────────────────────────────────────────────
    if task == "timeseries":
        # Delegate to forecast_agent (short-circuit with a helpful message)
        return {
            "math_result":    results,
            "final_response": (
                "This looks like a time-series forecasting request. "
                "SENTINEL's Forecast Agent handles future predictions — "
                "please rephrase as: *'Forecast [metric] for the next N days/months'*."
            ),
        }

    try:
        if task == "regression":
            _run_regression(
                df, sub, available_feats, target_col, q_low,
                model, model_name, results, all_analysis,
            )
        else:
            _run_classification(
                sub, available_feats, target_col, q_low,
                model, model_name, results, all_analysis,
            )
    except Exception as exc:
        import traceback
        print(f"  [PredictionAgent] Model error: {exc}")
        traceback.print_exc()
        return {"math_result": {"error": str(exc)},
                "final_response": f"ML model failed: {str(exc)[:200]}"}

    # ── Narrative ──────────────────────────────────────────────────────────────
    metrics_str = ""
    if task == "regression":
        metrics_str = (
            f"R²(test)={results.get('model_r2')}, CV-R²={results.get('model_cv_r2')}, "
            f"MAE={results.get('model_mae')}, MAPE={results.get('model_mape_pct')}%"
        )
    else:
        metrics_str = (
            f"Accuracy={results.get('accuracy')}, F1={results.get('f1_score')}"
            + (f", AUC={results.get('auc_roc')}" if results.get('auc_roc') else "")
        )

    # Include concrete prediction in the narrative prompt
    prediction_str = ""
    if results.get("scenario_prediction") is not None:
        prediction_str = (
            f"\nCONCRETE PREDICTION: {results['scenario_prediction']:,.2f} {results.get('target', '')}"
            f"\nScenario: {results.get('scenario_description', '')}"
        )
        if results.get("scenario_breakdown"):
            breakdown_parts = [f"{b['feature']}: {b['delta']:+,.2f}" for b in results['scenario_breakdown']]
            prediction_str += f"\nBreakdown from baseline: {', '.join(breakdown_parts)}"

    prompt = (
        f"Summarize these ML results for a business analyst in 3-5 bullet points "
        f"with concrete numbers and actionable insight.\n"
        f"IMPORTANT: Start your summary with the concrete prediction/estimate if available.\n"
        f"Query: {query}\n"
        f"Table: {table} | Target: {target_col} | Task: {task} | Model: {model_name}\n"
        f"Metrics: {metrics_str}\n"
        f"{prediction_str}\n"
        f"Feature importance (top): "
        f"{json.dumps(dict(list(results.get('feature_importance', {}).items())[:5]))}\n"
        f"Other findings: {json.dumps({k: v for k, v in results.items() if k not in ('feature_importance', 'per_1std_impact', 'feature_premiums', 'scenario_breakdown')}, default=str)}"
    )

    narrative = call_llm(prompt, model=FAST_MODEL, temperature=0.1)

    if not narrative or narrative.startswith("LLM_ERROR") or len(narrative.strip()) < 10:
        lines = []
        if results.get("scenario_prediction") is not None:
            lines.append(
                f"- **🎯 Prediction**: {results['scenario_prediction']:,.2f} {target_col}"
                f" — {results.get('scenario_description', '')}"
            )
        lines.append(f"- **Model**: {model_name} ({task}) — {metrics_str}")
        top_feats = list(results.get("feature_importance", {}).items())[:3]
        if top_feats:
            lines.append("- **Top features**: " + "; ".join(
                f"{k} ({v:.1f}%)" for k, v in top_feats
            ))
        if results.get("overvalued_count") is not None:
            lines.append(
                f"- **Valuation**: {results['overvalued_count']} overvalued (>10%), "
                f"{results['undervalued_count']} undervalued."
            )
        narrative = "\n".join(lines)

    chart_section = "\n\n".join(all_analysis)
    full_response = (
        f"{narrative}\n\n{'─'*50}\nChart Analysis:\n{chart_section}"
        if chart_section else narrative
    )

    l2_store(state["query"], "/* ML prediction */", str(results)[:300], score=1.0)
    return {"math_result": results, "final_response": full_response}
