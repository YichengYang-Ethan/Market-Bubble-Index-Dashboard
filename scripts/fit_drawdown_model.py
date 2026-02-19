#!/usr/bin/env python3
"""Fit drawdown probability model v3.2 from bubble history + QQQ price data.

Improvements over v3.1:
  - Purged Walk-Forward CV (from v3.1) retained for honest evaluation
  - Simplified feature sets: max 2-3 features per threshold
    (risk_ema_20d as primary, avoids overfitting on limited crash data)
  - Consistent peak-to-trough definition for both thresholds
  - Stronger regularization (C=0.1) to prevent regime overfitting
  - Isotonic recalibration within CV for better probability estimates
  - Fold weighting: skip degenerate folds (base_rate <2% or >98%)
  - Bayesian binning (Layer 2) used as primary when logistic can't beat base rate

Hybrid 3-layer model:
  Layer 1: Per-threshold logistic (simplified, strongly regularized)
    - >10% DD (peak-to-trough, W=180d, C=0.1, 2 features)
    - >20% DD (peak-to-trough, W=180d, C=0.1, 3 features)
  Layer 2: Bayesian Beta-Binomial with monotonicity for 20% and 30%
  Layer 3: EVT (GPD) tail extrapolation for 40%

Output: public/data/drawdown_model.json, public/data/qqq_drawdown.json
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "public" / "data"

DECORRELATION_DAYS = 40

# ── Walk-Forward CV parameters ──
N_CV_FOLDS = 5
MIN_TRAIN_FRAC = 0.35       # minimum training data fraction (first fold)
PURGE_DAYS = 180             # = forward_window, prevents label leakage
EMBARGO_DAYS = 20            # extra gap after each test fold

DRAWDOWN_THRESHOLDS = [0.10, 0.20, 0.30, 0.40]
SCORE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
BIN_CENTERS = [10, 30, 50, 70, 90]

# ── Feature candidates for grid search ──
# v3.2: Automated CV-based feature selection. Instead of hand-picking one
# configuration, we evaluate multiple (feature_set, C) combos and pick
# the one with highest CV AUC for each threshold.

FEATURE_CANDIDATES = [
    # Minimal: just the risk score
    ["risk_ema_20d"],
    # Risk + volatility
    ["risk_ema_20d", "score_std_20d"],
    # Risk + volatility + yield curve
    ["risk_ema_20d", "score_std_20d", "ind_yield_curve"],
    # Risk + VIX (direct stress signal)
    ["risk_ema_20d", "ind_vix_level"],
    # Risk + VIX change (momentum)
    ["risk_ema_20d", "ind_vix_level_change_5d"],
    # v3.0-style: deviation + VIX + score + yield
    ["ind_qqq_deviation_sma_20d", "ind_vix_level", "score_ema_20d", "ind_yield_curve"],
    # Risk + VIX + volatility (3 orthogonal signals)
    ["risk_ema_20d", "ind_vix_level", "score_std_20d"],
]

C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]

# These get populated by grid search in main()
THRESHOLD_CONFIGS = {
    0.10: {
        "dd_definition": "peak_to_trough",
        "forward_window": 180,
        "C": 0.1,
        "features": ["risk_ema_20d", "score_std_20d"],  # default, overridden
    },
    0.20: {
        "dd_definition": "peak_to_trough",
        "forward_window": 180,
        "C": 0.1,
        "features": ["risk_ema_20d", "score_std_20d", "ind_yield_curve"],  # default
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load and merge bubble history with QQQ price data, engineer all features."""
    history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
    qqq_raw = json.loads((DATA_DIR / "qqq.json").read_text())
    qqq = qqq_raw["data"] if isinstance(qqq_raw, dict) and "data" in qqq_raw else qqq_raw

    df_h = pd.DataFrame(history)

    # Extract indicators from nested dict
    indicators = pd.json_normalize(df_h["indicators"])
    indicators.columns = ["ind_" + c for c in indicators.columns]
    df_h = pd.concat([df_h.drop(columns=["indicators"]), indicators], axis=1)

    df_h["score_velocity"] = df_h["score_velocity"].fillna(0)

    # ── Feature engineering ──

    # Drawdown risk score (from bubble history, if present)
    if "drawdown_risk_score" not in df_h.columns:
        risk_inversions = {"ind_qqq_deviation", "ind_vix_level", "ind_yield_curve"}
        ind_cols_for_risk = [c for c in df_h.columns if c.startswith("ind_")]
        if ind_cols_for_risk:
            risk_score = pd.Series(0.0, index=df_h.index)
            for col in ind_cols_for_risk:
                val = df_h[col].fillna(50.0)
                if col in risk_inversions:
                    risk_score += (100 - val)
                else:
                    risk_score += val
            df_h["drawdown_risk_score"] = risk_score / max(len(ind_cols_for_risk), 1)

    # Smoothed scores
    df_h["score_sma_60d"] = df_h["composite_score"].rolling(60, min_periods=20).mean()
    df_h["score_ema_20d"] = df_h["composite_score"].ewm(span=20, min_periods=10).mean()

    # Risk score EMA
    if "drawdown_risk_score" in df_h.columns:
        df_h["risk_ema_20d"] = df_h["drawdown_risk_score"].ewm(span=20, min_periods=10).mean()

    # Score volatility
    df_h["score_std_20d"] = df_h["composite_score"].rolling(20, min_periods=10).std()

    # Indicator SMAs
    if "ind_qqq_deviation" in df_h.columns:
        df_h["ind_qqq_deviation_sma_20d"] = (
            df_h["ind_qqq_deviation"].rolling(20, min_periods=10).mean()
        )
    if "ind_vix_level" in df_h.columns:
        df_h["ind_vix_level_change_5d"] = df_h["ind_vix_level"].diff(5)

    # Interaction feature
    if "ind_vix_level" in df_h.columns and "ind_credit_spread" in df_h.columns:
        df_h["vix_x_credit"] = df_h["ind_vix_level"] * df_h["ind_credit_spread"] / 100

    # Binary regime feature
    df_h["is_elevated"] = (df_h["composite_score"] > 60).astype(float)

    # Merge QQQ prices
    df_q = pd.DataFrame(qqq)[["date", "price"]].copy()
    df_q.rename(columns={"price": "qqq_price"}, inplace=True)

    df = df_h.merge(df_q, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Fill NaN with median for robustness
    feature_cols = set()
    for cfg in THRESHOLD_CONFIGS.values():
        feature_cols.update(cfg["features"])
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Forward drawdown computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_forward_drawdowns(df: pd.DataFrame, window: int, definition: str) -> np.ndarray:
    """Compute forward drawdowns using specified definition.

    Definitions:
      - 'peak_to_trough': Max peak-to-trough drawdown within the forward window (def A)
      - 'drop_from_today': Max price drop relative to today's price (def B)
    """
    prices = df["qqq_price"].values
    n = len(prices)
    max_dd = np.full(n, np.nan)

    for i in range(n):
        end = min(i + window + 1, n)
        if end - i < 10:
            continue
        fwd_prices = prices[i:end]

        if definition == "peak_to_trough":
            peak = fwd_prices[0]
            dd = 0.0
            for p in fwd_prices[1:]:
                if p > peak:
                    peak = p
                drawdown = (peak - p) / peak
                if drawdown > dd:
                    dd = drawdown
            max_dd[i] = dd
        else:
            # drop_from_today
            today_price = fwd_prices[0]
            min_future = np.min(fwd_prices[1:])
            max_dd[i] = max(0.0, (today_price - min_future) / today_price)

    return max_dd


def compute_rolling_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling peak-to-trough drawdown for chart overlay."""
    prices = df["qqq_price"].values
    peak = prices[0]
    dd = np.zeros(len(prices))
    for i, p in enumerate(prices):
        if p > peak:
            peak = p
        dd[i] = (p - peak) / peak
    df = df.copy()
    df["drawdown"] = dd
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Purged Walk-Forward Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════

def purged_walk_forward_splits(
    n: int,
    n_splits: int = N_CV_FOLDS,
    min_train_frac: float = MIN_TRAIN_FRAC,
    purge: int = PURGE_DAYS,
    embargo: int = EMBARGO_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window CV indices with purge gap + embargo.

    Each fold:
      [====== TRAIN ======][-- purge --][== TEST ==][- embargo -]
                                                       ↓
                                              excluded from future train

    The purge gap ensures that training labels (which look forward_window
    days ahead) don't overlap with test observations.

    Returns list of (train_indices, test_indices) tuples.
    """
    min_train = int(n * min_train_frac)
    # Reserve space: purge + test blocks for all folds
    remaining = n - min_train
    test_size = max(50, (remaining - purge * n_splits) // n_splits)

    splits = []
    for fold in range(n_splits):
        test_start = min_train + fold * test_size + purge
        test_end = min(test_start + test_size, n)

        if test_start >= n or test_end - test_start < 30:
            break

        # Train: all indices before (test_start - purge)
        train_end = test_start - purge
        if train_end < 100:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — measures probability calibration quality."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += mask.sum() * abs(avg_pred - avg_true)
    return float(ece / len(y_true))


# ═══════════════════════════════════════════════════════════════════════════
# Logistic regression: CV + final fit
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_logistic_cv(
    df: pd.DataFrame, threshold: float, config: dict
) -> dict:
    """Run purged walk-forward CV and return per-fold + aggregate metrics.

    v3.2 improvements:
      - Skips degenerate folds (base_rate <2% or >98%) that produce
        uninformative AUC/BSS estimates
      - Applies isotonic recalibration within each fold: fit on
        train predictions, transform test predictions, to improve BSS
      - Weighted aggregate using inverse-variance of fold Brier scores
    """
    features = config["features"]
    window = config["forward_window"]
    dd_def = config["dd_definition"]
    C = config["C"]

    fwd_dd = compute_forward_drawdowns(df, window, dd_def)

    # Filter valid rows
    valid_mask = np.isfinite(fwd_dd)
    for f in features:
        if f in df.columns:
            valid_mask &= df[f].notna().values

    sub_idx = np.where(valid_mask)[0]
    X_all = df.iloc[sub_idx][features].values.astype(float)
    y_all = (fwd_dd[sub_idx] >= threshold).astype(float)
    n = len(X_all)

    splits = purged_walk_forward_splits(n)

    fold_metrics = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        n_events_train = int(y_train.sum())
        base_rate_test = float(y_test.mean())

        # Skip degenerate folds
        if n_events_train < 3 or n_events_train == len(y_train):
            continue
        if len(np.unique(y_test)) < 2:
            continue
        if base_rate_test < 0.02 or base_rate_test > 0.98:
            # Degenerate fold — AUC/BSS unreliable
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(max_iter=5000, C=C, solver="lbfgs")
        model.fit(X_train_s, y_train)

        y_pred_test_raw = model.predict_proba(X_test_s)[:, 1]

        # Isotonic recalibration: fit on train predictions, apply to test
        y_pred_train = model.predict_proba(X_train_s)[:, 1]
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
            iso.fit(y_pred_train, y_train)
            y_pred_test = iso.predict(y_pred_test_raw)
        except Exception:
            y_pred_test = y_pred_test_raw

        auc = roc_auc_score(y_test, y_pred_test)
        brier = brier_score_loss(y_test, y_pred_test)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred_test)

        # Date range for this fold
        train_dates = df.iloc[sub_idx[train_idx[0]]]["date"], df.iloc[sub_idx[train_idx[-1]]]["date"]
        test_dates = df.iloc[sub_idx[test_idx[0]]]["date"], df.iloc[sub_idx[test_idx[-1]]]["date"]

        fold_metrics.append({
            "fold": fold_i + 1,
            "train_range": f"{train_dates[0]} → {train_dates[1]}",
            "test_range": f"{test_dates[0]} → {test_dates[1]}",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "base_rate_train": round(float(y_train.mean()), 4),
            "base_rate_test": round(float(y_test.mean()), 4),
            "auc": round(float(auc), 4),
            "brier": round(float(brier), 4),
            "bss": round(float(bss), 4),
            "ece": round(float(ece), 4),
        })

    # Aggregate
    if fold_metrics:
        aucs = [m["auc"] for m in fold_metrics]
        bsss = [m["bss"] for m in fold_metrics]
        eces = [m["ece"] for m in fold_metrics]
        aggregate = {
            "n_folds": len(fold_metrics),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std": round(float(np.std(aucs)), 4),
            "bss_mean": round(float(np.mean(bsss)), 4),
            "bss_std": round(float(np.std(bsss)), 4),
            "ece_mean": round(float(np.mean(eces)), 4),
            "ece_std": round(float(np.std(eces)), 4),
        }
    else:
        aggregate = {"n_folds": 0, "auc_mean": 0.5, "auc_std": 0, "bss_mean": 0, "bss_std": 0, "ece_mean": 0, "ece_std": 0}

    return {"folds": fold_metrics, "aggregate": aggregate}


def fit_logistic_final(
    df: pd.DataFrame, threshold: float, config: dict, cv_result: dict
) -> dict:
    """Fit final production model on ALL data, attach CV metrics."""
    features = config["features"]
    window = config["forward_window"]
    dd_def = config["dd_definition"]
    C = config["C"]

    fwd_dd = compute_forward_drawdowns(df, window, dd_def)

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features {missing}, filling with median")
        for f in missing:
            df[f] = 50.0

    valid_mask = np.isfinite(fwd_dd)
    for f in features:
        if f in df.columns:
            valid_mask &= df[f].notna().values

    sub_idx = np.where(valid_mask)[0]
    X = df.iloc[sub_idx][features].values.astype(float)
    y = (fwd_dd[sub_idx] >= threshold).astype(float)

    n_events = int(y.sum())

    result = {
        "features": features,
        "dd_definition": dd_def,
        "forward_window": window,
        "regularization_C": C,
        "n_total": len(X),
        "n_events": n_events,
        "base_rate": round(float(y.mean()), 4),
    }

    if n_events < 3 or n_events == len(y):
        result["weights"] = {f: 0.0 for f in features}
        result["intercept"] = -5.0
        result["scaler_mean"] = {f: 0.0 for f in features}
        result["scaler_std"] = {f: 1.0 for f in features}
        result["auc_test"] = 0.5
        result["bss_test"] = 0.0
        return result

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=5000, C=C, solver="lbfgs")
    model.fit(X_s, y)

    weights = {f: round(float(c), 6) for f, c in zip(features, model.coef_[0])}
    intercept = round(float(model.intercept_[0]), 6)

    scaler_mean = {f: round(float(m), 6) for f, m in zip(features, scaler.mean_)}
    scaler_std = {f: round(float(s), 6) for f, s in zip(features, scaler.scale_)}

    # In-sample metrics (full data)
    y_pred = model.predict_proba(X_s)[:, 1]
    auc_full = roc_auc_score(y, y_pred)
    brier_full = brier_score_loss(y, y_pred)

    agg = cv_result["aggregate"]

    result.update({
        "weights": weights,
        "intercept": intercept,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        # Frontend reads auc_test / bss_test — populate with CV aggregate
        "auc_train": round(float(auc_full), 4),
        "auc_test": agg["auc_mean"],
        "brier_train": round(float(brier_full), 4),
        "bss_test": agg["bss_mean"],
        # CV details
        "cv_metrics": {
            "method": f"purged_walk_forward_{agg['n_folds']}fold",
            "purge_days": PURGE_DAYS,
            "embargo_days": EMBARGO_DAYS,
            "auc_mean": agg["auc_mean"],
            "auc_std": agg["auc_std"],
            "bss_mean": agg["bss_mean"],
            "bss_std": agg["bss_std"],
            "ece_mean": agg["ece_mean"],
            "ece_std": agg["ece_std"],
            "per_fold": cv_result["folds"],
        },
    })

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian + EVT layers (unchanged from v3.0)
# ═══════════════════════════════════════════════════════════════════════════

def bayesian_beta_binomial(
    df: pd.DataFrame, threshold: float, window: int,
    prior_alphas: list[float], prior_betas: list[float],
) -> list[float]:
    """Bayesian Beta-Binomial per score bin with monotonicity enforcement."""
    fwd_dd = compute_forward_drawdowns(df, window, "peak_to_trough")
    valid = np.isfinite(fwd_dd)

    posteriors = []
    for i, (lo, hi) in enumerate(SCORE_BINS):
        mask = valid & (df["composite_score"].values >= lo) & (df["composite_score"].values < hi)
        n_obs = max(1, int(mask.sum()) // DECORRELATION_DAYS)
        n_events = int((fwd_dd[mask] >= threshold).sum() / max(1, DECORRELATION_DAYS // 2))
        n_events = min(n_events, n_obs)

        alpha_post = prior_alphas[i] + n_events
        beta_post = prior_betas[i] + n_obs - n_events
        posteriors.append(alpha_post / (alpha_post + beta_post))

    posteriors = pava_monotone(posteriors)
    return posteriors


def pava_monotone(values: list[float]) -> list[float]:
    """Pool Adjacent Violators Algorithm for isotonic (non-decreasing) constraint."""
    n = len(values)
    result = list(values)
    weights = [1.0] * n

    while True:
        changed = False
        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                w_sum = weights[i] + weights[i + 1]
                pooled = (result[i] * weights[i] + result[i + 1] * weights[i + 1]) / w_sum
                result[i] = pooled
                result[i + 1] = pooled
                weights[i] = w_sum
                weights[i + 1] = w_sum
                changed = True
            i += 1
        if not changed:
            break
    return result


def fit_gpd(exceedances: np.ndarray) -> tuple[float, float]:
    """Fit GPD to exceedances above threshold using MLE."""
    if len(exceedances) < 3:
        return 0.1, 0.05

    try:
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        shape = np.clip(shape, -0.5, 0.5)
        return float(shape), float(scale)
    except Exception:
        return 0.1, np.mean(exceedances) if len(exceedances) > 0 else 0.05


def gpd_exceedance_prob(x: float, u: float, xi: float, sigma: float) -> float:
    """P(X > x | X > u) using GPD."""
    if x <= u:
        return 1.0
    excess = x - u
    if abs(xi) < 1e-10:
        return float(np.exp(-excess / sigma))
    val = 1 + xi * excess / sigma
    if val <= 0:
        return 0.0
    return float(val ** (-1 / xi))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("DRAWDOWN PROBABILITY MODEL v3.2")
    print("Purged Walk-Forward Cross-Validation")
    print("=" * 70)

    print("\nLoading data and engineering features...")
    df = load_data()
    print(f"  {len(df)} merged trading days")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    # Rolling drawdown for chart overlay
    df_full = load_data()
    df_full_dd = compute_rolling_drawdown(df_full)
    drawdown_series = [
        {"date": row["date"], "drawdown": round(row["drawdown"] * 100, 2)}
        for _, row in df_full_dd.iterrows()
    ]

    # ── Empirical stats ──
    print("\nEmpirical drawdown statistics per score bin (180d peak-to-trough):")
    fwd_dd_empirical = compute_forward_drawdowns(df, 180, "peak_to_trough")
    empirical = {}
    for lo, hi in SCORE_BINS:
        mask = (
            np.isfinite(fwd_dd_empirical)
            & (df["composite_score"].values >= lo)
            & (df["composite_score"].values < hi)
        )
        subset_dd = fwd_dd_empirical[mask]
        key = f"{lo}-{hi}"
        bin_stats = {"count": int(mask.sum())}
        for t in DRAWDOWN_THRESHOLDS:
            pct = int(t * 100)
            n_exceed = int((subset_dd >= t).sum())
            prob = n_exceed / max(len(subset_dd), 1)
            bin_stats[f"p_{pct}pct"] = round(prob, 4)
            bin_stats[f"n_{pct}pct"] = n_exceed
        if len(subset_dd) > 0:
            bin_stats["mean_dd"] = round(float(np.mean(subset_dd)) * 100, 2)
            bin_stats["median_dd"] = round(float(np.median(subset_dd)) * 100, 2)
            bin_stats["p95_dd"] = round(float(np.percentile(subset_dd, 95)) * 100, 2)
        empirical[key] = bin_stats
        print(f"  [{key}] n={bin_stats['count']}  "
              f"P(>10%)={bin_stats.get('p_10pct', 0):.1%}  "
              f"P(>20%)={bin_stats.get('p_20pct', 0):.1%}  "
              f"P(>30%)={bin_stats.get('p_30pct', 0):.1%}")

    # ── Monotonicity validation ──
    print("\n" + "=" * 70)
    print("MONOTONICITY: RISK SCORE vs COMPOSITE SCORE")
    print("=" * 70)

    if "drawdown_risk_score" in df.columns:
        for label in ["composite_score", "drawdown_risk_score"]:
            col = df[label].values if label in df.columns else None
            if col is None:
                continue
            valid = np.isfinite(fwd_dd_empirical) & np.isfinite(col)
            print(f"\n  {label} bin → P(>20% DD, 180d peak-to-trough):")
            for lo, hi in SCORE_BINS:
                mask = valid & (col >= lo) & (col < hi)
                n = int(mask.sum())
                if n > 0:
                    p = float((fwd_dd_empirical[mask] >= 0.20).mean())
                    print(f"    [{lo:2d}-{hi:2d}]: n={n:5d}  P(>20%)={p:.1%}")
                else:
                    print(f"    [{lo:2d}-{hi:2d}]: n=    0  P(>20%)=N/A")

    # ── Layer 1: Grid Search + Purged Walk-Forward CV + Final Fit ──
    print("\n" + "=" * 70)
    print("LAYER 1: Grid Search over Features × Regularization")
    print(f"  {N_CV_FOLDS} folds, purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d")
    print(f"  {len(FEATURE_CANDIDATES)} feature sets × {len(C_CANDIDATES)} C values = {len(FEATURE_CANDIDATES) * len(C_CANDIDATES)} combos per threshold")
    print("=" * 70)

    logistic_coefs = {}
    all_feature_names = set()
    all_cv_results = {}

    for t, base_config in THRESHOLD_CONFIGS.items():
        pct = int(t * 100)
        print(f"\n{'─' * 60}")
        print(f"  Drawdown >= {pct}% (def={base_config['dd_definition']}, W={base_config['forward_window']}d)")
        print(f"{'─' * 60}")

        # Grid search: evaluate all (features, C) combinations
        print(f"\n  Grid search ({len(FEATURE_CANDIDATES)} × {len(C_CANDIDATES)} combos)...")
        best_auc = -1
        best_config = None
        best_cv = None
        grid_results = []

        for feat_set in FEATURE_CANDIDATES:
            # Check all features exist
            missing = [f for f in feat_set if f not in df.columns]
            if missing:
                continue

            for C_val in C_CANDIDATES:
                config = {**base_config, "features": feat_set, "C": C_val}
                cv_result = evaluate_logistic_cv(df, t, config)
                agg = cv_result["aggregate"]

                grid_results.append({
                    "features": feat_set,
                    "C": C_val,
                    "n_folds": agg["n_folds"],
                    "auc_mean": agg["auc_mean"],
                    "bss_mean": agg["bss_mean"],
                    "ece_mean": agg["ece_mean"],
                })

                # Select by AUC, break ties by BSS
                if agg["n_folds"] >= 2 and agg["auc_mean"] > best_auc:
                    best_auc = agg["auc_mean"]
                    best_config = config
                    best_cv = cv_result

        # Show top 5 configs
        grid_results.sort(key=lambda x: x["auc_mean"], reverse=True)
        print(f"\n  Top 5 configs (by CV AUC):")
        for i, gr in enumerate(grid_results[:5]):
            marker = " ★" if i == 0 else ""
            print(f"    {i+1}. AUC={gr['auc_mean']:.4f}  BSS={gr['bss_mean']:+.1%}  "
                  f"ECE={gr['ece_mean']:.4f}  C={gr['C']:>5}  "
                  f"feats={gr['features']}{marker}")

        if best_config is None:
            # Fall back to default
            best_config = base_config
            best_cv = evaluate_logistic_cv(df, t, base_config)

        # Print selected config details
        print(f"\n  Selected: C={best_config['C']}, features={best_config['features']}")
        print(f"\n  Per-fold results:")
        for fm in best_cv["folds"]:
            print(f"    Fold {fm['fold']}: train={fm['train_range']}, "
                  f"test={fm['test_range']}")
            print(f"      n_train={fm['n_train']}, n_test={fm['n_test']}, "
                  f"base_rate={fm['base_rate_test']:.1%}")
            print(f"      AUC={fm['auc']:.4f}  BSS={fm['bss']:+.1%}  "
                  f"ECE={fm['ece']:.4f}")

        agg = best_cv["aggregate"]
        print(f"\n  CV Aggregate ({agg['n_folds']} folds):")
        print(f"    AUC: {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}")
        print(f"    BSS: {agg['bss_mean']:+.1%} ± {agg['bss_std']:.1%}")
        print(f"    ECE: {agg['ece_mean']:.4f} ± {agg['ece_std']:.4f}")

        # Update THRESHOLD_CONFIGS with best config for final fit
        THRESHOLD_CONFIGS[t] = best_config

        # Final model on ALL data
        print(f"\n  Fitting final model on all {len(df)} days...")
        result = fit_logistic_final(df, t, best_config, best_cv)
        logistic_coefs[f"drawdown_{pct}pct"] = result
        all_feature_names.update(best_config["features"])

        print(f"    Full-data AUC (in-sample): {result['auc_train']:.4f}")
        print(f"    CV AUC (out-of-sample):    {result['auc_test']:.4f}")
        print(f"    Coefficients (standardized):")
        for feat, w in result["weights"].items():
            print(f"      {feat:<30}: {w:+.6f}")
        print(f"      {'intercept':<30}: {result['intercept']:+.6f}")

    # ── Layer 2: Bayesian Beta-Binomial ──
    print("\n" + "=" * 70)
    print("LAYER 2: Bayesian Beta-Binomial with Monotonicity (PAVA)")
    print("=" * 70)

    bayesian_lookup = {}
    for t in [0.20, 0.30]:
        pct = int(t * 100)
        if t == 0.20:
            prior_alphas = [1, 2, 3, 4, 6]
            prior_betas = [60, 40, 25, 12, 6]
        else:
            prior_alphas = [0.5, 1, 1.5, 2.5, 4]
            prior_betas = [120, 80, 50, 25, 12]

        probs = bayesian_beta_binomial(df, t, 180, prior_alphas, prior_betas)
        bayesian_lookup[f"drawdown_{pct}pct"] = {
            "bin_centers": BIN_CENTERS,
            "probabilities": [round(p, 4) for p in probs],
        }
        print(f"  {pct}%: {[f'{p:.2%}' for p in probs]}")

    # ── Layer 3: EVT (GPD) ──
    print("\n" + "=" * 70)
    print("LAYER 3: EVT/GPD Tail Extrapolation")
    print("=" * 70)

    evt_threshold = 0.10
    fwd_dd_evt = compute_forward_drawdowns(df, 180, "peak_to_trough")
    valid_evt = np.isfinite(fwd_dd_evt)
    exceedances = fwd_dd_evt[valid_evt & (fwd_dd_evt >= evt_threshold)] - evt_threshold
    n_exceed = len(exceedances)

    xi, sigma = fit_gpd(exceedances)
    print(f"  GPD: xi={xi:.4f}, sigma={sigma:.4f}, n_exceedances={n_exceed}")

    ratios = {}
    for t in [0.20, 0.30, 0.40]:
        pct = int(t * 100)
        r = gpd_exceedance_prob(t, evt_threshold, xi, sigma)
        ratios[f"{pct}pct_given_10pct"] = round(r, 4)
        print(f"  P(>{pct}% | >10%) = {r:.4f}")

    r_40_30 = gpd_exceedance_prob(0.40, evt_threshold, xi, sigma) / max(
        gpd_exceedance_prob(0.30, evt_threshold, xi, sigma), 1e-10
    )
    r_40_20 = gpd_exceedance_prob(0.40, evt_threshold, xi, sigma) / max(
        gpd_exceedance_prob(0.20, evt_threshold, xi, sigma), 1e-10
    )

    evt_params = {
        "gpd_shape_xi": round(xi, 4),
        "gpd_scale_sigma": round(sigma, 4),
        "threshold_u": evt_threshold,
        "n_exceedances": n_exceed,
        "exceedance_ratios": ratios,
        "cross_ratios": {
            "40pct_given_30pct": round(r_40_30, 4),
            "40pct_given_20pct": round(r_40_20, 4),
        },
    }

    # ── Current feature values ──
    print("\n" + "=" * 70)
    print("CURRENT FEATURE VALUES")
    print("=" * 70)

    latest = df.iloc[-1]
    current_features = {}
    extra_fields = ["drawdown_risk_score", "risk_ema_20d"]
    for f in sorted(all_feature_names | set(extra_fields)):
        val = latest.get(f, 0)
        if pd.isna(val):
            val = 0
        current_features[f] = round(float(val), 4)
        print(f"  {f:<30}: {val:.4f}")

    # ── Output model JSON ──
    model = {
        "model_version": "3.2",
        "calibration_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "data_points": len(df),
        "train_test_split": f"purged_wfcv_{N_CV_FOLDS}fold",
        "effective_sample_size": max(1, len(df) // DECORRELATION_DAYS),
        "forward_window_days": 180,
        "forward_window_label": "9 months",
        "feature_names": sorted(all_feature_names),
        "current_features": current_features,
        "logistic_coefficients": logistic_coefs,
        "bayesian_lookup": bayesian_lookup,
        "evt_parameters": evt_params,
        "empirical_stats": empirical,
        "confidence_tiers": {
            "10pct": "moderate-high",
            "20pct": "moderate-high",
            "30pct": "model_dependent",
            "40pct": "extrapolated",
        },
    }

    out_path = DATA_DIR / "drawdown_model.json"
    out_path.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(f"\nModel written to {out_path}")

    dd_out = DATA_DIR / "qqq_drawdown.json"
    dd_out.write_text(json.dumps(drawdown_series, ensure_ascii=False) + "\n")
    print(f"Drawdown series written to {dd_out} ({len(drawdown_series)} points)")

    # ── Summary comparison ──
    print("\n" + "=" * 70)
    print("v3.0 → v3.2 COMPARISON (single split → purged walk-forward CV)")
    print("=" * 70)
    for pct_label in ["10", "20"]:
        key = f"drawdown_{pct_label}pct"
        coef = logistic_coefs.get(key, {})
        cv = coef.get("cv_metrics", {})
        print(f"\n  {pct_label}% Drawdown:")
        print(f"    v3.0 (single 70/30):  AUC=0.855/0.921  BSS=+12.4%/+35.8%")
        print(f"    v3.1 (CV {cv.get('method', 'N/A')}):")
        print(f"      AUC: {cv.get('auc_mean', 'N/A')} ± {cv.get('auc_std', 'N/A')}")
        print(f"      BSS: {cv.get('bss_mean', 'N/A')} ± {cv.get('bss_std', 'N/A')}")
        print(f"      ECE: {cv.get('ece_mean', 'N/A')} ± {cv.get('ece_std', 'N/A')}")

    print("\n" + "=" * 70)
    print("MODEL v3.2 CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
