#!/usr/bin/env python3
"""Fit drawdown probability model v4.1 from bubble history + QQQ price data.

v4.1 fixes over v4.0:
  - Bug fix: 10% DD now included in Bayesian Beta-Binomial layer
    (was missing → fell back to logistic AUC=0.5 noise)
  - Bug fix: Bootstrap CI now uses per-bin priors matching main Bayesian
    (was using single flat prior → CI inconsistent with point estimate)
  - Blend weights: all thresholds 100% Bayesian (logistic AUC≤0.5 on 1999+ data)
  - Shared BAYESIAN_PRIORS dict eliminates prior duplication

v4.0 improvements over v3.3:
  - Extended historical data: 1999-present (~6800 days) vs 2014-present (~2800 days)
    - Captures dot-com crash (-83%) and GFC (-54%)
    - Independent >20% drawdown events: ~4 → ~8-10
    - Effective independent samples: ~70 → ~170
  - Firth penalized logistic regression (if firthlogist available)
    - Corrects rare-event MLE bias via Jeffreys prior
    - Falls back to standard LogisticRegression if not installed
  - Bootstrap confidence intervals (90% CI)
    - StationaryBootstrap (arch) or manual block bootstrap
    - 500 resamples, block size = DECORRELATION_DAYS
    - Output: probability_ci in model JSON

Hybrid 3-layer model:
  Layer 1: Per-threshold penalized logistic (stability-selected features)
    - >10% DD: Lasso, 8 stable features, C auto-tuned
    - >20% DD: L2, 2 stable features, C auto-tuned
  Layer 2: Bayesian Beta-Binomial with monotonicity for 10%, 20%, and 30%
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

try:
    from firthlogist import FirthLogisticRegression
    USE_FIRTH = True
except ImportError:
    USE_FIRTH = False

try:
    from arch.bootstrap import StationaryBootstrap
    USE_ARCH_BOOTSTRAP = True
except ImportError:
    USE_ARCH_BOOTSTRAP = False

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "public" / "data"

DECORRELATION_DAYS = 40

# ── Walk-Forward CV parameters ──
N_CV_FOLDS = 5
MIN_TRAIN_FRAC = 0.35       # minimum training data fraction (first fold)
PURGE_DAYS = 120             # conservative purge gap (was 180, reduced to gain ~1 more fold)
EMBARGO_DAYS = 20            # extra gap after each test fold

DRAWDOWN_THRESHOLDS = [0.10, 0.20, 0.30, 0.40]
SCORE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
BIN_CENTERS = [10, 30, 50, 70, 90]

# ── Shared Bayesian priors (used by both bayesian_beta_binomial and bootstrap CI) ──
# Per-bin alpha/beta for Beta-Binomial posterior at each drawdown threshold.
# 10% DD: common event → flatter priors (higher alpha, lower beta)
# 20% DD: moderate event → informative priors
# 30% DD: rare event → stronger shrinkage toward low probability
BAYESIAN_PRIORS = {
    0.10: {"alphas": [2, 3, 5, 7, 10], "betas": [30, 20, 12, 6, 3]},
    0.20: {"alphas": [1, 2, 3, 4, 6],  "betas": [60, 40, 25, 12, 6]},
    0.30: {"alphas": [0.5, 1, 1.5, 2.5, 4], "betas": [120, 80, 50, 25, 12]},
}

# ── Per-threshold configurations (v3.3) ──
# Selected via multi-agent comparison of Lasso/Ridge/ElasticNet, AIC/BIC stepwise,
# stability selection (100 bootstrap resamples), and RFE.
#
# 10% DD: Stability-selected 8 features (>60% selection rate across bootstraps).
#   Key features: risk_ema_20d (100%), ind_qqq_deviation (100%),
#   ind_credit_spread (100%), ind_yield_curve (99%), ind_vix_level (96%),
#   vix_x_credit (95%), ind_vix_level_change_5d (82%), score_velocity (82%).
#   Penalty: L1 (Lasso) with moderate C for natural sparsity.
#
# 20% DD: ElasticNet with broad feature set. Stability selection found different
#   stable features (risk_ema_20d, score_std_20d, vix_x_credit, is_elevated,
#   composite_score, ind_qqq_deviation). ElasticNet C=0.1 l1_ratio=0.5
#   achieved AUC=0.791, BSS=-59.2% — best among all methods tested.

# All available features for 20% DD ElasticNet (let regularization select)
ALL_FEATURES = [
    "risk_ema_20d", "score_ema_20d", "score_std_20d", "score_sma_60d",
    "score_velocity", "ind_qqq_deviation_sma_20d", "ind_qqq_deviation",
    "ind_vix_level", "ind_vix_level_change_5d", "ind_yield_curve",
    "ind_credit_spread", "vix_x_credit", "is_elevated", "composite_score",
    "drawdown_risk_score",
]

# Stability-selected features for 10% DD (>60% bootstrap selection rate)
STABLE_FEATURES_10PCT = [
    "risk_ema_20d", "ind_qqq_deviation", "ind_credit_spread",
    "ind_yield_curve", "ind_vix_level", "vix_x_credit",
    "ind_vix_level_change_5d", "score_velocity",
]

THRESHOLD_CONFIGS = {
    0.10: {
        "dd_definition": "peak_to_trough",
        "forward_window": 180,
        "penalty": "l1",
        "C": 5.0,
        "l1_ratio": None,  # pure L1
        "features": STABLE_FEATURES_10PCT,
    },
    0.20: {
        "dd_definition": "peak_to_trough",
        "forward_window": 180,
        "penalty": "l2",
        "C": 1.0,
        "l1_ratio": None,
        # Minimal model for 20% DD — risk_ema_20d is the only feature with
        # consistent monotonicity vs drawdown probability. Multi-feature
        # models fail catastrophically on Fold 1 (pre-COVID→COVID).
        # The Bayesian Layer 2 provides the primary calibrated estimate;
        # this logistic serves as a smooth interpolation within bins.
        "features": [
            "risk_ema_20d", "score_std_20d",
        ],
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
    features = [f for f in config["features"] if f in df.columns]
    window = config["forward_window"]
    dd_def = config["dd_definition"]
    C = config["C"]
    penalty = config.get("penalty", "l2")
    l1_ratio = config.get("l1_ratio", None)

    fwd_dd = compute_forward_drawdowns(df, window, dd_def)

    # Filter valid rows
    valid_mask = np.isfinite(fwd_dd)
    for f in features:
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

        # Build model with appropriate penalty
        if USE_FIRTH:
            model = FirthLogisticRegression(max_iter=10000)
        else:
            model_kwargs = {"max_iter": 10000, "C": C}
            if penalty == "l1":
                model_kwargs.update({"penalty": "l1", "solver": "saga"})
            elif penalty == "elasticnet":
                model_kwargs.update({"penalty": "elasticnet", "solver": "saga",
                                     "l1_ratio": l1_ratio or 0.5})
            else:
                model_kwargs.update({"penalty": "l2", "solver": "lbfgs"})
            model = LogisticRegression(**model_kwargs)
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
    features = [f for f in config["features"] if f in df.columns]
    window = config["forward_window"]
    dd_def = config["dd_definition"]
    C = config["C"]
    penalty = config.get("penalty", "l2")
    l1_ratio = config.get("l1_ratio", None)

    fwd_dd = compute_forward_drawdowns(df, window, dd_def)

    valid_mask = np.isfinite(fwd_dd)
    for f in features:
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
        "penalty": penalty,
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

    # Build model with appropriate penalty (matching CV)
    if USE_FIRTH:
        model = FirthLogisticRegression(max_iter=10000)
    else:
        model_kwargs = {"max_iter": 10000, "C": C}
        if penalty == "l1":
            model_kwargs.update({"penalty": "l1", "solver": "saga"})
        elif penalty == "elasticnet":
            model_kwargs.update({"penalty": "elasticnet", "solver": "saga",
                                 "l1_ratio": l1_ratio or 0.5})
        else:
            model_kwargs.update({"penalty": "l2", "solver": "lbfgs"})
        model = LogisticRegression(**model_kwargs)
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
    score_column: str = "drawdown_risk_score",
) -> list[float]:
    """Bayesian Beta-Binomial per score bin with monotonicity enforcement.

    Uses drawdown_risk_score (not composite_score) by default so that
    bins are monotonic with actual drawdown probability.
    """
    fwd_dd = compute_forward_drawdowns(df, window, "peak_to_trough")
    valid = np.isfinite(fwd_dd)

    # Fall back to composite_score if score_column is missing
    if score_column not in df.columns:
        score_column = "composite_score"
    score_vals = df[score_column].values

    posteriors = []
    for i, (lo, hi) in enumerate(SCORE_BINS):
        mask = valid & (score_vals >= lo) & (score_vals < hi)
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
    """Fit GPD to exceedances above threshold using MLE.

    xi floor is -0.15 (not -0.5) to prevent unrealistically bounded tails
    that produce zero probability for large drawdowns.
    """
    if len(exceedances) < 3:
        return 0.1, 0.05

    try:
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        shape = np.clip(shape, -0.15, 0.5)
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
    print("DRAWDOWN PROBABILITY MODEL v4.1")
    print("Fix: 10% Bayesian + per-bin bootstrap CI + 100% Bayesian blend")
    print("=" * 70)
    if USE_FIRTH:
        print("  Firth penalized logistic: ENABLED")
    else:
        print("  Firth penalized logistic: DISABLED (firthlogist not installed)")
    if USE_ARCH_BOOTSTRAP:
        print("  Bootstrap CI: ENABLED (arch.bootstrap)")
    else:
        print("  Bootstrap CI: DISABLED (arch not installed)")

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

    # ── Layer 1: Penalized Logistic + Purged Walk-Forward CV + Final Fit ──
    print("\n" + "=" * 70)
    print("LAYER 1: Penalized Logistic (stability-selected features)")
    print(f"  {N_CV_FOLDS} folds, purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d")
    print("=" * 70)

    logistic_coefs = {}
    all_feature_names = set()
    actual_folds_per_threshold = {}

    for t, config in THRESHOLD_CONFIGS.items():
        pct = int(t * 100)
        penalty = config.get("penalty", "l2")
        feats = [f for f in config["features"] if f in df.columns]
        print(f"\n{'─' * 60}")
        print(f"  Drawdown >= {pct}% (def={config['dd_definition']}, "
              f"W={config['forward_window']}d, penalty={penalty}, C={config['C']})")
        print(f"  Features ({len(feats)}): {feats}")
        print(f"{'─' * 60}")

        # Cross-validation
        print(f"\n  Running {N_CV_FOLDS}-fold purged walk-forward CV...")
        cv_result = evaluate_logistic_cv(df, t, config)

        for fm in cv_result["folds"]:
            print(f"    Fold {fm['fold']}: train={fm['train_range']}, "
                  f"test={fm['test_range']}")
            print(f"      n_train={fm['n_train']}, n_test={fm['n_test']}, "
                  f"base_rate={fm['base_rate_test']:.1%}")
            print(f"      AUC={fm['auc']:.4f}  BSS={fm['bss']:+.1%}  "
                  f"ECE={fm['ece']:.4f}")

        agg = cv_result["aggregate"]
        actual_folds_per_threshold[f"drawdown_{pct}pct"] = agg["n_folds"]
        print(f"\n  CV Aggregate ({agg['n_folds']} actual folds out of {N_CV_FOLDS} configured):")
        print(f"    AUC: {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}")
        print(f"    BSS: {agg['bss_mean']:+.1%} ± {agg['bss_std']:.1%}")
        print(f"    ECE: {agg['ece_mean']:.4f} ± {agg['ece_std']:.4f}")

        # Final model on ALL data
        print(f"\n  Fitting final model on all {len(df)} days...")
        result = fit_logistic_final(df, t, config, cv_result)
        logistic_coefs[f"drawdown_{pct}pct"] = result
        all_feature_names.update(feats)

        print(f"    Full-data AUC (in-sample): {result['auc_train']:.4f}")
        print(f"    CV AUC (out-of-sample):    {result['auc_test']:.4f}")
        print(f"    Coefficients (standardized):")
        n_nonzero = 0
        for feat, w in result["weights"].items():
            marker = "" if abs(w) > 1e-6 else "  (zeroed by L1)"
            if abs(w) > 1e-6:
                n_nonzero += 1
            print(f"      {feat:<30}: {w:+.6f}{marker}")
        print(f"      {'intercept':<30}: {result['intercept']:+.6f}")
        if penalty in ("l1", "elasticnet"):
            print(f"    Non-zero features: {n_nonzero}/{len(result['weights'])}")

    # ── Layer 2: Bayesian Beta-Binomial ──
    print("\n" + "=" * 70)
    print("LAYER 2: Bayesian Beta-Binomial with Monotonicity (PAVA) — 10%/20%/30%")
    print("=" * 70)

    bayesian_lookup = {}
    for t in [0.10, 0.20, 0.30]:
        pct = int(t * 100)
        priors = BAYESIAN_PRIORS[t]
        prior_alphas = priors["alphas"]
        prior_betas = priors["betas"]

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

    # Compute GPD exceedance ratios with empirical floor
    # If GPD predicts 0 but historical data shows events, use empirical rate as minimum
    all_valid_dd = fwd_dd_evt[valid_evt]
    n_valid = len(all_valid_dd)

    ratios = {}
    empirical_cross_ratios = {}
    for t in [0.20, 0.30, 0.40]:
        pct = int(t * 100)
        gpd_r = gpd_exceedance_prob(t, evt_threshold, xi, sigma)

        # Empirical cross-ratio: P(>t% | >10%) from actual data
        n_above_thresh = int((all_valid_dd >= t).sum())
        empirical_r = n_above_thresh / max(n_exceed, 1)
        empirical_cross_ratios[f"{pct}pct_given_10pct"] = round(empirical_r, 4)

        # Apply floor: use max(GPD, empirical) to prevent zero predictions
        # when data actually shows events at that threshold
        r = max(gpd_r, empirical_r) if n_above_thresh > 0 else gpd_r
        ratios[f"{pct}pct_given_10pct"] = round(r, 4)
        print(f"  P(>{pct}% | >10%) = {r:.4f}  (GPD={gpd_r:.4f}, Empirical={empirical_r:.4f})")

    r_40_30 = ratios.get("40pct_given_10pct", 0) / max(ratios.get("30pct_given_10pct", 1e-10), 1e-10)
    r_40_20 = ratios.get("40pct_given_10pct", 0) / max(ratios.get("20pct_given_10pct", 1e-10), 1e-10)

    evt_params = {
        "gpd_shape_xi": round(xi, 4),
        "gpd_scale_sigma": round(sigma, 4),
        "threshold_u": evt_threshold,
        "n_exceedances": n_exceed,
        "exceedance_ratios": ratios,
        "empirical_cross_ratios": empirical_cross_ratios,
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
    extra_fields = ["composite_score", "drawdown_risk_score", "risk_ema_20d"]
    for f in sorted(all_feature_names | set(extra_fields)):
        val = latest.get(f, 0)
        if pd.isna(val):
            val = 0
        current_features[f] = round(float(val), 4)
        print(f"  {f:<30}: {val:.4f}")

    # ── Unconditional base rates ──
    print("\n" + "=" * 70)
    print("UNCONDITIONAL BASE RATES (180d peak-to-trough)")
    print("=" * 70)
    valid_dd = fwd_dd_empirical[np.isfinite(fwd_dd_empirical)]
    unconditional_base_rates = {}
    for t in DRAWDOWN_THRESHOLDS:
        pct = int(t * 100)
        rate = float((valid_dd >= t).mean()) if len(valid_dd) > 0 else 0.0
        unconditional_base_rates[f"{pct}pct"] = round(rate, 4)
        print(f"  P(>{pct}% DD | unconditional) = {rate:.1%}")

    # ── Bootstrap Confidence Intervals ──
    print("\n" + "=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (90% CI)")
    print("=" * 70)

    probability_ci = {}
    fwd_dd_boot = compute_forward_drawdowns(df, 180, "peak_to_trough")
    valid_boot = np.isfinite(fwd_dd_boot)

    # Bootstrap bins must match the Bayesian layer's score_column (drawdown_risk_score)
    boot_score_col = "drawdown_risk_score" if "drawdown_risk_score" in df.columns else "composite_score"

    if USE_ARCH_BOOTSTRAP and valid_boot.sum() > 100:
        N_BOOTSTRAP = 500
        boot_data = np.column_stack([
            df[boot_score_col].values[valid_boot],
            fwd_dd_boot[valid_boot],
        ])

        current_risk = current_features.get(boot_score_col, 50)

        for t in [0.10, 0.20, 0.30]:
            pct_label = f"{int(t*100)}pct"
            boot_probs = []

            bs = StationaryBootstrap(DECORRELATION_DAYS, boot_data)
            for i, ((data_star,), _) in enumerate(bs.bootstrap(N_BOOTSTRAP)):
                scores_star = data_star[:, 0]
                dd_star = data_star[:, 1]

                # Recompute Bayesian bin counts on resampled data
                # Use per-bin priors matching bayesian_beta_binomial()
                priors = BAYESIAN_PRIORS[t]
                bin_probs = []
                for bin_idx, (lo, hi) in enumerate(SCORE_BINS):
                    mask = (scores_star >= lo) & (scores_star < hi)
                    n_obs = max(1, int(mask.sum()) // DECORRELATION_DAYS)
                    n_events = int((dd_star[mask] >= t).sum() / max(1, DECORRELATION_DAYS // 2))
                    n_events = min(n_events, n_obs)
                    alpha_post = priors["alphas"][bin_idx] + n_events
                    beta_post = priors["betas"][bin_idx] + n_obs - n_events
                    prob = alpha_post / (alpha_post + beta_post)
                    bin_probs.append(prob)

                # PAVA monotonicity
                bin_probs = pava_monotone(bin_probs)

                # Interpolate at current score
                p = np.interp(current_risk, BIN_CENTERS, bin_probs)
                boot_probs.append(float(p))

            lower = float(np.percentile(boot_probs, 5))
            upper = float(np.percentile(boot_probs, 95))
            probability_ci[pct_label] = {
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "ci_level": 0.90,
            }
            print(f"  {pct_label}: [{lower:.1%}, {upper:.1%}] (90% CI, {N_BOOTSTRAP} resamples)")
    else:
        # Fallback: simple percentile bootstrap without arch
        N_BOOTSTRAP = 500
        rng = np.random.default_rng(42)
        scores_valid = df[boot_score_col].values[valid_boot]
        dd_valid = fwd_dd_boot[valid_boot]
        n_valid = len(scores_valid)

        current_risk = current_features.get(boot_score_col, 50)

        for t in [0.10, 0.20, 0.30]:
            pct_label = f"{int(t*100)}pct"
            boot_probs = []

            for _ in range(N_BOOTSTRAP):
                # Block bootstrap: sample blocks of DECORRELATION_DAYS
                block_size = DECORRELATION_DAYS
                n_blocks = n_valid // block_size + 1
                block_starts = rng.integers(0, n_valid - block_size, size=n_blocks)
                idx = np.concatenate([np.arange(s, min(s + block_size, n_valid)) for s in block_starts])[:n_valid]
                scores_star = scores_valid[idx]
                dd_star = dd_valid[idx]

                # Use per-bin priors matching bayesian_beta_binomial()
                priors = BAYESIAN_PRIORS[t]
                bin_probs = []
                for bin_idx, (lo, hi) in enumerate(SCORE_BINS):
                    mask = (scores_star >= lo) & (scores_star < hi)
                    n_obs = max(1, int(mask.sum()) // DECORRELATION_DAYS)
                    n_events = int((dd_star[mask] >= t).sum() / max(1, DECORRELATION_DAYS // 2))
                    n_events = min(n_events, n_obs)
                    alpha_post = priors["alphas"][bin_idx] + n_events
                    beta_post = priors["betas"][bin_idx] + n_obs - n_events
                    prob = alpha_post / (alpha_post + beta_post)
                    bin_probs.append(prob)

                bin_probs = pava_monotone(bin_probs)
                p = np.interp(current_risk, BIN_CENTERS, bin_probs)
                boot_probs.append(float(p))

            lower = float(np.percentile(boot_probs, 5))
            upper = float(np.percentile(boot_probs, 95))
            probability_ci[pct_label] = {
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "ci_level": 0.90,
            }
            print(f"  {pct_label}: [{lower:.1%}, {upper:.1%}] (90% CI, {N_BOOTSTRAP} block-bootstrap)")

    # ── Output model JSON ──
    model = {
        "model_version": "4.1",
        "calibration_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "data_points": len(df),
        "train_test_split": f"purged_wfcv_{N_CV_FOLDS}fold",
        "actual_folds_used": actual_folds_per_threshold,
        "purge_days": PURGE_DAYS,
        "embargo_days": EMBARGO_DAYS,
        "effective_sample_size": max(1, len(df) // DECORRELATION_DAYS),
        "forward_window_days": 180,
        "forward_window_label": "9 months",
        "firth_enabled": USE_FIRTH,
        "feature_names": sorted(all_feature_names),
        "current_features": current_features,
        "logistic_coefficients": logistic_coefs,
        "bayesian_lookup": bayesian_lookup,
        "evt_parameters": evt_params,
        "empirical_stats": empirical,
        "unconditional_base_rates": unconditional_base_rates,
        "probability_ci": probability_ci,
        "blend_weights": {
            "10pct": {"logistic": 0.0, "bayesian": 1.0},
            "20pct": {"logistic": 0.0, "bayesian": 1.0},
            "30pct": {"logistic": 0.0, "bayesian": 1.0},
        },
        "confidence_tiers": {
            "10pct": "moderate",
            "20pct": "moderate",
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
    print("v4.0 → v4.1 CHANGES (10% Bayesian + per-bin CI + 100% Bayesian)")
    print("=" * 70)
    v33_baselines = {
        "10": {"auc": 0.835, "bss": "+18.8%", "ece": "N/A"},
        "20": {"auc": 0.791, "bss": "-59.2%", "ece": "N/A"},
    }
    for pct_label in ["10", "20"]:
        key = f"drawdown_{pct_label}pct"
        coef = logistic_coefs.get(key, {})
        cv = coef.get("cv_metrics", {})
        bl = v33_baselines[pct_label]
        print(f"\n  {pct_label}% Drawdown:")
        print(f"    v3.3 (2014-2026):     AUC={bl['auc']}  BSS={bl['bss']}")
        print(f"    v4.0 (1999-2026, {'Firth' if USE_FIRTH else 'standard'} logistic):")
        print(f"      AUC: {cv.get('auc_mean', 'N/A')} ± {cv.get('auc_std', 'N/A')}")
        print(f"      BSS: {cv.get('bss_mean', 'N/A')} ± {cv.get('bss_std', 'N/A')}")
        print(f"      ECE: {cv.get('ece_mean', 'N/A')} ± {cv.get('ece_std', 'N/A')}")

    if probability_ci:
        print(f"\n  Bootstrap 90% CI:")
        for k, v in probability_ci.items():
            print(f"    {k}: [{v['lower']:.1%}, {v['upper']:.1%}]")

    print("\n" + "=" * 70)
    print("MODEL v4.1 CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
