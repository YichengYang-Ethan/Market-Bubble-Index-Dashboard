#!/usr/bin/env python3
"""Fit drawdown probability model v2.0 from bubble history + QQQ price data.

Improvements over v1.0:
  - Forward window: 126 trading days (~6 months) instead of 63
  - Multi-feature logistic regression (5 features instead of composite_score only)
  - Proper 70/30 chronological train/test split with OOS metrics
  - Derived features: score_sma_60d, qqq_vol_60d

Hybrid 3-layer model:
  Layer 1: Multi-feature logistic regression for 10% and 20% thresholds
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
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "public" / "data"

FORWARD_WINDOW = 126  # trading days (~6 months) — v2.0 change from 63
DECORRELATION_DAYS = 40
MIN_EPISODE_GAP = 30
TRAIN_RATIO = 0.70

DRAWDOWN_THRESHOLDS = [0.10, 0.20, 0.30, 0.40]
SCORE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
BIN_CENTERS = [10, 30, 50, 70, 90]

# Features used in the multi-feature logistic model
FEATURE_COLS = [
    "composite_score",
    "score_velocity",
    "ind_vix_level",
    "ind_credit_spread",
    "ind_qqq_deviation",
    "score_sma_60d",
]


def load_data() -> pd.DataFrame:
    """Load and merge bubble history with QQQ price data, extract all features."""
    history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
    qqq_raw = json.loads((DATA_DIR / "qqq.json").read_text())
    qqq = qqq_raw["data"] if isinstance(qqq_raw, dict) and "data" in qqq_raw else qqq_raw

    df_h = pd.DataFrame(history)

    # Extract indicators from nested dict
    indicators = pd.json_normalize(df_h["indicators"])
    indicators.columns = ["ind_" + c for c in indicators.columns]
    df_h = pd.concat([df_h.drop(columns=["indicators"]), indicators], axis=1)

    df_h["score_velocity"] = df_h["score_velocity"].fillna(0)

    # Derived feature: 60-day SMA of composite score
    df_h["score_sma_60d"] = df_h["composite_score"].rolling(60, min_periods=20).mean()

    df_q = pd.DataFrame(qqq)[["date", "price"]].copy()
    df_q.rename(columns={"price": "qqq_price"}, inplace=True)

    # Derived feature: 60-day annualized volatility of QQQ
    df_q["qqq_returns"] = df_q["qqq_price"].pct_change()
    df_q["qqq_vol_60d"] = df_q["qqq_returns"].rolling(60, min_periods=20).std() * np.sqrt(252) * 100

    df = df_h.merge(df_q, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Fill NaN indicators with median for robustness
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def compute_forward_drawdowns(df: pd.DataFrame) -> pd.DataFrame:
    """For each day, compute max forward drawdown over FORWARD_WINDOW days."""
    prices = df["qqq_price"].values
    n = len(prices)

    max_dd = np.full(n, np.nan)
    for i in range(n):
        end = min(i + FORWARD_WINDOW + 1, n)
        if end - i < 10:
            continue
        window = prices[i:end]
        peak = prices[i]
        dd = 0.0
        for p in window[1:]:
            if p > peak:
                peak = p
            drawdown = (peak - p) / peak
            if drawdown > dd:
                dd = drawdown
        max_dd[i] = dd

    df = df.copy()
    df["max_forward_dd"] = max_dd
    return df.dropna(subset=["max_forward_dd"])


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


def fit_multi_feature_logistic(
    df: pd.DataFrame, threshold: float, features: list[str]
) -> dict:
    """Fit L2-regularized logistic regression with train/test split and OOS metrics."""
    # Prepare data
    valid_mask = df[features].notna().all(axis=1)
    sub = df[valid_mask].copy()
    X = sub[features].values
    y = (sub["max_forward_dd"] >= threshold).astype(float).values

    n = len(X)
    split_idx = int(n * TRAIN_RATIO)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    n_events_train = int(y_train.sum())
    n_events_test = int(y_test.sum())

    result = {
        "features": features,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_events_train": n_events_train,
        "n_events_test": n_events_test,
        "base_rate_train": round(float(y_train.mean()), 4),
        "base_rate_test": round(float(y_test.mean()), 4),
    }

    if n_events_train < 3 or n_events_train == len(y_train):
        # Can't fit model
        result["weights"] = {f: 0.0 for f in features}
        result["intercept"] = -5.0
        result["auc_train"] = 0.5
        result["auc_test"] = 0.5
        result["brier_train"] = 0.25
        result["brier_test"] = 0.25
        return result

    # Fit L2-regularized logistic regression (C=1.0 = moderate regularization)
    model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    model.fit(X_train, y_train)

    # Coefficients
    weights = {f: round(float(c), 6) for f, c in zip(features, model.coef_[0])}
    intercept = round(float(model.intercept_[0]), 4)

    # Train metrics
    y_pred_train = model.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred_train)
    brier_train = brier_score_loss(y_train, y_pred_train)

    # Test metrics
    y_pred_test = model.predict_proba(X_test)[:, 1]
    if len(np.unique(y_test)) == 2:
        auc_test = roc_auc_score(y_test, y_pred_test)
    else:
        auc_test = float("nan")
    brier_test = brier_score_loss(y_test, y_pred_test)

    # Brier Skill Score (BSS) = 1 - Brier / Brier_climatology
    brier_clim = y_test.mean() * (1 - y_test.mean())
    bss_test = round(1 - brier_test / max(brier_clim, 1e-10), 4) if brier_clim > 0 else 0.0

    result.update({
        "weights": weights,
        "intercept": intercept,
        "auc_train": round(float(auc_train), 4),
        "auc_test": round(float(auc_test), 4) if not np.isnan(auc_test) else 0.5,
        "brier_train": round(float(brier_train), 4),
        "brier_test": round(float(brier_test), 4),
        "bss_test": bss_test,
    })

    return result


def bayesian_beta_binomial(
    df: pd.DataFrame, threshold: float, prior_alphas: list[float], prior_betas: list[float]
) -> list[float]:
    """Bayesian Beta-Binomial per score bin with monotonicity enforcement."""
    posteriors = []
    for i, (lo, hi) in enumerate(SCORE_BINS):
        mask = (df["composite_score"] >= lo) & (df["composite_score"] < hi)
        subset = df[mask]
        n_obs = max(1, len(subset) // DECORRELATION_DAYS)
        n_events = int((subset["max_forward_dd"] >= threshold).sum() / max(1, DECORRELATION_DAYS // 2))
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


def main():
    print("=" * 70)
    print("DRAWDOWN PROBABILITY MODEL v2.0")
    print("=" * 70)

    print("\nLoading data...")
    df = load_data()
    print(f"  {len(df)} merged trading days")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"  Features: {FEATURE_COLS}")

    print("\nComputing forward drawdowns (126-day window)...")
    df = compute_forward_drawdowns(df)
    print(f"  {len(df)} days with forward drawdown data")
    print(f"  Forward DD stats: mean={df['max_forward_dd'].mean():.2%}, "
          f"median={df['max_forward_dd'].median():.2%}, "
          f"max={df['max_forward_dd'].max():.2%}")

    # Rolling drawdown for chart overlay
    df_full = load_data()
    df_full = compute_rolling_drawdown(df_full)
    drawdown_series = [
        {"date": row["date"], "drawdown": round(row["drawdown"] * 100, 2)}
        for _, row in df_full.iterrows()
    ]

    # -----------------------------------------------------------------------
    # Empirical stats per bin
    # -----------------------------------------------------------------------
    print("\nEmpirical drawdown statistics per score bin (126d window):")
    empirical = {}
    for lo, hi in SCORE_BINS:
        mask = (df["composite_score"] >= lo) & (df["composite_score"] < hi)
        subset = df[mask]
        key = f"{lo}-{hi}"
        bin_stats = {"count": int(len(subset))}
        for t in DRAWDOWN_THRESHOLDS:
            pct = int(t * 100)
            n_exceed = int((subset["max_forward_dd"] >= t).sum())
            prob = n_exceed / max(len(subset), 1)
            bin_stats[f"p_{pct}pct"] = round(prob, 4)
            bin_stats[f"n_{pct}pct"] = n_exceed
        if len(subset) > 0:
            bin_stats["mean_dd"] = round(float(subset["max_forward_dd"].mean()) * 100, 2)
            bin_stats["median_dd"] = round(float(subset["max_forward_dd"].median()) * 100, 2)
            bin_stats["p95_dd"] = round(float(subset["max_forward_dd"].quantile(0.95)) * 100, 2)
        empirical[key] = bin_stats
        print(f"  [{key}] n={bin_stats['count']}  "
              f"P(>10%)={bin_stats.get('p_10pct', 0):.1%}  "
              f"P(>20%)={bin_stats.get('p_20pct', 0):.1%}  "
              f"P(>30%)={bin_stats.get('p_30pct', 0):.1%}")

    # -----------------------------------------------------------------------
    # Layer 1: Multi-feature logistic regression for 10% and 20%
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LAYER 1: Multi-feature Logistic Regression (L2-regularized)")
    print("=" * 70)

    logistic_coefs = {}
    for t in [0.10, 0.20]:
        pct = int(t * 100)
        print(f"\n--- Drawdown >= {pct}% ---")

        result = fit_multi_feature_logistic(df, t, FEATURE_COLS)
        logistic_coefs[f"drawdown_{pct}pct"] = result

        print(f"  Train: {result['n_train']} obs, {result['n_events_train']} events ({result['base_rate_train']:.1%})")
        print(f"  Test:  {result['n_test']} obs, {result['n_events_test']} events ({result['base_rate_test']:.1%})")
        print(f"  AUC train: {result['auc_train']:.4f}")
        print(f"  AUC test:  {result['auc_test']:.4f}")
        print(f"  Brier train: {result['brier_train']:.4f}")
        print(f"  Brier test:  {result['brier_test']:.4f}")
        print(f"  BSS test:    {result.get('bss_test', 'N/A')}")
        print(f"  Coefficients:")
        for feat, w in result["weights"].items():
            print(f"    {feat:<22}: {w:+.6f}")
        print(f"    {'intercept':<22}: {result['intercept']:+.4f}")

    # -----------------------------------------------------------------------
    # Layer 2: Bayesian Beta-Binomial for 20% and 30%
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LAYER 2: Bayesian Beta-Binomial with Monotonicity (PAVA)")
    print("=" * 70)

    bayesian_lookup = {}
    for t in [0.20, 0.30]:
        pct = int(t * 100)
        # Priors informed by extended NASDAQ history
        # With 126d window, drawdown probabilities are higher than 63d
        if t == 0.20:
            prior_alphas = [1, 2, 3, 4, 6]
            prior_betas = [60, 40, 25, 12, 6]
        else:  # 0.30
            prior_alphas = [0.5, 1, 1.5, 2.5, 4]
            prior_betas = [120, 80, 50, 25, 12]

        probs = bayesian_beta_binomial(df, t, prior_alphas, prior_betas)
        bayesian_lookup[f"drawdown_{pct}pct"] = {
            "bin_centers": BIN_CENTERS,
            "probabilities": [round(p, 4) for p in probs],
        }
        print(f"  {pct}%: {[f'{p:.2%}' for p in probs]}")

    # -----------------------------------------------------------------------
    # Layer 3: EVT (GPD) for tail extrapolation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LAYER 3: EVT/GPD Tail Extrapolation")
    print("=" * 70)

    evt_threshold = 0.10
    exceedances = df[df["max_forward_dd"] >= evt_threshold]["max_forward_dd"].values - evt_threshold
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

    # -----------------------------------------------------------------------
    # Current predictions
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CURRENT PREDICTIONS")
    print("=" * 70)

    latest = df.iloc[-1]
    current_features = {}
    for f in FEATURE_COLS:
        val = latest.get(f, 0)
        if pd.isna(val):
            val = 0
        current_features[f] = round(float(val), 2)
        print(f"  {f:<22}: {val:.2f}")

    # -----------------------------------------------------------------------
    # Output model
    # -----------------------------------------------------------------------
    model = {
        "model_version": "2.0",
        "calibration_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "data_points": len(df),
        "train_test_split": f"{int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}",
        "effective_sample_size": max(1, len(df) // DECORRELATION_DAYS),
        "forward_window_days": FORWARD_WINDOW,
        "forward_window_label": "6 months",
        "feature_names": FEATURE_COLS,
        "current_features": current_features,
        "logistic_coefficients": logistic_coefs,
        "bayesian_lookup": bayesian_lookup,
        "evt_parameters": evt_params,
        "empirical_stats": empirical,
        "confidence_tiers": {
            "10pct": "moderate",
            "20pct": "low",
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

    print("\n" + "=" * 70)
    print("MODEL v2.0 CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
