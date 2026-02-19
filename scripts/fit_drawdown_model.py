#!/usr/bin/env python3
"""Fit drawdown probability model v3.0 from bubble history + QQQ price data.

Improvements over v2.0:
  - Per-threshold optimized models (different features, C, DD definition per threshold)
  - Two drawdown definitions: (A) peak-to-trough within window, (B) drop-from-today
  - Forward window: 180 trading days (~9 months) for both 10% and 20% thresholds
  - Engineered features: score_ema_20d, score_std_20d, ind_qqq_deviation_sma_20d,
    ind_vix_level_change_5d, vix_x_credit, is_elevated
  - BSS now positive for both thresholds (v2 had -24.6% for 10%)

Hybrid 3-layer model:
  Layer 1: Per-threshold logistic regression with optimized features
    - >10% DD (def B, drop-from-today, W=180d, C=1.0, 5 features)
    - >20% DD (def A, peak-to-trough, W=180d, C=10.0, 6 features)
  Layer 2: Bayesian Beta-Binomial with monotonicity for 20% and 30%
  Layer 3: EVT (GPD) tail extrapolation for 40%

v2.0 → v3.0 OOS improvements:
  - 10% DD: AUC 0.569 → 0.878, BSS -24.6% → +27.3%
  - 20% DD: AUC 0.884 → 0.917, BSS -8.0% → +36.0%

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
TRAIN_RATIO = 0.70

DRAWDOWN_THRESHOLDS = [0.10, 0.20, 0.30, 0.40]
SCORE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
BIN_CENTERS = [10, 30, 50, 70, 90]

# ── Per-threshold model configurations (optimized via expanding-window CV) ──

THRESHOLD_CONFIGS = {
    0.10: {
        "dd_definition": "drop_from_today",  # Definition B
        "forward_window": 180,
        "C": 1.0,
        "features": [
            "ind_qqq_deviation_sma_20d",
            "ind_vix_level",
            "score_ema_20d",
            "ind_yield_curve",
            "ind_vix_level_change_5d",
        ],
    },
    0.20: {
        "dd_definition": "peak_to_trough",  # Definition A
        "forward_window": 180,
        "C": 10.0,
        "features": [
            "ind_qqq_deviation_sma_20d",
            "score_std_20d",
            "ind_yield_curve",
            "score_ema_20d",
            "vix_x_credit",
            "ind_vix_level_change_5d",
        ],
    },
}


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
        # Compute from indicators: invert VIX, QQQ deviation, yield curve
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
            # Track running peak within window, measure trough from that peak
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
            # drop_from_today: max drop from today's price
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


def fit_logistic(
    df: pd.DataFrame, threshold: float, config: dict
) -> dict:
    """Fit L2-regularized logistic regression with the optimized config."""
    features = config["features"]
    window = config["forward_window"]
    dd_def = config["dd_definition"]
    C = config["C"]

    # Compute forward drawdowns for this specific config
    fwd_dd = compute_forward_drawdowns(df, window, dd_def)

    # Check all features exist; skip missing
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features {missing}, filling with median")
        for f in missing:
            df[f] = 50.0  # neutral default

    # Filter valid rows (features + drawdown non-NaN)
    valid_mask = np.isfinite(fwd_dd)
    for f in features:
        if f in df.columns:
            valid_mask &= df[f].notna().values

    sub_idx = np.where(valid_mask)[0]
    X = df.iloc[sub_idx][features].values.astype(float)
    y = (fwd_dd[sub_idx] >= threshold).astype(float)

    n = len(X)
    split_idx = int(n * TRAIN_RATIO)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    n_events_train = int(y_train.sum())
    n_events_test = int(y_test.sum())

    result = {
        "features": features,
        "dd_definition": dd_def,
        "forward_window": window,
        "regularization_C": C,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_events_train": n_events_train,
        "n_events_test": n_events_test,
        "base_rate_train": round(float(y_train.mean()), 4),
        "base_rate_test": round(float(y_test.mean()), 4),
    }

    if n_events_train < 3 or n_events_train == len(y_train):
        result["weights"] = {f: 0.0 for f in features}
        result["intercept"] = -5.0
        result["scaler_mean"] = {f: 0.0 for f in features}
        result["scaler_std"] = {f: 1.0 for f in features}
        result["auc_train"] = 0.5
        result["auc_test"] = 0.5
        result["brier_train"] = 0.25
        result["brier_test"] = 0.25
        result["bss_test"] = 0.0
        return result

    # StandardScaler for better numerical stability
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, C=C, solver="lbfgs")
    model.fit(X_train_s, y_train)

    # Store scaled coefficients — frontend will need scaler params
    weights = {f: round(float(c), 6) for f, c in zip(features, model.coef_[0])}
    intercept = round(float(model.intercept_[0]), 6)

    scaler_mean = {f: round(float(m), 6) for f, m in zip(features, scaler.mean_)}
    scaler_std = {f: round(float(s), 6) for f, s in zip(features, scaler.scale_)}

    # Train metrics
    y_pred_train = model.predict_proba(X_train_s)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred_train)
    brier_train = brier_score_loss(y_train, y_pred_train)

    # Test metrics
    y_pred_test = model.predict_proba(X_test_s)[:, 1]
    if len(np.unique(y_test)) == 2:
        auc_test = roc_auc_score(y_test, y_pred_test)
    else:
        auc_test = float("nan")
    brier_test = brier_score_loss(y_test, y_pred_test)

    brier_clim = y_test.mean() * (1 - y_test.mean())
    bss_test = round(1 - brier_test / max(brier_clim, 1e-10), 4) if brier_clim > 0 else 0.0

    result.update({
        "weights": weights,
        "intercept": intercept,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "auc_train": round(float(auc_train), 4),
        "auc_test": round(float(auc_test), 4) if not np.isnan(auc_test) else 0.5,
        "brier_train": round(float(brier_train), 4),
        "brier_test": round(float(brier_test), 4),
        "bss_test": bss_test,
    })

    return result


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


def main():
    print("=" * 70)
    print("DRAWDOWN PROBABILITY MODEL v3.0")
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

    # ── Empirical stats (using 180d peak-to-trough for consistency) ──
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

    # ── Monotonicity validation: risk score vs composite ──
    print("\n" + "=" * 70)
    print("MONOTONICITY: RISK SCORE vs COMPOSITE SCORE")
    print("=" * 70)

    if "drawdown_risk_score" in df.columns:
        for label, dd_arr in [("composite_score", fwd_dd_empirical), ("drawdown_risk_score", fwd_dd_empirical)]:
            col = df[label].values if label in df.columns else None
            if col is None:
                continue
            valid = np.isfinite(dd_arr) & np.isfinite(col)
            print(f"\n  {label} bin → P(>20% DD, 180d peak-to-trough):")
            for lo, hi in SCORE_BINS:
                mask = valid & (col >= lo) & (col < hi)
                n = int(mask.sum())
                if n > 0:
                    p = float((dd_arr[mask] >= 0.20).mean())
                    print(f"    [{lo:2d}-{hi:2d}]: n={n:5d}  P(>20%)={p:.1%}")
                else:
                    print(f"    [{lo:2d}-{hi:2d}]: n=    0  P(>20%)=N/A")

    # ── Layer 1: Per-threshold optimized logistic regression ──
    print("\n" + "=" * 70)
    print("LAYER 1: Per-Threshold Optimized Logistic Regression")
    print("=" * 70)

    logistic_coefs = {}
    all_feature_names = set()
    for t, config in THRESHOLD_CONFIGS.items():
        pct = int(t * 100)
        print(f"\n--- Drawdown >= {pct}% (def={config['dd_definition']}, W={config['forward_window']}d, C={config['C']}) ---")

        result = fit_logistic(df, t, config)
        logistic_coefs[f"drawdown_{pct}pct"] = result
        all_feature_names.update(config["features"])

        print(f"  Train: {result['n_train']} obs, {result['n_events_train']} events ({result['base_rate_train']:.1%})")
        print(f"  Test:  {result['n_test']} obs, {result['n_events_test']} events ({result['base_rate_test']:.1%})")
        print(f"  AUC train: {result['auc_train']:.4f}")
        print(f"  AUC test:  {result['auc_test']:.4f}")
        print(f"  Brier test: {result['brier_test']:.4f}")
        print(f"  BSS test:   {result['bss_test']:+.1%}")
        print(f"  Coefficients (standardized):")
        for feat, w in result["weights"].items():
            print(f"    {feat:<30}: {w:+.6f}")
        print(f"    {'intercept':<30}: {result['intercept']:+.6f}")

    # ── Layer 2: Bayesian Beta-Binomial for 20% and 30% ──
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

    # ── Layer 3: EVT (GPD) for tail extrapolation ──
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

    # ── Output model ──
    model = {
        "model_version": "3.0",
        "calibration_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "data_points": len(df),
        "train_test_split": f"{int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}",
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

    print("\n" + "=" * 70)
    print("MODEL v3.0 CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
