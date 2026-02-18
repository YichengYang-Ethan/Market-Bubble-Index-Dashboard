#!/usr/bin/env python3
"""Fit drawdown probability model from bubble history + QQQ price data.

Hybrid 3-layer model:
  Layer 1: Logistic regression for 10% and 20% thresholds
  Layer 2: Bayesian Beta-Binomial with monotonicity for 20% and 30%
  Layer 3: EVT (GPD) tail extrapolation for 40%

Output: public/data/drawdown_model.json
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

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "public" / "data"

FORWARD_WINDOW = 63  # trading days (~3 months)
DECORRELATION_DAYS = 40  # for effective sample size
MIN_EPISODE_GAP = 30  # min gap between drawdown episodes

DRAWDOWN_THRESHOLDS = [0.10, 0.20, 0.30, 0.40]
SCORE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
BIN_CENTERS = [10, 30, 50, 70, 90]


def load_data() -> pd.DataFrame:
    """Load and merge bubble history with QQQ price data."""
    history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
    qqq_raw = json.loads((DATA_DIR / "qqq.json").read_text())
    qqq = qqq_raw["data"] if isinstance(qqq_raw, dict) and "data" in qqq_raw else qqq_raw

    df_h = pd.DataFrame(history)[["date", "composite_score", "score_velocity"]].copy()
    df_h["score_velocity"] = df_h["score_velocity"].fillna(0)

    df_q = pd.DataFrame(qqq)[["date", "price"]].copy()
    df_q.rename(columns={"price": "qqq_price"}, inplace=True)

    df = df_h.merge(df_q, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return df


def compute_forward_drawdowns(df: pd.DataFrame) -> pd.DataFrame:
    """For each day, compute max forward drawdown over FORWARD_WINDOW days."""
    prices = df["qqq_price"].values
    n = len(prices)

    max_dd = np.full(n, np.nan)
    for i in range(n):
        end = min(i + FORWARD_WINDOW + 1, n)
        if end - i < 5:
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
        dd[i] = (p - peak) / peak  # negative values
    df = df.copy()
    df["drawdown"] = dd
    return df


def subsample_episodes(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Subsample to approximately independent episodes for logistic regression."""
    # For event days: group consecutive event days, take first of each cluster
    # For non-event days: take one per DECORRELATION_DAYS block
    is_event = df["max_forward_dd"] >= threshold
    rows = []

    # Events: cluster with MIN_EPISODE_GAP
    event_indices = df.index[is_event].tolist()
    if event_indices:
        clusters = [[event_indices[0]]]
        for idx in event_indices[1:]:
            if idx - clusters[-1][-1] <= MIN_EPISODE_GAP:
                clusters[-1].append(idx)
            else:
                clusters.append([idx])
        for cluster in clusters:
            rows.append(cluster[0])

    # Non-events: one per decorrelation window
    non_event = df[~is_event]
    if len(non_event) > 0:
        step = max(1, DECORRELATION_DAYS)
        rows.extend(non_event.index[::step].tolist())

    return df.loc[sorted(set(rows))].copy()


def fit_logistic(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit logistic regression with Firth-like penalty for small samples."""
    # P(y=1) = sigmoid(a*x + b)
    # Add small L2 penalty for stability

    def neg_log_likelihood(params):
        a, b = params
        z = a * X + b
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        # Firth-like penalty
        penalty = 0.5 * (a**2 + b**2) * 0.01
        return -ll + penalty

    result = minimize(neg_log_likelihood, [0.01, -2.0], method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])


def bayesian_beta_binomial(
    df: pd.DataFrame, threshold: float, prior_alphas: list[float], prior_betas: list[float]
) -> list[float]:
    """Bayesian Beta-Binomial per score bin with monotonicity enforcement."""
    posteriors = []
    for i, (lo, hi) in enumerate(SCORE_BINS):
        mask = (df["composite_score"] >= lo) & (df["composite_score"] < hi)
        subset = df[mask]
        n_obs = max(1, len(subset) // DECORRELATION_DAYS)  # effective sample size
        n_events = int((subset["max_forward_dd"] >= threshold).sum() / max(1, DECORRELATION_DAYS // 2))
        n_events = min(n_events, n_obs)

        alpha_post = prior_alphas[i] + n_events
        beta_post = prior_betas[i] + n_obs - n_events
        posteriors.append(alpha_post / (alpha_post + beta_post))

    # Enforce monotonicity via Pool Adjacent Violators (PAVA)
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
                # Pool
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
        # Not enough data, return conservative defaults
        return 0.1, 0.05

    try:
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        # Bound shape parameter for stability
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
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} merged trading days")

    print("Computing forward drawdowns...")
    df = compute_forward_drawdowns(df)
    print(f"  {len(df)} days with forward drawdown data")

    # Also compute rolling drawdown for the chart overlay data
    df_full = load_data()
    df_full = compute_rolling_drawdown(df_full)
    drawdown_series = [
        {"date": row["date"], "drawdown": round(row["drawdown"] * 100, 2)}
        for _, row in df_full.iterrows()
    ]

    # Empirical stats per bin
    print("\nEmpirical drawdown statistics per score bin:")
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
    # Layer 1: Logistic regression for 10% and 20%
    # -----------------------------------------------------------------------
    print("\nFitting logistic regression (Firth-penalized)...")
    logistic_coefs = {}
    for t in [0.10, 0.20]:
        pct = int(t * 100)
        sub = subsample_episodes(df, t)
        X = sub["composite_score"].values
        y = (sub["max_forward_dd"] >= t).astype(float).values
        a, b = fit_logistic(X, y)
        n_events = int(y.sum())
        n_total = len(y)

        # Also fit with velocity
        # P(y=1) = sigmoid(a1*score + a2*velocity + b)
        def neg_ll_2d(params):
            a1, a2, b0 = params
            z = a1 * sub["composite_score"].values + a2 * sub["score_velocity"].values + b0
            z = np.clip(z, -30, 30)
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            penalty = 0.5 * (a1**2 + a2**2 + b0**2) * 0.01
            return -ll + penalty

        res2 = minimize(neg_ll_2d, [a, 0.0, b], method="Nelder-Mead")
        a1, a2, b0 = res2.x

        logistic_coefs[f"drawdown_{pct}pct"] = {
            "a": round(float(a), 6),
            "b": round(float(b), 4),
            "a_velocity": round(float(a2), 6),
            "b_with_velocity": round(float(b0), 4),
            "a_score_with_velocity": round(float(a1), 6),
            "n_events": n_events,
            "n_total": n_total,
        }
        print(f"  {pct}%: a={a:.4f}, b={b:.3f} (events={n_events}/{n_total})")
        print(f"    +velocity: a_score={a1:.4f}, a_vel={a2:.4f}, b={b0:.3f}")

    # -----------------------------------------------------------------------
    # Layer 2: Bayesian Beta-Binomial for 20% and 30%
    # -----------------------------------------------------------------------
    print("\nFitting Bayesian Beta-Binomial with monotonicity...")
    bayesian_lookup = {}

    # Priors informed by longer NASDAQ history (1971-present):
    # 20% drawdown within 63 days: ~15-25% unconditional probability in bear markets
    # Higher score bins get stronger priors toward higher probability
    for t in [0.20, 0.30]:
        pct = int(t * 100)
        if t == 0.20:
            prior_alphas = [1, 1.5, 2, 3, 5]
            prior_betas = [80, 50, 30, 15, 8]
        else:  # 0.30
            prior_alphas = [0.5, 0.8, 1, 2, 3]
            prior_betas = [150, 100, 60, 30, 15]

        probs = bayesian_beta_binomial(df, t, prior_alphas, prior_betas)
        bayesian_lookup[f"drawdown_{pct}pct"] = {
            "bin_centers": BIN_CENTERS,
            "probabilities": [round(p, 4) for p in probs],
        }
        print(f"  {pct}%: {[f'{p:.2%}' for p in probs]}")

    # -----------------------------------------------------------------------
    # Layer 3: EVT (GPD) for tail extrapolation
    # -----------------------------------------------------------------------
    print("\nFitting GPD for tail extrapolation...")
    evt_threshold = 0.10  # exceedances above 10%
    exceedances = df[df["max_forward_dd"] >= evt_threshold]["max_forward_dd"].values - evt_threshold
    n_exceed = len(exceedances)

    xi, sigma = fit_gpd(exceedances)
    print(f"  GPD: xi={xi:.4f}, sigma={sigma:.4f}, n_exceedances={n_exceed}")

    # Compute extrapolation ratios
    ratios = {}
    for t in [0.20, 0.30, 0.40]:
        pct = int(t * 100)
        r = gpd_exceedance_prob(t, evt_threshold, xi, sigma)
        ratios[f"{pct}pct_given_10pct"] = round(r, 4)
        print(f"  P(>{pct}% | >10%) = {r:.4f}")

    # Cross-threshold ratios
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
    # Current score probabilities
    # -----------------------------------------------------------------------
    current_score = df["composite_score"].iloc[-1]
    current_velocity = df["score_velocity"].iloc[-1]
    print(f"\nCurrent score: {current_score:.1f}, velocity: {current_velocity:.1f}")

    current_probs = {}
    for pct_key, coefs in logistic_coefs.items():
        z = coefs["a"] * current_score + coefs["b"]
        p = 1 / (1 + np.exp(-z))
        z_v = coefs["a_score_with_velocity"] * current_score + coefs["a_velocity"] * current_velocity + coefs["b_with_velocity"]
        p_v = 1 / (1 + np.exp(-z_v))
        current_probs[pct_key] = {
            "logistic": round(float(p), 4),
            "logistic_with_velocity": round(float(p_v), 4),
        }
        print(f"  {pct_key}: logistic={p:.2%}, +velocity={p_v:.2%}")

    # -----------------------------------------------------------------------
    # Output model
    # -----------------------------------------------------------------------
    model = {
        "model_version": "1.0",
        "calibration_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "data_points": len(df),
        "effective_sample_size": max(1, len(df) // DECORRELATION_DAYS),
        "forward_window_days": FORWARD_WINDOW,
        "forward_window_label": "3 months",
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
        "current_probabilities": current_probs,
    }

    # Write model coefficients
    out_path = DATA_DIR / "drawdown_model.json"
    out_path.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(f"\nModel written to {out_path}")

    # Write drawdown series for chart overlay
    dd_out = DATA_DIR / "qqq_drawdown.json"
    dd_out.write_text(json.dumps(drawdown_series, ensure_ascii=False) + "\n")
    print(f"Drawdown series written to {dd_out} ({len(drawdown_series)} points)")


if __name__ == "__main__":
    main()
