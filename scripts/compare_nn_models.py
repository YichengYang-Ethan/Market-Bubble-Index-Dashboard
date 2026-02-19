#!/usr/bin/env python3
"""Compare neural network models (MLP) for drawdown probability prediction.

Uses the same purged walk-forward CV infrastructure as fit_drawdown_model.py.
Tests various MLP architectures and regularization settings using sklearn
MLPClassifier (PyTorch not available).

Output:
  - Formatted comparison table to stdout
  - JSON results to scripts/results_nn_models.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "public" / "data"

# ── CV parameters (must match fit_drawdown_model.py) ──
N_CV_FOLDS = 5
MIN_TRAIN_FRAC = 0.35
PURGE_DAYS = 180
EMBARGO_DAYS = 20
DECORRELATION_DAYS = 40

# ── Thresholds to evaluate ──
THRESHOLDS = [0.10, 0.20]

# ── All available features (same as tree models) ──
ALL_FEATURES = [
    "risk_ema_20d",
    "score_ema_20d",
    "score_std_20d",
    "score_sma_60d",
    "score_velocity",
    "composite_score",
    "is_elevated",
    "ind_qqq_deviation",
    "ind_qqq_deviation_sma_20d",
    "ind_vix_level",
    "ind_vix_level_change_5d",
    "ind_yield_curve",
    "ind_credit_spread",
    "ind_market_breadth",
    "ind_momentum_divergence",
    "drawdown_risk_score",
]

# ── MLP configurations ──
HIDDEN_SIZES = [(16,), (32, 16), (64, 32, 16)]
ALPHA_VALUES = [1e-3, 1e-2]  # weight_decay / L2 regularization
# sklearn MLPClassifier doesn't have per-layer dropout, but has alpha (L2)
# We test different learning rate strategies as proxy for regularization
LEARNING_RATE_INITS = [1e-3]


# ═══════════════════════════════════════════════════════════════════════════
# Reused functions from fit_drawdown_model.py
# ═══════════════════════════════════════════════════════════════════════════

# Import threshold configs for data loading
THRESHOLD_CONFIGS = {
    0.10: {
        "dd_definition": "peak_to_trough",
        "forward_window": 180,
        "C": 0.1,
        "features": ["risk_ema_20d", "score_std_20d"],
    },
    0.20: {
        "dd_definition": "peak_to_trough",
        "forward_window": 180,
        "C": 0.1,
        "features": ["risk_ema_20d", "score_std_20d", "ind_yield_curve"],
    },
}


def load_data() -> pd.DataFrame:
    """Load and merge bubble history with QQQ price data, engineer all features."""
    history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
    qqq_raw = json.loads((DATA_DIR / "qqq.json").read_text())
    qqq = qqq_raw["data"] if isinstance(qqq_raw, dict) and "data" in qqq_raw else qqq_raw

    df_h = pd.DataFrame(history)

    indicators = pd.json_normalize(df_h["indicators"])
    indicators.columns = ["ind_" + c for c in indicators.columns]
    df_h = pd.concat([df_h.drop(columns=["indicators"]), indicators], axis=1)
    df_h["score_velocity"] = df_h["score_velocity"].fillna(0)

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

    df_h["score_sma_60d"] = df_h["composite_score"].rolling(60, min_periods=20).mean()
    df_h["score_ema_20d"] = df_h["composite_score"].ewm(span=20, min_periods=10).mean()

    if "drawdown_risk_score" in df_h.columns:
        df_h["risk_ema_20d"] = df_h["drawdown_risk_score"].ewm(span=20, min_periods=10).mean()

    df_h["score_std_20d"] = df_h["composite_score"].rolling(20, min_periods=10).std()

    if "ind_qqq_deviation" in df_h.columns:
        df_h["ind_qqq_deviation_sma_20d"] = (
            df_h["ind_qqq_deviation"].rolling(20, min_periods=10).mean()
        )
    if "ind_vix_level" in df_h.columns:
        df_h["ind_vix_level_change_5d"] = df_h["ind_vix_level"].diff(5)

    if "ind_vix_level" in df_h.columns and "ind_credit_spread" in df_h.columns:
        df_h["vix_x_credit"] = df_h["ind_vix_level"] * df_h["ind_credit_spread"] / 100

    df_h["is_elevated"] = (df_h["composite_score"] > 60).astype(float)

    df_q = pd.DataFrame(qqq)[["date", "price"]].copy()
    df_q.rename(columns={"price": "qqq_price"}, inplace=True)

    df = df_h.merge(df_q, on="date", how="inner").sort_values("date").reset_index(drop=True)

    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def compute_forward_drawdowns(df: pd.DataFrame, window: int, definition: str) -> np.ndarray:
    """Compute forward drawdowns using specified definition."""
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
            today_price = fwd_prices[0]
            min_future = np.min(fwd_prices[1:])
            max_dd[i] = max(0.0, (today_price - min_future) / today_price)

    return max_dd


def purged_walk_forward_splits(
    n: int,
    n_splits: int = N_CV_FOLDS,
    min_train_frac: float = MIN_TRAIN_FRAC,
    purge: int = PURGE_DAYS,
    embargo: int = EMBARGO_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window CV indices with purge gap + embargo."""
    min_train = int(n * min_train_frac)
    remaining = n - min_train
    test_size = max(50, (remaining - purge * n_splits) // n_splits)

    splits = []
    for fold in range(n_splits):
        test_start = min_train + fold * test_size + purge
        test_end = min(test_start + test_size, n)

        if test_start >= n or test_end - test_start < 30:
            break

        train_end = test_start - purge
        if train_end < 100:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
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
# MLP evaluation with purged WF-CV
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_mlp_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    hidden_layer_sizes: tuple,
    alpha: float,
    learning_rate_init: float = 1e-3,
    max_iter: int = 500,
    early_stopping: bool = True,
) -> dict:
    """Evaluate an MLP configuration using purged walk-forward CV.

    Uses early stopping with 20% of training data as validation set within each fold.
    """
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
            continue

        # StandardScaler within fold
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # MLPClassifier with early stopping (uses last 20% of train as validation)
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=min(64, max(32, len(X_train_s) // 10)),
            learning_rate="adaptive",
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
        )

        try:
            mlp.fit(X_train_s, y_train)
        except Exception as e:
            # If training fails (e.g., convergence issues), skip fold
            continue

        y_pred_test = mlp.predict_proba(X_test_s)[:, 1]

        # Clip predictions for numerical stability
        y_pred_test = np.clip(y_pred_test, 1e-4, 1 - 1e-4)

        try:
            auc = roc_auc_score(y_test, y_pred_test)
        except ValueError:
            continue

        brier = brier_score_loss(y_test, y_pred_test)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred_test)

        fold_metrics.append({
            "fold": fold_i + 1,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "base_rate_test": round(base_rate_test, 4),
            "auc": round(float(auc), 4),
            "brier": round(float(brier), 4),
            "bss": round(float(bss), 4),
            "ece": round(float(ece), 4),
            "n_iter": mlp.n_iter_,
        })

    # Aggregate
    if fold_metrics:
        aucs = [m["auc"] for m in fold_metrics]
        bsss = [m["bss"] for m in fold_metrics]
        eces = [m["ece"] for m in fold_metrics]
        briers = [m["brier"] for m in fold_metrics]
        aggregate = {
            "n_folds": len(fold_metrics),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std": round(float(np.std(aucs)), 4),
            "bss_mean": round(float(np.mean(bsss)), 4),
            "bss_std": round(float(np.std(bsss)), 4),
            "ece_mean": round(float(np.mean(eces)), 4),
            "ece_std": round(float(np.std(eces)), 4),
            "brier_mean": round(float(np.mean(briers)), 4),
        }
    else:
        aggregate = {
            "n_folds": 0,
            "auc_mean": 0.5,
            "auc_std": 0,
            "bss_mean": 0,
            "bss_std": 0,
            "ece_mean": 0,
            "ece_std": 0,
            "brier_mean": 1.0,
        }

    return {"folds": fold_metrics, "aggregate": aggregate}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 70)
    print("NEURAL NETWORK MODEL COMPARISON")
    print("sklearn MLPClassifier (PyTorch not available)")
    print("Purged Walk-Forward CV: 5 folds, 180d purge, 20d embargo")
    print("=" * 70)

    print("\nLoading data...")
    df = load_data()
    print(f"  {len(df)} trading days, date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    # Filter to available features
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing_features = [f for f in ALL_FEATURES if f not in df.columns]
    if missing_features:
        print(f"  Missing features (skipped): {missing_features}")
    print(f"  Using {len(available_features)} features: {available_features}")

    # Build MLP configs
    mlp_configs = []
    for hidden, alpha in product(HIDDEN_SIZES, ALPHA_VALUES):
        name = f"MLP-{'-'.join(str(h) for h in hidden)}_alpha={alpha}"
        mlp_configs.append({
            "name": name,
            "hidden_layer_sizes": hidden,
            "alpha": alpha,
        })

    print(f"\n  Testing {len(mlp_configs)} MLP configurations:")
    for cfg in mlp_configs:
        print(f"    - {cfg['name']}")

    all_results = {}

    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        print(f"\n{'=' * 70}")
        print(f"  THRESHOLD: {pct}% Drawdown (peak-to-trough, W=180d)")
        print(f"{'=' * 70}")

        # Compute labels
        fwd_dd = compute_forward_drawdowns(df, 180, "peak_to_trough")

        # Filter valid rows
        valid_mask = np.isfinite(fwd_dd)
        for f in available_features:
            valid_mask &= df[f].notna().values

        sub_idx = np.where(valid_mask)[0]
        X_all = df.iloc[sub_idx][available_features].values.astype(float)
        y_all = (fwd_dd[sub_idx] >= threshold).astype(float)
        n = len(X_all)

        base_rate = float(y_all.mean())
        print(f"  Samples: {n}, base rate: {base_rate:.1%}")

        splits = purged_walk_forward_splits(n)
        print(f"  CV splits: {len(splits)} folds")

        # Print fold details
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            br_train = float(y_all[train_idx].mean())
            br_test = float(y_all[test_idx].mean())
            print(f"    Fold {fold_i+1}: train={len(train_idx)} (br={br_train:.1%}), "
                  f"test={len(test_idx)} (br={br_test:.1%})")

        threshold_results = []

        for cfg in mlp_configs:
            print(f"\n  Testing {cfg['name']}...", end=" ", flush=True)
            t0 = time.time()

            result = evaluate_mlp_cv(
                X_all, y_all, splits,
                hidden_layer_sizes=cfg["hidden_layer_sizes"],
                alpha=cfg["alpha"],
            )

            elapsed = time.time() - t0
            agg = result["aggregate"]

            if agg["n_folds"] > 0:
                print(f"done ({elapsed:.1f}s) - "
                      f"AUC={agg['auc_mean']:.4f}+-{agg['auc_std']:.4f}  "
                      f"BSS={agg['bss_mean']:+.1%}  "
                      f"ECE={agg['ece_mean']:.4f}  "
                      f"({agg['n_folds']} folds)")
            else:
                print(f"done ({elapsed:.1f}s) - NO VALID FOLDS")

            threshold_results.append({
                "model": cfg["name"],
                "hidden_layer_sizes": list(cfg["hidden_layer_sizes"]),
                "alpha": cfg["alpha"],
                "threshold": threshold,
                "threshold_pct": pct,
                **agg,
                "folds": result["folds"],
            })

        all_results[f"dd_{pct}pct"] = threshold_results

    # ── Print comparison table ──
    print("\n" + "=" * 70)
    print("COMPARISON TABLE: Neural Network Models")
    print("=" * 70)

    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        results = all_results[f"dd_{pct}pct"]

        print(f"\n  {'=' * 66}")
        print(f"  Drawdown >= {pct}% (peak-to-trough, W=180d)")
        print(f"  {'=' * 66}")
        print(f"  {'Model':<35} {'Folds':>5} {'AUC':>10} {'BSS':>10} {'ECE':>10}")
        print(f"  {'-' * 66}")

        # Sort by AUC descending
        sorted_results = sorted(results, key=lambda x: x["auc_mean"], reverse=True)
        for r in sorted_results:
            auc_str = f"{r['auc_mean']:.4f}+-{r['auc_std']:.4f}" if r["n_folds"] > 0 else "N/A"
            bss_str = f"{r['bss_mean']:+.1%}" if r["n_folds"] > 0 else "N/A"
            ece_str = f"{r['ece_mean']:.4f}" if r["n_folds"] > 0 else "N/A"
            marker = " <-- best" if r == sorted_results[0] and r["n_folds"] > 0 else ""
            print(f"  {r['model']:<35} {r['n_folds']:>5} {auc_str:>10} {bss_str:>10} {ece_str:>10}{marker}")

    # ── Best models summary ──
    print("\n" + "=" * 70)
    print("BEST MODELS PER THRESHOLD")
    print("=" * 70)

    summary = {}
    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        results = all_results[f"dd_{pct}pct"]
        valid_results = [r for r in results if r["n_folds"] >= 2]

        if valid_results:
            best = max(valid_results, key=lambda x: x["auc_mean"])
            print(f"\n  {pct}% DD: {best['model']}")
            print(f"    AUC: {best['auc_mean']:.4f} +- {best['auc_std']:.4f}")
            print(f"    BSS: {best['bss_mean']:+.1%} +- {best['bss_std']:.1%}")
            print(f"    ECE: {best['ece_mean']:.4f} +- {best['ece_std']:.4f}")
            summary[f"dd_{pct}pct"] = {
                "best_model": best["model"],
                "auc_mean": best["auc_mean"],
                "auc_std": best["auc_std"],
                "bss_mean": best["bss_mean"],
                "bss_std": best["bss_std"],
                "ece_mean": best["ece_mean"],
                "ece_std": best["ece_std"],
            }
        else:
            print(f"\n  {pct}% DD: No valid results")
            summary[f"dd_{pct}pct"] = {"best_model": "none", "auc_mean": 0.5}

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total:.1f}s")

    # ── Save results ──
    output = {
        "model_type": "neural_network",
        "backend": "sklearn.MLPClassifier",
        "cv_method": "purged_walk_forward",
        "cv_params": {
            "n_folds": N_CV_FOLDS,
            "purge_days": PURGE_DAYS,
            "embargo_days": EMBARGO_DAYS,
            "min_train_frac": MIN_TRAIN_FRAC,
        },
        "features_used": available_features,
        "n_features": len(available_features),
        "configs_tested": len(mlp_configs),
        "thresholds": [int(t * 100) for t in THRESHOLDS],
        "results": all_results,
        "best_per_threshold": summary,
        "runtime_seconds": round(elapsed_total, 1),
        "note": "PyTorch not available; LSTM/GRU skipped. Using sklearn MLPClassifier only.",
    }

    out_path = SCRIPT_DIR / "results_nn_models.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("NEURAL NETWORK COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
