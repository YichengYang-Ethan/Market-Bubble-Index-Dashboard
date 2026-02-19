#!/usr/bin/env python3
"""Compare Lasso / Ridge / ElasticNet logistic regression for drawdown prediction.

Uses the same purged walk-forward CV infrastructure from fit_drawdown_model.py.
Tests all 14 features with regularization doing the selection.
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Import shared infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_drawdown_model import (
    compute_ece,
    compute_forward_drawdowns,
    load_data,
    purged_walk_forward_splits,
)

SCRIPT_DIR = Path(__file__).resolve().parent

ALL_FEATURES = [
    "risk_ema_20d",
    "score_ema_20d",
    "score_std_20d",
    "score_sma_60d",
    "score_velocity",
    "ind_qqq_deviation_sma_20d",
    "ind_vix_level",
    "ind_vix_level_change_5d",
    "ind_yield_curve",
    "ind_credit_spread",
    "vix_x_credit",
    "is_elevated",
    "drawdown_risk_score",
    "composite_score",
]

THRESHOLDS = [0.10, 0.20]
DD_DEFINITION = "peak_to_trough"
FORWARD_WINDOW = 180

# Hyperparameter grids
RIDGE_C = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
LASSO_C = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
ENET_C = [0.01, 0.1, 1.0, 10.0]
ENET_L1_RATIO = [0.1, 0.3, 0.5, 0.7, 0.9]

BASELINE = {
    0.10: {"AUC": 0.692, "BSS": 0.113, "ECE": 0.147},
    0.20: {"AUC": 0.650, "BSS": -0.827, "ECE": 0.274},
}


def prepare_data(df: pd.DataFrame, threshold: float):
    """Prepare X, y arrays from dataframe for a given threshold."""
    fwd_dd = compute_forward_drawdowns(df, FORWARD_WINDOW, DD_DEFINITION)

    # Use only features that exist
    features = [f for f in ALL_FEATURES if f in df.columns]

    valid_mask = np.isfinite(fwd_dd)
    for f in features:
        valid_mask &= df[f].notna().values

    sub_idx = np.where(valid_mask)[0]
    X = df.iloc[sub_idx][features].values.astype(float)
    y = (fwd_dd[sub_idx] >= threshold).astype(float)

    return X, y, features, sub_idx


def evaluate_config(X, y, penalty, C, l1_ratio=None, solver=None):
    """Run purged WF-CV for a single config. Returns metrics dict or None."""
    n = len(X)
    splits = purged_walk_forward_splits(n)

    fold_aucs, fold_bsss, fold_eces = [], [], []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        n_events = int(y_train.sum())
        base_rate = float(y_test.mean())

        # Skip degenerate folds
        if n_events < 3 or n_events == len(y_train):
            continue
        if len(np.unique(y_test)) < 2:
            continue
        if base_rate < 0.02 or base_rate > 0.98:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        kwargs = {"penalty": penalty, "C": C, "max_iter": 10000}
        if solver:
            kwargs["solver"] = solver
        if l1_ratio is not None:
            kwargs["l1_ratio"] = l1_ratio

        model = LogisticRegression(**kwargs)
        try:
            model.fit(X_train_s, y_train)
        except Exception:
            continue

        y_pred_raw = model.predict_proba(X_test_s)[:, 1]

        # Isotonic recalibration
        y_pred_train = model.predict_proba(X_train_s)[:, 1]
        try:
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(y_pred_train, y_train)
            y_pred = iso.predict(y_pred_raw)
        except Exception:
            y_pred = y_pred_raw

        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred)

        fold_aucs.append(auc)
        fold_bsss.append(bss)
        fold_eces.append(ece)

    if len(fold_aucs) < 2:
        return None

    return {
        "n_folds": len(fold_aucs),
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "bss_mean": float(np.mean(fold_bsss)),
        "bss_std": float(np.std(fold_bsss)),
        "ece_mean": float(np.mean(fold_eces)),
        "ece_std": float(np.std(fold_eces)),
    }


def get_coefs(X, y, penalty, C, l1_ratio=None, solver=None, feature_names=None):
    """Fit on full data, return coefficients."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    kwargs = {"penalty": penalty, "C": C, "max_iter": 10000}
    if solver:
        kwargs["solver"] = solver
    if l1_ratio is not None:
        kwargs["l1_ratio"] = l1_ratio

    model = LogisticRegression(**kwargs)
    model.fit(X_s, y)

    coefs = dict(zip(feature_names or range(X.shape[1]), model.coef_[0]))
    return coefs


def is_pareto_optimal(results):
    """Find Pareto-optimal configs (maximize AUC and BSS simultaneously)."""
    n = len(results)
    is_optimal = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i on both and > on at least one
            if (results[j]["auc_mean"] >= results[i]["auc_mean"] and
                results[j]["bss_mean"] >= results[i]["bss_mean"] and
                (results[j]["auc_mean"] > results[i]["auc_mean"] or
                 results[j]["bss_mean"] > results[i]["bss_mean"])):
                is_optimal[i] = False
                break
    return is_optimal


def main():
    print("=" * 72)
    print("PENALIZED LINEAR MODEL COMPARISON")
    print("Ridge (L2) / Lasso (L1) / ElasticNet vs v3.2 Baseline")
    print("Purged Walk-Forward CV | 5 folds | 180d purge | 20d embargo")
    print("=" * 72)

    df = load_data()
    print(f"\nData: {len(df)} days, {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    all_results = {}

    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        print(f"\n{'=' * 72}")
        print(f"  THRESHOLD: {pct}% DRAWDOWN (peak-to-trough, W=180d)")
        print(f"{'=' * 72}")

        X, y, features, sub_idx = prepare_data(df, threshold)
        base_rate = float(y.mean())
        print(f"  Samples: {len(X)}, Events: {int(y.sum())}, Base rate: {base_rate:.1%}")
        print(f"  Features: {len(features)}")

        results = []

        # --- Ridge ---
        print(f"\n  Ridge (L2): testing {len(RIDGE_C)} C values...")
        for c in RIDGE_C:
            m = evaluate_config(X, y, "l2", c, solver="lbfgs")
            if m:
                m["model"] = "Ridge"
                m["C"] = c
                m["l1_ratio"] = None
                results.append(m)
                print(f"    C={c:<8} AUC={m['auc_mean']:.4f}  BSS={m['bss_mean']:+.1%}  ECE={m['ece_mean']:.4f}")

        # --- Lasso ---
        print(f"\n  Lasso (L1): testing {len(LASSO_C)} C values...")
        for c in LASSO_C:
            m = evaluate_config(X, y, "l1", c, solver="saga")
            if m:
                m["model"] = "Lasso"
                m["C"] = c
                m["l1_ratio"] = None
                results.append(m)
                print(f"    C={c:<8} AUC={m['auc_mean']:.4f}  BSS={m['bss_mean']:+.1%}  ECE={m['ece_mean']:.4f}")

        # --- ElasticNet ---
        combos = list(product(ENET_C, ENET_L1_RATIO))
        print(f"\n  ElasticNet: testing {len(combos)} (C, l1_ratio) combos...")
        for c, lr in combos:
            m = evaluate_config(X, y, "elasticnet", c, l1_ratio=lr, solver="saga")
            if m:
                m["model"] = "ElasticNet"
                m["C"] = c
                m["l1_ratio"] = lr
                results.append(m)
                print(f"    C={c:<6} l1r={lr:<4} AUC={m['auc_mean']:.4f}  BSS={m['bss_mean']:+.1%}  ECE={m['ece_mean']:.4f}")

        if not results:
            print("  No valid results!")
            continue

        # --- Rankings ---
        baseline = BASELINE[threshold]

        print(f"\n  {'─' * 68}")
        print(f"  BASELINE (v3.2): AUC={baseline['AUC']:.3f}  BSS={baseline['BSS']:+.1%}  ECE={baseline['ECE']:.3f}")
        print(f"  {'─' * 68}")

        # Top 10 by AUC
        by_auc = sorted(results, key=lambda r: r["auc_mean"], reverse=True)
        print(f"\n  TOP 10 BY AUC:")
        for i, r in enumerate(by_auc[:10]):
            label = f"{r['model']} C={r['C']}"
            if r["l1_ratio"] is not None:
                label += f" l1r={r['l1_ratio']}"
            delta_auc = r["auc_mean"] - baseline["AUC"]
            print(f"    {i+1:>2}. {label:<32} AUC={r['auc_mean']:.4f} ({delta_auc:+.4f})  "
                  f"BSS={r['bss_mean']:+.1%}  ECE={r['ece_mean']:.4f}")

        # Top 10 by BSS
        by_bss = sorted(results, key=lambda r: r["bss_mean"], reverse=True)
        print(f"\n  TOP 10 BY BSS:")
        for i, r in enumerate(by_bss[:10]):
            label = f"{r['model']} C={r['C']}"
            if r["l1_ratio"] is not None:
                label += f" l1r={r['l1_ratio']}"
            delta_bss = r["bss_mean"] - baseline["BSS"]
            print(f"    {i+1:>2}. {label:<32} BSS={r['bss_mean']:+.1%} ({delta_bss:+.1%})  "
                  f"AUC={r['auc_mean']:.4f}  ECE={r['ece_mean']:.4f}")

        # Pareto-optimal
        pareto_mask = is_pareto_optimal(results)
        pareto = [r for r, opt in zip(results, pareto_mask) if opt]
        pareto.sort(key=lambda r: r["auc_mean"], reverse=True)
        print(f"\n  PARETO-OPTIMAL (AUC vs BSS): {len(pareto)} configs")
        for i, r in enumerate(pareto):
            label = f"{r['model']} C={r['C']}"
            if r["l1_ratio"] is not None:
                label += f" l1r={r['l1_ratio']}"
            print(f"    {i+1:>2}. {label:<32} AUC={r['auc_mean']:.4f}  BSS={r['bss_mean']:+.1%}  ECE={r['ece_mean']:.4f}")

        # Best Lasso: surviving features
        lasso_results = [r for r in results if r["model"] == "Lasso"]
        if lasso_results:
            best_lasso = max(lasso_results, key=lambda r: r["auc_mean"])
            coefs = get_coefs(X, y, "l1", best_lasso["C"], solver="saga", feature_names=features)
            nonzero = {k: v for k, v in coefs.items() if abs(v) > 1e-6}
            zero = [k for k, v in coefs.items() if abs(v) <= 1e-6]

            print(f"\n  BEST LASSO (C={best_lasso['C']}) — SURVIVING FEATURES:")
            for feat, w in sorted(nonzero.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"    {feat:<35} {w:+.6f}")
            print(f"  Eliminated ({len(zero)}): {', '.join(zero)}")

        # Best ElasticNet: surviving features
        enet_results = [r for r in results if r["model"] == "ElasticNet"]
        if enet_results:
            best_enet = max(enet_results, key=lambda r: r["auc_mean"])
            coefs = get_coefs(X, y, "elasticnet", best_enet["C"],
                              l1_ratio=best_enet["l1_ratio"], solver="saga",
                              feature_names=features)
            nonzero = {k: v for k, v in coefs.items() if abs(v) > 1e-6}
            zero = [k for k, v in coefs.items() if abs(v) <= 1e-6]

            print(f"\n  BEST ELASTICNET (C={best_enet['C']}, l1r={best_enet['l1_ratio']}) — SURVIVING FEATURES:")
            for feat, w in sorted(nonzero.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"    {feat:<35} {w:+.6f}")
            print(f"  Eliminated ({len(zero)}): {', '.join(zero)}")

        all_results[f"{pct}pct"] = results

    # --- Save JSON ---
    output = {}
    for key, results in all_results.items():
        output[key] = []
        for r in results:
            output[key].append({
                "model": r["model"],
                "C": r["C"],
                "l1_ratio": r["l1_ratio"],
                "n_folds": r["n_folds"],
                "auc_mean": round(r["auc_mean"], 4),
                "auc_std": round(r["auc_std"], 4),
                "bss_mean": round(r["bss_mean"], 4),
                "bss_std": round(r["bss_std"], 4),
                "ece_mean": round(r["ece_mean"], 4),
                "ece_std": round(r["ece_std"], 4),
            })

    out_path = SCRIPT_DIR / "results_penalized_linear.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
