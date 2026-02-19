#!/usr/bin/env python3
"""Compare ensemble and stacking models for drawdown prediction.

Tests:
  - Simple averaging of base learner predictions
  - Weighted ensemble (inverse Brier score weights)
  - Stacking with logistic meta-learner
  - SVM (RBF) and KNN as additional base learners

Uses purged walk-forward CV (5 folds, 180d purge, 20d embargo).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Import shared infrastructure from the existing model script
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fit_drawdown_model import (
    load_data,
    compute_forward_drawdowns,
    purged_walk_forward_splits,
    compute_ece,
    THRESHOLD_CONFIGS,
    N_CV_FOLDS,
    PURGE_DAYS,
    EMBARGO_DAYS,
)

OUTPUT_PATH = SCRIPT_DIR / "results_ensemble_models.json"

# All available features for the comparison
ALL_FEATURES = [
    "risk_ema_20d",
    "score_std_20d",
    "ind_yield_curve",
    "ind_vix_level",
    "ind_vix_level_change_5d",
    "ind_qqq_deviation_sma_20d",
    "score_ema_20d",
    "is_elevated",
    "composite_score",
    "score_velocity",
]


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return features that exist in the dataframe."""
    return [f for f in ALL_FEATURES if f in df.columns and df[f].notna().sum() > 100]


def prepare_data(df: pd.DataFrame, threshold: float, features: list[str]):
    """Prepare X, y arrays with valid mask applied."""
    config = THRESHOLD_CONFIGS.get(threshold, THRESHOLD_CONFIGS[0.10])
    window = config["forward_window"]
    dd_def = config["dd_definition"]

    fwd_dd = compute_forward_drawdowns(df, window, dd_def)

    valid_mask = np.isfinite(fwd_dd)
    for f in features:
        valid_mask &= df[f].notna().values

    sub_idx = np.where(valid_mask)[0]
    X = df.iloc[sub_idx][features].values.astype(float)
    y = (fwd_dd[sub_idx] >= threshold).astype(float)

    return X, y, sub_idx


def is_degenerate_fold(y_train, y_test):
    """Check if fold is degenerate (skip it)."""
    n_events_train = int(y_train.sum())
    if n_events_train < 3 or n_events_train == len(y_train):
        return True
    if len(np.unique(y_test)) < 2:
        return True
    base_rate_test = float(y_test.mean())
    if base_rate_test < 0.02 or base_rate_test > 0.98:
        return True
    return False


def make_base_learners():
    """Create dict of base learner factories."""
    return {
        "Logistic_C0.1": lambda: LogisticRegression(max_iter=5000, C=0.1, solver="lbfgs"),
        "Logistic_C1": lambda: LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs"),
        "XGBoost": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42
        ),
        "RF": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            max_features="sqrt", random_state=42
        ),
        "SVM_C0.1": lambda: SVC(kernel="rbf", C=0.1, probability=True, random_state=42),
        "SVM_C1": lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "SVM_C10": lambda: SVC(kernel="rbf", C=10.0, probability=True, random_state=42),
        "KNN_5": lambda: KNeighborsClassifier(n_neighbors=5),
        "KNN_10": lambda: KNeighborsClassifier(n_neighbors=10),
        "KNN_20": lambda: KNeighborsClassifier(n_neighbors=20),
    }


def evaluate_single_model(name, model_factory, X, y, splits):
    """Evaluate a single model across CV folds."""
    fold_aucs, fold_briers, fold_bsss, fold_eces = [], [], [], []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if is_degenerate_fold(y_train, y_test):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train_s, y_train)
        y_pred = model.predict_proba(X_test_s)[:, 1]

        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred)

        fold_aucs.append(auc)
        fold_briers.append(brier)
        fold_bsss.append(bss)
        fold_eces.append(ece)

    if not fold_aucs:
        return None

    return {
        "model": name,
        "n_folds": len(fold_aucs),
        "auc_mean": round(float(np.mean(fold_aucs)), 4),
        "auc_std": round(float(np.std(fold_aucs)), 4),
        "brier_mean": round(float(np.mean(fold_briers)), 4),
        "bss_mean": round(float(np.mean(fold_bsss)), 4),
        "bss_std": round(float(np.std(fold_bsss)), 4),
        "ece_mean": round(float(np.mean(fold_eces)), 4),
        "ece_std": round(float(np.std(fold_eces)), 4),
    }


def evaluate_simple_average(base_names, X, y, splits):
    """Simple average of base learner predictions."""
    fold_aucs, fold_briers, fold_bsss, fold_eces = [], [], [], []
    learners = make_base_learners()

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if is_degenerate_fold(y_train, y_test):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        preds = []
        for name in base_names:
            model = learners[name]()
            model.fit(X_train_s, y_train)
            preds.append(model.predict_proba(X_test_s)[:, 1])

        y_pred = np.mean(preds, axis=0)
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred)

        fold_aucs.append(auc)
        fold_briers.append(brier)
        fold_bsss.append(bss)
        fold_eces.append(ece)

    if not fold_aucs:
        return None

    return {
        "model": f"SimpleAvg({'+'.join(base_names)})",
        "n_folds": len(fold_aucs),
        "auc_mean": round(float(np.mean(fold_aucs)), 4),
        "auc_std": round(float(np.std(fold_aucs)), 4),
        "brier_mean": round(float(np.mean(fold_briers)), 4),
        "bss_mean": round(float(np.mean(fold_bsss)), 4),
        "bss_std": round(float(np.std(fold_bsss)), 4),
        "ece_mean": round(float(np.mean(fold_eces)), 4),
        "ece_std": round(float(np.std(fold_eces)), 4),
    }


def evaluate_weighted_ensemble(base_names, X, y, splits):
    """Weighted ensemble: weights = inverse Brier score from each fold."""
    fold_aucs, fold_briers, fold_bsss, fold_eces = [], [], [], []
    learners = make_base_learners()

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if is_degenerate_fold(y_train, y_test):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Get train-set Brier scores for weighting
        preds_test = []
        weights = []
        for name in base_names:
            model = learners[name]()
            model.fit(X_train_s, y_train)
            pred_train = model.predict_proba(X_train_s)[:, 1]
            pred_test = model.predict_proba(X_test_s)[:, 1]

            brier_train = brier_score_loss(y_train, pred_train)
            # Inverse Brier as weight (lower Brier = better = higher weight)
            w = 1.0 / max(brier_train, 1e-6)
            weights.append(w)
            preds_test.append(pred_test)

        # Normalize weights
        w_arr = np.array(weights)
        w_arr = w_arr / w_arr.sum()

        y_pred = np.zeros(len(y_test))
        for w, pred in zip(w_arr, preds_test):
            y_pred += w * pred

        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred)

        fold_aucs.append(auc)
        fold_briers.append(brier)
        fold_bsss.append(bss)
        fold_eces.append(ece)

    if not fold_aucs:
        return None

    return {
        "model": f"WeightedEns({'+'.join(base_names)})",
        "n_folds": len(fold_aucs),
        "auc_mean": round(float(np.mean(fold_aucs)), 4),
        "auc_std": round(float(np.std(fold_aucs)), 4),
        "brier_mean": round(float(np.mean(fold_briers)), 4),
        "bss_mean": round(float(np.mean(fold_bsss)), 4),
        "bss_std": round(float(np.std(fold_bsss)), 4),
        "ece_mean": round(float(np.mean(fold_eces)), 4),
        "ece_std": round(float(np.std(fold_eces)), 4),
    }


def evaluate_stacking(base_names, X, y, splits, n_inner_folds=3):
    """Stacking: inner CV to generate OOF predictions, logistic meta-learner."""
    fold_aucs, fold_briers, fold_bsss, fold_eces = [], [], [], []
    learners = make_base_learners()

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if is_degenerate_fold(y_train, y_test):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_base = len(base_names)

        # Generate OOF predictions via inner CV
        oof_preds = np.zeros((len(X_train_s), n_base))
        test_preds = np.zeros((len(X_test_s), n_base))

        inner_cv = KFold(n_splits=n_inner_folds, shuffle=False)

        for base_i, name in enumerate(base_names):
            test_pred_accum = np.zeros(len(X_test_s))

            for inner_train, inner_val in inner_cv.split(X_train_s):
                X_it, X_iv = X_train_s[inner_train], X_train_s[inner_val]
                y_it, y_iv = y_train[inner_train], y_train[inner_val]

                # Skip if inner fold is degenerate
                if y_it.sum() < 2 or y_it.sum() == len(y_it):
                    oof_preds[inner_val, base_i] = y_train.mean()
                    test_pred_accum += y_train.mean()
                    continue

                model = learners[name]()
                model.fit(X_it, y_it)
                oof_preds[inner_val, base_i] = model.predict_proba(X_iv)[:, 1]
                test_pred_accum += model.predict_proba(X_test_s)[:, 1]

            test_preds[:, base_i] = test_pred_accum / n_inner_folds

        # Train meta-learner on OOF predictions
        meta = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
        meta.fit(oof_preds, y_train)
        y_pred = meta.predict_proba(test_preds)[:, 1]

        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred)

        fold_aucs.append(auc)
        fold_briers.append(brier)
        fold_bsss.append(bss)
        fold_eces.append(ece)

    if not fold_aucs:
        return None

    return {
        "model": f"Stacking({'+'.join(base_names)})",
        "n_folds": len(fold_aucs),
        "auc_mean": round(float(np.mean(fold_aucs)), 4),
        "auc_std": round(float(np.std(fold_aucs)), 4),
        "brier_mean": round(float(np.mean(fold_briers)), 4),
        "bss_mean": round(float(np.mean(fold_bsss)), 4),
        "bss_std": round(float(np.std(fold_bsss)), 4),
        "ece_mean": round(float(np.mean(fold_eces)), 4),
        "ece_std": round(float(np.std(fold_eces)), 4),
    }


def main():
    print("=" * 70)
    print("ENSEMBLE & STACKING MODEL COMPARISON")
    print(f"Purged Walk-Forward CV: {N_CV_FOLDS} folds, "
          f"purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d")
    print("=" * 70)

    print("\nLoading data...")
    df = load_data()
    features = get_available_features(df)
    print(f"  {len(df)} trading days, {len(features)} features available")
    print(f"  Features: {features}")

    all_results = {}

    for threshold in [0.10, 0.20]:
        pct = int(threshold * 100)
        print(f"\n{'=' * 70}")
        print(f"  THRESHOLD: {pct}% DRAWDOWN")
        print(f"{'=' * 70}")

        X, y, sub_idx = prepare_data(df, threshold, features)
        n = len(X)
        base_rate = float(y.mean())
        print(f"  N={n}, base_rate={base_rate:.1%}")

        splits = purged_walk_forward_splits(n)
        print(f"  {len(splits)} CV folds")

        results = []

        # ── Individual base learners ──
        print(f"\n  --- Individual Base Learners ---")
        learner_dict = make_base_learners()
        for name, factory in learner_dict.items():
            r = evaluate_single_model(name, factory, X, y, splits)
            if r:
                results.append(r)
                print(f"    {name:<20s}  AUC={r['auc_mean']:.4f}+-{r['auc_std']:.4f}  "
                      f"BSS={r['bss_mean']:+.1%}  ECE={r['ece_mean']:.4f}")
            else:
                print(f"    {name:<20s}  SKIPPED (all folds degenerate)")

        # ── Ensemble combinations ──
        core_base = ["Logistic_C0.1", "XGBoost", "RF"]
        extended_base = ["Logistic_C0.1", "XGBoost", "RF", "SVM_C1", "KNN_10"]

        print(f"\n  --- Simple Averaging ---")
        for combo in [core_base, extended_base]:
            r = evaluate_simple_average(combo, X, y, splits)
            if r:
                results.append(r)
                print(f"    {r['model'][:50]:<50s}")
                print(f"      AUC={r['auc_mean']:.4f}+-{r['auc_std']:.4f}  "
                      f"BSS={r['bss_mean']:+.1%}  ECE={r['ece_mean']:.4f}")

        print(f"\n  --- Weighted Ensemble (inv. Brier) ---")
        for combo in [core_base, extended_base]:
            r = evaluate_weighted_ensemble(combo, X, y, splits)
            if r:
                results.append(r)
                print(f"    {r['model'][:50]:<50s}")
                print(f"      AUC={r['auc_mean']:.4f}+-{r['auc_std']:.4f}  "
                      f"BSS={r['bss_mean']:+.1%}  ECE={r['ece_mean']:.4f}")

        print(f"\n  --- Stacking (Logistic Meta-Learner) ---")
        for combo in [core_base, extended_base]:
            r = evaluate_stacking(combo, X, y, splits)
            if r:
                results.append(r)
                print(f"    {r['model'][:50]:<50s}")
                print(f"      AUC={r['auc_mean']:.4f}+-{r['auc_std']:.4f}  "
                      f"BSS={r['bss_mean']:+.1%}  ECE={r['ece_mean']:.4f}")

        # ── Summary table ──
        print(f"\n  {'=' * 65}")
        print(f"  SUMMARY TABLE: {pct}% Drawdown Threshold")
        print(f"  {'=' * 65}")
        print(f"  {'Model':<45s} {'AUC':>7s} {'BSS':>8s} {'ECE':>7s} {'Folds':>5s}")
        print(f"  {'-' * 65}")

        results.sort(key=lambda x: x["auc_mean"], reverse=True)
        for r in results:
            model_name = r["model"][:44]
            print(f"  {model_name:<45s} {r['auc_mean']:.4f} {r['bss_mean']:+7.1%} "
                  f"{r['ece_mean']:.4f} {r['n_folds']:>5d}")

        all_results[f"drawdown_{pct}pct"] = results

    # ── Save JSON ──
    output = {
        "description": "Ensemble and stacking model comparison for drawdown prediction",
        "cv_method": f"purged_walk_forward_{N_CV_FOLDS}fold",
        "purge_days": PURGE_DAYS,
        "embargo_days": EMBARGO_DAYS,
        "features_used": features,
        "results": {}
    }
    for key, results in all_results.items():
        output["results"][key] = results

    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResults saved to {OUTPUT_PATH}")

    # ── Final Recommendation ──
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    for threshold_key in ["drawdown_10pct", "drawdown_20pct"]:
        pct = threshold_key.split("_")[1]
        results = all_results[threshold_key]
        if not results:
            continue

        best = results[0]  # already sorted by AUC
        # Find logistic baseline
        logistic_baseline = next(
            (r for r in results if r["model"] == "Logistic_C0.1"), None
        )

        print(f"\n  {pct} Drawdown:")
        print(f"    Best model: {best['model']}")
        print(f"      AUC={best['auc_mean']:.4f}, BSS={best['bss_mean']:+.1%}, ECE={best['ece_mean']:.4f}")
        if logistic_baseline:
            auc_diff = best['auc_mean'] - logistic_baseline['auc_mean']
            bss_diff = best['bss_mean'] - logistic_baseline['bss_mean']
            print(f"    vs Logistic baseline: AUC delta={auc_diff:+.4f}, BSS delta={bss_diff:+.1%}")

            if auc_diff < 0.01 and abs(bss_diff) < 0.05:
                print(f"    --> RECOMMENDATION: Keep logistic regression. "
                      f"Ensemble gains are marginal (<1% AUC improvement).")
                print(f"        Logistic is simpler, more interpretable, and better calibrated.")
            elif auc_diff >= 0.01:
                print(f"    --> RECOMMENDATION: Consider {best['model']}.")
                print(f"        AUC improvement of {auc_diff:+.4f} is meaningful.")
                if best['ece_mean'] > logistic_baseline['ece_mean'] + 0.02:
                    print(f"        WARNING: ECE is worse ({best['ece_mean']:.4f} vs "
                          f"{logistic_baseline['ece_mean']:.4f}). "
                          f"May need recalibration.")
            else:
                print(f"    --> RECOMMENDATION: Keep logistic. Complex models do not justify "
                      f"the added complexity.")

    print()


if __name__ == "__main__":
    main()
