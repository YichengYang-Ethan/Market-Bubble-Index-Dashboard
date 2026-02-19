#!/usr/bin/env python3
"""Compare regularization path analysis + stability selection approaches.

Tests:
  a) Lasso regularization path: sweep C from 0.001..100, show feature survival
  b) Stability selection: 100 bootstrap resamples, identify robust features
  c) Cross-validated regularization: LogisticRegressionCV with inner CV
  d) RFE: Recursive Feature Elimination with n_features_to_select=[2,3,4,5]

All evaluated with purged walk-forward CV, isotonic recalibration, StandardScaler.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import RFE
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Import shared infrastructure
from fit_drawdown_model import (
    load_data,
    compute_forward_drawdowns,
    purged_walk_forward_splits,
    compute_ece,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# All available features (superset)
ALL_FEATURES = [
    "risk_ema_20d",
    "score_std_20d",
    "ind_yield_curve",
    "ind_vix_level",
    "ind_vix_level_change_5d",
    "ind_qqq_deviation_sma_20d",
    "score_ema_20d",
    "ind_qqq_deviation",
    "ind_credit_spread",
    "vix_x_credit",
    "is_elevated",
    "composite_score",
    "score_sma_60d",
    "score_velocity",
]

THRESHOLDS = [0.10, 0.20]
DD_DEF = "peak_to_trough"
FWD_WINDOW = 180

# Baseline v3.2 for comparison
BASELINE = {
    0.10: {"AUC": 0.692, "BSS": 0.113, "ECE": 0.147},
    0.20: {"AUC": 0.650, "BSS": -0.827, "ECE": 0.274},
}


def get_valid_features(df: pd.DataFrame) -> list[str]:
    """Return features that actually exist in the dataframe."""
    return [f for f in ALL_FEATURES if f in df.columns and df[f].notna().sum() > 100]


def prepare_data(df: pd.DataFrame, threshold: float, features: list[str]):
    """Prepare X, y arrays with valid mask."""
    fwd_dd = compute_forward_drawdowns(df, FWD_WINDOW, DD_DEF)
    valid_mask = np.isfinite(fwd_dd)
    for f in features:
        valid_mask &= df[f].notna().values

    sub_idx = np.where(valid_mask)[0]
    X = df.iloc[sub_idx][features].values.astype(float)
    y = (fwd_dd[sub_idx] >= threshold).astype(float)
    return X, y, sub_idx


def evaluate_with_cv(X, y, features, C=0.1, penalty="l1", feature_subset=None):
    """Run purged walk-forward CV on given data. Returns aggregate metrics dict."""
    if feature_subset is not None:
        X = X[:, feature_subset]

    n = len(X)
    splits = purged_walk_forward_splits(n)

    aucs, bsss, eces = [], [], []
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if y_train.sum() < 3 or len(np.unique(y_test)) < 2:
            continue
        br = y_test.mean()
        if br < 0.02 or br > 0.98:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        solver = "liblinear" if penalty == "l1" else "lbfgs"
        model = LogisticRegression(
            max_iter=5000, C=C, penalty=penalty, solver=solver
        )
        model.fit(X_tr_s, y_train)

        y_pred_raw = model.predict_proba(X_te_s)[:, 1]

        # Isotonic recalibration
        y_pred_train = model.predict_proba(X_tr_s)[:, 1]
        try:
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(y_pred_train, y_train)
            y_pred = iso.predict(y_pred_raw)
        except Exception:
            y_pred = y_pred_raw

        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        brier_clim = br * (1 - br)
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred)

        aucs.append(auc)
        bsss.append(bss)
        eces.append(ece)

    if not aucs:
        return {"n_folds": 0, "auc": 0.5, "bss": 0.0, "ece": 1.0}

    return {
        "n_folds": len(aucs),
        "auc": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "bss": round(float(np.mean(bsss)), 4),
        "bss_std": round(float(np.std(bsss)), 4),
        "ece": round(float(np.mean(eces)), 4),
        "ece_std": round(float(np.std(eces)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# (a) Regularization path
# ═══════════════════════════════════════════════════════════════════════════

def regularization_path(X, y, features):
    """Sweep C values and show which features survive at each level."""
    C_values = np.logspace(-3, 2, 20)
    path_results = []

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print("\n  C value    | # non-zero | Surviving features")
    print("  " + "-" * 70)

    for C in C_values:
        model = LogisticRegression(
            max_iter=5000, C=C, penalty="l1", solver="liblinear"
        )
        model.fit(X_s, y)
        coefs = model.coef_[0]
        nonzero = np.abs(coefs) > 1e-6
        surviving = [features[i] for i in range(len(features)) if nonzero[i]]
        coef_vals = {features[i]: round(float(coefs[i]), 4) for i in range(len(features)) if nonzero[i]}

        path_results.append({
            "C": round(float(C), 6),
            "n_nonzero": int(nonzero.sum()),
            "surviving_features": surviving,
            "coefficients": coef_vals,
        })

        print(f"  {C:10.4f}  | {nonzero.sum():10d} | {', '.join(surviving) if surviving else '(none)'}")

    # Identify elbow: biggest drop in feature count
    counts = [r["n_nonzero"] for r in path_results]
    max_drop_idx = 0
    max_drop = 0
    for i in range(1, len(counts)):
        drop = counts[i - 1] - counts[i]
        if drop > max_drop:
            max_drop = drop
            max_drop_idx = i

    elbow_C = path_results[max_drop_idx]["C"]
    elbow_features = path_results[max_drop_idx]["surviving_features"]
    print(f"\n  Elbow at C={elbow_C:.4f}: {len(elbow_features)} features survive")
    print(f"  Features at elbow: {elbow_features}")

    return path_results, elbow_C, elbow_features


# ═══════════════════════════════════════════════════════════════════════════
# (b) Stability selection
# ═══════════════════════════════════════════════════════════════════════════

def stability_selection(X, y, features, n_bootstrap=100, subsample_frac=0.5, C=0.1):
    """Bootstrap stability selection for L1 logistic regression."""
    n = len(X)
    n_features = len(features)
    selection_counts = np.zeros(n_features)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    rng = np.random.RandomState(42)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=int(n * subsample_frac), replace=False)
        X_b, y_b = X_s[idx], y[idx]

        if y_b.sum() < 2 or y_b.sum() == len(y_b):
            continue

        model = LogisticRegression(
            max_iter=5000, C=C, penalty="l1", solver="liblinear"
        )
        model.fit(X_b, y_b)
        nonzero = np.abs(model.coef_[0]) > 1e-6
        selection_counts += nonzero

    frequencies = selection_counts / n_bootstrap
    freq_dict = {features[i]: round(float(frequencies[i]), 3) for i in range(n_features)}

    # Sort by frequency
    sorted_feats = sorted(freq_dict.items(), key=lambda x: -x[1])

    print("\n  Feature stability selection frequencies (100 bootstrap, C=0.1):")
    for feat, freq in sorted_feats:
        bar = "#" * int(freq * 40)
        stable = " STABLE" if freq > 0.6 else ""
        print(f"    {feat:<30}: {freq:.1%} {bar}{stable}")

    # Stable features: selected > 60% of the time
    stable_features = [f for f, freq in sorted_feats if freq > 0.6]
    print(f"\n  Stable features (>60%): {stable_features}")

    return freq_dict, stable_features


# ═══════════════════════════════════════════════════════════════════════════
# (c) Cross-validated regularization (LogisticRegressionCV)
# ═══════════════════════════════════════════════════════════════════════════

def cv_regularization(X, y, features):
    """Use LogisticRegressionCV with inner CV to auto-select C."""
    results = {}
    Cs = np.logspace(-3, 2, 20)

    for penalty in ["l1", "l2"]:
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        # Inner CV to select C
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        model = LogisticRegressionCV(
            Cs=Cs, cv=3, penalty=penalty, solver=solver,
            max_iter=5000, scoring="roc_auc",
        )
        model.fit(X_s, y)

        best_C = float(model.C_[0])
        coefs = model.coef_[0]
        nonzero = np.abs(coefs) > 1e-6
        active = [features[i] for i in range(len(features)) if nonzero[i]]
        coef_dict = {features[i]: round(float(coefs[i]), 4) for i in range(len(features))}

        # Now evaluate with outer purged WF-CV using the selected C
        outer = evaluate_with_cv(X, y, features, C=best_C, penalty=penalty)

        results[penalty] = {
            "best_C": round(best_C, 6),
            "active_features": active,
            "n_active": len(active),
            "coefficients": coef_dict,
            "outer_cv": outer,
        }

        print(f"\n  LogisticRegressionCV penalty={penalty}:")
        print(f"    Best C (inner 3-fold): {best_C:.6f}")
        print(f"    Active features ({len(active)}): {active}")
        print(f"    Outer CV: AUC={outer['auc']:.4f}  BSS={outer['bss']:+.1%}  ECE={outer['ece']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# (d) Recursive Feature Elimination
# ═══════════════════════════════════════════════════════════════════════════

def recursive_feature_elimination(X, y, features):
    """RFE with LogisticRegression, test n_features_to_select=[2,3,4,5]."""
    results = {}

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Get full-feature ranking
    base_model = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
    rfe_full = RFE(base_model, n_features_to_select=1)
    rfe_full.fit(X_s, y)
    ranking = {features[i]: int(rfe_full.ranking_[i]) for i in range(len(features))}
    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1])

    print("\n  RFE Feature Rankings (1=best):")
    for feat, rank in sorted_ranking:
        print(f"    {rank:2d}. {feat}")

    for n_select in [2, 3, 4, 5]:
        rfe = RFE(
            LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs"),
            n_features_to_select=n_select,
        )
        rfe.fit(X_s, y)

        selected_mask = rfe.support_
        selected_features = [features[i] for i in range(len(features)) if selected_mask[i]]
        selected_idx = [i for i in range(len(features)) if selected_mask[i]]

        # Evaluate with purged WF-CV
        outer = evaluate_with_cv(X, y, features, C=1.0, penalty="l2", feature_subset=selected_idx)

        results[n_select] = {
            "selected_features": selected_features,
            "outer_cv": outer,
        }

        print(f"\n  RFE n_features={n_select}: {selected_features}")
        print(f"    Outer CV: AUC={outer['auc']:.4f}  BSS={outer['bss']:+.1%}  ECE={outer['ece']:.4f}")

    return results, ranking


# ═══════════════════════════════════════════════════════════════════════════
# Evaluate stability-selected model
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_stable_model(X, y, features, stable_features):
    """Build final model using only stable features and evaluate."""
    if not stable_features:
        return None

    feat_idx = [i for i, f in enumerate(features) if f in stable_features]
    result = evaluate_with_cv(X, y, features, C=0.1, penalty="l1", feature_subset=feat_idx)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("STABILITY SELECTION & REGULARIZATION PATH COMPARISON")
    print("=" * 70)

    df = load_data()
    features = get_valid_features(df)
    print(f"\nData: {len(df)} days, {len(features)} available features")
    print(f"Features: {features}")

    # Fill NaN with median for all features
    for f in features:
        df[f] = df[f].fillna(df[f].median())

    all_results = {}

    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        print(f"\n{'=' * 70}")
        print(f"  THRESHOLD: {pct}% DRAWDOWN (peak-to-trough, W=180d)")
        print(f"{'=' * 70}")

        base = BASELINE[threshold]
        print(f"\n  v3.2 Baseline: AUC={base['AUC']:.3f}  BSS={base['BSS']:+.1%}  ECE={base['ECE']:.3f}")

        X, y, sub_idx = prepare_data(df, threshold, features)
        print(f"  Samples: {len(X)}, Events: {int(y.sum())}, Base rate: {y.mean():.1%}")

        thresh_results = {}

        # (a) Regularization path
        print(f"\n{'─' * 60}")
        print(f"  (a) REGULARIZATION PATH (Lasso)")
        print(f"{'─' * 60}")
        path_results, elbow_C, elbow_features = regularization_path(X, y, features)
        thresh_results["reg_path"] = {
            "path": path_results,
            "elbow_C": elbow_C,
            "elbow_features": elbow_features,
        }

        # Evaluate elbow config
        if elbow_features:
            elbow_idx = [i for i, f in enumerate(features) if f in elbow_features]
            elbow_cv = evaluate_with_cv(X, y, features, C=elbow_C, penalty="l1", feature_subset=elbow_idx)
            thresh_results["reg_path"]["elbow_cv"] = elbow_cv
            print(f"  Elbow CV: AUC={elbow_cv['auc']:.4f}  BSS={elbow_cv['bss']:+.1%}  ECE={elbow_cv['ece']:.4f}")

        # (b) Stability selection
        print(f"\n{'─' * 60}")
        print(f"  (b) STABILITY SELECTION (100 bootstrap, C=0.1)")
        print(f"{'─' * 60}")
        freq_dict, stable_features = stability_selection(X, y, features)
        thresh_results["stability"] = {
            "frequencies": freq_dict,
            "stable_features": stable_features,
        }

        # Evaluate stable model
        if stable_features:
            stable_cv = evaluate_stable_model(X, y, features, stable_features)
            thresh_results["stability"]["stable_cv"] = stable_cv
            print(f"  Stable model CV: AUC={stable_cv['auc']:.4f}  BSS={stable_cv['bss']:+.1%}  ECE={stable_cv['ece']:.4f}")

        # (c) Cross-validated regularization
        print(f"\n{'─' * 60}")
        print(f"  (c) CROSS-VALIDATED REGULARIZATION (LogisticRegressionCV)")
        print(f"{'─' * 60}")
        cv_reg_results = cv_regularization(X, y, features)
        thresh_results["cv_regularization"] = cv_reg_results

        # (d) RFE
        print(f"\n{'─' * 60}")
        print(f"  (d) RECURSIVE FEATURE ELIMINATION")
        print(f"{'─' * 60}")
        rfe_results, rfe_ranking = recursive_feature_elimination(X, y, features)
        thresh_results["rfe"] = {
            "ranking": rfe_ranking,
            "by_n_features": {str(k): v for k, v in rfe_results.items()},
        }

        # ── Summary comparison ──
        print(f"\n{'─' * 60}")
        print(f"  SUMMARY: {pct}% DD Threshold")
        print(f"{'─' * 60}")
        print(f"  {'Method':<45} {'AUC':>6} {'BSS':>8} {'ECE':>6}")
        print(f"  {'-'*65}")
        print(f"  {'v3.2 Baseline':<45} {base['AUC']:>6.3f} {base['BSS']:>+7.1%} {base['ECE']:>6.3f}")

        if elbow_features and "elbow_cv" in thresh_results["reg_path"]:
            ec = thresh_results["reg_path"]["elbow_cv"]
            print(f"  {'Lasso path (elbow)':<45} {ec['auc']:>6.3f} {ec['bss']:>+7.1%} {ec['ece']:>6.3f}")

        if stable_features and thresh_results["stability"].get("stable_cv"):
            sc = thresh_results["stability"]["stable_cv"]
            print(f"  {'Stability selection (>60%)':<45} {sc['auc']:>6.3f} {sc['bss']:>+7.1%} {sc['ece']:>6.3f}")

        for pen in ["l1", "l2"]:
            if pen in cv_reg_results:
                cr = cv_reg_results[pen]["outer_cv"]
                label = f"LogisticRegressionCV ({pen}, C={cv_reg_results[pen]['best_C']:.4f})"
                print(f"  {label:<45} {cr['auc']:>6.3f} {cr['bss']:>+7.1%} {cr['ece']:>6.3f}")

        for n_sel in [2, 3, 4, 5]:
            if n_sel in rfe_results:
                rr = rfe_results[n_sel]["outer_cv"]
                feats = rfe_results[n_sel]["selected_features"]
                label = f"RFE n={n_sel} ({', '.join(feats[:3])}{'...' if len(feats) > 3 else ''})"
                print(f"  {label:<45} {rr['auc']:>6.3f} {rr['bss']:>+7.1%} {rr['ece']:>6.3f}")

        all_results[f"dd_{pct}pct"] = thresh_results

    # ── Best config summary ──
    print(f"\n{'=' * 70}")
    print("BEST CONFIGURATIONS PER THRESHOLD")
    print(f"{'=' * 70}")

    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        tr = all_results[f"dd_{pct}pct"]

        # Collect all configs with AUC
        configs = []
        if "elbow_cv" in tr.get("reg_path", {}):
            ec = tr["reg_path"]["elbow_cv"]
            configs.append(("Lasso elbow", ec, tr["reg_path"]["elbow_features"]))

        if tr.get("stability", {}).get("stable_cv"):
            sc = tr["stability"]["stable_cv"]
            configs.append(("Stability", sc, tr["stability"]["stable_features"]))

        for pen in ["l1", "l2"]:
            if pen in tr.get("cv_regularization", {}):
                cr = tr["cv_regularization"][pen]
                configs.append((f"LRCV-{pen}", cr["outer_cv"], cr["active_features"]))

        for n_sel in [2, 3, 4, 5]:
            key = str(n_sel)
            if key in tr.get("rfe", {}).get("by_n_features", {}):
                rr = tr["rfe"]["by_n_features"][key]
                configs.append((f"RFE-{n_sel}", rr["outer_cv"], rr["selected_features"]))

        configs.sort(key=lambda x: -x[1].get("auc", 0))
        best = configs[0] if configs else None

        if best:
            print(f"\n  {pct}% DD: Best = {best[0]}")
            print(f"    AUC={best[1]['auc']:.4f}  BSS={best[1]['bss']:+.1%}  ECE={best[1]['ece']:.4f}")
            print(f"    Features: {best[2]}")

    # ── Save results ──
    out_path = SCRIPT_DIR / "results_stability_selection.json"

    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    out_data = make_serializable(all_results)
    out_path.write_text(json.dumps(out_data, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
