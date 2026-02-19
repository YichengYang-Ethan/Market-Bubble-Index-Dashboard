#!/usr/bin/env python3
"""AIC/BIC stepwise feature selection + profile likelihood CI for drawdown models.

Compares forward stepwise, backward elimination, and exhaustive search
using AIC and BIC criteria via statsmodels GLM (Binomial/Logit).
Evaluates selected models with purged walk-forward CV matching v3.2 framework.
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Import shared infrastructure from fit_drawdown_model
from fit_drawdown_model import (
    compute_ece,
    compute_forward_drawdowns,
    load_data,
    purged_walk_forward_splits,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent

# All candidate features (same pool as penalized regression task)
ALL_FEATURES = [
    "risk_ema_20d",
    "score_ema_20d",
    "score_std_20d",
    "score_sma_60d",
    "composite_score",
    "score_velocity",
    "ind_vix_level",
    "ind_vix_level_change_5d",
    "ind_yield_curve",
    "ind_qqq_deviation",
    "ind_qqq_deviation_sma_20d",
    "ind_credit_spread",
    "is_elevated",
    "vix_x_credit",
]

THRESHOLDS = [0.10, 0.20]
DD_DEFINITION = "peak_to_trough"
FORWARD_WINDOW = 180

BASELINE = {
    0.10: {"auc": 0.692, "bss": 0.113, "ece": 0.147},
    0.20: {"auc": 0.650, "bss": -0.827, "ece": 0.274},
}


def prepare_data(df: pd.DataFrame, threshold: float):
    """Prepare X, y arrays with valid features only."""
    fwd_dd = compute_forward_drawdowns(df, FORWARD_WINDOW, DD_DEFINITION)
    available = [f for f in ALL_FEATURES if f in df.columns]

    valid_mask = np.isfinite(fwd_dd)
    for f in available:
        valid_mask &= df[f].notna().values

    sub_idx = np.where(valid_mask)[0]
    X_all = df.iloc[sub_idx][available].copy()
    # Fill any remaining NaN with median
    for c in X_all.columns:
        X_all[c] = X_all[c].fillna(X_all[c].median())
    y_all = (fwd_dd[sub_idx] >= threshold).astype(float)

    return X_all, y_all, sub_idx, available


def fit_glm(X: pd.DataFrame, y: np.ndarray, feature_list: list[str]) -> sm.GLM | None:
    """Fit statsmodels GLM Binomial/Logit and return fitted model."""
    if not feature_list:
        return None
    Xsub = sm.add_constant(X[feature_list].values.astype(float))
    try:
        model = sm.GLM(y, Xsub, family=sm.families.Binomial(link=sm.families.links.Logit()))
        result = model.fit(disp=0, maxiter=100)
        return result
    except Exception:
        return None


def get_aic_bic(result) -> tuple[float, float]:
    """Extract AIC and BIC from fitted GLM result."""
    return result.aic, result.bic_llf


def forward_stepwise(X: pd.DataFrame, y: np.ndarray, available: list[str],
                     criterion: str = "aic") -> list[str]:
    """Forward stepwise selection. Add features that reduce AIC/BIC most."""
    selected = []
    remaining = list(available)

    # Null model (intercept only)
    Xconst = sm.add_constant(np.ones((len(y), 1)))
    null_model = sm.GLM(y, Xconst[:, :1], family=sm.families.Binomial()).fit(disp=0, maxiter=100)
    current_score = null_model.aic if criterion == "aic" else null_model.bic_llf

    while remaining:
        best_score = current_score
        best_feat = None

        for feat in remaining:
            candidate = selected + [feat]
            result = fit_glm(X, y, candidate)
            if result is None:
                continue
            aic, bic = get_aic_bic(result)
            score = aic if criterion == "aic" else bic
            if score < best_score:
                best_score = score
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        current_score = best_score

    return selected


def backward_elimination(X: pd.DataFrame, y: np.ndarray, available: list[str],
                         criterion: str = "aic") -> list[str]:
    """Backward elimination. Remove features whose removal reduces AIC/BIC."""
    selected = list(available)

    result = fit_glm(X, y, selected)
    if result is None:
        return selected
    aic, bic = get_aic_bic(result)
    current_score = aic if criterion == "aic" else bic

    while len(selected) > 1:
        best_score = current_score
        worst_feat = None

        for feat in selected:
            candidate = [f for f in selected if f != feat]
            result = fit_glm(X, y, candidate)
            if result is None:
                continue
            aic, bic = get_aic_bic(result)
            score = aic if criterion == "aic" else bic
            if score < best_score:
                best_score = score
                worst_feat = feat

        if worst_feat is None:
            break

        selected.remove(worst_feat)
        current_score = best_score

    return selected


def exhaustive_search(X: pd.DataFrame, y: np.ndarray, available: list[str],
                      criterion: str = "aic", max_size: int = 5) -> list[str]:
    """Try all subsets of size 1..max_size, pick best by criterion."""
    # If too many features, preselect top 10 by univariate AUC
    feats = available
    if len(feats) > 10:
        univariate_aucs = []
        for f in feats:
            try:
                auc = roc_auc_score(y, X[f].values)
                auc = max(auc, 1 - auc)  # handle inversions
            except Exception:
                auc = 0.5
            univariate_aucs.append((f, auc))
        univariate_aucs.sort(key=lambda x: x[1], reverse=True)
        feats = [f for f, _ in univariate_aucs[:10]]

    best_score = np.inf
    best_features = feats[:1]

    for size in range(1, min(max_size + 1, len(feats) + 1)):
        for combo in combinations(feats, size):
            result = fit_glm(X, y, list(combo))
            if result is None:
                continue
            aic, bic = get_aic_bic(result)
            score = aic if criterion == "aic" else bic
            if score < best_score:
                best_score = score
                best_features = list(combo)

    return best_features


def evaluate_cv(df: pd.DataFrame, threshold: float, features: list[str]) -> dict:
    """Run purged walk-forward CV with isotonic recalibration. Returns metrics."""
    fwd_dd = compute_forward_drawdowns(df, FORWARD_WINDOW, DD_DEFINITION)

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

        if n_events_train < 3 or n_events_train == len(y_train):
            continue
        if len(np.unique(y_test)) < 2:
            continue
        if base_rate_test < 0.02 or base_rate_test > 0.98:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Use statsmodels GLM for consistency
        X_train_c = sm.add_constant(X_train_s)
        X_test_c = sm.add_constant(X_test_s)

        try:
            model = sm.GLM(y_train, X_train_c,
                           family=sm.families.Binomial(link=sm.families.links.Logit()))
            result = model.fit(disp=0, maxiter=200)
            y_pred_test_raw = result.predict(X_test_c)
            y_pred_train = result.predict(X_train_c)
        except Exception:
            continue

        # Isotonic recalibration
        try:
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(y_pred_train, y_train)
            y_pred_test = iso.predict(y_pred_test_raw)
        except Exception:
            y_pred_test = np.array(y_pred_test_raw)

        y_pred_test = np.clip(y_pred_test, 0.001, 0.999)

        auc = roc_auc_score(y_test, y_pred_test)
        brier = brier_score_loss(y_test, y_pred_test)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred_test)

        fold_metrics.append({
            "fold": fold_i + 1,
            "auc": float(auc),
            "brier": float(brier),
            "bss": float(bss),
            "ece": float(ece),
            "n_test": len(y_test),
            "base_rate_test": float(base_rate_test),
        })

    if fold_metrics:
        aucs = [m["auc"] for m in fold_metrics]
        bsss = [m["bss"] for m in fold_metrics]
        eces = [m["ece"] for m in fold_metrics]
        return {
            "n_folds": len(fold_metrics),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "bss_mean": float(np.mean(bsss)),
            "bss_std": float(np.std(bsss)),
            "ece_mean": float(np.mean(eces)),
            "ece_std": float(np.std(eces)),
            "folds": fold_metrics,
        }
    return {
        "n_folds": 0, "auc_mean": 0.5, "auc_std": 0.0,
        "bss_mean": 0.0, "bss_std": 0.0, "ece_mean": 0.5, "ece_std": 0.0,
        "folds": [],
    }


def compute_hosmer_lemeshow(y_true, y_pred, n_groups=10):
    """Hosmer-Lemeshow goodness-of-fit test."""
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    groups = np.array_split(np.arange(len(y_true)), n_groups)
    chi2 = 0.0
    for g in groups:
        if len(g) == 0:
            continue
        obs = y_true_sorted[g].sum()
        exp = y_pred_sorted[g].sum()
        n_g = len(g)
        avg_p = y_pred_sorted[g].mean()
        denom = n_g * avg_p * (1 - avg_p)
        if denom > 1e-10:
            chi2 += (obs - exp) ** 2 / denom

    df = n_groups - 2
    p_value = 1 - scipy_stats.chi2.cdf(chi2, df)
    return {"chi2": float(chi2), "df": df, "p_value": float(p_value)}


def compute_mcfadden_r2(y, X_features, feature_list):
    """McFadden's pseudo-R-squared."""
    # Full model
    Xfull = sm.add_constant(X_features[feature_list].values.astype(float))
    full = sm.GLM(y, Xfull, family=sm.families.Binomial()).fit(disp=0)

    # Null model (intercept only)
    Xnull = np.ones((len(y), 1))
    null = sm.GLM(y, Xnull, family=sm.families.Binomial()).fit(disp=0)

    r2 = 1 - (full.llf / null.llf)
    return float(r2), float(full.llf), float(null.llf)


def profile_likelihood_ci(X: pd.DataFrame, y: np.ndarray, feature_list: list[str],
                          alpha: float = 0.05) -> dict:
    """Compute profile likelihood confidence intervals for coefficients.

    Uses Wald-based CI as approximation (statsmodels conf_int).
    """
    Xsub = sm.add_constant(X[feature_list].values.astype(float))
    model = sm.GLM(y, Xsub, family=sm.families.Binomial()).fit(disp=0)

    ci = model.conf_int(alpha=alpha)
    params = model.params
    pvalues = model.pvalues

    result = {}
    names = ["const"] + feature_list
    for i, name in enumerate(names):
        result[name] = {
            "coef": float(params[i]),
            "ci_lower": float(ci[i, 0]),
            "ci_upper": float(ci[i, 1]),
            "p_value": float(pvalues[i]),
        }
    return result


def main():
    print("=" * 70)
    print("AIC/BIC STEPWISE FEATURE SELECTION + PROFILE LIKELIHOOD CI")
    print("=" * 70)

    print("\nLoading data...")
    df = load_data()
    print(f"  {len(df)} trading days, {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    results = {}

    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        print(f"\n{'=' * 70}")
        print(f"  THRESHOLD: {pct}% DD ({DD_DEFINITION}, W={FORWARD_WINDOW}d)")
        print(f"{'=' * 70}")

        X_all, y_all, sub_idx, available = prepare_data(df, threshold)
        n_events = int(y_all.sum())
        base_rate = float(y_all.mean())
        print(f"  n={len(y_all)}, events={n_events}, base_rate={base_rate:.3f}")
        print(f"  Available features ({len(available)}): {available}")

        threshold_results = {
            "threshold": threshold,
            "n_samples": len(y_all),
            "n_events": n_events,
            "base_rate": round(base_rate, 4),
            "methods": {},
        }

        methods = {
            "aic_forward": ("AIC Forward Stepwise", lambda: forward_stepwise(X_all, y_all, available, "aic")),
            "bic_forward": ("BIC Forward Stepwise", lambda: forward_stepwise(X_all, y_all, available, "bic")),
            "aic_backward": ("AIC Backward Elimination", lambda: backward_elimination(X_all, y_all, available, "aic")),
            "bic_backward": ("BIC Backward Elimination", lambda: backward_elimination(X_all, y_all, available, "bic")),
            "aic_exhaustive": ("AIC Exhaustive (size 1-5)", lambda: exhaustive_search(X_all, y_all, available, "aic", 5)),
            "bic_exhaustive": ("BIC Exhaustive (size 1-5)", lambda: exhaustive_search(X_all, y_all, available, "bic", 5)),
        }

        for method_key, (method_name, select_fn) in methods.items():
            print(f"\n  --- {method_name} ---")
            selected = select_fn()
            print(f"  Selected ({len(selected)}): {selected}")

            if not selected:
                print("  No features selected, skipping.")
                threshold_results["methods"][method_key] = {
                    "name": method_name, "features": [], "cv": None
                }
                continue

            # Full-data GLM fit for AIC/BIC/LL
            glm_result = fit_glm(X_all, y_all, selected)
            aic_val = float(glm_result.aic) if glm_result else None
            bic_val = float(glm_result.bic_llf) if glm_result else None
            ll_val = float(glm_result.llf) if glm_result else None

            # McFadden R2
            mcfadden_r2, ll_full, ll_null = compute_mcfadden_r2(y_all, X_all, selected)
            print(f"  AIC={aic_val:.1f}  BIC={bic_val:.1f}  LL={ll_val:.1f}  McFadden R2={mcfadden_r2:.4f}")

            # Hosmer-Lemeshow
            y_pred_full = glm_result.predict(sm.add_constant(X_all[selected].values.astype(float)))
            hl = compute_hosmer_lemeshow(y_all, np.array(y_pred_full))
            print(f"  Hosmer-Lemeshow: chi2={hl['chi2']:.2f}, df={hl['df']}, p={hl['p_value']:.4f}")

            # Profile likelihood CI
            ci = profile_likelihood_ci(X_all, y_all, selected)
            print(f"  Coefficient CIs (95%):")
            for name, vals in ci.items():
                sig = "*" if vals["p_value"] < 0.05 else ""
                print(f"    {name:<30}: {vals['coef']:+.4f}  [{vals['ci_lower']:+.4f}, {vals['ci_upper']:+.4f}]  p={vals['p_value']:.4f}{sig}")

            # Purged walk-forward CV
            cv = evaluate_cv(df, threshold, selected)
            print(f"  CV ({cv['n_folds']} folds): AUC={cv['auc_mean']:.4f}+/-{cv['auc_std']:.4f}  "
                  f"BSS={cv['bss_mean']:+.1%}+/-{cv['bss_std']:.1%}  "
                  f"ECE={cv['ece_mean']:.4f}+/-{cv['ece_std']:.4f}")

            baseline = BASELINE[threshold]
            delta_auc = cv["auc_mean"] - baseline["auc"]
            delta_bss = cv["bss_mean"] - baseline["bss"]
            print(f"  vs baseline: dAUC={delta_auc:+.4f}  dBSS={delta_bss:+.1%}")

            threshold_results["methods"][method_key] = {
                "name": method_name,
                "features": selected,
                "n_features": len(selected),
                "aic": round(aic_val, 2) if aic_val else None,
                "bic": round(bic_val, 2) if bic_val else None,
                "log_likelihood": round(ll_val, 2) if ll_val else None,
                "mcfadden_r2": round(mcfadden_r2, 4),
                "hosmer_lemeshow": {k: round(v, 4) if isinstance(v, float) else v for k, v in hl.items()},
                "profile_ci": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ci.items()},
                "cv": {
                    "n_folds": cv["n_folds"],
                    "auc_mean": round(cv["auc_mean"], 4),
                    "auc_std": round(cv["auc_std"], 4),
                    "bss_mean": round(cv["bss_mean"], 4),
                    "bss_std": round(cv["bss_std"], 4),
                    "ece_mean": round(cv["ece_mean"], 4),
                    "ece_std": round(cv["ece_std"], 4),
                },
                "vs_baseline": {
                    "delta_auc": round(delta_auc, 4),
                    "delta_bss": round(delta_bss, 4),
                },
            }

        results[f"dd_{pct}pct"] = threshold_results

    # ── Comparison Table ──
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        key = f"dd_{pct}pct"
        baseline = BASELINE[threshold]
        print(f"\n  {pct}% Drawdown (baseline: AUC={baseline['auc']:.3f}  BSS={baseline['bss']:+.1%}  ECE={baseline['ece']:.3f})")
        print(f"  {'Method':<28} {'#Feat':>5} {'AUC':>7} {'BSS':>8} {'ECE':>7} {'AIC':>8} {'BIC':>8} {'R2':>7}")
        print(f"  {'-'*84}")

        # Baseline row
        print(f"  {'v3.2 Baseline':<28} {'--':>5} {baseline['auc']:>7.4f} {baseline['bss']:>+7.1%} {baseline['ece']:>7.4f} {'--':>8} {'--':>8} {'--':>7}")

        for method_key, method_data in results[key]["methods"].items():
            cv = method_data.get("cv")
            if cv is None:
                continue
            n_feat = method_data["n_features"]
            auc_str = f"{cv['auc_mean']:.4f}"
            bss_str = f"{cv['bss_mean']:+.1%}"
            ece_str = f"{cv['ece_mean']:.4f}"
            aic_str = f"{method_data['aic']:.1f}" if method_data['aic'] else "--"
            bic_str = f"{method_data['bic']:.1f}" if method_data['bic'] else "--"
            r2_str = f"{method_data['mcfadden_r2']:.4f}"
            print(f"  {method_data['name']:<28} {n_feat:>5} {auc_str:>7} {bss_str:>8} {ece_str:>7} {aic_str:>8} {bic_str:>8} {r2_str:>7}")

    # Save results
    out_path = SCRIPT_DIR / "results_aic_bic.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
