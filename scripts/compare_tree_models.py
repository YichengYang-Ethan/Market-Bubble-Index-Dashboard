#!/usr/bin/env python3
"""Compare tree-based models (XGBoost, Random Forest) against Logistic baseline.

Uses the same purged walk-forward CV infrastructure as fit_drawdown_model.py.
Tests grid of hyperparameters for each model type across 10% and 20% DD thresholds.
"""

from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "public" / "data"

# CV parameters (same as fit_drawdown_model.py)
N_CV_FOLDS = 5
MIN_TRAIN_FRAC = 0.35
PURGE_DAYS = 180
EMBARGO_DAYS = 20

# All available features for tree models
ALL_FEATURES = [
    "risk_ema_20d",
    "score_std_20d",
    "ind_vix_level",
    "ind_qqq_deviation_sma_20d",
    "score_ema_20d",
    "ind_yield_curve",
    "ind_vix_level_change_5d",
    "vix_x_credit",
    "is_elevated",
    "drawdown_risk_score",
]

# Thresholds to test
THRESHOLDS = [0.10, 0.20]
DD_DEFINITION = "peak_to_trough"
FORWARD_WINDOW = 180

# Hyperparameter grids
XGBOOST_GRID = {
    "max_depth": [2, 3, 4],
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1],
    "min_child_weight": [5, 10, 20],
}

RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [10, 20, 50],
}

# Logistic baseline config (best from fit_drawdown_model.py grid search)
LOGISTIC_CONFIGS = {
    0.10: {"features": ["risk_ema_20d", "score_std_20d"], "C": 0.1},
    0.20: {"features": ["risk_ema_20d", "score_std_20d", "ind_yield_curve"], "C": 0.1},
}


# ── Reused functions from fit_drawdown_model.py ──


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

    # Drawdown risk score
    if "drawdown_risk_score" not in df_h.columns:
        risk_inversions = {"ind_qqq_deviation", "ind_vix_level", "ind_yield_curve"}
        ind_cols = [c for c in df_h.columns if c.startswith("ind_")]
        if ind_cols:
            risk_score = pd.Series(0.0, index=df_h.index)
            for col in ind_cols:
                val = df_h[col].fillna(50.0)
                if col in risk_inversions:
                    risk_score += (100 - val)
                else:
                    risk_score += val
            df_h["drawdown_risk_score"] = risk_score / max(len(ind_cols), 1)

    # Smoothed scores
    df_h["score_sma_60d"] = df_h["composite_score"].rolling(60, min_periods=20).mean()
    df_h["score_ema_20d"] = df_h["composite_score"].ewm(span=20, min_periods=10).mean()
    if "drawdown_risk_score" in df_h.columns:
        df_h["risk_ema_20d"] = df_h["drawdown_risk_score"].ewm(span=20, min_periods=10).mean()
    df_h["score_std_20d"] = df_h["composite_score"].rolling(20, min_periods=10).std()
    if "ind_qqq_deviation" in df_h.columns:
        df_h["ind_qqq_deviation_sma_20d"] = df_h["ind_qqq_deviation"].rolling(20, min_periods=10).mean()
    if "ind_vix_level" in df_h.columns:
        df_h["ind_vix_level_change_5d"] = df_h["ind_vix_level"].diff(5)
    if "ind_vix_level" in df_h.columns and "ind_credit_spread" in df_h.columns:
        df_h["vix_x_credit"] = df_h["ind_vix_level"] * df_h["ind_credit_spread"] / 100
    df_h["is_elevated"] = (df_h["composite_score"] > 60).astype(float)

    # Merge QQQ prices
    df_q = pd.DataFrame(qqq)[["date", "price"]].copy()
    df_q.rename(columns={"price": "qqq_price"}, inplace=True)
    df = df_h.merge(df_q, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Fill NaN with median
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
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece / len(y_true))


# ── Model evaluation ──


def evaluate_model_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    splits: list,
    model_factory,
    model_name: str,
    needs_scaling: bool = False,
) -> dict:
    """Run purged walk-forward CV for a given model. Returns per-fold and aggregate metrics."""
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        base_rate_test = float(y_test.mean())
        n_events_train = int(y_train.sum())

        # Skip degenerate folds
        if n_events_train < 3 or n_events_train == len(y_train):
            continue
        if len(np.unique(y_test)) < 2:
            continue
        if base_rate_test < 0.02 or base_rate_test > 0.98:
            continue

        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred_test_raw = model.predict_proba(X_test)[:, 1]

        # Isotonic recalibration
        y_pred_train = model.predict_proba(X_train)[:, 1]
        try:
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(y_pred_train, y_train)
            y_pred_test = iso.predict(y_pred_test_raw)
        except Exception:
            y_pred_test = y_pred_test_raw

        auc = roc_auc_score(y_test, y_pred_test)
        brier = brier_score_loss(y_test, y_pred_test)
        brier_clim = y_test.mean() * (1 - y_test.mean())
        bss = 1 - brier / max(brier_clim, 1e-10) if brier_clim > 0 else 0.0
        ece = compute_ece(y_test, y_pred_test)

        fold_metrics.append({
            "fold": fold_i + 1,
            "n_test": len(X_test),
            "base_rate": round(base_rate_test, 4),
            "auc": round(float(auc), 4),
            "bss": round(float(bss), 4),
            "ece": round(float(ece), 4),
        })

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
        aggregate = {
            "n_folds": 0, "auc_mean": 0.5, "auc_std": 0,
            "bss_mean": 0, "bss_std": 0, "ece_mean": 0, "ece_std": 0,
        }

    return {"model": model_name, "folds": fold_metrics, "aggregate": aggregate}


def get_feature_importances(model, feature_names: list[str], model_type: str) -> dict:
    """Extract feature importances from a fitted model."""
    if model_type == "xgboost":
        imp = model.feature_importances_
    elif model_type == "rf":
        imp = model.feature_importances_
    elif model_type == "logistic":
        imp = np.abs(model.coef_[0])
    else:
        return {}
    pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    return {name: round(float(val), 4) for name, val in pairs}


def main():
    print("=" * 74)
    print("TREE-BASED MODEL COMPARISON")
    print(f"Purged Walk-Forward CV: {N_CV_FOLDS} folds, purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d")
    print("=" * 74)

    df = load_data()
    print(f"\nData: {len(df)} trading days, {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    # Determine available features
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    print(f"Features ({len(available_features)}): {available_features}")
    if missing:
        print(f"Missing features (skipped): {missing}")

    fwd_dd = compute_forward_drawdowns(df, FORWARD_WINDOW, DD_DEFINITION)

    all_results = {}

    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        print(f"\n{'=' * 74}")
        print(f"THRESHOLD: >={pct}% Drawdown ({DD_DEFINITION}, W={FORWARD_WINDOW}d)")
        print(f"{'=' * 74}")

        # Prepare data
        valid_mask = np.isfinite(fwd_dd)
        for f in available_features:
            valid_mask &= df[f].notna().values

        sub_idx = np.where(valid_mask)[0]
        X_all = df.iloc[sub_idx][available_features].values.astype(float)
        y_all = (fwd_dd[sub_idx] >= threshold).astype(float)
        n = len(X_all)
        base_rate = float(y_all.mean())

        print(f"  Samples: {n}, Base rate: {base_rate:.1%}")

        splits = purged_walk_forward_splits(n)
        print(f"  CV splits: {len(splits)}")

        results_for_threshold = []

        # ── 1. Logistic Baseline ──
        print(f"\n  {'─' * 60}")
        print(f"  LOGISTIC REGRESSION (baseline)")
        print(f"  {'─' * 60}")

        log_cfg = LOGISTIC_CONFIGS[threshold]
        log_features = log_cfg["features"]
        log_feat_idx = [available_features.index(f) for f in log_features if f in available_features]
        X_log = X_all[:, log_feat_idx]

        log_cv = evaluate_model_cv(
            X_log, y_all, splits,
            model_factory=lambda: LogisticRegression(max_iter=5000, C=log_cfg["C"], solver="lbfgs"),
            model_name=f"Logistic(C={log_cfg['C']}, feats={log_features})",
            needs_scaling=True,
        )
        agg = log_cv["aggregate"]
        print(f"    AUC: {agg['auc_mean']:.4f} +/- {agg['auc_std']:.4f}  "
              f"BSS: {agg['bss_mean']:+.1%} +/- {agg['bss_std']:.1%}  "
              f"ECE: {agg['ece_mean']:.4f}")
        results_for_threshold.append({
            "model": "Logistic",
            "params": {"C": log_cfg["C"], "features": log_features},
            **agg,
        })

        # ── 2. XGBoost Grid Search ──
        print(f"\n  {'─' * 60}")
        print(f"  XGBOOST (grid search: {len(list(product(*XGBOOST_GRID.values())))} combos)")
        print(f"  {'─' * 60}")

        best_xgb_auc = -1
        best_xgb_result = None
        best_xgb_params = None
        xgb_count = 0

        for md, ne, lr, mcw in product(
            XGBOOST_GRID["max_depth"],
            XGBOOST_GRID["n_estimators"],
            XGBOOST_GRID["learning_rate"],
            XGBOOST_GRID["min_child_weight"],
        ):
            xgb_count += 1
            params = {"max_depth": md, "n_estimators": ne, "learning_rate": lr, "min_child_weight": mcw}

            def make_xgb(md=md, ne=ne, lr=lr, mcw=mcw):
                return XGBClassifier(
                    max_depth=md, n_estimators=ne, learning_rate=lr,
                    min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1.0, reg_lambda=1.0,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0, random_state=42,
                )

            cv = evaluate_model_cv(X_all, y_all, splits, make_xgb, f"XGB({params})")
            a = cv["aggregate"]

            if a["n_folds"] >= 2 and a["auc_mean"] > best_xgb_auc:
                best_xgb_auc = a["auc_mean"]
                best_xgb_result = cv
                best_xgb_params = params

        if best_xgb_result:
            agg = best_xgb_result["aggregate"]
            print(f"    Best: {best_xgb_params}")
            print(f"    AUC: {agg['auc_mean']:.4f} +/- {agg['auc_std']:.4f}  "
                  f"BSS: {agg['bss_mean']:+.1%} +/- {agg['bss_std']:.1%}  "
                  f"ECE: {agg['ece_mean']:.4f}")
            for fm in best_xgb_result["folds"]:
                print(f"      Fold {fm['fold']}: AUC={fm['auc']:.4f}  BSS={fm['bss']:+.1%}  "
                      f"ECE={fm['ece']:.4f}  base_rate={fm['base_rate']:.1%}")

            results_for_threshold.append({
                "model": "XGBoost",
                "params": best_xgb_params,
                **agg,
            })

            # Fit best XGBoost on all data for feature importances
            best_xgb_model = XGBClassifier(
                **best_xgb_params, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0,
                use_label_encoder=False, eval_metric="logloss",
                verbosity=0, random_state=42,
            )
            best_xgb_model.fit(X_all, y_all)
            xgb_importances = get_feature_importances(best_xgb_model, available_features, "xgboost")
            print(f"    Feature importances:")
            for feat, imp in xgb_importances.items():
                bar = "#" * int(imp * 50)
                print(f"      {feat:<30} {imp:.4f}  {bar}")

        # ── 3. Random Forest Grid Search ──
        print(f"\n  {'─' * 60}")
        print(f"  RANDOM FOREST (grid search: {len(list(product(*RF_GRID.values())))} combos)")
        print(f"  {'─' * 60}")

        best_rf_auc = -1
        best_rf_result = None
        best_rf_params = None

        for ne, md, msl in product(
            RF_GRID["n_estimators"],
            RF_GRID["max_depth"],
            RF_GRID["min_samples_leaf"],
        ):
            params = {"n_estimators": ne, "max_depth": md, "min_samples_leaf": msl}

            def make_rf(ne=ne, md=md, msl=msl):
                return RandomForestClassifier(
                    n_estimators=ne, max_depth=md, min_samples_leaf=msl,
                    max_features="sqrt", random_state=42, n_jobs=-1,
                )

            cv = evaluate_model_cv(X_all, y_all, splits, make_rf, f"RF({params})")
            a = cv["aggregate"]

            if a["n_folds"] >= 2 and a["auc_mean"] > best_rf_auc:
                best_rf_auc = a["auc_mean"]
                best_rf_result = cv
                best_rf_params = params

        if best_rf_result:
            agg = best_rf_result["aggregate"]
            print(f"    Best: {best_rf_params}")
            print(f"    AUC: {agg['auc_mean']:.4f} +/- {agg['auc_std']:.4f}  "
                  f"BSS: {agg['bss_mean']:+.1%} +/- {agg['bss_std']:.1%}  "
                  f"ECE: {agg['ece_mean']:.4f}")
            for fm in best_rf_result["folds"]:
                print(f"      Fold {fm['fold']}: AUC={fm['auc']:.4f}  BSS={fm['bss']:+.1%}  "
                      f"ECE={fm['ece']:.4f}  base_rate={fm['base_rate']:.1%}")

            results_for_threshold.append({
                "model": "RandomForest",
                "params": best_rf_params,
                **agg,
            })

            # Feature importances
            best_rf_model = RandomForestClassifier(
                **best_rf_params, max_features="sqrt", random_state=42, n_jobs=-1,
            )
            best_rf_model.fit(X_all, y_all)
            rf_importances = get_feature_importances(best_rf_model, available_features, "rf")
            print(f"    Feature importances:")
            for feat, imp in rf_importances.items():
                bar = "#" * int(imp * 50)
                print(f"      {feat:<30} {imp:.4f}  {bar}")

        all_results[f"dd_{pct}pct"] = results_for_threshold

    # ── Summary Table ──
    print(f"\n{'=' * 74}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'=' * 74}")
    print(f"{'Model':<20} {'Threshold':<12} {'AUC':>8} {'AUC_std':>8} {'BSS':>8} {'BSS_std':>8} {'ECE':>8} {'Folds':>6}")
    print("-" * 74)

    for thresh_key, models in all_results.items():
        for m in models:
            print(f"{m['model']:<20} {thresh_key:<12} "
                  f"{m['auc_mean']:>8.4f} {m['auc_std']:>8.4f} "
                  f"{m['bss_mean']:>+8.1%} {m['bss_std']:>8.1%} "
                  f"{m['ece_mean']:>8.4f} {m['n_folds']:>6}")

    # ── Save JSON ──
    output_path = SCRIPT_DIR / "results_tree_models.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
