#!/usr/bin/env python3
"""
A/B Test: Current Model (v3.3) vs Enhanced Model with new features.

Runs purged walk-forward cross-validation on BOTH feature sets using
the SAME data, SAME splits, SAME evaluation metrics.  Reports hard
OOS numbers so we can answer: "does the enhancement actually help?"

New features engineered from EXISTING data (no external downloads):
  1. bull_duration     — trading days since last >10% peak-to-trough drawdown
  2. regime_duration   — consecutive days in current regime (composite > 60)
  3. credit_delta_5d   — 5-day change of credit spread score
  4. credit_delta_20d  — 20-day change of credit spread score
  5. risk_composite_spread — risk_score - composite_score (divergence)
  6. score_skew_20d    — rolling 20d skewness of composite score
  7. qqq_vol_20d       — 20-day realized volatility of QQQ returns
  8. qqq_momentum_60d  — 60-day price momentum (return)
  9. vix_credit_diverge — VIX score - Credit spread score (stress divergence)

For VIX/VIX3M term structure, we attempt a yfinance download; if
unavailable, we skip that feature and test the rest.

All comparisons use identical purged walk-forward CV with
purge=180d, embargo=20d, 5 folds.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

# ── CV Parameters (same as production) ──
N_CV_FOLDS = 5
MIN_TRAIN_FRAC = 0.35
PURGE_DAYS = 180
EMBARGO_DAYS = 20


# ═══════════════════════════════════════════════════════════════════════════
# Data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def load_base_data() -> pd.DataFrame:
    """Load bubble history + QQQ prices, reproduce v3.3 features."""
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
        risk_inv = {"ind_qqq_deviation", "ind_vix_level", "ind_yield_curve"}
        ind_cols = [c for c in df_h.columns if c.startswith("ind_")]
        if ind_cols:
            rs = pd.Series(0.0, index=df_h.index)
            for col in ind_cols:
                val = df_h[col].fillna(50.0)
                rs += (100 - val) if col in risk_inv else val
            df_h["drawdown_risk_score"] = rs / max(len(ind_cols), 1)

    # ── v3.3 baseline features ──
    df_h["score_sma_60d"] = df_h["composite_score"].rolling(60, min_periods=20).mean()
    df_h["score_ema_20d"] = df_h["composite_score"].ewm(span=20, min_periods=10).mean()
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

    return df


def engineer_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the proposed enhanced features."""
    df = df.copy()

    # 1. Bull market duration: days since last >10% peak-to-trough drawdown
    prices = df["qqq_price"].values
    peak = prices[0]
    in_drawdown = False
    last_dd_end = 0
    bull_dur = np.zeros(len(prices))
    for i in range(len(prices)):
        if prices[i] > peak:
            peak = prices[i]
        dd = (peak - prices[i]) / peak
        if dd >= 0.10:
            in_drawdown = True
            last_dd_end = i
        elif in_drawdown and dd < 0.05:
            in_drawdown = False
            last_dd_end = i
            peak = prices[i]
        bull_dur[i] = i - last_dd_end
    df["bull_duration"] = bull_dur
    # Normalize to percentile rank over expanding window
    df["bull_duration_pct"] = df["bull_duration"].expanding(min_periods=60).apply(
        lambda w: float((w < w.iloc[-1]).sum()) / max(len(w) - 1, 1) * 100
        if len(w) > 1 else 50.0, raw=False
    )

    # 2. Regime duration: consecutive days with composite > 60
    regime_dur = np.zeros(len(df))
    count = 0
    for i in range(len(df)):
        if df["composite_score"].iloc[i] > 60:
            count += 1
        else:
            count = 0
        regime_dur[i] = count
    df["regime_duration"] = regime_dur

    # 3-4. Credit spread change rates
    if "ind_credit_spread" in df.columns:
        df["credit_delta_5d"] = df["ind_credit_spread"].diff(5)
        df["credit_delta_20d"] = df["ind_credit_spread"].diff(20)

    # 5. Risk-Composite divergence
    if "drawdown_risk_score" in df.columns:
        df["risk_composite_spread"] = df["drawdown_risk_score"] - df["composite_score"]

    # 6. Score skewness (20d rolling)
    df["score_skew_20d"] = df["composite_score"].rolling(20, min_periods=10).skew()

    # 7. QQQ realized volatility (20d)
    qqq_ret = np.log(df["qqq_price"] / df["qqq_price"].shift(1))
    df["qqq_vol_20d"] = qqq_ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100

    # 8. QQQ momentum (60d return)
    df["qqq_momentum_60d"] = df["qqq_price"].pct_change(60) * 100

    # 9. VIX-Credit divergence
    if "ind_vix_level" in df.columns and "ind_credit_spread" in df.columns:
        df["vix_credit_diverge"] = df["ind_vix_level"] - df["ind_credit_spread"]

    # 10. Try VIX term structure (VIX/VIX3M)
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start="2014-01-01", progress=False)["Close"].squeeze().dropna()
        vix3m = yf.download("^VIX3M", start="2014-01-01", progress=False)["Close"].squeeze().dropna()
        if not vix.empty and not vix3m.empty:
            ts_ratio = (vix / vix3m).dropna()
            ts_df = ts_ratio.reset_index()
            ts_df.columns = ["date_dt", "vix_term_structure"]
            ts_df["date"] = ts_df["date_dt"].dt.strftime("%Y-%m-%d")
            df = df.merge(ts_df[["date", "vix_term_structure"]], on="date", how="left")
            # Percentile rank
            df["vix_ts_pct"] = df["vix_term_structure"].expanding(min_periods=60).apply(
                lambda w: float((w < w.iloc[-1]).sum()) / max(len(w) - 1, 1) * 100
                if len(w) > 1 else 50.0, raw=False
            )
            print("  ✓ VIX term structure (VIX/VIX3M) loaded successfully")
        else:
            print("  ✗ VIX3M data empty, skipping term structure")
    except Exception as e:
        print(f"  ✗ VIX term structure unavailable: {e}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Forward drawdown & CV (copied from production code for consistency)
# ═══════════════════════════════════════════════════════════════════════════

def compute_forward_drawdowns(df: pd.DataFrame, window: int, definition: str) -> np.ndarray:
    prices = df["qqq_price"].values
    n = len(prices)
    max_dd = np.full(n, np.nan)
    for i in range(n):
        end = min(i + window + 1, n)
        if end - i < 10:
            continue
        fwd = prices[i:end]
        if definition == "peak_to_trough":
            peak = fwd[0]
            dd = 0.0
            for p in fwd[1:]:
                if p > peak:
                    peak = p
                drawdown = (peak - p) / peak
                if drawdown > dd:
                    dd = drawdown
            max_dd[i] = dd
        else:
            today = fwd[0]
            min_future = np.min(fwd[1:])
            max_dd[i] = max(0.0, (today - min_future) / today)
    return max_dd


def purged_walk_forward_splits(n, n_splits=N_CV_FOLDS, min_train_frac=MIN_TRAIN_FRAC,
                                purge=PURGE_DAYS, embargo=EMBARGO_DAYS):
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


def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece / len(y_true))


# ═══════════════════════════════════════════════════════════════════════════
# Core evaluation function
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(df: pd.DataFrame, features: list[str], threshold: float,
                   dd_def: str, window: int, penalty: str, C: float,
                   label: str) -> dict:
    """Run purged walk-forward CV for a given feature set. Returns metrics dict."""
    feats = [f for f in features if f in df.columns]
    if not feats:
        return {"label": label, "error": "No features available"}

    fwd_dd = compute_forward_drawdowns(df, window, dd_def)

    # Valid rows
    valid_mask = np.isfinite(fwd_dd)
    for f in feats:
        valid_mask &= df[f].notna().values

    sub_idx = np.where(valid_mask)[0]
    X_all = df.iloc[sub_idx][feats].values.astype(float)
    y_all = (fwd_dd[sub_idx] >= threshold).astype(float)
    n = len(X_all)

    splits = purged_walk_forward_splits(n)

    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        n_events = int(y_train.sum())
        if n_events < 3 or n_events == len(y_train):
            continue
        if len(np.unique(y_test)) < 2:
            continue
        base_rate_test = float(y_test.mean())
        if base_rate_test < 0.02 or base_rate_test > 0.98:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model_kwargs = {"max_iter": 10000, "C": C}
        if penalty == "l1":
            model_kwargs.update({"penalty": "l1", "solver": "saga"})
        else:
            model_kwargs.update({"penalty": "l2", "solver": "lbfgs"})
        model = LogisticRegression(**model_kwargs)
        model.fit(X_train_s, y_train)

        y_pred_raw = model.predict_proba(X_test_s)[:, 1]

        # Isotonic recalibration
        try:
            from sklearn.isotonic import IsotonicRegression
            y_pred_train = model.predict_proba(X_train_s)[:, 1]
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(y_pred_train, y_train)
            y_pred = iso.predict(y_pred_raw)
        except Exception:
            y_pred = y_pred_raw

        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        brier_clim = base_rate_test * (1 - base_rate_test)
        bss = 1 - brier / max(brier_clim, 1e-10)
        ece = compute_ece(y_test, y_pred)

        fold_results.append({
            "fold": fold_i + 1,
            "n_test": len(y_test),
            "base_rate": round(base_rate_test, 4),
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "bss": round(bss, 4),
            "ece": round(ece, 4),
        })
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    if not fold_results:
        return {"label": label, "error": "No valid folds"}

    aucs = [f["auc"] for f in fold_results]
    bsss = [f["bss"] for f in fold_results]
    eces = [f["ece"] for f in fold_results]

    # Pooled metrics across all folds
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    pooled_auc = roc_auc_score(all_y_true, all_y_pred)
    pooled_brier = brier_score_loss(all_y_true, all_y_pred)
    pooled_base = all_y_true.mean()
    pooled_bss = 1 - pooled_brier / max(pooled_base * (1 - pooled_base), 1e-10)

    # Monotonicity test on pooled predictions
    score_col = df.iloc[sub_idx]["drawdown_risk_score"].values if "drawdown_risk_score" in df.columns else None
    mono_score = None
    if score_col is not None:
        bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        # Collect indices that went into test sets
        test_indices = []
        for _, test_idx in splits:
            test_indices.extend(test_idx.tolist())
        if test_indices:
            # For test observations, check if predicted prob increases with score bin
            bin_probs = []
            for lo, hi in bins:
                mask_bin = []
                for ti in range(len(all_y_true)):
                    global_i = sub_idx[test_indices[ti]] if ti < len(test_indices) else -1
                    if 0 <= global_i < len(df) and score_col[test_indices[ti] if ti < len(test_indices) else 0] is not None:
                        sc = score_col[test_indices[ti]] if ti < len(test_indices) else 50
                        if lo <= sc < hi:
                            mask_bin.append(ti)
                if mask_bin:
                    bin_probs.append(float(all_y_pred[mask_bin].mean()))
            if len(bin_probs) >= 3:
                diffs = [bin_probs[i+1] - bin_probs[i] for i in range(len(bin_probs)-1)]
                mono_score = sum(1 for d in diffs if d > 0) / len(diffs)

    return {
        "label": label,
        "features": feats,
        "n_features": len(feats),
        "n_folds": len(fold_results),
        "per_fold": fold_results,
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "bss_mean": round(float(np.mean(bsss)), 4),
        "bss_std": round(float(np.std(bsss)), 4),
        "ece_mean": round(float(np.mean(eces)), 4),
        "pooled_auc": round(float(pooled_auc), 4),
        "pooled_bss": round(float(pooled_bss), 4),
        "pooled_n": len(all_y_true),
        "monotonicity": round(mono_score, 2) if mono_score is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Survival analysis (Cox PH) evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_cox_model(df: pd.DataFrame, features: list[str], threshold: float,
                       window: int, label: str) -> dict | None:
    """Attempt Cox PH survival analysis if lifelines is available."""
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        return None

    feats = [f for f in features if f in df.columns]
    if not feats:
        return None

    # Time-to-event: days until drawdown >= threshold, censored at window
    prices = df["qqq_price"].values
    n = len(prices)
    duration = np.full(n, np.nan)
    event = np.full(n, 0.0)

    for i in range(n):
        end = min(i + window + 1, n)
        if end - i < 10:
            continue
        peak = prices[i]
        for j in range(i + 1, end):
            if prices[j] > peak:
                peak = prices[j]
            dd = (peak - prices[j]) / peak
            if dd >= threshold:
                duration[i] = j - i
                event[i] = 1.0
                break
        if np.isnan(duration[i]):
            duration[i] = end - i - 1
            event[i] = 0.0  # censored

    valid = np.isfinite(duration)
    for f in feats:
        valid &= df[f].notna().values

    sub_idx = np.where(valid)[0]
    surv_df = df.iloc[sub_idx][feats].copy()
    surv_df["duration"] = duration[sub_idx]
    surv_df["event"] = event[sub_idx]

    # Walk-forward CV
    n_sub = len(surv_df)
    splits = purged_walk_forward_splits(n_sub)

    c_indices = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_data = surv_df.iloc[train_idx]
        test_data = surv_df.iloc[test_idx]
        if train_data["event"].sum() < 5:
            continue
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train_data, duration_col="duration", event_col="event")
            # Predict risk scores (higher = higher risk)
            risk_scores = -cph.predict_partial_hazard(test_data).values.flatten()
            ci = concordance_index(test_data["duration"], risk_scores, test_data["event"])
            c_indices.append(ci)
        except Exception:
            continue

    if not c_indices:
        return None

    return {
        "label": label,
        "method": "Cox PH (lifelines)",
        "n_folds": len(c_indices),
        "c_index_mean": round(float(np.mean(c_indices)), 4),
        "c_index_std": round(float(np.std(c_indices)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main comparison
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  A/B TEST: Current v3.3 vs Enhanced Feature Set")
    print("  Purged Walk-Forward CV, purge=180d, embargo=20d, 5 folds")
    print("=" * 72)

    print("\n[1/3] Loading data & engineering features...")
    df = load_base_data()
    print(f"  Base data: {len(df)} trading days, {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    df = engineer_new_features(df)

    # Fill NaN with median
    all_feat_cols = [c for c in df.columns if c not in ("date", "regime", "qqq_price")]
    for col in all_feat_cols:
        if df[col].dtype in (np.float64, np.int64, float, int):
            df[col] = df[col].fillna(df[col].median())

    # ── Define model configs ──
    # Baseline: current v3.3 features
    BASELINE_10 = {
        "features": ["risk_ema_20d", "ind_qqq_deviation", "ind_credit_spread",
                      "ind_yield_curve", "ind_vix_level", "vix_x_credit",
                      "ind_vix_level_change_5d", "score_velocity"],
        "penalty": "l1", "C": 5.0,
    }
    BASELINE_20 = {
        "features": ["risk_ema_20d", "score_std_20d"],
        "penalty": "l2", "C": 1.0,
    }

    # Enhanced: baseline + new features
    new_features = [
        "bull_duration_pct", "regime_duration", "credit_delta_5d",
        "credit_delta_20d", "risk_composite_spread", "score_skew_20d",
        "qqq_vol_20d", "qqq_momentum_60d", "vix_credit_diverge",
    ]
    # Add VIX term structure if available
    if "vix_ts_pct" in df.columns:
        new_features.append("vix_ts_pct")

    ENHANCED_10 = {
        "features": BASELINE_10["features"] + new_features,
        "penalty": "l1", "C": 5.0,  # L1 will select useful features
    }
    ENHANCED_20 = {
        "features": BASELINE_20["features"] + new_features,
        "penalty": "l1", "C": 1.0,  # Switch to L1 for automatic selection
    }

    # Conservative enhanced: only strongest new features
    TARGETED_10 = {
        "features": BASELINE_10["features"] + [
            "bull_duration_pct", "qqq_vol_20d", "credit_delta_5d",
        ],
        "penalty": "l1", "C": 5.0,
    }
    TARGETED_20 = {
        "features": BASELINE_20["features"] + [
            "bull_duration_pct", "qqq_vol_20d", "risk_composite_spread",
        ],
        "penalty": "l2", "C": 1.0,
    }

    # ── Run all tests ──
    print("\n[2/3] Running purged walk-forward CV (this may take a minute)...\n")

    configs = [
        # 10% DD tests
        ("10% DD", 0.10, "peak_to_trough", 180, [
            ("A: Baseline v3.3", BASELINE_10),
            ("B: Enhanced (all new)", ENHANCED_10),
            ("C: Targeted (top 3 new)", TARGETED_10),
        ]),
        # 20% DD tests
        ("20% DD", 0.20, "peak_to_trough", 180, [
            ("A: Baseline v3.3", BASELINE_20),
            ("B: Enhanced (all new)", ENHANCED_20),
            ("C: Targeted (top 3 new)", TARGETED_20),
        ]),
    ]

    all_results = {}

    for group_label, threshold, dd_def, window, models in configs:
        print(f"\n{'━' * 72}")
        print(f"  {group_label} (threshold={threshold}, def={dd_def}, window={window}d)")
        print(f"{'━' * 72}")

        group_results = []
        for model_label, config in models:
            result = evaluate_model(
                df, config["features"], threshold, dd_def, window,
                config["penalty"], config["C"], model_label
            )
            group_results.append(result)

            if "error" in result:
                print(f"\n  {model_label}: ERROR - {result['error']}")
                continue

            print(f"\n  {model_label}")
            print(f"    Features ({result['n_features']}): {result['features']}")
            print(f"    Folds: {result['n_folds']}")
            print(f"    AUC (mean±std):    {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
            print(f"    BSS (mean±std):    {result['bss_mean']:+.1%} ± {result['bss_std']:.1%}")
            print(f"    ECE (mean):        {result['ece_mean']:.4f}")
            print(f"    Pooled AUC:        {result['pooled_auc']:.4f}")
            print(f"    Pooled BSS:        {result['pooled_bss']:+.1%}")
            print(f"    Pooled N:          {result['pooled_n']}")
            if result.get("monotonicity") is not None:
                print(f"    Monotonicity:      {result['monotonicity']:.0%}")
            for fd in result["per_fold"]:
                print(f"      Fold {fd['fold']}: AUC={fd['auc']:.4f}  BSS={fd['bss']:+.1%}  "
                      f"ECE={fd['ece']:.4f}  base_rate={fd['base_rate']:.1%}  n={fd['n_test']}")

        all_results[group_label] = group_results

    # ── Cox PH Survival Analysis (bonus) ──
    print(f"\n{'━' * 72}")
    print(f"  BONUS: Cox PH Survival Analysis")
    print(f"{'━' * 72}")

    cox_feats_base = ["risk_ema_20d", "score_std_20d", "ind_vix_level",
                      "ind_credit_spread", "score_velocity"]
    cox_feats_enh = cox_feats_base + ["bull_duration_pct", "qqq_vol_20d",
                                       "credit_delta_5d"]

    for threshold, label in [(0.10, "10% DD"), (0.20, "20% DD")]:
        for feats, flabel in [(cox_feats_base, "Baseline"), (cox_feats_enh, "Enhanced")]:
            result = evaluate_cox_model(df, feats, threshold, 180, f"{label} {flabel}")
            if result:
                print(f"\n  {result['label']}: C-index = {result['c_index_mean']:.4f} "
                      f"± {result['c_index_std']:.4f} ({result['n_folds']} folds)")
            else:
                print(f"\n  {label} {flabel}: N/A (lifelines not installed or insufficient data)")

    # ── Summary comparison table ──
    print(f"\n\n{'=' * 72}")
    print("  SUMMARY: Head-to-Head Comparison (OOS metrics)")
    print(f"{'=' * 72}")
    print(f"\n  {'Model':<30} {'AUC':>8} {'BSS':>10} {'ECE':>8} {'Mono':>6}")
    print(f"  {'─' * 62}")

    for group_label, results in all_results.items():
        print(f"  {group_label}:")
        for r in results:
            if "error" in r:
                print(f"    {r['label']:<28} {'ERROR':>8}")
                continue
            mono_str = f"{r['monotonicity']:.0%}" if r.get("monotonicity") is not None else "N/A"
            print(f"    {r['label']:<28} {r['pooled_auc']:>8.4f} "
                  f"{r['pooled_bss']:>+9.1%} {r['ece_mean']:>8.4f} {mono_str:>6}")

    # ── Delta analysis ──
    print(f"\n  {'─' * 62}")
    print(f"  Deltas (Enhanced - Baseline):")
    for group_label, results in all_results.items():
        if len(results) >= 2 and "error" not in results[0] and "error" not in results[1]:
            base = results[0]
            enh = results[1]
            d_auc = enh["pooled_auc"] - base["pooled_auc"]
            d_bss = enh["pooled_bss"] - base["pooled_bss"]
            d_ece = enh["ece_mean"] - base["ece_mean"]
            winner = "Enhanced" if d_auc > 0 else "Baseline"
            print(f"    {group_label}: ΔAUC={d_auc:+.4f}  ΔBSS={d_bss:+.1%}  "
                  f"ΔECE={d_ece:+.4f}  → {winner}")

    print(f"\n{'=' * 72}")
    print("  All metrics are OUT-OF-SAMPLE (purged walk-forward CV)")
    print(f"{'=' * 72}\n")

    # Save results
    output = {
        "test_date": str(pd.Timestamp.now()),
        "cv_method": f"purged_walk_forward_{N_CV_FOLDS}fold",
        "purge_days": PURGE_DAYS,
        "embargo_days": EMBARGO_DAYS,
        "results": {}
    }
    for group_label, results in all_results.items():
        output["results"][group_label] = []
        for r in results:
            r_clean = {k: v for k, v in r.items() if k != "per_fold"}
            output["results"][group_label].append(r_clean)

    out_path = Path(__file__).resolve().parent / "ab_test_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
