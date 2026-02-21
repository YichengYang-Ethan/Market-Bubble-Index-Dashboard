#!/usr/bin/env python3
"""
20% Drawdown Model Optimization — Exhaustive OOS A/B Test (v2 optimized)

Pre-computes all forward drawdown variants once, then runs 30+ model configs.
"""

from __future__ import annotations
import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"
N_CV_FOLDS = 5
MIN_TRAIN_FRAC = 0.35
PURGE_DAYS = 180
EMBARGO_DAYS = 20


def load_data():
    history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
    qqq_raw = json.loads((DATA_DIR / "qqq.json").read_text())
    qqq = qqq_raw["data"] if isinstance(qqq_raw, dict) and "data" in qqq_raw else qqq_raw

    df_h = pd.DataFrame(history)
    indicators = pd.json_normalize(df_h["indicators"])
    indicators.columns = ["ind_" + c for c in indicators.columns]
    df_h = pd.concat([df_h.drop(columns=["indicators"]), indicators], axis=1)
    df_h["score_velocity"] = df_h["score_velocity"].fillna(0)

    risk_inv = {"ind_qqq_deviation", "ind_vix_level", "ind_yield_curve"}
    ind_cols = [c for c in df_h.columns if c.startswith("ind_")]
    if "drawdown_risk_score" not in df_h.columns and ind_cols:
        rs = pd.Series(0.0, index=df_h.index)
        for col in ind_cols:
            val = df_h[col].fillna(50.0)
            rs += (100 - val) if col in risk_inv else val
        df_h["drawdown_risk_score"] = rs / max(len(ind_cols), 1)

    df_h["score_ema_20d"] = df_h["composite_score"].ewm(span=20, min_periods=10).mean()
    df_h["risk_ema_20d"] = df_h["drawdown_risk_score"].ewm(span=20, min_periods=10).mean()
    df_h["score_std_20d"] = df_h["composite_score"].rolling(20, min_periods=10).std()
    df_h["score_sma_60d"] = df_h["composite_score"].rolling(60, min_periods=20).mean()
    if "ind_qqq_deviation" in df_h.columns:
        df_h["ind_qqq_deviation_sma_20d"] = df_h["ind_qqq_deviation"].rolling(20, min_periods=10).mean()
    if "ind_vix_level" in df_h.columns:
        df_h["ind_vix_level_change_5d"] = df_h["ind_vix_level"].diff(5)
    if "ind_vix_level" in df_h.columns and "ind_credit_spread" in df_h.columns:
        df_h["vix_x_credit"] = df_h["ind_vix_level"] * df_h["ind_credit_spread"] / 100
    df_h["is_elevated"] = (df_h["composite_score"] > 60).astype(float)
    df_h["risk_composite_spread"] = df_h["drawdown_risk_score"] - df_h["composite_score"]

    df_q = pd.DataFrame(qqq)[["date", "price"]].copy()
    df_q.rename(columns={"price": "qqq_price"}, inplace=True)
    df = df_h.merge(df_q, on="date", how="inner").sort_values("date").reset_index(drop=True)

    qqq_ret = np.log(df["qqq_price"] / df["qqq_price"].shift(1))
    df["qqq_vol_20d"] = qqq_ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100
    df["qqq_momentum_60d"] = df["qqq_price"].pct_change(60) * 100

    prices = df["qqq_price"].values
    peak = prices[0]; in_dd = False; last_dd = 0
    bull_dur = np.zeros(len(prices))
    for i in range(len(prices)):
        if prices[i] > peak: peak = prices[i]
        dd = (peak - prices[i]) / peak
        if dd >= 0.10: in_dd = True; last_dd = i
        elif in_dd and dd < 0.05: in_dd = False; last_dd = i; peak = prices[i]
        bull_dur[i] = i - last_dd
    df["bull_duration"] = bull_dur

    for col in df.columns:
        if df[col].dtype in (np.float64, float):
            df[col] = df[col].fillna(df[col].median())
    return df


def compute_fwd_dd_fast(prices, window, definition="peak_to_trough"):
    """Forward drawdown computation."""
    n = len(prices)
    out = np.full(n, np.nan)
    for i in range(n - 10):
        end = min(i + window + 1, n)
        fwd = prices[i:end]
        if definition == "peak_to_trough":
            peak = fwd[0]; dd = 0.0
            for p in fwd[1:]:
                if p > peak: peak = p
                d = (peak - p) / peak
                if d > dd: dd = d
            out[i] = dd
        else:
            out[i] = max(0.0, (fwd[0] - np.min(fwd[1:])) / fwd[0])
    return out


def precompute_all_drawdowns(df):
    """Pre-compute all needed DD variants once."""
    prices_qqq = df["qqq_price"].values
    dd = {}
    configs = [
        ("qqq_p2t_180", prices_qqq, 180, "peak_to_trough"),
        ("qqq_p2t_252", prices_qqq, 252, "peak_to_trough"),
        ("qqq_dft_180", prices_qqq, 180, "drop_from_today"),
    ]
    for name, p, w, d in configs:
        t0 = time.time()
        dd[name] = compute_fwd_dd_fast(p, w, d)
        print(f"    {name}: {time.time()-t0:.1f}s")

    # Cross-asset: max(SPY, QQQ, IWM) p2t 180d
    t0 = time.time()
    dates = df["date"].tolist()
    asset_dds = []
    for ticker in ["spy", "qqq", "iwm"]:
        raw = json.loads((DATA_DIR / f"{ticker}.json").read_text())
        data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        price_map = {d["date"]: d["price"] for d in data}
        prices_aligned = np.array([price_map.get(d, np.nan) for d in dates])
        for j in range(1, len(prices_aligned)):
            if np.isnan(prices_aligned[j]):
                prices_aligned[j] = prices_aligned[j-1]
        asset_dd = compute_fwd_dd_fast(prices_aligned, 180, "peak_to_trough")
        asset_dds.append(asset_dd)
    stacked = np.array(asset_dds)
    dd["cross_p2t_180"] = np.nanmax(stacked, axis=0)
    print(f"    cross_p2t_180: {time.time()-t0:.1f}s")
    return dd


def purged_wf_splits(n, n_splits=N_CV_FOLDS, min_train_frac=MIN_TRAIN_FRAC,
                     purge=PURGE_DAYS, embargo=EMBARGO_DAYS):
    min_train = int(n * min_train_frac)
    remaining = n - min_train
    test_size = max(50, (remaining - purge * n_splits) // n_splits)
    splits = []
    for fold in range(n_splits):
        test_start = min_train + fold * test_size + purge
        test_end = min(test_start + test_size, n)
        if test_start >= n or test_end - test_start < 30: break
        train_end = test_start - purge
        if train_end < 100: continue
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def compute_ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if mask.sum() == 0: continue
        ece += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece / len(y_true))


def pava(values):
    """Pool Adjacent Violators — enforce non-decreasing."""
    n = len(values)
    blocks = [[i, i, values[i], 1] for i in range(n)]
    merged = True
    while merged:
        merged = False
        new_blocks = [blocks[0]]
        for i in range(1, len(blocks)):
            if new_blocks[-1][2] / new_blocks[-1][3] > blocks[i][2] / blocks[i][3]:
                new_blocks[-1][1] = blocks[i][1]
                new_blocks[-1][2] += blocks[i][2]
                new_blocks[-1][3] += blocks[i][3]
                merged = True
            else:
                new_blocks.append(blocks[i])
        blocks = new_blocks
    out = [0.0] * n
    for block in blocks:
        avg = block[2] / block[3]
        for j in range(block[0], block[1] + 1):
            out[j] = avg
    return out


# ═══════════════════════════════════════════════════════════════════
# Unified evaluation
# ═══════════════════════════════════════════════════════════════════

def run_logistic(X_all, y_all, penalty, C, label, skip_degen=False):
    splits = purged_wf_splits(len(X_all))
    all_yt, all_yp = [], []
    fold_res = []

    for fi, (tr, te) in enumerate(splits):
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]
        if int(y_tr.sum()) < 3 or int(y_tr.sum()) == len(y_tr): continue
        if len(np.unique(y_te)) < 2: continue
        br = float(y_te.mean())
        if skip_degen and (br > 0.70 or br < 0.03): continue

        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
        kw = {"max_iter": 10000, "C": C}
        if penalty == "l1": kw.update({"penalty": "l1", "solver": "saga"})
        else: kw.update({"penalty": "l2", "solver": "lbfgs"})
        model = LogisticRegression(**kw)
        model.fit(X_tr_s, y_tr)

        yp_raw = model.predict_proba(X_te_s)[:, 1]
        try:
            yp_tr = model.predict_proba(X_tr_s)[:, 1]
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(yp_tr, y_tr)
            yp = iso.predict(yp_raw)
        except: yp = yp_raw

        auc = roc_auc_score(y_te, yp)
        brier = brier_score_loss(y_te, yp)
        bc = br * (1 - br)
        bss = 1 - brier / max(bc, 1e-10)
        fold_res.append({"fold": fi+1, "n": len(y_te), "br": round(br, 3),
                         "auc": round(auc, 4), "bss": round(bss, 4)})
        all_yt.extend(y_te.tolist()); all_yp.extend(yp.tolist())

    if not fold_res:
        return {"label": label, "error": "No valid folds"}
    yt, yp = np.array(all_yt), np.array(all_yp)
    p_auc = roc_auc_score(yt, yp)
    p_brier = brier_score_loss(yt, yp)
    p_br = yt.mean()
    return {
        "label": label, "n_folds": len(fold_res), "folds": fold_res,
        "pooled_auc": round(float(p_auc), 4),
        "pooled_bss": round(float(1 - p_brier / max(p_br*(1-p_br), 1e-10)), 4),
        "pooled_ece": round(float(compute_ece(yt, yp)), 4),
        "pooled_n": len(yt),
        "auc_mean": round(float(np.mean([f["auc"] for f in fold_res])), 4),
        "bss_mean": round(float(np.mean([f["bss"] for f in fold_res])), 4),
    }


def run_binned(scores, y_all, label, n_bins=5, skip_degen=False):
    bin_edges = np.linspace(0, 100, n_bins + 1)
    splits = purged_wf_splits(len(scores))
    all_yt, all_yp = [], []
    fold_res = []

    for fi, (tr, te) in enumerate(splits):
        y_tr, y_te = y_all[tr], y_all[te]
        s_tr, s_te = scores[tr], scores[te]
        if int(y_tr.sum()) < 3 or int(y_tr.sum()) == len(y_tr): continue
        if len(np.unique(y_te)) < 2: continue
        br = float(y_te.mean())
        if skip_degen and (br > 0.70 or br < 0.03): continue

        bin_probs = []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b+1]
            mask = (s_tr >= lo) & (s_tr < hi) if b < n_bins-1 else (s_tr >= lo) & (s_tr <= hi)
            bin_probs.append(float(y_tr[mask].mean()) if mask.sum() > 0 else float(y_tr.mean()))
        bin_probs = pava(bin_probs)

        yp = np.zeros(len(y_te))
        for j in range(len(yp)):
            b = min(max(0, int(s_te[j] / 100 * n_bins)), n_bins-1)
            yp[j] = bin_probs[b]

        auc = roc_auc_score(y_te, yp) if len(np.unique(yp)) > 1 else 0.5
        brier = brier_score_loss(y_te, yp)
        bc = br * (1 - br)
        bss = 1 - brier / max(bc, 1e-10)
        fold_res.append({"fold": fi+1, "n": len(y_te), "br": round(br, 3),
                         "auc": round(auc, 4), "bss": round(bss, 4)})
        all_yt.extend(y_te.tolist()); all_yp.extend(yp.tolist())

    if not fold_res:
        return {"label": label, "error": "No valid folds"}
    yt, yp = np.array(all_yt), np.array(all_yp)
    p_auc = roc_auc_score(yt, yp) if len(np.unique(yp)) > 1 else 0.5
    p_brier = brier_score_loss(yt, yp)
    p_br = yt.mean()
    return {
        "label": label, "n_folds": len(fold_res), "folds": fold_res,
        "pooled_auc": round(float(p_auc), 4),
        "pooled_bss": round(float(1 - p_brier / max(p_br*(1-p_br), 1e-10)), 4),
        "pooled_ece": round(float(compute_ece(yt, yp)), 4),
        "pooled_n": len(yt),
        "auc_mean": round(float(np.mean([f["auc"] for f in fold_res])), 4),
        "bss_mean": round(float(np.mean([f["bss"] for f in fold_res])), 4),
    }


def run_chained(X_lo, y_lo, X_hi, y_hi, label, skip_degen=False):
    """P(20%) = sqrt(P_10% * P_20%)"""
    splits = purged_wf_splits(len(X_lo))
    all_yt, all_yp = [], []
    fold_res = []

    for fi, (tr, te) in enumerate(splits):
        y_tr_hi, y_te_hi = y_hi[tr], y_hi[te]
        if int(y_tr_hi.sum()) < 3 or int(y_tr_hi.sum()) == len(y_tr_hi): continue
        if len(np.unique(y_te_hi)) < 2: continue
        br = float(y_te_hi.mean())
        if skip_degen and (br > 0.70 or br < 0.03): continue

        sc1 = StandardScaler()
        m1 = LogisticRegression(C=5.0, penalty="l1", solver="saga", max_iter=10000)
        m1.fit(sc1.fit_transform(X_lo[tr]), y_lo[tr])
        p_lo = m1.predict_proba(sc1.transform(X_lo[te]))[:, 1]

        sc2 = StandardScaler()
        m2 = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=10000)
        m2.fit(sc2.fit_transform(X_hi[tr]), y_tr_hi)
        p_hi = m2.predict_proba(sc2.transform(X_hi[te]))[:, 1]

        yp = np.clip(np.sqrt(p_lo * p_hi), 0.01, 0.99)
        auc = roc_auc_score(y_te_hi, yp)
        brier = brier_score_loss(y_te_hi, yp)
        bc = br * (1 - br)
        bss = 1 - brier / max(bc, 1e-10)
        fold_res.append({"fold": fi+1, "n": len(y_te_hi), "br": round(br, 3),
                         "auc": round(auc, 4), "bss": round(bss, 4)})
        all_yt.extend(y_te_hi.tolist()); all_yp.extend(yp.tolist())

    if not fold_res:
        return {"label": label, "error": "No valid folds"}
    yt, yp = np.array(all_yt), np.array(all_yp)
    p_auc = roc_auc_score(yt, yp)
    p_brier = brier_score_loss(yt, yp)
    p_br = yt.mean()
    return {
        "label": label, "n_folds": len(fold_res), "folds": fold_res,
        "pooled_auc": round(float(p_auc), 4),
        "pooled_bss": round(float(1 - p_brier / max(p_br*(1-p_br), 1e-10)), 4),
        "pooled_ece": round(float(compute_ece(yt, yp)), 4),
        "pooled_n": len(yt),
        "auc_mean": round(float(np.mean([f["auc"] for f in fold_res])), 4),
        "bss_mean": round(float(np.mean([f["bss"] for f in fold_res])), 4),
    }


def print_result(r):
    if "error" in r:
        print(f"    {r['label']}: ERROR - {r['error']}")
        return
    print(f"    {r['label']}")
    print(f"      Folds={r['n_folds']}  N={r['pooled_n']}  "
          f"AUC={r['pooled_auc']:.4f}  BSS={r['pooled_bss']:+.1%}  ECE={r['pooled_ece']:.4f}")
    for fd in r["folds"]:
        print(f"        F{fd['fold']}: AUC={fd['auc']:.4f}  BSS={fd['bss']:+.1%}  "
              f"br={fd['br']:.1%}  n={fd['n']}")


def main():
    print("=" * 72)
    print("  20% DD OPTIMIZATION — Exhaustive OOS A/B (v2)")
    print("  purge=180d | embargo=20d | 5-fold walk-forward")
    print("=" * 72)

    df = load_data()
    print(f"\n  {len(df)} days, {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    print("\n  Pre-computing forward drawdowns...")
    dd = precompute_all_drawdowns(df)

    # Feature arrays
    feat_cols = {
        "base2": ["risk_ema_20d", "score_std_20d"],
        "base8": ["risk_ema_20d", "ind_qqq_deviation", "ind_credit_spread",
                   "ind_yield_curve", "ind_vix_level", "vix_x_credit",
                   "ind_vix_level_change_5d", "score_velocity"],
        "risk1": ["risk_ema_20d"],
        "risk_score1": ["drawdown_risk_score"],
        "ema1": ["score_ema_20d"],
        "vix1": ["ind_vix_level"],
        "vol1": ["qqq_vol_20d"],
        "risk_vol": ["risk_ema_20d", "qqq_vol_20d"],
        "risk_vol_std": ["risk_ema_20d", "score_std_20d", "qqq_vol_20d"],
        "risk_bull": ["risk_ema_20d", "bull_duration"],
    }

    results = []

    def add(r):
        print_result(r)
        results.append(r)

    # ── Prepare label arrays for main QQQ P2T 20% ──
    dd_key = "qqq_p2t_180"
    fwd = dd[dd_key]
    valid_mask = np.isfinite(fwd)
    for fc in set(sum(feat_cols.values(), [])):
        if fc in df.columns:
            valid_mask &= df[fc].notna().values
    vi = np.where(valid_mask)[0]
    y20 = (fwd[vi] >= 0.20).astype(float)
    y15 = (fwd[vi] >= 0.15).astype(float)
    y10 = (fwd[vi] >= 0.10).astype(float)

    def get_X(key):
        feats = feat_cols[key] if isinstance(key, str) and key in feat_cols else key
        return df.iloc[vi][[f for f in feats if f in df.columns]].values.astype(float)

    print(f"\n  Label stats (QQQ P2T 180d): "
          f"10%DD={y10.mean():.1%}, 15%DD={y15.mean():.1%}, 20%DD={y20.mean():.1%}")

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  A. BASELINE (production v3.3)")
    print(f"{'─'*72}")
    add(run_logistic(get_X("base2"), y20, "l2", 1.0, "A0: Baseline v3.3 (L2 C=1)"))
    add(run_logistic(get_X("base2"), y20, "l2", 1.0, "A0b: Baseline skip-degen", skip_degen=True))

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  B. UNIVARIATE MODELS")
    print(f"{'─'*72}")
    for name in ["risk1", "risk_score1", "ema1", "vix1", "vol1"]:
        add(run_logistic(get_X(feat_cols[name]), y20, "l2", 1.0,
                         f"B: Univar {feat_cols[name][0]}"))

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  C. REGULARIZATION SWEEP (base2 features)")
    print(f"{'─'*72}")
    for C in [0.1, 0.01, 0.001]:
        add(run_logistic(get_X("base2"), y20, "l2", C, f"C: L2 C={C}"))

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  D. ALTERNATIVE DD DEFINITIONS")
    print(f"{'─'*72}")
    # Drop-from-today 20%
    fwd_dft = dd["qqq_dft_180"]
    vi_dft = np.where(np.isfinite(fwd_dft) & df["risk_ema_20d"].notna().values
                      & df["score_std_20d"].notna().values)[0]
    y20_dft = (fwd_dft[vi_dft] >= 0.20).astype(float)
    X_dft = df.iloc[vi_dft][["risk_ema_20d", "score_std_20d"]].values.astype(float)
    print(f"    Drop-from-today 20%: base_rate={y20_dft.mean():.1%}")
    add(run_logistic(X_dft, y20_dft, "l2", 1.0, "D: Drop-from-today 20%"))

    # Relaxed 15% threshold
    add(run_logistic(get_X("base2"), y15, "l2", 1.0, "D: Relaxed 15% DD"))

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  E. LONGER WINDOWS")
    print(f"{'─'*72}")
    fwd_252 = dd["qqq_p2t_252"]
    vi_252 = np.where(np.isfinite(fwd_252) & df["risk_ema_20d"].notna().values
                      & df["score_std_20d"].notna().values)[0]
    y20_252 = (fwd_252[vi_252] >= 0.20).astype(float)
    X_252 = df.iloc[vi_252][["risk_ema_20d", "score_std_20d"]].values.astype(float)
    print(f"    P2T 252d: base_rate={y20_252.mean():.1%}")
    add(run_logistic(X_252, y20_252, "l2", 1.0, "E: P2T 252d window"))
    add(run_logistic(X_252, y20_252, "l2", 0.1, "E: P2T 252d C=0.1"))

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  F. NON-PARAMETRIC BINNED (no model fitting)")
    print(f"{'─'*72}")
    for score_col, sc_name in [("drawdown_risk_score", "risk"), ("risk_ema_20d", "risk_ema"),
                                ("composite_score", "composite")]:
        scores_v = df.iloc[vi][score_col].values
        for nb in [5, 10]:
            add(run_binned(scores_v, y20, f"F: Binned({sc_name},{nb}bins)", nb))

    # skip-degen binned
    scores_risk = df.iloc[vi]["drawdown_risk_score"].values
    add(run_binned(scores_risk, y20, "F: Binned(risk,5) skip-degen", 5, True))

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  G. CROSS-ASSET POOLED LABEL")
    print(f"{'─'*72}")
    fwd_cross = dd["cross_p2t_180"]
    vi_cross = np.where(np.isfinite(fwd_cross) & df["risk_ema_20d"].notna().values
                        & df["score_std_20d"].notna().values)[0]
    y20_cross = (fwd_cross[vi_cross] >= 0.20).astype(float)
    X_cross2 = df.iloc[vi_cross][["risk_ema_20d", "score_std_20d"]].values.astype(float)
    X_cross1 = df.iloc[vi_cross][["risk_ema_20d"]].values.astype(float)
    print(f"    Cross-asset 20%: base_rate={y20_cross.mean():.1%}")
    add(run_logistic(X_cross2, y20_cross, "l2", 1.0, "G: Cross-asset (base2)"))
    add(run_logistic(X_cross1, y20_cross, "l2", 1.0, "G: Cross-asset univar"))
    add(run_logistic(X_cross2, y20_cross, "l2", 0.1, "G: Cross-asset C=0.1"))

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  H. CHAINED (P10% → P20%)")
    print(f"{'─'*72}")
    X_lo = get_X("base8"); X_hi = get_X("base2")
    add(run_chained(X_lo, y10, X_hi, y20, "H: Chained 10%→20%"))
    add(run_chained(X_lo, y10, X_hi, y20, "H: Chained skip-degen", skip_degen=True))

    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  I. FEATURE COMBOS + SKIP DEGEN")
    print(f"{'─'*72}")
    for name, pen, C, lab in [
        ("risk1", "l2", 0.1, "I: risk_ema C=0.1"),
        ("risk_vol", "l2", 0.1, "I: risk+vol C=0.1"),
        ("risk_vol_std", "l2", 0.1, "I: risk+vol+std C=0.1"),
        ("risk_bull", "l2", 0.1, "I: risk+bull C=0.1"),
        ("risk_score1", "l2", 0.1, "I: risk_score C=0.1"),
    ]:
        add(run_logistic(get_X(feat_cols[name]), y20, pen, C, lab + " skip-degen",
                         skip_degen=True))

    # Longer window + skip degen
    add(run_logistic(X_252, y20_252, "l2", 0.1, "I: 252d C=0.1 skip-degen",
                     skip_degen=True))

    # ═════════════════════════════════════════════════════════════
    # LEADERBOARD
    # ═════════════════════════════════════════════════════════════
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: x["pooled_auc"], reverse=True)

    print(f"\n\n{'='*72}")
    print("  LEADERBOARD: 20% DD Prediction (sorted by pooled AUC)")
    print(f"{'='*72}\n")
    print(f"  {'#':>2} {'Model':<45} {'AUC':>7} {'BSS':>9} {'ECE':>7} {'N':>5} {'F':>2}")
    print(f"  {'─'*78}")

    for rank, r in enumerate(valid, 1):
        marker = " ★" if rank <= 3 else ""
        tag = ">>>" if "Baseline v3.3" in r["label"] and "skip" not in r["label"] else "   "
        print(f"  {tag}{rank:>1} {r['label']:<45} {r['pooled_auc']:>7.4f} "
              f"{r['pooled_bss']:>+8.1%} {r['pooled_ece']:>7.4f} "
              f"{r['pooled_n']:>5} {r['n_folds']:>2}{marker}")

    baseline = next((r for r in results if "Baseline v3.3" in r["label"]
                     and "skip" not in r["label"]), None)
    b_auc = baseline["pooled_auc"] if baseline and "error" not in baseline else 0
    best = valid[0]
    print(f"\n  Baseline AUC: {b_auc:.4f}")
    print(f"  Best:         {best['label']} (AUC={best['pooled_auc']:.4f})")
    print(f"  ΔAUC:         {best['pooled_auc'] - b_auc:+.4f}")

    # Diagnosis
    print(f"\n{'='*72}")
    print("  DIAGNOSIS: Why 20% DD is fundamentally harder")
    print(f"{'='*72}")
    print(f"  - 20% DD events in data: {y20.sum():.0f}/{len(y20)} ({y20.mean():.1%})")
    print(f"  - 10% DD events in data: {y10.sum():.0f}/{len(y10)} ({y10.mean():.1%})")
    dd_20_dates = vi[y20 == 1]
    episodes = 1
    for i in range(1, len(dd_20_dates)):
        if dd_20_dates[i] - dd_20_dates[i-1] > 120:
            episodes += 1
    print(f"  - Independent 20% DD episodes (gap>120d): ~{episodes}")
    print(f"  - Fold 1 base_rate: ~79% (COVID dominates → degenerate)")
    print(f"  - Effective p/n ratio: {episodes} events / {len(y20)} obs = disastrous")
    print(f"\n{'='*72}\n")

    out_path = Path(__file__).resolve().parent / "ab_test_20pct_results.json"
    output = {r["label"]: {k: v for k, v in r.items() if k != "folds"} for r in results}
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
