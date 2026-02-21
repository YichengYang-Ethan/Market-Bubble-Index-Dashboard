#!/usr/bin/env python3
"""Validate whether Risk Score / Composite Score is a leading indicator for drawdowns."""
import json
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "public" / "data"

h = json.load(open(DATA / "bubble_history.json"))["history"]
qqq = json.load(open(DATA / "qqq.json"))
prices = {d["date"]: d["price"] for d in qqq["data"]}

risk_scores = [pt.get("drawdown_risk_score") for pt in h]
comp_scores = [pt.get("composite_score") for pt in h]
dates = [pt["date"] for pt in h]

bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

print("=" * 70)
print("MONOTONICITY TEST: Score bin → Forward 180-day Peak-to-Trough Drawdown")
print("=" * 70)

for label, scores in [("Drawdown Risk Score", risk_scores), ("Composite Score", comp_scores)]:
    print(f"\n--- {label} ---")
    bin_probs = []
    for lo, hi in bins:
        fwd_dd = []
        for i, s in enumerate(scores):
            if s is None or s < lo or s >= hi:
                continue
            d = dates[i]
            if d not in prices:
                continue
            # Forward 180 trading days peak-to-trough
            peak = prices[d]
            max_dd = 0
            count = 0
            for j in range(i + 1, min(i + 181, len(h))):
                dj = dates[j]
                if dj in prices:
                    pj = prices[dj]
                    if pj > peak:
                        peak = pj
                    dd = (peak - pj) / peak
                    if dd > max_dd:
                        max_dd = dd
                    count += 1
            if count > 100:
                fwd_dd.append(max_dd)
        if fwd_dd:
            arr = np.array(fwd_dd)
            p10 = float((arr >= 0.10).mean())
            p20 = float((arr >= 0.20).mean())
            mean_dd = float(arr.mean())
            bin_probs.append(p20)
            print(f"  [{lo:2d}-{hi:3d}] n={len(fwd_dd):5d}  P(>10%)={p10:.1%}  P(>20%)={p20:.1%}  mean_dd={mean_dd:.1%}")
        else:
            bin_probs.append(None)
            print(f"  [{lo:2d}-{hi:3d}] n=    0")

    # Check monotonicity
    valid = [x for x in bin_probs if x is not None]
    if len(valid) >= 3:
        diffs = [valid[i+1] - valid[i] for i in range(len(valid)-1)]
        mono = sum(1 for d in diffs if d > 0) / len(diffs)
        print(f"  Monotonicity (P(>20%)): {mono:.0%} of adjacent bins are increasing")

print()
print("=" * 70)
print("LEAD-LAG TEST: Risk Score peaks vs actual drawdown troughs")
print("=" * 70)

# Find risk score peaks (>75) and see if drawdowns follow
high_risk_episodes = []
in_episode = False
episode_start = None
for i, s in enumerate(risk_scores):
    if s is None:
        continue
    if s >= 75 and not in_episode:
        in_episode = True
        episode_start = i
    elif s < 60 and in_episode:
        in_episode = False
        high_risk_episodes.append((episode_start, i))

print(f"\nHigh-risk episodes (Risk Score >= 75):")
for start_i, end_i in high_risk_episodes:
    start_date = dates[start_i]
    end_date = dates[end_i]
    peak_risk = max(risk_scores[start_i:end_i+1])

    # Check what happened to QQQ in the next 180 days after episode START
    if start_date in prices:
        p0 = prices[start_date]
        peak_p = p0
        max_dd = 0
        dd_date = start_date
        for j in range(start_i, min(start_i + 250, len(h))):
            dj = dates[j]
            if dj in prices:
                pj = prices[dj]
                if pj > peak_p:
                    peak_p = pj
                dd = (peak_p - pj) / peak_p
                if dd > max_dd:
                    max_dd = dd
                    dd_date = dj
        print(f"  {start_date} to {end_date}: peak_risk={peak_risk:.1f}  "
              f"max_dd_within_1yr={max_dd:.1%} (trough={dd_date})")

print()
print("=" * 70)
print("SIGNAL QUALITY: SPY forward returns after Composite score thresholds")
print("=" * 70)

spy = json.load(open(DATA / "spy.json"))
spy_prices = {d["date"]: d["price"] for d in spy["data"]}

for threshold_label, threshold_fn in [
    ("Composite < 30 (Buy)", lambda s: s < 30),
    ("Composite > 70 (Sell)", lambda s: s > 70),
    ("Risk Score > 70 (Danger)", lambda s: s > 70),
    ("Risk Score < 30 (Safe)", lambda s: s < 30),
]:
    use_risk = "Risk" in threshold_label
    scores_to_use = risk_scores if use_risk else comp_scores

    fwd_returns = {20: [], 60: [], 126: []}
    for i, s in enumerate(scores_to_use):
        if s is None or not threshold_fn(s):
            continue
        d = dates[i]
        if d not in spy_prices:
            continue
        p0 = spy_prices[d]
        for hz in fwd_returns:
            if i + hz < len(h):
                dj = dates[i + hz]
                if dj in spy_prices:
                    ret = (spy_prices[dj] - p0) / p0 * 100
                    fwd_returns[hz].append(ret)

    print(f"\n  {threshold_label} (n signals):")
    for hz, rets in fwd_returns.items():
        if rets:
            arr = np.array(rets)
            hit = float((arr > 0).mean()) * 100
            print(f"    {hz:3d}d: n={len(rets):4d}  mean={arr.mean():+.2f}%  "
                  f"median={np.median(arr):+.2f}%  hit_rate={hit:.0f}%  "
                  f"sharpe={arr.mean()/max(arr.std(),0.01):.2f}")

print()
print("=" * 70)
print("GRANGER-LIKE LEAD TEST: Does Risk Score lead drawdown events?")
print("=" * 70)

# Compute rolling drawdown
qqq_dates = sorted(prices.keys())
qqq_peak = {}
qqq_dd = {}
peak = 0
for d in qqq_dates:
    p = prices[d]
    if p > peak:
        peak = p
    qqq_dd[d] = (p - peak) / peak * 100  # negative percentage
    qqq_peak[d] = peak

# Cross-correlation: risk_score[t] vs drawdown[t+lag]
for lag in [0, 20, 40, 60, 90, 120]:
    pairs = []
    for i, s in enumerate(risk_scores):
        if s is None:
            continue
        d = dates[i]
        j = i + lag
        if j < len(dates):
            dj = dates[j]
            if dj in qqq_dd:
                pairs.append((s, -qqq_dd[dj]))  # flip sign: positive = deeper drawdown
    if len(pairs) > 50:
        x, y = zip(*pairs)
        corr = np.corrcoef(x, y)[0, 1]
        print(f"  lag={lag:3d}d: corr(RiskScore[t], |Drawdown|[t+lag]) = {corr:+.4f}  (n={len(pairs)})")
