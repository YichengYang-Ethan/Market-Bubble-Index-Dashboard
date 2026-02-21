#!/usr/bin/env python3
"""Analyze Risk>65 sell-call signal quality and optimal parameters."""

import json, numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

# Load V1 results
v1 = json.loads(Path(Path(__file__).resolve().parent / "options_backtest_results.json").read_text())
print("=== V1 Best Configs (all call-only) ===")
# V1 is a dict keyed by label
v1_list = [{"label": k, **v} for k, v in v1.items() if isinstance(v, dict) and "ann_return" in v]
for r in sorted(v1_list, key=lambda x: x.get("ann_return", 0), reverse=True)[:5]:
    print(f"  {r['label']:<40} ann={r['ann_return']:>6.1f}%  sharpe={r['sharpe']:>5.2f}  "
          f"maxdd={r['max_dd_pct']:>6.1f}%  wr={r['win_rate']:>4.0f}%  pf={r['profit_factor']:>4.1f}  trades={r['trades']}")

# Load data
history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
tqqq_raw = json.loads((DATA_DIR / "tqqq.json").read_text())
tqqq = tqqq_raw["data"] if isinstance(tqqq_raw, dict) else tqqq_raw

risk_map = {h["date"]: h.get("drawdown_risk_score", 50) for h in history}
tqqq_map = {t["date"]: t["price"] for t in tqqq}

dates = sorted(set(risk_map.keys()) & set(tqqq_map.keys()))
data = [(d, risk_map[d], tqqq_map[d]) for d in dates]

# Forward returns after Risk>65
print("\n=== Risk>65 信号后 TQQQ 表现 ===")
fwd_periods = [5, 10, 20, 40]
fwd_returns = {p: [] for p in fwd_periods}
for i, (d, risk, price) in enumerate(data):
    if risk > 65:
        for fwd in fwd_periods:
            if i + fwd < len(data):
                ret = (data[i + fwd][2] - price) / price * 100
                fwd_returns[fwd].append(ret)

for fwd, rets in fwd_returns.items():
    rets = np.array(rets)
    print(f"  {fwd:>2}d fwd: mean={np.mean(rets):>+6.2f}%  median={np.median(rets):>+6.2f}%  "
          f"neg={sum(r < 0 for r in rets) / len(rets) * 100:.0f}%  worst={np.min(rets):>+7.2f}%  n={len(rets)}")

# Risk>65 distribution
high_risks = [risk for d, risk, p in data if risk > 65]
print(f"\n=== Risk>65 分布 ===")
print(f"  天数: {len(high_risks)} / {len(data)} ({len(high_risks) / len(data) * 100:.1f}%)")
print(f"  均值: {np.mean(high_risks):.1f}")
print(f"  P25={np.percentile(high_risks, 25):.1f}  P50={np.percentile(high_risks, 50):.1f}  "
      f"P75={np.percentile(high_risks, 75):.1f}  Max={np.max(high_risks):.1f}")

# Signal clustering: distinct episodes
print(f"\n=== 信号触发时段 (Risk>65 连续段) ===")
episodes = []
current_ep = []
for i, (d, risk, p) in enumerate(data):
    if risk > 65:
        if not current_ep or (i - current_ep[-1][0]) <= 5:
            current_ep.append((i, d, risk, p))
        else:
            episodes.append(current_ep)
            current_ep = [(i, d, risk, p)]
if current_ep:
    episodes.append(current_ep)

print(f"  共 {len(episodes)} 个独立信号段\n")
print(f"  {'Start':<12} {'End':<12} {'Days':>5} {'AvgRisk':>8} {'TQQQ Start':>11} {'TQQQ End':>9} {'Return':>8}")
print(f"  {'─' * 70}")
for ep in episodes:
    start_d, end_d = ep[0][1], ep[-1][1]
    days = len(ep)
    avg_risk = np.mean([e[2] for e in ep])
    start_p, end_p = ep[0][3], ep[-1][3]
    ret = (end_p - start_p) / start_p * 100
    # Also check max drawdown during episode
    prices = [e[3] for e in ep]
    peak = prices[0]
    max_dd = 0
    for p in prices:
        if p > peak:
            peak = p
        dd = (p - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    print(f"  {start_d:<12} {end_d:<12} {days:>5} {avg_risk:>7.0f} ${start_p:>9.2f} ${end_p:>7.2f} {ret:>+7.1f}%")

# Analyze optimal DTE: what's the ideal holding period?
print(f"\n=== 最佳持仓天数分析 ===")
# For each Risk>65 entry, find max decline in next N days
for hold in [14, 21, 28, 35, 45]:
    declines = []
    for i, (d, risk, price) in enumerate(data):
        if risk > 65:
            max_decline = 0
            for j in range(1, hold + 1):
                if i + j < len(data):
                    chg = (data[i + j][2] - price) / price * 100
                    if chg < max_decline:
                        max_decline = chg
            declines.append(max_decline)
    d = np.array(declines)
    print(f"  {hold:>2}d hold: avg max decline={np.mean(d):>+6.2f}%  "
          f"median={np.median(d):>+6.2f}%  <-10%={sum(d < -10) / len(d) * 100:.0f}%  "
          f"<-20%={sum(d < -20) / len(d) * 100:.0f}%")

# Current status
latest = data[-1]
print(f"\n=== 当前状态 ({latest[0]}) ===")
print(f"  Risk Score: {latest[1]:.1f}")
print(f"  TQQQ: ${latest[2]:.2f}")
print(f"  信号: {'SELL CALL ✓' if latest[1] > 65 else '观望 (Risk<65)'}")
