#!/usr/bin/env python3
"""Sweep Risk threshold x Delta to find optimal CC parameters."""

import json, numpy as np, math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, S - K)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm_cdf(d1) - K*math.exp(-r*T)*norm_cdf(d2)

def bs_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm_cdf(d1)

def find_strike(S, T, r, sigma, target_delta):
    lo, hi = S*0.8, S*3.0
    for _ in range(80):
        mid = (lo+hi)/2
        d = bs_delta(S, mid, T, r, sigma)
        if d > target_delta: lo = mid
        else: hi = mid
    return round((lo+hi)/2, 2)

def compute_vol(prices, window=20):
    lr = np.log(np.array(prices[1:])/np.array(prices[:-1]))
    vol = np.full(len(prices), np.nan)
    for i in range(window, len(lr)):
        vol[i+1] = float(np.std(lr[i-window+1:i+1])*np.sqrt(252))
    fv = window+1
    if fv < len(vol): vol[:fv] = vol[fv]
    return vol

# Load data
history = json.loads((DATA_DIR/'bubble_history.json').read_text())['history']
tqqq_raw = json.loads((DATA_DIR/'tqqq.json').read_text())
tqqq_data = tqqq_raw['data'] if isinstance(tqqq_raw, dict) else tqqq_raw
risk_map = {h['date']: h['drawdown_risk_score'] for h in history}
records = []
for d in tqqq_data:
    if d['date'] in risk_map:
        records.append({'date': d['date'], 'price': d['price'], 'risk': risk_map[d['date']]})
records.sort(key=lambda x: x['date'])

prices = [r['price'] for r in records]
risks = np.array([r['risk'] for r in records])
vol = compute_vol(prices, 20)
n = len(records)
start = 30
years = (np.datetime64(records[-1]['date']) - np.datetime64(records[0]['date'])).astype(int)/365.25
start_price = prices[start]
init_shares = max(100, int(10000/start_price/100)*100)
init_cash = max(0, 10000 - init_shares*start_price)
init_val = init_shares*start_price + init_cash
bh_shares = init_val / start_price

# Risk distribution
print("=== Risk Score 分布 ===")
for pct in [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90]:
    val = np.percentile(risks, pct)
    days_above = np.sum(risks > val)
    print(f"  P{pct:>2} = {val:.1f}  (>{val:.0f}: {days_above} days, {days_above/len(risks)*100:.1f}%)")

print(f"\nThreshold sweep: {init_shares} shares, init=${init_val:,.0f}, {years:.1f}y")
print()
print(f"{'Thresh':>6} {'Delta':>5} {'AnnRet':>7} {'BH_Ann':>7} {'Alpha':>7} {'MaxDD':>7} "
      f"{'Trades':>6} {'WR':>5} {'Asgn':>5} {'CCpnl':>10}")
print('-'*80)

results = []
best_alpha = -999
best_combo = ''

for thresh in range(60, 76):
    for delta in [0.15, 0.20, 0.25, 0.30]:
        shares = init_shares
        cash = init_cash
        active = None
        last_open = -999
        cc_pnl = 0
        trades_n = 0
        wins = 0
        assigned = 0
        pv_list = []

        for i in range(start, n):
            S = prices[i]
            risk = risks[i]
            sigma = vol[i] if not np.isnan(vol[i]) else 0.60
            sigma = max(sigma, 0.30)

            # Manage active CC
            if active is not None:
                oi, K, prem, dte, cov = active
                days_in = i - oi
                T_rem = max((dte - days_in)/252, 1/252)
                cv = bs_price(S, K, T_rem, 0.045, sigma)
                unreal = prem - cv
                close = None
                if unreal >= prem*0.50: close = 'pt'
                elif dte - days_in <= 14: close = 'roll'
                elif S > K*1.15: close = 'asgn'

                if close:
                    if close == 'asgn':
                        pnl = prem*cov - (S-K)*cov
                        assigned += 1
                    else:
                        pnl = (prem - cv)*cov
                    cash += pnl
                    cc_pnl += pnl
                    trades_n += 1
                    if pnl > 0: wins += 1
                    active = None

            # Open new CC
            if active is None and i - last_open >= 5:
                if risk > thresh and shares >= 100:
                    T = 35/252
                    K = find_strike(S, T, 0.045, sigma, delta)
                    prem = bs_price(S, K, T, 0.045, sigma)
                    if prem > 0.01:
                        cov = (shares//100)*100
                        active = (i, K, prem, 35, cov)
                        last_open = i

            # Portfolio value
            sv = shares * S
            cl = 0
            if active:
                oi, K, prem, dte, cov = active
                days_in = i - oi
                T_rem = max((dte-days_in)/252, 1/252)
                cl = bs_price(S, K, T_rem, 0.045, sigma)*cov
            pv_list.append(sv + cash - cl)

        pv = np.array(pv_list)
        fv = pv[-1]
        bh_final = bh_shares * prices[n-1]
        ann = ((fv/init_val)**(1/years)-1)*100
        bh_ann = ((bh_final/init_val)**(1/years)-1)*100
        alpha = ann - bh_ann

        peak = np.maximum.accumulate(pv)
        dd = ((pv-peak)/peak).min()*100

        wr = wins/trades_n*100 if trades_n > 0 else 0

        results.append({
            'thresh': thresh, 'delta': delta, 'ann': ann, 'alpha': alpha,
            'dd': dd, 'trades': trades_n, 'wr': wr, 'asgn': assigned, 'ccpnl': cc_pnl
        })

        if alpha > best_alpha:
            best_alpha = alpha
            best_combo = f'thresh={thresh} delta={delta}'

        print(f'{thresh:>6} {delta:>5.2f} {ann:>6.1f}% {bh_ann:>6.1f}% {alpha:>+6.2f}% '
              f'{dd:>6.1f}% {trades_n:>6} {wr:>4.0f}% {assigned:>5} ${cc_pnl:>9,.0f}')

# Summary: best per delta
print(f"\n{'='*60}")
print("BEST THRESHOLD PER DELTA")
print(f"{'='*60}")
for d in [0.15, 0.20, 0.25, 0.30]:
    subset = [r for r in results if r['delta'] == d]
    best = max(subset, key=lambda x: x['alpha'])
    print(f"  delta={d:.2f}: thresh={best['thresh']}  alpha={best['alpha']:+.2f}%  "
          f"ann={best['ann']:.1f}%  maxdd={best['dd']:.1f}%  trades={best['trades']}  wr={best['wr']:.0f}%")

print(f"\nOVERALL BEST: {best_combo}  alpha={best_alpha:+.2f}%")

# Also show: top 10 by alpha
print(f"\nTOP 10 BY ALPHA:")
results.sort(key=lambda x: x['alpha'], reverse=True)
for i, r in enumerate(results[:10]):
    print(f"  {i+1}. thresh={r['thresh']} delta={r['delta']:.2f}  "
          f"alpha={r['alpha']:+.2f}%  ann={r['ann']:.1f}%  dd={r['dd']:.1f}%  "
          f"trades={r['trades']}  wr={r['wr']:.0f}%  asgn={r['asgn']}")
