#!/usr/bin/env python3
"""
Covered Call Backtest — TQQQ holder + Risk Score signal
========================================================

Scenario: You HOLD TQQQ shares throughout.
When Risk Score > threshold → sell covered calls to collect premium.
When Risk Score < threshold → just hold shares (no CC).

This measures:
  1. Buy & Hold TQQQ alone (baseline)
  2. Buy & Hold + always sell CC (every month)
  3. Buy & Hold + sell CC only when Risk > 65 (signal-guided)

Key differences from naked selling:
  - No margin needed (shares are collateral)
  - "Loss" = opportunity cost (miss upside above strike when called away)
  - Premium = pure income on top of share gains
  - If assigned → sell shares at strike → buy back next day (simulated)
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes
# ═══════════════════════════════════════════════════════════════════

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, r, sigma):
    """Call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_delta(S, K, T, r, sigma):
    """Call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def find_call_strike_by_delta(S, T, r, sigma, target_delta):
    """Find OTM call strike with given delta. Higher strike = lower delta."""
    lo, hi = S * 0.8, S * 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma)
        if d > target_delta:
            lo = mid   # strike too low, delta too high → raise strike
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CCTrade:
    open_date: str
    open_idx: int
    strike: float
    premium: float            # per share
    dte: int
    underlying_at_open: float
    risk_at_open: float
    iv_at_open: float
    shares_covered: int       # number of shares covered (multiples of 100)

    close_date: Optional[str] = None
    close_idx: Optional[int] = None
    close_reason: Optional[str] = None
    pnl: Optional[float] = None           # total call P&L
    days_held: Optional[int] = None
    assigned: bool = False


@dataclass
class CCConfig:
    # Signal
    risk_threshold: float = 65.0      # sell CC when risk > this
    always_sell: bool = False          # True = ignore signal, sell every month

    # Options parameters
    target_delta: float = 0.20
    target_dte: int = 35
    roll_dte: int = 14
    profit_target_pct: float = 0.50   # buy back at 50% profit
    assignment_threshold: float = 1.15  # close if price > strike * 1.15

    # Position
    initial_shares: int = 100         # start with 100 shares
    initial_capital: float = 10_000.0 # additional cash for buying back etc.
    reinvest_premium: bool = True     # use premium to buy more shares?

    # Risk-free rate
    risk_free_rate: float = 0.045


def load_data():
    history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
    tqqq_raw = json.loads((DATA_DIR / "tqqq.json").read_text())
    tqqq_data = tqqq_raw["data"] if isinstance(tqqq_raw, dict) else tqqq_raw

    risk_map = {h["date"]: h["drawdown_risk_score"] for h in history}
    records = []
    for d in tqqq_data:
        date = d["date"]
        if date in risk_map:
            records.append({
                "date": date,
                "price": d["price"],
                "risk_score": risk_map[date],
            })
    records.sort(key=lambda x: x["date"])
    return records


def compute_historical_vol(prices, window=20):
    log_ret = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
    vol = np.full(len(prices), np.nan)
    for i in range(window, len(log_ret)):
        vol[i + 1] = float(np.std(log_ret[i - window + 1:i + 1]) * np.sqrt(252))
    first_valid = window + 1
    if first_valid < len(vol):
        vol[:first_valid] = vol[first_valid]
    return vol


# ═══════════════════════════════════════════════════════════════════
# Covered Call Backtest
# ═══════════════════════════════════════════════════════════════════

def run_cc_backtest(data, config: CCConfig):
    """
    Simulate: hold TQQQ shares + sell covered calls.

    Returns:
        trades: list of CCTrade
        portfolio_values: np.array of daily portfolio value
        bh_values: np.array of buy-and-hold comparison
    """
    prices = [d["price"] for d in data]
    risk_scores = np.array([d["risk_score"] for d in data])
    vol = compute_historical_vol(prices, 20)

    n = len(data)
    start_idx = 30  # skip initial period

    # Portfolio state
    shares = config.initial_shares
    cash = config.initial_capital
    initial_value = shares * prices[start_idx] + cash

    # Buy-and-hold baseline: same initial value, all in shares
    bh_shares = initial_value / prices[start_idx]

    active_cc: Optional[CCTrade] = None
    closed_trades: list[CCTrade] = []
    last_open_idx = -999

    portfolio_values = []
    bh_values = []
    cc_premium_total = 0.0

    for i in range(start_idx, n):
        d = data[i]
        S = d["price"]
        risk = risk_scores[i]
        sigma = vol[i] if not np.isnan(vol[i]) else 0.60
        sigma = max(sigma, 0.30)

        # ── Manage active covered call ──
        if active_cc is not None:
            days_in = i - active_cc.open_idx
            T_rem = max((active_cc.dte - days_in) / 252, 1 / 252)

            call_value = bs_price(S, active_cc.strike, T_rem,
                                   config.risk_free_rate, sigma)

            unrealized_profit = active_cc.premium - call_value
            close_reason = None

            # Profit target: buy back cheap
            if unrealized_profit >= active_cc.premium * config.profit_target_pct:
                close_reason = "profit_target"

            # Roll: approaching expiry
            elif active_cc.dte - days_in <= config.roll_dte:
                close_reason = "roll_expiry"

            # Assignment risk: price way above strike
            elif S > active_cc.strike * config.assignment_threshold:
                close_reason = "assignment"

            if close_reason:
                # Buy back the call
                buyback_cost = call_value * active_cc.shares_covered
                pnl = (active_cc.premium - call_value) * active_cc.shares_covered

                if close_reason == "assignment":
                    # Assigned: sell shares at strike, then buy back at market
                    assigned_shares = active_cc.shares_covered
                    sell_proceeds = active_cc.strike * assigned_shares
                    buyback_cost_shares = S * assigned_shares
                    # Net: premium + (strike - market) per share
                    # We keep shares but "lose" the upside above strike
                    # Simulate: sell at strike, buy back at market next day
                    # The loss is (S - strike) per share, offset by premium
                    assignment_loss = (S - active_cc.strike) * assigned_shares
                    pnl = active_cc.premium * active_cc.shares_covered - assignment_loss
                    active_cc.assigned = True

                active_cc.close_date = d["date"]
                active_cc.close_idx = i
                active_cc.close_reason = close_reason
                active_cc.pnl = pnl
                active_cc.days_held = days_in
                closed_trades.append(active_cc)
                cash += pnl
                cc_premium_total += pnl
                active_cc = None

        # ── Open new covered call ──
        if active_cc is None and i - last_open_idx >= 5:
            should_sell = False
            if config.always_sell:
                should_sell = True
            elif risk > config.risk_threshold:
                should_sell = True

            if should_sell and shares >= 100:
                T = config.target_dte / 252
                K = find_call_strike_by_delta(S, T, config.risk_free_rate,
                                               sigma, config.target_delta)
                premium = bs_price(S, K, T, config.risk_free_rate, sigma)

                if premium > 0.01:
                    covered_shares = (shares // 100) * 100
                    active_cc = CCTrade(
                        open_date=d["date"],
                        open_idx=i,
                        strike=K,
                        premium=premium,
                        dte=config.target_dte,
                        underlying_at_open=S,
                        risk_at_open=risk,
                        iv_at_open=sigma,
                        shares_covered=covered_shares,
                    )
                    last_open_idx = i

                    # If reinvesting premium into more shares
                    if config.reinvest_premium and premium * covered_shares > S:
                        extra_shares = int((premium * covered_shares) / S)
                        # Don't actually buy more for simplicity; just track cash
                        pass

        # ── Daily portfolio value ──
        # Shares value + cash - active call liability
        share_val = shares * S
        call_liability = 0.0
        if active_cc is not None:
            days_in = i - active_cc.open_idx
            T_rem = max((active_cc.dte - days_in) / 252, 1 / 252)
            call_liability = bs_price(S, active_cc.strike, T_rem,
                                       config.risk_free_rate, sigma) * active_cc.shares_covered

        portfolio_val = share_val + cash - call_liability
        portfolio_values.append(portfolio_val)

        bh_val = bh_shares * S
        bh_values.append(bh_val)

    return closed_trades, np.array(portfolio_values), np.array(bh_values), initial_value, cc_premium_total


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def report_cc(trades, pv, bh, initial_value, cc_premium_total, config, label, years):
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    mode = "Always sell CC" if config.always_sell else f"Sell CC when Risk > {config.risk_threshold}"
    print(f"  Mode: {mode} | δ={config.target_delta} | DTE={config.target_dte}")
    print(f"  Initial: {config.initial_shares} shares + ${config.initial_capital:,.0f} cash")
    print(f"{'=' * 72}")

    # Portfolio performance
    final_pv = pv[-1]
    final_bh = bh[-1]
    pv_return = (final_pv / initial_value - 1) * 100
    bh_return = (final_bh / initial_value - 1) * 100
    pv_ann = ((final_pv / initial_value) ** (1 / years) - 1) * 100
    bh_ann = ((final_bh / initial_value) ** (1 / years) - 1) * 100

    # Max drawdown
    peak_pv = np.maximum.accumulate(pv)
    dd_pv = (pv - peak_pv) / peak_pv
    maxdd_pv = float(dd_pv.min()) * 100

    peak_bh = np.maximum.accumulate(bh)
    dd_bh = (bh - peak_bh) / peak_bh
    maxdd_bh = float(dd_bh.min()) * 100

    # Monthly Sharpe
    step = max(1, len(pv) // int(years * 12))
    monthly_pv = pv[::step]
    monthly_bh = bh[::step]
    if len(monthly_pv) > 1:
        mr_pv = np.diff(monthly_pv) / monthly_pv[:-1]
        mr_bh = np.diff(monthly_bh) / monthly_bh[:-1]
        sharpe_pv = float(np.mean(mr_pv) / np.std(mr_pv) * np.sqrt(12)) if np.std(mr_pv) > 0 else 0
        sharpe_bh = float(np.mean(mr_bh) / np.std(mr_bh) * np.sqrt(12)) if np.std(mr_bh) > 0 else 0
    else:
        sharpe_pv = sharpe_bh = 0

    print(f"\n  {'METRIC':<30} {'CC Strategy':>15} {'Buy & Hold':>15}")
    print(f"  {'─' * 62}")
    print(f"  {'Final value':<30} ${final_pv:>14,.2f} ${final_bh:>14,.2f}")
    print(f"  {'Total return':<30} {pv_return:>14.1f}% {bh_return:>14.1f}%")
    print(f"  {'Annualized return':<30} {pv_ann:>14.2f}% {bh_ann:>14.2f}%")
    print(f"  {'Sharpe ratio':<30} {sharpe_pv:>14.2f} {sharpe_bh:>14.2f}")
    print(f"  {'Max drawdown':<30} {maxdd_pv:>14.1f}% {maxdd_bh:>14.1f}%")
    print(f"  {'CC premium income':<30} ${cc_premium_total:>14,.2f} {'—':>15}")
    print(f"  {'Alpha (CC - BH ann.)':<30} {pv_ann - bh_ann:>+14.2f}% {'—':>15}")

    # Trade stats
    if trades:
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100
        assigned = sum(1 for t in trades if t.assigned)
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        print(f"\n  CC TRADE STATS")
        print(f"  {'─' * 50}")
        print(f"  Total CC trades:      {len(trades):>8}")
        print(f"  Win/Loss:             {len(wins):>3} / {len(losses):<3}   ({win_rate:.0f}%)")
        print(f"  Profit factor:        {pf:>8.2f}")
        if wins:
            print(f"  Avg win:              ${np.mean(wins):>8,.2f}")
        if losses:
            print(f"  Avg loss:             ${np.mean(losses):>8,.2f}")
        print(f"  Assigned:             {assigned:>8}  ({assigned/len(trades)*100:.0f}%)")
        print(f"  Avg days held:        {np.mean([t.days_held for t in trades]):>8.1f}")
        print(f"  Total premium P&L:    ${sum(pnls):>8,.2f}")

        # Close reasons
        reasons = {}
        for t in trades:
            r = t.close_reason
            reasons[r] = reasons.get(r, {"count": 0, "pnl": 0.0})
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t.pnl

        print(f"\n  CLOSE REASONS")
        print(f"  {'─' * 55}")
        for r, v in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            print(f"  {r:<20} {v['count']:>4} ({v['count']/len(trades)*100:>5.1f}%)  "
                  f"P&L=${v['pnl']:>10,.2f}")

        # Annual breakdown
        from collections import defaultdict
        yearly = defaultdict(lambda: {"pnl": 0, "n": 0, "wins": 0, "assigned": 0})
        for t in trades:
            yr = t.open_date[:4]
            yearly[yr]["pnl"] += t.pnl
            yearly[yr]["n"] += 1
            if t.pnl > 0: yearly[yr]["wins"] += 1
            if t.assigned: yearly[yr]["assigned"] += 1

        print(f"\n  ANNUAL CC BREAKDOWN")
        print(f"  {'─' * 65}")
        print(f"  {'Year':<6} {'Trades':>7} {'WR':>5} {'Assigned':>9} {'Premium P&L':>14}")
        print(f"  {'─' * 65}")
        for yr in sorted(yearly.keys()):
            y = yearly[yr]
            wr = y["wins"] / y["n"] * 100 if y["n"] > 0 else 0
            print(f"  {yr:<6} {y['n']:>7} {wr:>4.0f}% {y['assigned']:>9} ${y['pnl']:>13,.2f}")

    return {
        "label": label,
        "final_value": round(final_pv, 2),
        "bh_final": round(final_bh, 2),
        "total_return": round(pv_return, 2),
        "bh_return": round(bh_return, 2),
        "ann_return": round(pv_ann, 2),
        "bh_ann_return": round(bh_ann, 2),
        "alpha": round(pv_ann - bh_ann, 2),
        "sharpe": round(sharpe_pv, 2),
        "bh_sharpe": round(sharpe_bh, 2),
        "max_dd": round(maxdd_pv, 2),
        "bh_max_dd": round(maxdd_bh, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 1) if trades else 0,
        "assigned": sum(1 for t in trades if t.assigned) if trades else 0,
        "cc_premium_total": round(cc_premium_total, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  COVERED CALL BACKTEST — TQQQ Holder + Risk Signal")
    print("=" * 72)

    data = load_data()
    years = (np.datetime64(data[-1]["date"]) - np.datetime64(data[0]["date"])).astype(int) / 365.25
    start_price = data[30]["price"]
    print(f"\n  Data: {len(data)} days, {data[0]['date']} to {data[-1]['date']} ({years:.1f} years)")
    print(f"  TQQQ start: ${start_price:.2f}  end: ${data[-1]['price']:.2f}")

    # We simulate holding 100 shares + $10K cash
    # At start (~$2/share), 100 shares = $200 + $10K cash = $10.2K initial
    # More realistic: invest $10K → buy shares at start price
    initial_shares_for_10k = int(10_000 / start_price / 100) * 100  # round to 100s
    if initial_shares_for_10k < 100:
        initial_shares_for_10k = 100
    initial_cash = 10_000 - initial_shares_for_10k * start_price
    if initial_cash < 0:
        initial_cash = 0
        initial_shares_for_10k = 100
        
    print(f"  Starting position: {initial_shares_for_10k} shares @ ${start_price:.2f} = "
          f"${initial_shares_for_10k * start_price:,.2f} + ${initial_cash:,.2f} cash")

    configs = [
        # ── Baseline: Buy & Hold only (no CC) ──
        # We need a special config that never sells CC 
        (CCConfig(
            risk_threshold=999,  # never triggers
            always_sell=False,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "A: Buy & Hold only (no CC)"),

        # ── Always sell CC every month ──
        (CCConfig(
            always_sell=True,
            target_delta=0.20,
            target_dte=35,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "B: Always CC δ=0.20 DTE=35"),

        # ── Signal: Risk > 65, δ=0.20 ──
        (CCConfig(
            risk_threshold=65,
            target_delta=0.20,
            target_dte=35,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "C: Risk>65 CC δ=0.20"),

        # ── Signal: Risk > 65, δ=0.15 (more OTM) ──
        (CCConfig(
            risk_threshold=65,
            target_delta=0.15,
            target_dte=35,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "D: Risk>65 CC δ=0.15"),

        # ── Signal: Risk > 65, δ=0.25 (closer ATM, more premium) ──
        (CCConfig(
            risk_threshold=65,
            target_delta=0.25,
            target_dte=35,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "E: Risk>65 CC δ=0.25"),

        # ── Signal: Risk > 65, δ=0.30 (aggressive, high premium) ──
        (CCConfig(
            risk_threshold=65,
            target_delta=0.30,
            target_dte=35,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "F: Risk>65 CC δ=0.30"),

        # ── Signal: Risk > 60, δ=0.20 (earlier entry) ──
        (CCConfig(
            risk_threshold=60,
            target_delta=0.20,
            target_dte=35,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "G: Risk>60 CC δ=0.20"),

        # ── Signal: Risk > 70, δ=0.20 (stricter) ──
        (CCConfig(
            risk_threshold=70,
            target_delta=0.20,
            target_dte=35,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "H: Risk>70 CC δ=0.20"),

        # ── Always CC but δ=0.15 (very OTM, less upside cap) ──
        (CCConfig(
            always_sell=True,
            target_delta=0.15,
            target_dte=35,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "I: Always CC δ=0.15"),

        # ── Signal: Risk > 65, shorter DTE=21 ──
        (CCConfig(
            risk_threshold=65,
            target_delta=0.20,
            target_dte=21,
            roll_dte=7,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "J: Risk>65 CC δ=0.20 DTE=21"),

        # ── Signal: Risk > 65, δ=0.20, 70% PT (let more premium decay) ──
        (CCConfig(
            risk_threshold=65,
            target_delta=0.20,
            target_dte=35,
            profit_target_pct=0.70,
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "K: Risk>65 CC δ=0.20 PT=70%"),

        # ── Signal: Risk > 65, δ=0.20, no early close (hold to roll) ──
        (CCConfig(
            risk_threshold=65,
            target_delta=0.20,
            target_dte=35,
            profit_target_pct=0.95,  # almost never triggers
            initial_shares=initial_shares_for_10k,
            initial_capital=initial_cash,
        ), "L: Risk>65 CC δ=0.20 no PT"),
    ]

    all_results = []
    for cfg, label in configs:
        trades, pv, bh, iv, cpt = run_cc_backtest(data, cfg)
        r = report_cc(trades, pv, bh, iv, cpt, cfg, label, years)
        if r:
            all_results.append(r)

    # Comparison table
    print(f"\n\n{'=' * 80}")
    print("  COVERED CALL STRATEGY COMPARISON")
    print(f"{'=' * 80}\n")
    print(f"  {'Strategy':<32} {'AnnRet':>7} {'BH Ann':>7} {'Alpha':>7} {'Sharpe':>7} "
          f"{'MaxDD':>7} {'Trades':>6} {'WR':>5} {'Asgn':>5}")
    print(f"  {'─' * 90}")

    all_results.sort(key=lambda x: x["ann_return"], reverse=True)
    for r in all_results:
        print(f"  {r['label']:<32} {r['ann_return']:>6.1f}% {r['bh_ann_return']:>6.1f}% "
              f"{r['alpha']:>+6.2f}% {r['sharpe']:>6.2f} "
              f"{r['max_dd']:>6.1f}% {r['trades']:>6} {r['win_rate']:>4.0f}% {r['assigned']:>5}")

    # Save
    out_path = Path(__file__).resolve().parent / "cc_backtest_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Results saved to {out_path}")

    # Current signal
    latest = data[-1]
    risk = latest["risk_score"]
    print(f"\n{'=' * 72}")
    print(f"  CURRENT SIGNAL ({latest['date']})")
    print(f"{'=' * 72}")
    print(f"  Risk Score: {risk:.1f}")
    print(f"  TQQQ: ${latest['price']:.2f}")
    if risk > 65:
        sigma = compute_historical_vol([d["price"] for d in data], 20)[-1]
        T = 35 / 252
        K20 = find_call_strike_by_delta(latest["price"], T, 0.045, sigma, 0.20)
        K15 = find_call_strike_by_delta(latest["price"], T, 0.045, sigma, 0.15)
        P20 = bs_price(latest["price"], K20, T, 0.045, sigma)
        P15 = bs_price(latest["price"], K15, T, 0.045, sigma)
        print(f"\n  SELL CC SIGNAL ACTIVE:")
        print(f"  δ=0.20: Strike=${K20:.2f} ({(K20/latest['price']-1)*100:+.1f}% OTM)  "
              f"Premium≈${P20:.2f}/share  (${P20*100:.0f}/contract)")
        print(f"  δ=0.15: Strike=${K15:.2f} ({(K15/latest['price']-1)*100:+.1f}% OTM)  "
              f"Premium≈${P15:.2f}/share  (${P15*100:.0f}/contract)")
    else:
        print(f"  Signal: HOLD ONLY — no CC (Risk {risk:.0f} < 65)")


if __name__ == "__main__":
    main()
