#!/usr/bin/env python3
"""
Risk-Guided Options Strategy v2 — Optimized
=============================================

Improvements over v1:
  1. Adaptive thresholds: rolling percentile-based instead of fixed
  2. Conviction scaling: more contracts when risk is extreme  
  3. Vol-adjusted delta: wider OTM in high vol, tighter in low vol
  4. Credit spreads: defined risk, lower margin, higher capital efficiency
  5. Score velocity filter: only open when trend confirms direction
  6. Dynamic position sizing: scale with equity (compounding)
  7. Straddle/strangle in neutral zone with high vol

Data: TQQQ daily prices + Bubble Risk Score (drawdown_risk_score)
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes
# ═══════════════════════════════════════════════════════════════════

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, r, sigma, opt="P"):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, (K - S) if opt == "P" else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "C":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_delta(S, K, T, r, sigma, opt="P"):
    if T <= 0 or sigma <= 0:
        if opt == "C": return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if opt == "C" else norm_cdf(d1) - 1.0

def find_strike_by_delta(S, T, r, sigma, target_delta, opt="P"):
    target = abs(target_delta)
    lo, hi = S * 0.3, S * 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        d = abs(bs_delta(S, mid, T, r, sigma, opt))
        if opt == "P":
            if d > target: hi = mid
            else: lo = mid
        else:
            if d > target: lo = mid
            else: hi = mid
    return round((lo + hi) / 2, 2)


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    open_date: str
    open_idx: int
    trade_type: str           # "naked_put", "naked_call", "put_spread", "call_spread"
    short_strike: float
    long_strike: Optional[float]   # None for naked, defined for spreads
    premium: float            # net premium received per share
    dte_at_open: int
    underlying_at_open: float
    risk_at_open: float
    iv_at_open: float
    contracts: int
    conviction: float         # 0-1 conviction score

    close_date: Optional[str] = None
    close_idx: Optional[int] = None
    close_premium: Optional[float] = None
    close_reason: Optional[str] = None
    pnl: Optional[float] = None
    days_held: Optional[int] = None
    max_loss_per_share: float = 0.0   # for position sizing


@dataclass
class V2Config:
    # Threshold mode
    use_adaptive_thresholds: bool = True
    percentile_lookback: int = 252     # 1 year rolling window
    low_percentile: float = 20.0      # bottom 20% → sell put
    high_percentile: float = 80.0     # top 80% → sell call
    # Fixed fallback
    fixed_low: float = 40.0
    fixed_high: float = 60.0

    # Options
    base_delta: float = 0.20
    target_dte: int = 35
    spread_width_pct: float = 0.05    # 5% spread width (e.g., 50→47.5 for put spread)
    use_spreads: bool = False         # credit spreads vs naked

    # Conviction scaling
    use_conviction_scaling: bool = True
    max_contracts: int = 3            # max contracts per trade
    base_contracts: int = 1

    # Vol-adjusted delta
    vol_adjust_delta: bool = True
    vol_high_threshold: float = 0.80  # annualized vol > 80% → widen delta
    vol_low_threshold: float = 0.40   # vol < 40% → tighten delta

    # Score velocity filter
    use_velocity_filter: bool = True
    velocity_lookback: int = 10       # days
    # For sell put: require risk decreasing (velocity < 0)
    # For sell call: require risk increasing (velocity > 0)

    # Position management
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 2.00
    roll_dte: int = 14                # tighter roll (was 21)

    # Limits
    max_open_positions: int = 3
    min_days_between_opens: int = 7

    # Capital
    initial_capital: float = 10_000.0
    use_dynamic_sizing: bool = True   # scale contracts with equity
    margin_pct: float = 0.20          # 20% margin per contract (portfolio margin)
    max_capital_per_trade: float = 0.90  # max 90% of equity per trade (full position)

    # Risk-free rate
    risk_free_rate: float = 0.045

    ticker: str = "TQQQ"


# ═══════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════

def load_data():
    history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
    tqqq_raw = json.loads((DATA_DIR / "tqqq.json").read_text())
    tqqq_data = tqqq_raw["data"] if isinstance(tqqq_raw, dict) else tqqq_raw

    risk_map = {}
    composite_map = {}
    for h in history:
        risk_map[h["date"]] = h["drawdown_risk_score"]
        composite_map[h["date"]] = h["composite_score"]

    records = []
    for d in tqqq_data:
        date = d["date"]
        if date in risk_map:
            records.append({
                "date": date,
                "price": d["price"],
                "risk_score": risk_map[date],
                "composite_score": composite_map[date],
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


def compute_rolling_percentiles(scores, window, lo_pct, hi_pct):
    """Compute rolling percentile thresholds."""
    n = len(scores)
    lo_thresh = np.full(n, np.nan)
    hi_thresh = np.full(n, np.nan)
    for i in range(window, n):
        w = scores[i - window:i]
        lo_thresh[i] = np.percentile(w, lo_pct)
        hi_thresh[i] = np.percentile(w, hi_pct)
    # Backfill
    if window < n:
        lo_thresh[:window] = lo_thresh[window]
        hi_thresh[:window] = hi_thresh[window]
    return lo_thresh, hi_thresh


# ═══════════════════════════════════════════════════════════════════
# Backtest engine v2
# ═══════════════════════════════════════════════════════════════════

def run_backtest_v2(data, config: V2Config):
    prices = [d["price"] for d in data]
    risk_scores = np.array([d["risk_score"] for d in data])
    vol = compute_historical_vol(prices, 20)

    # Adaptive thresholds
    if config.use_adaptive_thresholds:
        lo_thresh, hi_thresh = compute_rolling_percentiles(
            risk_scores, config.percentile_lookback,
            config.low_percentile, config.high_percentile)
    else:
        lo_thresh = np.full(len(data), config.fixed_low)
        hi_thresh = np.full(len(data), config.fixed_high)

    # Score velocity
    velocity = np.full(len(data), 0.0)
    for i in range(config.velocity_lookback, len(data)):
        velocity[i] = risk_scores[i] - risk_scores[i - config.velocity_lookback]

    open_positions: list[Trade] = []
    closed_positions: list[Trade] = []
    last_open_idx = -999

    equity = config.initial_capital
    equity_curve = [equity]
    peak_equity = equity

    start_idx = max(30, config.percentile_lookback)

    for i in range(start_idx, len(data)):
        d = data[i]
        S = d["price"]
        risk = risk_scores[i]
        sigma = vol[i] if not np.isnan(vol[i]) else 0.60
        sigma = max(sigma, 0.30)

        # ── Manage existing positions ──
        to_close = []
        for pos in open_positions:
            days_in = i - pos.open_idx
            T_rem = max((pos.dte_at_open - days_in) / 252, 1/252)

            # Price short leg
            short_val = bs_price(S, pos.short_strike, T_rem,
                                  config.risk_free_rate, sigma,
                                  "P" if "put" in pos.trade_type else "C")
            # Price long leg (if spread)
            long_val = 0.0
            if pos.long_strike is not None:
                long_val = bs_price(S, pos.long_strike, T_rem,
                                     config.risk_free_rate, sigma,
                                     "P" if "put" in pos.trade_type else "C")

            # Net value to close (buy back spread)
            net_value = short_val - long_val
            unrealized = pos.premium - net_value  # positive = profit

            close_reason = None
            if unrealized >= pos.premium * config.profit_target_pct:
                close_reason = "profit_target"
            elif unrealized <= -pos.premium * config.stop_loss_pct:
                close_reason = "stop_loss"
            elif pos.dte_at_open - days_in <= config.roll_dte:
                close_reason = "roll_dte"
            elif "put" in pos.trade_type and S < pos.short_strike * 0.85:
                close_reason = "assignment_risk"
            elif "call" in pos.trade_type and S > pos.short_strike * 1.15:
                close_reason = "assignment_risk"

            if close_reason:
                pnl = (pos.premium - net_value) * 100 * pos.contracts
                pos.close_date = d["date"]
                pos.close_idx = i
                pos.close_premium = net_value
                pos.close_reason = close_reason
                pos.pnl = pnl
                pos.days_held = days_in
                to_close.append(pos)

        for pos in to_close:
            open_positions.remove(pos)
            closed_positions.append(pos)
            equity += pos.pnl

        # ── Signal generation ──
        if (len(open_positions) < config.max_open_positions and
                i - last_open_idx >= config.min_days_between_opens):

            is_low = risk < lo_thresh[i]
            is_high = risk > hi_thresh[i]

            # Velocity filter
            vel = velocity[i]
            if config.use_velocity_filter:
                if is_low and vel > 2:  # risk rising fast → don't sell put
                    is_low = False
                if is_high and vel < -2:  # risk falling fast → don't sell call
                    is_high = False

            signal = None
            if is_low:
                signal = "SELL_PUT"
            elif is_high:
                signal = "SELL_CALL"

            if signal:
                T = config.target_dte / 252
                is_put = signal == "SELL_PUT"
                opt_type = "P" if is_put else "C"

                # ── Vol-adjusted delta ──
                delta = config.base_delta
                if config.vol_adjust_delta:
                    if sigma > config.vol_high_threshold:
                        delta = max(0.10, delta - 0.05)  # wider OTM in high vol
                    elif sigma < config.vol_low_threshold:
                        delta = min(0.35, delta + 0.05)  # tighter in low vol

                # ── Conviction score ──
                # How extreme is the risk score vs threshold?
                if is_put:
                    raw_conv = (lo_thresh[i] - risk) / max(lo_thresh[i], 1)
                else:
                    raw_conv = (risk - hi_thresh[i]) / max(100 - hi_thresh[i], 1)
                conviction = min(1.0, max(0.0, raw_conv * 3))  # scale 0-1

                # ── Position sizing ──
                notional_per = S * 100
                margin_per = notional_per * config.margin_pct

                if config.use_dynamic_sizing:
                    # Full-capital: size contracts to use available equity
                    margin_used = sum(
                        p.underlying_at_open * 100 * p.contracts * config.margin_pct
                        for p in open_positions
                    )
                    available = max(0, equity - margin_used)
                    contracts = max(1, int(
                        available * config.max_capital_per_trade / margin_per
                    ))
                else:
                    contracts = config.base_contracts

                if config.use_conviction_scaling:
                    # Scale up/down based on conviction (cap at max_contracts)
                    conv_multiplier = 0.5 + conviction * 0.5  # 0.5x to 1.0x
                    contracts = max(1, int(contracts * conv_multiplier))

                contracts = min(contracts, config.max_contracts)

                # ── Find strikes ──
                short_strike = find_strike_by_delta(S, T, config.risk_free_rate,
                                                     sigma, delta, opt_type)

                long_strike = None
                if config.use_spreads:
                    # Credit spread: buy protection further OTM
                    if is_put:
                        long_strike = round(short_strike * (1 - config.spread_width_pct), 2)
                    else:
                        long_strike = round(short_strike * (1 + config.spread_width_pct), 2)

                # Premium
                short_prem = bs_price(S, short_strike, T, config.risk_free_rate,
                                       sigma, opt_type)
                long_prem = 0.0
                if long_strike:
                    long_prem = bs_price(S, long_strike, T, config.risk_free_rate,
                                          sigma, opt_type)

                net_premium = short_prem - long_prem  # credit received

                # Max loss per share for spreads
                if long_strike:
                    max_loss = abs(short_strike - long_strike) - net_premium
                else:
                    max_loss = short_strike if is_put else S  # approximate

                if net_premium >= 0.05:
                    trade = Trade(
                        open_date=d["date"],
                        open_idx=i,
                        trade_type=("put_spread" if is_put else "call_spread") if config.use_spreads
                                    else ("naked_put" if is_put else "naked_call"),
                        short_strike=short_strike,
                        long_strike=long_strike,
                        premium=net_premium,
                        dte_at_open=config.target_dte,
                        underlying_at_open=S,
                        risk_at_open=risk,
                        iv_at_open=sigma,
                        contracts=contracts,
                        conviction=conviction,
                        max_loss_per_share=max_loss,
                    )
                    open_positions.append(trade)
                    last_open_idx = i

        peak_equity = max(peak_equity, equity)
        equity_curve.append(equity)

    # Close remaining
    last = data[-1]
    S_last = last["price"]
    sig_last = vol[-1] if not np.isnan(vol[-1]) else 0.60
    for pos in open_positions:
        days_in = len(data) - 1 - pos.open_idx
        T_rem = max((pos.dte_at_open - days_in) / 252, 1/252)
        opt_t = "P" if "put" in pos.trade_type else "C"
        sv = bs_price(S_last, pos.short_strike, T_rem, config.risk_free_rate, sig_last, opt_t)
        lv = 0.0
        if pos.long_strike:
            lv = bs_price(S_last, pos.long_strike, T_rem, config.risk_free_rate, sig_last, opt_t)
        nv = sv - lv
        pos.close_date = last["date"]
        pos.close_idx = len(data) - 1
        pos.close_premium = nv
        pos.close_reason = "end_of_data"
        pos.pnl = (pos.premium - nv) * 100 * pos.contracts
        pos.days_held = days_in
        closed_positions.append(pos)
        equity += pos.pnl

    return closed_positions, np.array(equity_curve), config


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def report(trades, eq_curve, config, label, years):
    print(f"\n{'='*72}")
    print(f"  {label}")
    adapt = "adaptive" if config.use_adaptive_thresholds else f"fixed {config.fixed_low}/{config.fixed_high}"
    spread = "credit spread" if config.use_spreads else "naked"
    print(f"  Mode: {spread} | Thresholds: {adapt} | Delta: {config.base_delta}")
    vel = "ON" if config.use_velocity_filter else "OFF"
    conv = "ON" if config.use_conviction_scaling else "OFF"
    dyn = "ON" if config.use_dynamic_sizing else "OFF"
    print(f"  Velocity filter: {vel} | Conviction scaling: {conv} | Dynamic sizing: {dyn}")
    print(f"  Initial capital: ${config.initial_capital:,.0f}")
    print(f"{'='*72}")

    if not trades:
        print("  No trades.")
        return {}

    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100

    final_eq = eq_curve[-1]
    total_return = (final_eq / config.initial_capital - 1)
    ann_return = (final_eq / config.initial_capital) ** (1 / years) - 1

    # Max drawdown from equity curve
    peak = np.maximum.accumulate(eq_curve)
    dd = (eq_curve - peak) / peak
    max_dd = float(dd.min())
    max_dd_dollar = float((eq_curve - peak).min())

    # Sharpe on monthly equity returns
    monthly_eq = [eq_curve[0]]
    month_label = ""
    for i, t in enumerate(trades):
        m = t.close_date[:7] if t.close_date else ""
        if m != month_label and month_label:
            monthly_eq.append(eq_curve[min(t.close_idx or 0, len(eq_curve)-1)])
        month_label = m
    monthly_eq.append(eq_curve[-1])

    # Better: sample equity curve monthly
    step = max(1, len(eq_curve) // (int(years * 12)))
    monthly_samples = eq_curve[::step]
    if len(monthly_samples) > 1:
        monthly_rets = np.diff(monthly_samples) / monthly_samples[:-1]
        sharpe = float(np.mean(monthly_rets) / np.std(monthly_rets) * np.sqrt(12)) if np.std(monthly_rets) > 0 else 0
    else:
        sharpe = 0

    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    put_trades = [t for t in trades if "put" in t.trade_type]
    call_trades = [t for t in trades if "call" in t.trade_type]

    print(f"\n  PERFORMANCE")
    print(f"  {'─'*50}")
    print(f"  Final equity:       ${final_eq:>12,.2f}")
    print(f"  Total P&L:          ${total_pnl:>12,.2f}")
    print(f"  Total return:       {total_return:>11.1%}")
    print(f"  Annualized return:  {ann_return:>11.2%}")
    print(f"  Sharpe ratio:       {sharpe:>11.2f}")
    print(f"  Max drawdown:       {max_dd:>11.1%}  (${max_dd_dollar:>,.0f})")

    print(f"\n  TRADES")
    print(f"  {'─'*50}")
    print(f"  Total:              {len(trades):>10}")
    print(f"  Win/Loss:           {len(wins):>4} / {len(losses):<4}  ({win_rate:.0f}%)")
    print(f"  Avg win:            ${np.mean(wins):>10,.2f}" if wins else "")
    print(f"  Avg loss:           ${np.mean(losses):>10,.2f}" if losses else "")
    print(f"  Profit factor:      {pf:>10.2f}")
    print(f"  Avg days held:      {np.mean([t.days_held for t in trades]):>10.1f}")
    print(f"  Avg contracts:      {np.mean([t.contracts for t in trades]):>10.1f}")

    print(f"\n  BY TYPE")
    print(f"  {'─'*50}")
    if put_trades:
        pp = sum(t.pnl for t in put_trades)
        pwr = sum(1 for t in put_trades if t.pnl > 0) / len(put_trades) * 100
        print(f"  Puts:  {len(put_trades):>4}  P&L=${pp:>10,.2f}  WR={pwr:.0f}%")
    if call_trades:
        cp = sum(t.pnl for t in call_trades)
        cwr = sum(1 for t in call_trades if t.pnl > 0) / len(call_trades) * 100
        print(f"  Calls: {len(call_trades):>4}  P&L=${cp:>10,.2f}  WR={cwr:.0f}%")

    # Close reasons
    reasons = {}
    for t in trades:
        r = t.close_reason
        reasons[r] = reasons.get(r, {"count": 0, "pnl": 0})
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t.pnl

    print(f"\n  CLOSE REASONS")
    print(f"  {'─'*50}")
    for r, v in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        print(f"  {r:<20} {v['count']:>4} ({v['count']/len(trades)*100:>5.1f}%)  P&L=${v['pnl']:>10,.2f}")

    # Annual breakdown
    from collections import defaultdict
    yearly = defaultdict(lambda: {"pnl": 0, "n": 0, "wins": 0,
                                   "puts": 0, "calls": 0, "contracts": 0})
    for t in trades:
        yr = t.open_date[:4]
        yearly[yr]["pnl"] += t.pnl
        yearly[yr]["n"] += 1
        yearly[yr]["contracts"] += t.contracts
        if t.pnl > 0: yearly[yr]["wins"] += 1
        if "put" in t.trade_type: yearly[yr]["puts"] += 1
        else: yearly[yr]["calls"] += 1

    print(f"\n  ANNUAL BREAKDOWN")
    print(f"  {'─'*70}")
    print(f"  {'Year':<6} {'Trades':>7} {'Puts':>5} {'Calls':>6} {'Contracts':>10} {'WR':>5} {'P&L':>12}")
    print(f"  {'─'*70}")
    cum = 0
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        wr = y["wins"] / y["n"] * 100 if y["n"] > 0 else 0
        cum += y["pnl"]
        print(f"  {yr:<6} {y['n']:>7} {y['puts']:>5} {y['calls']:>6} "
              f"{y['contracts']:>10} {wr:>4.0f}% ${y['pnl']:>11,.2f}")
    print(f"  {'─'*70}")
    print(f"  {'TOTAL':<6} {len(trades):>7} {len(put_trades):>5} {len(call_trades):>6} "
          f"{sum(t.contracts for t in trades):>10} {win_rate:>4.0f}% ${total_pnl:>11,.2f}")

    return {
        "label": label,
        "total_pnl": round(total_pnl, 2),
        "total_return": round(total_return * 100, 2),
        "ann_return": round(ann_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "max_dd_dollar": round(max_dd_dollar, 2),
        "trades": len(trades),
        "n_puts": len(put_trades),
        "n_calls": len(call_trades),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  RISK-GUIDED OPTIONS STRATEGY v2 — OPTIMIZED")
    print("=" * 72)

    data = load_data()
    years = (np.datetime64(data[-1]["date"]) - np.datetime64(data[0]["date"])).astype(int) / 365.25
    print(f"\n  Data: {len(data)} days, {data[0]['date']} to {data[-1]['date']} ({years:.1f} years)")

    # Full-position sizing: use up to 90% of available equity per trade
    FULL = dict(use_dynamic_sizing=True, max_capital_per_trade=0.90)
    HALF = dict(use_dynamic_sizing=True, max_capital_per_trade=0.50)
    THIRD = dict(use_dynamic_sizing=True, max_capital_per_trade=0.33)

    configs = [
        # ══════════════════════════════════════════════════════════
        # Group 1: 提高 put 阈值，让中等风险也卖 put
        # risk均值=57, P20=49, P50=56, P80=64
        # ══════════════════════════════════════════════════════════

        # ── 基线: 旧版 35/65 (put几乎不触发) ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=35, fixed_high=65,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=10,
            max_contracts=999, base_contracts=1, **FULL,
        ), "A: old 35/65 (call only)"),

        # ── 50/65: risk<50卖put, >65卖call ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=50, fixed_high=65,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **FULL,
        ), "B: 50/65 full"),

        # ── 55/65: 更激进put, risk<55都卖 ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=55, fixed_high=65,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **FULL,
        ), "C: 55/65 full"),

        # ── 55/60: 双边紧, 5分中性区 ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=55, fixed_high=60,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **FULL,
        ), "D: 55/60 full (tight)"),

        # ── 50/60: 双边, 10分中性区 ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=50, fixed_high=60,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **FULL,
        ), "E: 50/60 full"),

        # ── 60/65: 很高才卖call, <60都卖put ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=60, fixed_high=65,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **FULL,
        ), "F: 60/65 full (put-heavy)"),

        # ══════════════════════════════════════════════════════════
        # Group 2: 最佳阈值 + 参数调优
        # ══════════════════════════════════════════════════════════

        # ── 50/65 + delta 0.15 (更远OTM) ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=50, fixed_high=65,
            base_delta=0.15, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **FULL,
        ), "G: 50/65 delta=0.15"),

        # ── 55/65 + delta 0.15 ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=55, fixed_high=65,
            base_delta=0.15, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **FULL,
        ), "H: 55/65 delta=0.15"),

        # ── 50/65 半仓 ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=50, fixed_high=65,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **HALF,
        ), "I: 50/65 half(50%)"),

        # ── 55/65 半仓 ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=55, fixed_high=65,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            max_contracts=999, base_contracts=1, **HALF,
        ), "J: 55/65 half(50%)"),

        # ── 50/65 + 2个仓位交替 ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=50, fixed_high=65,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=2, min_days_between_opens=5,
            max_contracts=999, base_contracts=1, **HALF,
        ), "K: 50/65 2pos x 50%"),

        # ── 55/65 + tight stop ──
        (V2Config(
            use_adaptive_thresholds=False, fixed_low=55, fixed_high=65,
            base_delta=0.20, use_conviction_scaling=False,
            vol_adjust_delta=False, use_velocity_filter=False,
            use_spreads=False,
            max_open_positions=1, min_days_between_opens=7,
            profit_target_pct=0.50, stop_loss_pct=1.50,
            max_contracts=999, base_contracts=1, **FULL,
        ), "L: 55/65 tight stop(1.5x)"),
    ]

    all_results = []
    for cfg, label in configs:
        trades, eq, _ = run_backtest_v2(data, cfg)
        r = report(trades, eq, cfg, label, years)
        if r:
            all_results.append(r)

    # Comparison table
    print(f"\n\n{'='*72}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*72}\n")
    print(f"  {'Strategy':<35} {'AnnRet':>7} {'Sharpe':>7} {'MaxDD':>7} {'WR':>5} {'PF':>5} {'Puts':>5} {'Calls':>5} {'Total':>5}")
    print(f"  {'─'*92}")

    all_results.sort(key=lambda x: x["ann_return"], reverse=True)
    for r in all_results:
        print(f"  {r['label']:<35} {r['ann_return']:>6.1f}% {r['sharpe']:>6.2f} "
              f"{r['max_dd_pct']:>6.1f}% {r['win_rate']:>4.0f}% "
              f"{r['profit_factor']:>4.1f} {r.get('n_puts',0):>5} {r.get('n_calls',0):>5} {r['trades']:>5}")

    # Save
    out_path = Path(__file__).resolve().parent / "options_v2_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Results saved to {out_path}")

    # Current signal
    latest = data[-1]
    risk = latest["risk_score"]
    risk_scores = np.array([d["risk_score"] for d in data])
    lo_t, hi_t = compute_rolling_percentiles(risk_scores, 252, 20, 80)

    print(f"\n{'='*72}")
    print(f"  CURRENT SIGNAL ({latest['date']})")
    print(f"{'='*72}")
    print(f"  Risk Score: {risk:.1f}")
    print(f"  Adaptive thresholds: Low={lo_t[-1]:.1f} (P20), High={hi_t[-1]:.1f} (P80)")
    print(f"  TQQQ: ${latest['price']:.2f}")

    if risk < lo_t[-1]:
        print(f"  Signal: SELL PUT (risk {risk:.0f} < {lo_t[-1]:.0f})")
    elif risk > hi_t[-1]:
        print(f"  Signal: SELL CALL (risk {risk:.0f} > {hi_t[-1]:.0f})")
    else:
        print(f"  Signal: NEUTRAL (risk in {lo_t[-1]:.0f}-{hi_t[-1]:.0f} range)")


if __name__ == "__main__":
    main()
