#!/usr/bin/env python3
"""
Risk-Guided Options Strategy Backtest
======================================

Combines Market Bubble Index risk scores with clawdfolio's options playbook:

  Risk LOW  (< 40) → Sell OTM Put  on TQQQ  (bullish, collect theta)
  Risk MID  (40-60) → No new positions (hold existing, let decay)
  Risk HIGH (> 60) → Sell OTM Call on TQQQ  (bearish, collect theta)

Options Parameters (matching clawdfolio's playbook):
  - DTE target: 30-45 days
  - Delta target: ~0.20-0.30 OTM
  - Position size: 1 contract per signal (100 shares notional)
  - Management: close at 50% profit, roll at 21 DTE, stop at 200% loss
  - Assignment: accepted for puts (own shares), avoided for calls

Since we don't have historical options chain data, we use Black-Scholes
to synthesize realistic option premiums from historical price + volatility.

Data: TQQQ daily prices (2013-2026) + Bubble Risk Score (2015-2026)
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
# Black-Scholes for synthetic options pricing
# ═══════════════════════════════════════════════════════════════════

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(S, K, T, r, sigma, option_type="P"):
    """Black-Scholes price for European option."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, (K - S) if option_type == "P" else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "C":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_delta(S, K, T, r, sigma, option_type="P"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == "C":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if option_type == "C":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0

def find_strike_by_delta(S, T, r, sigma, target_delta, option_type="P"):
    """Binary search for strike that gives target |delta|.
    
    For puts: OTM = strike < S. Higher strike → more ITM → higher |delta|.
    For calls: OTM = strike > S. Higher strike → more OTM → lower |delta|.
    """
    target = abs(target_delta)
    lo, hi = S * 0.3, S * 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        d = abs(bs_delta(S, mid, T, r, sigma, option_type))
        if option_type == "P":
            # Put: higher K → higher |delta| (more ITM)
            if d > target:
                hi = mid   # too ITM, lower the strike
            else:
                lo = mid   # too OTM, raise the strike
        else:
            # Call: higher K → lower |delta| (more OTM)
            if d > target:
                lo = mid   # too ITM, raise the strike
            else:
                hi = mid   # too OTM, lower the strike
    return round((lo + hi) / 2, 2)


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OptionPosition:
    """A short option position."""
    open_date: str
    open_idx: int
    option_type: str          # "P" or "C"
    strike: float
    premium: float            # premium received per share
    dte_at_open: int
    underlying_price_at_open: float
    risk_score_at_open: float
    iv_at_open: float

    close_date: Optional[str] = None
    close_idx: Optional[int] = None
    close_premium: Optional[float] = None
    close_reason: Optional[str] = None
    pnl: Optional[float] = None
    days_held: Optional[int] = None

    def is_open(self):
        return self.close_date is None


@dataclass
class StrategyConfig:
    """Configurable strategy parameters."""
    # Risk thresholds
    risk_low_threshold: float = 40.0    # below → sell put
    risk_high_threshold: float = 60.0   # above → sell call
    
    # Options parameters (matching clawdfolio playbook)
    target_dte: int = 35                # 30-45 day sweet spot
    target_delta: float = 0.25          # ~0.20-0.30 OTM
    
    # Position management
    profit_target_pct: float = 0.50     # close at 50% of premium collected
    stop_loss_pct: float = 2.00         # close at 200% loss (3x premium)
    roll_dte: int = 21                  # roll when DTE <= this
    
    # Position limits
    max_open_positions: int = 3         # max concurrent positions
    min_days_between_opens: int = 5     # don't open too frequently
    
    # Underlying
    ticker: str = "TQQQ"
    contracts_per_signal: int = 1       # 1 contract = 100 shares
    
    # Risk-free rate
    risk_free_rate: float = 0.045       # ~4.5% (current environment)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: StrategyConfig
    positions: list
    
    # Summary stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_premium_collected: float = 0.0
    
    # By type
    put_trades: int = 0
    put_pnl: float = 0.0
    call_trades: int = 0
    call_pnl: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Time stats
    avg_days_held: float = 0.0
    
    # Close reasons
    close_reasons: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Main backtest engine
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load TQQQ prices + bubble risk scores."""
    history = json.loads((DATA_DIR / "bubble_history.json").read_text())["history"]
    tqqq_raw = json.loads((DATA_DIR / "tqqq.json").read_text())
    tqqq_data = tqqq_raw["data"] if isinstance(tqqq_raw, dict) else tqqq_raw

    # Build aligned dataset
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
    """Compute rolling annualized volatility."""
    log_ret = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
    vol = np.full(len(prices), np.nan)
    for i in range(window, len(log_ret)):
        vol[i + 1] = float(np.std(log_ret[i - window + 1:i + 1]) * np.sqrt(252))
    # Backfill
    first_valid = window + 1
    if first_valid < len(vol):
        vol[:first_valid] = vol[first_valid]
    return vol


def run_backtest(data, config: StrategyConfig) -> BacktestResult:
    """Run the risk-guided options strategy backtest."""
    prices = [d["price"] for d in data]
    vol = compute_historical_vol(prices, 20)
    
    open_positions: list[OptionPosition] = []
    closed_positions: list[OptionPosition] = []
    last_open_idx = -999
    
    equity_curve = [0.0]  # cumulative P&L
    
    for i in range(30, len(data)):  # start after vol warmup
        d = data[i]
        S = d["price"]
        risk = d["risk_score"]
        sigma = vol[i] if not np.isnan(vol[i]) else 0.60  # TQQQ vol ~60% typical
        
        # Ensure minimum vol for TQQQ (3x leveraged ETF)
        sigma = max(sigma, 0.30)
        
        # ── Step 1: Manage existing positions ──
        positions_to_close = []
        for pos in open_positions:
            days_in = i - pos.open_idx
            T_remaining = max((pos.dte_at_open - days_in) / 252, 1/252)
            
            # Current option value
            current_value = bs_price(S, pos.strike, T_remaining,
                                      config.risk_free_rate, sigma, pos.option_type)
            
            # P&L per share (we sold, so profit = premium - current_value)
            unrealized_pnl = pos.premium - current_value
            
            close_reason = None
            
            # Check profit target (50% of premium captured)
            if unrealized_pnl >= pos.premium * config.profit_target_pct:
                close_reason = "profit_target"
            
            # Check stop loss (loss exceeds 200% of premium)
            elif unrealized_pnl <= -pos.premium * config.stop_loss_pct:
                close_reason = "stop_loss"
            
            # Check expiry approach (roll at 21 DTE)
            elif pos.dte_at_open - days_in <= config.roll_dte:
                close_reason = "roll_dte"
            
            # Check assignment risk for deep ITM (> 15% ITM)
            elif pos.option_type == "P" and S < pos.strike * 0.85:
                close_reason = "assignment_risk"
            
            elif pos.option_type == "C" and S > pos.strike * 1.15:
                close_reason = "assignment_risk"
            
            if close_reason:
                pos.close_date = d["date"]
                pos.close_idx = i
                pos.close_premium = current_value
                pos.close_reason = close_reason
                pos.pnl = (pos.premium - current_value) * 100 * config.contracts_per_signal
                pos.days_held = days_in
                positions_to_close.append(pos)
        
        for pos in positions_to_close:
            open_positions.remove(pos)
            closed_positions.append(pos)
        
        # ── Step 2: Signal generation ──
        if (len(open_positions) < config.max_open_positions and
                i - last_open_idx >= config.min_days_between_opens):
            
            signal = None
            if risk < config.risk_low_threshold:
                signal = "SELL_PUT"
            elif risk > config.risk_high_threshold:
                signal = "SELL_CALL"
            
            if signal:
                T = config.target_dte / 252
                option_type = "P" if signal == "SELL_PUT" else "C"
                
                # Find strike at target delta
                strike = find_strike_by_delta(
                    S, T, config.risk_free_rate, sigma,
                    config.target_delta, option_type
                )
                
                # Calculate premium
                premium = bs_price(S, strike, T, config.risk_free_rate,
                                    sigma, option_type)
                
                # Minimum premium check ($0.10 per share)
                if premium >= 0.10:
                    pos = OptionPosition(
                        open_date=d["date"],
                        open_idx=i,
                        option_type=option_type,
                        strike=strike,
                        premium=premium,
                        dte_at_open=config.target_dte,
                        underlying_price_at_open=S,
                        risk_score_at_open=risk,
                        iv_at_open=sigma,
                    )
                    open_positions.append(pos)
                    last_open_idx = i
        
        # Track equity curve
        daily_pnl = sum(p.pnl for p in positions_to_close if p.pnl is not None)
        equity_curve.append(equity_curve[-1] + daily_pnl)
    
    # Close any remaining open positions at last price
    last = data[-1]
    S = last["price"]
    sigma_last = vol[-1] if not np.isnan(vol[-1]) else 0.60
    for pos in open_positions:
        days_in = len(data) - 1 - pos.open_idx
        T_remaining = max((pos.dte_at_open - days_in) / 252, 1/252)
        current_value = bs_price(S, pos.strike, T_remaining,
                                  config.risk_free_rate, sigma_last, pos.option_type)
        pos.close_date = last["date"]
        pos.close_idx = len(data) - 1
        pos.close_premium = current_value
        pos.close_reason = "end_of_data"
        pos.pnl = (pos.premium - current_value) * 100 * config.contracts_per_signal
        pos.days_held = days_in
        closed_positions.append(pos)
    
    # ── Compute results ──
    result = BacktestResult(config=config, positions=closed_positions)
    result.total_trades = len(closed_positions)
    
    if not closed_positions:
        return result
    
    pnls = [p.pnl for p in closed_positions]
    result.total_pnl = sum(pnls)
    result.total_premium_collected = sum(
        p.premium * 100 * config.contracts_per_signal for p in closed_positions
    )
    result.winning_trades = sum(1 for p in pnls if p > 0)
    result.losing_trades = sum(1 for p in pnls if p <= 0)
    result.win_rate = result.winning_trades / result.total_trades * 100
    
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    result.avg_win = np.mean(wins) if wins else 0
    result.avg_loss = np.mean(losses) if losses else 0
    result.profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    
    # By type
    for p in closed_positions:
        if p.option_type == "P":
            result.put_trades += 1
            result.put_pnl += p.pnl
        else:
            result.call_trades += 1
            result.call_pnl += p.pnl
    
    result.avg_days_held = np.mean([p.days_held for p in closed_positions])
    
    # Close reasons
    reasons = {}
    for p in closed_positions:
        r = p.close_reason
        reasons[r] = reasons.get(r, 0) + 1
    result.close_reasons = reasons
    
    # Equity curve metrics
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    result.max_drawdown = float(dd.min())
    
    # Sharpe on monthly P&L
    monthly_pnl = []
    month_pnl = 0
    current_month = data[30]["date"][:7]
    for i in range(30, len(data)):
        m = data[i]["date"][:7]
        if m != current_month:
            monthly_pnl.append(month_pnl)
            month_pnl = 0
            current_month = m
        day_pnl = equity_curve[i - 29] - (equity_curve[i - 30] if i > 30 else 0)
        month_pnl += day_pnl
    if month_pnl != 0:
        monthly_pnl.append(month_pnl)
    
    if monthly_pnl and np.std(monthly_pnl) > 0:
        result.sharpe_ratio = float(np.mean(monthly_pnl) / np.std(monthly_pnl) * np.sqrt(12))
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def print_report(result: BacktestResult, label: str = ""):
    r = result
    c = r.config
    
    print(f"\n{'='*72}")
    if label:
        print(f"  {label}")
    print(f"  Risk-Guided Options Strategy: Sell Put (risk<{c.risk_low_threshold}) / Sell Call (risk>{c.risk_high_threshold})")
    print(f"  Underlying: {c.ticker}  |  Delta: {c.target_delta}  |  DTE: {c.target_dte}")
    print(f"  Profit target: {c.profit_target_pct:.0%}  |  Stop: {c.stop_loss_pct:.0%}  |  Roll: {c.roll_dte} DTE")
    print(f"{'='*72}")
    
    if r.total_trades == 0:
        print("  No trades executed.")
        return
    
    print(f"\n  PERFORMANCE SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Total P&L:          ${r.total_pnl:>10,.2f}")
    print(f"  Premium collected:  ${r.total_premium_collected:>10,.2f}")
    print(f"  P&L / Premium:      {r.total_pnl/r.total_premium_collected*100:>9.1f}%")
    print(f"  Max drawdown:       ${r.max_drawdown:>10,.2f}")
    print(f"  Sharpe ratio:       {r.sharpe_ratio:>10.2f}")
    
    print(f"\n  TRADE STATISTICS")
    print(f"  {'─'*50}")
    print(f"  Total trades:       {r.total_trades:>10}")
    print(f"  Win / Loss:         {r.winning_trades:>4} / {r.losing_trades:<4}  ({r.win_rate:.1f}%)")
    print(f"  Avg win:            ${r.avg_win:>10,.2f}")
    print(f"  Avg loss:           ${r.avg_loss:>10,.2f}")
    print(f"  Profit factor:      {r.profit_factor:>10.2f}")
    print(f"  Avg days held:      {r.avg_days_held:>10.1f}")
    
    print(f"\n  BY OPTION TYPE")
    print(f"  {'─'*50}")
    if r.put_trades > 0:
        put_wr = sum(1 for p in r.positions if p.option_type == "P" and p.pnl > 0) / r.put_trades * 100
        print(f"  Sell Put:   {r.put_trades:>4} trades  P&L=${r.put_pnl:>10,.2f}  WR={put_wr:.0f}%")
    if r.call_trades > 0:
        call_wr = sum(1 for p in r.positions if p.option_type == "C" and p.pnl > 0) / r.call_trades * 100
        print(f"  Sell Call:  {r.call_trades:>4} trades  P&L=${r.call_pnl:>10,.2f}  WR={call_wr:.0f}%")
    
    print(f"\n  CLOSE REASONS")
    print(f"  {'─'*50}")
    for reason, count in sorted(r.close_reasons.items(), key=lambda x: -x[1]):
        pct = count / r.total_trades * 100
        rpnl = sum(p.pnl for p in r.positions if p.close_reason == reason)
        print(f"  {reason:<20} {count:>4} ({pct:>5.1f}%)  P&L=${rpnl:>10,.2f}")
    
    # Show worst trades
    worst = sorted(r.positions, key=lambda p: p.pnl)[:5]
    print(f"\n  WORST 5 TRADES")
    print(f"  {'─'*50}")
    for p in worst:
        print(f"  {p.open_date} {p.option_type} K={p.strike:>7.2f} "
              f"risk={p.risk_score_at_open:.0f} "
              f"P&L=${p.pnl:>8,.2f} ({p.close_reason})")
    
    # Show best trades
    best = sorted(r.positions, key=lambda p: -p.pnl)[:5]
    print(f"\n  BEST 5 TRADES")
    print(f"  {'─'*50}")
    for p in best:
        print(f"  {p.open_date} {p.option_type} K={p.strike:>7.2f} "
              f"risk={p.risk_score_at_open:.0f} "
              f"P&L=${p.pnl:>8,.2f} ({p.close_reason})")


def print_annual_breakdown(result: BacktestResult):
    """Print P&L by year."""
    from collections import defaultdict
    yearly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "puts": 0, "calls": 0})
    for p in result.positions:
        yr = p.open_date[:4]
        yearly[yr]["pnl"] += p.pnl
        yearly[yr]["trades"] += 1
        if p.pnl > 0:
            yearly[yr]["wins"] += 1
        if p.option_type == "P":
            yearly[yr]["puts"] += 1
        else:
            yearly[yr]["calls"] += 1
    
    print(f"\n  ANNUAL BREAKDOWN")
    print(f"  {'─'*65}")
    print(f"  {'Year':<6} {'Trades':>7} {'Puts':>5} {'Calls':>6} {'WinRate':>8} {'P&L':>12}")
    print(f"  {'─'*65}")
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        wr = y["wins"] / y["trades"] * 100 if y["trades"] > 0 else 0
        print(f"  {yr:<6} {y['trades']:>7} {y['puts']:>5} {y['calls']:>6} "
              f"{wr:>7.0f}% ${y['pnl']:>11,.2f}")
    print(f"  {'─'*65}")
    total = sum(y["pnl"] for y in yearly.values())
    print(f"  {'TOTAL':<6} {result.total_trades:>7} {result.put_trades:>5} {result.call_trades:>6} "
          f"{result.win_rate:>7.0f}% ${total:>11,.2f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  RISK-GUIDED OPTIONS STRATEGY BACKTEST")
    print("  Bubble Risk Score → Sell Put / Sell Call on TQQQ")
    print("=" * 72)
    
    data = load_data()
    print(f"\n  Data: {len(data)} days, {data[0]['date']} to {data[-1]['date']}")
    print(f"  TQQQ: ${data[0]['price']:.2f} → ${data[-1]['price']:.2f}")
    
    # Risk score stats
    scores = [d["risk_score"] for d in data]
    print(f"  Risk score: mean={np.mean(scores):.1f}, P20={np.percentile(scores,20):.1f}, "
          f"P80={np.percentile(scores,80):.1f}")
    
    low_days = sum(1 for s in scores if s < 40)
    high_days = sum(1 for s in scores if s > 60)
    print(f"  Signal days: Low(<40)={low_days} ({low_days/len(scores)*100:.0f}%), "
          f"High(>60)={high_days} ({high_days/len(scores)*100:.0f}%)")
    
    configs = [
        # ── A: Main strategy ──
        (StrategyConfig(), "A: Default (risk<40 put, >60 call)"),
        
        # ── B: More conservative thresholds ──
        (StrategyConfig(risk_low_threshold=30, risk_high_threshold=70),
         "B: Conservative (risk<30 put, >70 call)"),
        
        # ── C: Aggressive thresholds ──
        (StrategyConfig(risk_low_threshold=45, risk_high_threshold=55),
         "C: Aggressive (risk<45 put, >55 call)"),
        
        # ── D: Put-only (risk < 40) ──
        (StrategyConfig(risk_high_threshold=999),
         "D: Put-only (risk<40, never sell call)"),
        
        # ── E: Call-only (risk > 60) ──
        (StrategyConfig(risk_low_threshold=-999),
         "E: Call-only (risk>60, never sell put)"),
        
        # ── F: Wider delta (more OTM, safer) ──
        (StrategyConfig(target_delta=0.15),
         "F: Wide delta=0.15 (more OTM)"),
        
        # ── G: Tighter delta (more ATM, more premium) ──
        (StrategyConfig(target_delta=0.30),
         "G: Tight delta=0.30 (closer ATM)"),
        
        # ── H: Shorter DTE ──
        (StrategyConfig(target_dte=21),
         "H: Short DTE=21"),
        
        # ── I: Longer DTE ──
        (StrategyConfig(target_dte=45),
         "I: Long DTE=45"),
        
        # ── J: Tighter stop ──
        (StrategyConfig(stop_loss_pct=1.0),
         "J: Tight stop=100%"),
        
        # ── K: No stop ──
        (StrategyConfig(stop_loss_pct=10.0),
         "K: Loose stop=1000%"),
        
        # ── L: Best combo hypothesis ──
        (StrategyConfig(
            risk_low_threshold=35, risk_high_threshold=65,
            target_delta=0.20, target_dte=35,
            profit_target_pct=0.50, stop_loss_pct=2.0,
            max_open_positions=2, min_days_between_opens=10),
         "L: Tuned combo"),
    ]
    
    all_results = []
    for cfg, label in configs:
        result = run_backtest(data, cfg)
        print_report(result, label)
        print_annual_breakdown(result)
        all_results.append((label, result))
    
    # ═══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*72}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*72}\n")
    print(f"  {'Strategy':<45} {'P&L':>10} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD':>10} {'Trades':>7}")
    print(f"  {'─'*91}")
    
    all_results.sort(key=lambda x: x[1].total_pnl if x[1].total_trades > 0 else -999999,
                     reverse=True)
    
    for label, r in all_results:
        if r.total_trades == 0:
            print(f"  {label:<45} {'N/A':>10}")
            continue
        print(f"  {label:<45} ${r.total_pnl:>9,.0f} {r.win_rate:>5.0f}% "
              f"{r.profit_factor:>5.1f} {r.sharpe_ratio:>6.2f} "
              f"${r.max_drawdown:>9,.0f} {r.total_trades:>7}")
    
    # Save results
    out_path = Path(__file__).resolve().parent / "options_backtest_results.json"
    output = {}
    for label, r in all_results:
        output[label] = {
            "total_pnl": round(r.total_pnl, 2),
            "win_rate": round(r.win_rate, 1),
            "profit_factor": round(r.profit_factor, 2),
            "sharpe_ratio": round(r.sharpe_ratio, 2),
            "max_drawdown": round(r.max_drawdown, 2),
            "total_trades": r.total_trades,
            "put_trades": r.put_trades,
            "call_trades": r.call_trades,
            "put_pnl": round(r.put_pnl, 2),
            "call_pnl": round(r.call_pnl, 2),
            "avg_days_held": round(r.avg_days_held, 1),
            "close_reasons": r.close_reasons,
        }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {out_path}")
    
    # Current signal
    latest = data[-1]
    print(f"\n{'='*72}")
    print(f"  CURRENT SIGNAL ({latest['date']})")
    print(f"{'='*72}")
    print(f"  Risk Score: {latest['risk_score']:.1f}")
    print(f"  TQQQ Price: ${latest['price']:.2f}")
    risk = latest["risk_score"]
    if risk < 40:
        print(f"  Signal: SELL PUT (risk {risk:.0f} < 40)")
        S = latest["price"]
        sigma = 0.60  # approximate
        T = 35 / 252
        K = find_strike_by_delta(S, T, 0.045, sigma, 0.25, "P")
        prem = bs_price(S, K, T, 0.045, sigma, "P")
        print(f"  Suggested: Sell TQQQ {K:.0f}P ~35 DTE, premium ~${prem:.2f}/sh")
    elif risk > 60:
        print(f"  Signal: SELL CALL (risk {risk:.0f} > 60)")
        S = latest["price"]
        sigma = 0.60
        T = 35 / 252
        K = find_strike_by_delta(S, T, 0.045, sigma, 0.25, "C")
        prem = bs_price(S, K, T, 0.045, sigma, "C")
        print(f"  Suggested: Sell TQQQ {K:.0f}C ~35 DTE, premium ~${prem:.2f}/sh")
    else:
        print(f"  Signal: NEUTRAL (risk {risk:.0f}, hold existing positions)")
    
    print(f"\n{'='*72}")


if __name__ == "__main__":
    main()
