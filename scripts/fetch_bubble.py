"""Fetch bubble index indicators and generate static JSON files for the dashboard.

Standalone script -- no clawdfolio dependency.
Uses yfinance for market data and fredapi for FRED data (optional).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance", file=sys.stderr)
    sys.exit(1)

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDICATOR_CONFIG = {
    "qqq_deviation": {
        "label": "QQQ Deviation",
        "weight": 0.20,
        "lookback": 200,
    },
    "vix_level": {
        "label": "VIX Level",
        "weight": 0.18,
        "lookback": 252,
    },
    "sector_breadth": {
        "label": "Sector Breadth",
        "weight": 0.15,
        "lookback": 50,
    },
    "credit_spread": {
        "label": "Credit Spread",
        "weight": 0.15,
        "lookback": 252,
    },
    "put_call_ratio": {
        "label": "Put/Call Ratio (SKEW)",
        "weight": 0.17,
        "lookback": 252,
        "source": "yfinance",
        "ticker": "^SKEW",
    },
    "yield_curve": {
        "label": "Yield Curve",
        "weight": 0.15,
        "lookback": 252,
        "source": "fred",
        "series_id": "T10Y2Y",
    },
}

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]

REGIME_THRESHOLDS = {
    "LOW": 30,
    "MODERATE": 50,
    "ELEVATED": 70,
    "HIGH": 85,
}


def get_regime(score: float) -> str:
    if score < REGIME_THRESHOLDS["LOW"]:
        return "LOW"
    if score < REGIME_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    if score < REGIME_THRESHOLDS["ELEVATED"]:
        return "ELEVATED"
    if score < REGIME_THRESHOLDS["HIGH"]:
        return "HIGH"
    return "EXTREME"


# ---------------------------------------------------------------------------
# Percentile ranking helper
# ---------------------------------------------------------------------------

def percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling percentile rank (0-100) of each value within its lookback window."""
    def _rank(window):
        if len(window) < 2:
            return 50.0
        val = window.iloc[-1]
        return float((window < val).sum()) / (len(window) - 1) * 100

    return series.rolling(window=lookback, min_periods=max(20, lookback // 4)).apply(_rank, raw=False)


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def compute_qqq_deviation(lookback: int = 200) -> pd.Series:
    """QQQ deviation from 200-day SMA, percentile-ranked."""
    data = yf.download("QQQ", period="3y", progress=False)
    closes = data["Close"].squeeze().dropna()
    sma = closes.rolling(window=lookback).mean()
    deviation = (closes - sma) / sma
    return percentile_rank(deviation, lookback)


def compute_vix_level(lookback: int = 252) -> pd.Series:
    """VIX level inverted (high VIX = high bubble risk when inverted means complacency).
    Actually: low VIX = complacency = higher bubble risk. So we invert."""
    data = yf.download("^VIX", period="3y", progress=False)
    vix = data["Close"].squeeze().dropna()
    # Invert: low VIX -> high score (complacency / bubble-like)
    inverted = -vix
    return percentile_rank(inverted, lookback)


def compute_sector_breadth(lookback: int = 50) -> pd.Series:
    """Fraction of sector ETFs above their 50-day SMA, percentile-ranked."""
    tickers = SECTOR_ETFS
    data = yf.download(tickers, period="2y", progress=False)
    closes = data["Close"]

    # Count how many sectors are above their own SMA
    above_sma = pd.DataFrame()
    for col in closes.columns:
        sma = closes[col].rolling(window=lookback).mean()
        above_sma[col] = (closes[col] > sma).astype(float)

    breadth = above_sma.mean(axis=1)
    # High breadth = more euphoric = higher bubble score
    return percentile_rank(breadth, lookback)


def compute_credit_spread(lookback: int = 252) -> pd.Series:
    """HYG/IEF ratio as credit spread proxy. Tight spreads (high ratio) = risk-on = higher bubble score."""
    data = yf.download(["HYG", "IEF"], period="3y", progress=False)
    closes = data["Close"]
    ratio = (closes["HYG"] / closes["IEF"]).dropna()
    return percentile_rank(ratio, lookback)


def compute_put_call_ratio(lookback: int = 252) -> pd.Series | None:
    """CBOE SKEW index as sentiment proxy (replaces discontinued FRED PCCE).
    High SKEW = heavy tail-risk hedging = complacency/bubble signal."""
    try:
        data = yf.download("^SKEW", period="3y", progress=False)
        skew = data["Close"].squeeze().dropna()
        if skew.empty:
            return None
        # High SKEW -> higher bubble score (no inversion needed)
        return percentile_rank(skew, lookback)
    except Exception as e:
        print(f"  SKEW error: {e}", file=sys.stderr)
        return None


def compute_yield_curve(fred: "Fred | None", lookback: int = 252) -> pd.Series | None:
    """10Y-2Y spread from FRED. Positive/steepening = risk-on = higher bubble score."""
    if fred is None:
        return None
    try:
        spread = fred.get_series("T10Y2Y", observation_start="2023-01-01")
        spread = spread.dropna()
        if spread.empty:
            return None
        return percentile_rank(spread, lookback)
    except Exception as e:
        print(f"  FRED T10Y2Y error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

COMPUTE_FNS = {
    "qqq_deviation": lambda fred: compute_qqq_deviation(INDICATOR_CONFIG["qqq_deviation"]["lookback"]),
    "vix_level": lambda fred: compute_vix_level(INDICATOR_CONFIG["vix_level"]["lookback"]),
    "sector_breadth": lambda fred: compute_sector_breadth(INDICATOR_CONFIG["sector_breadth"]["lookback"]),
    "credit_spread": lambda fred: compute_credit_spread(INDICATOR_CONFIG["credit_spread"]["lookback"]),
    "put_call_ratio": lambda fred: compute_put_call_ratio(INDICATOR_CONFIG["put_call_ratio"]["lookback"]),
    "yield_curve": lambda fred: compute_yield_curve(fred, INDICATOR_CONFIG["yield_curve"]["lookback"]),
}


def build_bubble_index():
    """Compute all indicators, combine into composite score, and write JSON."""

    # Initialize FRED if key is available
    fred = None
    fred_api_key = os.environ.get("FRED_API_KEY", "")
    if fred_api_key and FRED_AVAILABLE:
        try:
            fred = Fred(api_key=fred_api_key)
            print("FRED API initialized.")
        except Exception as e:
            print(f"FRED init failed: {e}", file=sys.stderr)
    elif not FRED_AVAILABLE:
        print("fredapi not installed -- skipping FRED indicators.")
    else:
        print("FRED_API_KEY not set -- skipping FRED indicators.")

    # Compute each indicator
    indicator_series: dict[str, pd.Series] = {}
    for name, fn in COMPUTE_FNS.items():
        print(f"Computing {name}...")
        try:
            result = fn(fred)
            if result is not None and not result.dropna().empty:
                indicator_series[name] = result.dropna()
                print(f"  OK: {len(indicator_series[name])} points")
            else:
                print(f"  SKIPPED (no data)")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    if not indicator_series:
        print("ERROR: No indicators computed successfully.", file=sys.stderr)
        sys.exit(1)

    # Align all series to common dates
    combined = pd.DataFrame(indicator_series)
    combined = combined.dropna(how="all")

    # Re-normalize weights for available indicators only
    available = [k for k in combined.columns]
    total_weight = sum(INDICATOR_CONFIG[k]["weight"] for k in available)
    weights = {k: INDICATOR_CONFIG[k]["weight"] / total_weight for k in available}

    # Compute weighted composite score
    composite = pd.Series(0.0, index=combined.index)
    for col in available:
        composite += combined[col].fillna(50.0) * weights[col]

    # Categorize into sentiment vs liquidity sub-scores
    sentiment_keys = {"qqq_deviation", "vix_level", "put_call_ratio"}
    liquidity_keys = {"sector_breadth", "credit_spread", "yield_curve"}

    def sub_score(keys):
        cols = [k for k in keys if k in combined.columns]
        if not cols:
            return composite * np.nan
        w = {k: INDICATOR_CONFIG[k]["weight"] for k in cols}
        tw = sum(w.values())
        s = pd.Series(0.0, index=combined.index)
        for c in cols:
            s += combined[c].fillna(50.0) * (w[c] / tw)
        return s

    sentiment = sub_score(sentiment_keys)
    liquidity = sub_score(liquidity_keys)

    # Build snapshot (latest values)
    latest_idx = combined.index[-1]
    snapshot_indicators = {}
    for name in available:
        score_val = float(combined[name].iloc[-1])
        cfg = INDICATOR_CONFIG[name]
        snapshot_indicators[name] = {
            "score": round(score_val, 1),
            "raw_value": round(score_val / 100.0, 4),  # normalized 0-1
            "weight": round(weights[name], 4),
            "label": cfg["label"],
        }

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "composite_score": round(float(composite.iloc[-1]), 1),
        "sentiment_score": round(float(sentiment.iloc[-1]), 1) if not np.isnan(sentiment.iloc[-1]) else None,
        "liquidity_score": round(float(liquidity.iloc[-1]), 1) if not np.isnan(liquidity.iloc[-1]) else None,
        "regime": get_regime(float(composite.iloc[-1])),
        "indicators": snapshot_indicators,
    }

    # Build previous_day data (second-to-last entry) for trend arrows
    if len(composite) >= 2:
        prev_indicators = {}
        for name in available:
            val = combined[name].iloc[-2]
            prev_indicators[name] = round(float(val), 1) if not np.isnan(val) else None
        snapshot["previous_day"] = {
            "composite_score": round(float(composite.iloc[-2]), 1),
            "indicators": prev_indicators,
        }

    # Build history (last 365 trading days)
    history_points = []
    for i in range(max(0, len(composite) - 365), len(composite)):
        idx = composite.index[i]
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        c_val = float(composite.iloc[i])
        s_val = float(sentiment.iloc[i]) if not np.isnan(sentiment.iloc[i]) else None
        l_val = float(liquidity.iloc[i]) if not np.isnan(liquidity.iloc[i]) else None

        # Per-indicator scores for this day
        day_indicators = {}
        for name in available:
            val = combined[name].iloc[i]
            day_indicators[name] = round(float(val), 1) if not np.isnan(val) else None

        history_points.append({
            "date": date_str,
            "composite_score": round(c_val, 1),
            "sentiment_score": round(s_val, 1) if s_val is not None else None,
            "liquidity_score": round(l_val, 1) if l_val is not None else None,
            "regime": get_regime(c_val),
            "indicators": day_indicators,
        })

    history = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history": history_points,
    }

    # Write output files
    out_dir = Path(__file__).resolve().parent.parent / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_dir / "bubble_index.json"
    snapshot_path.write_text(json.dumps(snapshot, separators=(",", ":")))
    print(f"\nSnapshot written to {snapshot_path}")
    print(f"  Composite: {snapshot['composite_score']} | Regime: {snapshot['regime']}")

    history_path = out_dir / "bubble_history.json"
    history_path.write_text(json.dumps(history, separators=(",", ":")))
    print(f"History written to {history_path} ({len(history_points)} points)")


if __name__ == "__main__":
    build_bubble_index()
