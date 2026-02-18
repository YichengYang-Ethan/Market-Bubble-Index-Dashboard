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
import math


def _sanitize(obj):
    """Recursively replace float NaN/Inf with None so json.dumps never emits
    invalid tokens like NaN or Infinity."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _dump_json(obj) -> str:
    """Serialize to compact JSON, guaranteed free of NaN/Infinity."""
    return json.dumps(_sanitize(obj), separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDICATOR_CONFIG = {
    "qqq_deviation": {
        "label": "QQQ Deviation",
        "weight": 0.17,
        "lookback": 200,
        "category": "sentiment",
    },
    "vix_level": {
        "label": "VIX Level",
        "weight": 0.15,
        "lookback": 252,
        "category": "sentiment",
    },
    "sector_breadth": {
        "label": "Sector Breadth",
        "weight": 0.13,
        "lookback": 50,
        "category": "liquidity",
    },
    "credit_spread": {
        "label": "Credit Spread",
        "weight": 0.13,
        "lookback": 252,
        "category": "liquidity",
    },
    "put_call_ratio": {
        "label": "Put/Call Ratio (SKEW)",
        "weight": 0.14,
        "lookback": 252,
        "source": "yfinance",
        "ticker": "^SKEW",
        "category": "sentiment",
    },
    "yield_curve": {
        "label": "Yield Curve",
        "weight": 0.13,
        "lookback": 252,
        "source": "fred",
        "series_id": "T10Y2Y",
        "category": "liquidity",
    },
    "cape_ratio": {
        "label": "CAPE Valuation",
        "weight": 0.15,
        "lookback": 252,
        "category": "valuation",
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
# Percentile ranking helpers
# ---------------------------------------------------------------------------

def percentile_rank_rolling(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling percentile rank (0-100) of each value within its lookback window."""
    def _rank(window):
        if len(window) < 2:
            return 50.0
        val = window.iloc[-1]
        return float((window < val).sum()) / (len(window) - 1) * 100

    return series.rolling(window=lookback, min_periods=max(20, lookback // 4)).apply(_rank, raw=False)


def percentile_rank_hybrid(series: pd.Series, lookback: int, expanding_weight: float = 0.3) -> pd.Series:
    """Hybrid percentile: 70% rolling + 30% expanding for more stable rankings."""
    rolling_pct = percentile_rank_rolling(series, lookback)
    expanding_pct = series.expanding(min_periods=max(252, lookback)).apply(
        lambda w: float((w < w.iloc[-1]).sum()) / (len(w) - 1) * 100 if len(w) > 1 else 50.0, raw=False
    )
    return (1 - expanding_weight) * rolling_pct + expanding_weight * expanding_pct


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def compute_qqq_deviation(lookback: int = 200) -> pd.Series:
    """QQQ deviation from 200-day SMA, percentile-ranked."""
    data = yf.download("QQQ", start="2014-01-01", progress=False)
    closes = data["Close"].squeeze().dropna()
    sma = closes.rolling(window=lookback).mean()
    deviation = (closes - sma) / sma
    return percentile_rank_hybrid(deviation, lookback)


def compute_vix_level(lookback: int = 252) -> pd.Series:
    """VIX level inverted (high VIX = high bubble risk when inverted means complacency).
    Actually: low VIX = complacency = higher bubble risk. So we invert."""
    data = yf.download("^VIX", start="2014-01-01", progress=False)
    vix = data["Close"].squeeze().dropna()
    # Invert: low VIX -> high score (complacency / bubble-like)
    inverted = -vix
    return percentile_rank_hybrid(inverted, lookback)


def compute_sector_breadth(lookback: int = 50) -> pd.Series:
    """Fraction of sector ETFs above their 50-day SMA, percentile-ranked."""
    tickers = SECTOR_ETFS
    data = yf.download(tickers, start="2014-01-01", progress=False)
    closes = data["Close"]

    # Count how many sectors are above their own SMA
    above_sma = pd.DataFrame()
    for col in closes.columns:
        sma = closes[col].rolling(window=lookback).mean()
        above_sma[col] = (closes[col] > sma).astype(float)

    breadth = above_sma.mean(axis=1)
    # High breadth = more euphoric = higher bubble score
    return percentile_rank_hybrid(breadth, lookback)


def compute_credit_spread(lookback: int = 252) -> pd.Series:
    """HYG/IEF ratio as credit spread proxy. Tight spreads (high ratio) = risk-on = higher bubble score."""
    data = yf.download(["HYG", "IEF"], start="2014-01-01", progress=False)
    closes = data["Close"]
    ratio = (closes["HYG"] / closes["IEF"]).dropna()
    return percentile_rank_hybrid(ratio, lookback)


def compute_put_call_ratio(lookback: int = 252) -> pd.Series | None:
    """CBOE SKEW index as sentiment proxy (replaces discontinued FRED PCCE).
    High SKEW = heavy tail-risk hedging = complacency/bubble signal."""
    try:
        data = yf.download("^SKEW", start="2014-01-01", progress=False)
        skew = data["Close"].squeeze().dropna()
        if skew.empty:
            return None
        # High SKEW -> higher bubble score (no inversion needed)
        return percentile_rank_hybrid(skew, lookback)
    except Exception as e:
        print(f"  SKEW error: {e}", file=sys.stderr)
        return None


def compute_yield_curve(fred: "Fred | None", lookback: int = 252) -> pd.Series | None:
    """10Y-2Y spread from FRED. Positive/steepening = risk-on = higher bubble score."""
    if fred is None:
        return None
    try:
        spread = fred.get_series("T10Y2Y", observation_start="2014-01-01")
        spread = spread.dropna()
        if spread.empty:
            return None
        return percentile_rank_hybrid(spread, lookback)
    except Exception as e:
        print(f"  FRED T10Y2Y error: {e}", file=sys.stderr)
        return None


def compute_cape_ratio(lookback: int = 252) -> pd.Series | None:
    """Approximate CAPE using S&P 500 price relative to 10-year moving average.
    Higher values = more expensive = more bubble-like."""
    try:
        spy = yf.download("^GSPC", start="2014-01-01", progress=False)["Close"].squeeze().dropna()
        if spy.empty:
            return None
        # Use 10-year (2520 trading day) moving average as long-term earnings proxy
        sma_long = spy.rolling(window=min(2520, len(spy) - 1), min_periods=252).mean()
        cape_proxy = spy / sma_long  # Price relative to long-term average
        cape_proxy = cape_proxy.dropna()
        if cape_proxy.empty:
            return None
        return percentile_rank_hybrid(cape_proxy, lookback)
    except Exception as e:
        print(f"  CAPE ratio error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(combined_row, weights_dict, n=1000):
    vals = np.array([combined_row.get(c, 50.0) for c in weights_dict])
    w = np.array(list(weights_dict.values()))
    scores = []
    for _ in range(n):
        perturbed = np.clip(vals + np.random.normal(0, 3, len(vals)), 0, 100)
        scores.append(float(np.dot(perturbed, w)))
    return round(float(np.percentile(scores, 2.5)), 1), round(float(np.percentile(scores, 97.5)), 1)


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
    "cape_ratio": lambda fred: compute_cape_ratio(INDICATOR_CONFIG["cape_ratio"]["lookback"]),
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

    # --- PCA Orthogonalization (#2) ---
    corr_matrix = combined.corr()
    orthogonalized = combined.copy()
    processed = set()
    for i, col1 in enumerate(combined.columns):
        for col2 in combined.columns[i+1:]:
            if abs(corr_matrix.loc[col1, col2]) > 0.7 and col2 not in processed:
                # Regress col2 on col1, keep residuals
                mask = combined[[col1, col2]].dropna().index
                if len(mask) > 50:
                    x = combined.loc[mask, col1].values
                    y = combined.loc[mask, col2].values
                    coeffs = np.polyfit(x, y, 1)
                    residuals = y - np.polyval(coeffs, x)
                    # Scale residuals back to 0-100
                    r_min, r_max = residuals.min(), residuals.max()
                    if r_max > r_min:
                        scaled = (residuals - r_min) / (r_max - r_min) * 100
                        orthogonalized.loc[mask, col2] = scaled
                    processed.add(col2)
    combined = orthogonalized

    # Re-normalize weights for available indicators only
    available = [k for k in combined.columns]
    total_weight = sum(INDICATOR_CONFIG[k]["weight"] for k in available)
    weights = {k: INDICATOR_CONFIG[k]["weight"] / total_weight for k in available}

    # Compute weighted composite score
    composite = pd.Series(0.0, index=combined.index)
    for col in available:
        composite += combined[col].fillna(50.0) * weights[col]

    # --- Score Velocity & Acceleration (#6) ---
    score_velocity = composite.diff(5)    # 5-day change
    score_acceleration = score_velocity.diff(5)

    # Categorize into sub-scores by category
    sentiment_keys = {k for k in available if INDICATOR_CONFIG[k].get("category") == "sentiment"}
    liquidity_keys = {k for k in available if INDICATOR_CONFIG[k].get("category") == "liquidity"}
    valuation_keys = {k for k in available if INDICATOR_CONFIG[k].get("category") == "valuation"}

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
    valuation = sub_score(valuation_keys)

    # Build snapshot from the latest date where ALL indicators have data.
    complete_rows = combined.dropna()
    if not complete_rows.empty:
        snap_row = complete_rows.iloc[-1]
        snap_idx = complete_rows.index[-1]
        snap_comp = float(composite.loc[snap_idx])
        snap_sent = float(sentiment.loc[snap_idx])
        snap_liq  = float(liquidity.loc[snap_idx])
        snap_val  = float(valuation.loc[snap_idx])
    else:
        snap_row = combined.iloc[-1]
        snap_idx = combined.index[-1]
        snap_comp = float(composite.iloc[-1])
        snap_sent = float(sentiment.iloc[-1])
        snap_liq  = float(liquidity.iloc[-1])
        snap_val  = float(valuation.iloc[-1])

    snapshot_indicators = {}
    for name in available:
        raw_val = snap_row[name]
        is_nan = np.isnan(raw_val) if not isinstance(raw_val, type(None)) else True
        cfg = INDICATOR_CONFIG[name]
        snapshot_indicators[name] = {
            "score": round(float(raw_val), 1) if not is_nan else None,
            "raw_value": round(float(raw_val) / 100.0, 4) if not is_nan else None,
            "weight": round(weights[name], 4),
            "label": cfg["label"],
        }

    # Velocity & acceleration for snapshot
    vel = float(score_velocity.loc[snap_idx]) if not np.isnan(score_velocity.loc[snap_idx]) else 0.0
    acc = float(score_acceleration.loc[snap_idx]) if not np.isnan(score_acceleration.loc[snap_idx]) else 0.0

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "composite_score": round(snap_comp, 1),
        "sentiment_score": round(snap_sent, 1) if not np.isnan(snap_sent) else None,
        "liquidity_score": round(snap_liq, 1) if not np.isnan(snap_liq) else None,
        "valuation_score": round(snap_val, 1) if not np.isnan(snap_val) else None,
        "regime": get_regime(snap_comp),
        "score_velocity": round(vel, 2),
        "score_acceleration": round(acc, 2),
        "indicators": snapshot_indicators,
    }

    # --- Bootstrap CI (#10) ---
    ci_lower, ci_upper = bootstrap_ci(snap_row.to_dict(), weights)
    snapshot["confidence_interval"] = {"lower": ci_lower, "upper": ci_upper}

    # --- Data Quality Monitoring (#7) ---
    snapshot["data_quality"] = {
        "indicators_available": len(available),
        "indicators_total": len(INDICATOR_CONFIG),
        "completeness": round(len(available) / len(INDICATOR_CONFIG) * 100, 1),
        "data_end_dates": {name: str(indicator_series[name].index[-1])[:10] for name in available},
        "staleness_warning": any(
            (datetime.now().date() - indicator_series[name].index[-1].date()).days > 3 for name in available
        ),
    }

    # --- Correlation Matrix & Sensitivity (#8, #9) ---
    corr = combined.tail(252).corr()
    snapshot["diagnostics"] = {
        "correlation_matrix": {
            row: {col: round(float(corr.loc[row, col]), 3) for col in corr.columns}
            for row in corr.index
        },
    }

    # Sensitivity analysis
    sensitivity = {}
    for drop_col in available:
        remaining = [c for c in available if c != drop_col]
        tw = sum(INDICATOR_CONFIG[c]["weight"] for c in remaining)
        alt_composite = sum(combined[c].fillna(50) * INDICATOR_CONFIG[c]["weight"] / tw for c in remaining)
        diff = float((alt_composite - composite).abs().mean())
        sensitivity[drop_col] = round(diff, 2)
    snapshot["diagnostics"]["sensitivity"] = sensitivity

    # Build previous_day data for trend arrows.
    if len(complete_rows) >= 2:
        prev_row = complete_rows.iloc[-2]
        prev_indicators = {}
        for name in available:
            val = prev_row[name]
            prev_indicators[name] = round(float(val), 1) if not np.isnan(val) else None
        prev_idx = complete_rows.index[-2]
        snapshot["previous_day"] = {
            "composite_score": round(float(composite.loc[prev_idx]), 1),
            "indicators": prev_indicators,
        }
    elif len(composite) >= 2:
        prev_indicators = {}
        for name in available:
            val = combined[name].iloc[-2]
            prev_indicators[name] = round(float(val), 1) if not np.isnan(val) else None
        snapshot["previous_day"] = {
            "composite_score": round(float(composite.iloc[-2]), 1),
            "indicators": prev_indicators,
        }

    # Build history (all available trading days)
    history_points = []
    for i in range(len(composite)):
        idx = composite.index[i]
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        c_val = float(composite.iloc[i])
        s_val = float(sentiment.iloc[i]) if not np.isnan(sentiment.iloc[i]) else None
        l_val = float(liquidity.iloc[i]) if not np.isnan(liquidity.iloc[i]) else None
        v_val = float(valuation.iloc[i]) if not np.isnan(valuation.iloc[i]) else None

        # Velocity & acceleration for this day
        vel_i = float(score_velocity.iloc[i]) if not np.isnan(score_velocity.iloc[i]) else None
        acc_i = float(score_acceleration.iloc[i]) if not np.isnan(score_acceleration.iloc[i]) else None

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
            "valuation_score": round(v_val, 1) if v_val is not None else None,
            "regime": get_regime(c_val),
            "score_velocity": round(vel_i, 2) if vel_i is not None else None,
            "score_acceleration": round(acc_i, 2) if acc_i is not None else None,
            "indicators": day_indicators,
        })

    # Trim early points with poor indicator coverage (< 3 is noise)
    MIN_INDICATORS = 3
    history_points = [
        pt for pt in history_points
        if sum(1 for v in pt["indicators"].values() if v is not None) >= MIN_INDICATORS
    ]

    history = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history": history_points,
    }

    # Write output files
    out_dir = Path(__file__).resolve().parent.parent / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_dir / "bubble_index.json"
    snapshot_path.write_text(_dump_json(snapshot))
    print(f"\nSnapshot written to {snapshot_path}")
    print(f"  Composite: {snapshot['composite_score']} | Regime: {snapshot['regime']}")
    print(f"  Velocity: {snapshot['score_velocity']} | Acceleration: {snapshot['score_acceleration']}")
    print(f"  CI: [{snapshot['confidence_interval']['lower']}, {snapshot['confidence_interval']['upper']}]")
    print(f"  Data quality: {snapshot['data_quality']['completeness']}% complete")

    history_path = out_dir / "bubble_history.json"
    history_path.write_text(_dump_json(history))
    print(f"History written to {history_path} ({len(history_points)} points)")


if __name__ == "__main__":
    build_bubble_index()
