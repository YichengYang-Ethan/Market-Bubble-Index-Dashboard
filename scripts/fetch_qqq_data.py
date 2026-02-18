"""Fetch market data via yfinance and generate static JSON files for the dashboard."""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _sanitize(obj):
    """Recursively replace float NaN/Inf with None for valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance", file=sys.stderr)
    sys.exit(1)

# Deviation normalization ranges per ticker.
# Maps [-lower%, +upper%] -> [0, 100].
# TQQQ is 3x leveraged so its deviation range is much wider.
TICKER_CONFIG = {
    "QQQ":  {"lower": 0.15, "upper": 0.25},
    "SPY":  {"lower": 0.12, "upper": 0.20},
    "TQQQ": {"lower": 0.40, "upper": 0.65},
    "IWM":  {"lower": 0.15, "upper": 0.25},
}

DEFAULT_TICKERS = list(TICKER_CONFIG.keys())


def fetch_ticker(symbol: str) -> list[dict]:
    cfg = TICKER_CONFIG.get(symbol, {"lower": 0.15, "upper": 0.25})
    norm_range = cfg["lower"] + cfg["upper"]

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="10y", interval="1d")

    if hist.empty:
        print(f"WARNING: No data returned for {symbol}", file=sys.stderr)
        return []

    closes = hist["Close"].dropna()
    if len(closes) < 200:
        print(f"WARNING: Only {len(closes)} data points for {symbol}, need 200", file=sys.stderr)
        return []

    sma200 = closes.rolling(window=200).mean()

    data_points = []
    for i in range(len(closes)):
        if pd.isna(sma200.iloc[i]):
            continue
        price = float(closes.iloc[i])
        sma = float(sma200.iloc[i])
        raw_deviation = (price - sma) / sma

        index_value = ((raw_deviation + cfg["lower"]) / norm_range) * 100
        index_value = max(0.0, min(100.0, index_value))

        date_str = closes.index[i].strftime("%Y-%m-%d")
        data_points.append({
            "date": date_str,
            "price": round(price, 2),
            "sma200": round(sma, 2),
            "deviation": round(raw_deviation, 6),
            "index": round(index_value, 2),
        })

    return data_points


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    out_dir = Path(__file__).resolve().parent.parent / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for symbol in tickers:
        symbol = symbol.upper()
        print(f"Fetching {symbol}...")
        data_points = fetch_ticker(symbol)

        if not data_points:
            failed.append(symbol)
            continue

        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ticker": symbol,
            "data": data_points,
        }

        out_path = out_dir / f"{symbol.lower()}.json"
        out_path.write_text(json.dumps(_sanitize(output), separators=(",", ":"), allow_nan=False))

        print(f"  OK: {len(data_points)} points -> {out_path}")
        print(f"      Range: {data_points[0]['date']} to {data_points[-1]['date']}")
        print(f"      Latest: ${data_points[-1]['price']} (index {data_points[-1]['index']})")

    if failed:
        print(f"\nWARNING: Failed tickers: {', '.join(failed)}", file=sys.stderr)
        if len(failed) == len(tickers):
            sys.exit(1)


if __name__ == "__main__":
    main()
