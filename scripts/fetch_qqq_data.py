"""Fetch real QQQ data via yfinance and generate a static JSON file for the dashboard."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance", file=sys.stderr)
    sys.exit(1)


def main():
    # Fetch ~3 years of daily QQQ data (need 200+ extra days for SMA warm-up)
    ticker = yf.Ticker("QQQ")
    hist = ticker.history(period="3y", interval="1d")

    if hist.empty:
        print("ERROR: No data returned from yfinance", file=sys.stderr)
        sys.exit(1)

    closes = hist["Close"].dropna()
    if len(closes) < 200:
        print(f"ERROR: Only {len(closes)} data points, need at least 200", file=sys.stderr)
        sys.exit(1)

    # Calculate 200-day SMA
    sma200 = closes.rolling(window=200).mean()

    # Build data points (skip first 200 days where SMA is NaN)
    data_points = []
    for i in range(len(closes)):
        if sma200.iloc[i] != sma200.iloc[i]:  # NaN check
            continue
        price = float(closes.iloc[i])
        sma = float(sma200.iloc[i])
        raw_deviation = (price - sma) / sma

        # Normalize to 0-100 index: maps [-15%, +25%] -> [0, 100]
        index_value = ((raw_deviation + 0.15) / 0.40) * 100
        index_value = max(0.0, min(100.0, index_value))

        date_str = closes.index[i].strftime("%Y-%m-%d")
        data_points.append({
            "date": date_str,
            "price": round(price, 2),
            "sma200": round(sma, 2),
            "deviation": round(raw_deviation, 6),
            "index": round(index_value, 2),
        })

    if not data_points:
        print("ERROR: No valid data points generated", file=sys.stderr)
        sys.exit(1)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": "QQQ",
        "data": data_points,
    }

    # Write to public/data directory so it gets included in the build
    out_dir = Path(__file__).resolve().parent.parent / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qqq.json"
    out_path.write_text(json.dumps(output, separators=(",", ":")))

    print(f"OK: Wrote {len(data_points)} data points to {out_path}")
    print(f"    Date range: {data_points[0]['date']} to {data_points[-1]['date']}")
    print(f"    Latest price: ${data_points[-1]['price']}")
    print(f"    Latest index: {data_points[-1]['index']}")


if __name__ == "__main__":
    main()
