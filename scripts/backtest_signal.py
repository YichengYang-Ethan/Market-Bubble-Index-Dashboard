"""Backtest the bubble composite score as a trading signal."""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy not installed. Run: pip install numpy", file=sys.stderr)
    sys.exit(1)


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


def main():
    data_dir = Path(__file__).resolve().parent.parent / "public" / "data"

    # Load bubble history
    with open(data_dir / "bubble_history.json") as f:
        history = json.load(f)["history"]

    # Load SPY price data
    with open(data_dir / "spy.json") as f:
        spy_data = {d["date"]: d["price"] for d in json.load(f)["data"]}

    # 1. Compute autocorrelation of composite score
    scores = [h["composite_score"] for h in history]
    autocorrelations = {}
    for lag in [1, 5, 10, 20]:
        if len(scores) > lag:
            s1 = np.array(scores[lag:])
            s2 = np.array(scores[:-lag])
            autocorrelations[f"lag_{lag}"] = round(float(np.corrcoef(s1, s2)[0, 1]), 4)

    # 2. Signal analysis
    # Buy signal: composite < 30, Sell signal: composite > 70
    buy_signals = []
    sell_signals = []
    horizons = [1, 5, 20, 60]

    for i, h in enumerate(history):
        date = h["date"]
        score = h["composite_score"]
        if date not in spy_data:
            continue

        # Get forward returns at each horizon
        forward_returns = {}
        for hz in horizons:
            if i + hz < len(history):
                future_date = history[i + hz]["date"]
                if future_date in spy_data:
                    ret = (spy_data[future_date] - spy_data[date]) / spy_data[date] * 100
                    forward_returns[f"{hz}d"] = round(ret, 4)

        if not forward_returns:
            continue

        if score < 30:
            buy_signals.append({"date": date, "score": score, "returns": forward_returns})
        elif score > 70:
            sell_signals.append({"date": date, "score": score, "returns": forward_returns})

    # 3. Compute statistics
    def compute_stats(signals, horizon_key):
        returns = [s["returns"][horizon_key] for s in signals if horizon_key in s["returns"]]
        if not returns:
            return None
        arr = np.array(returns)
        n = len(arr)
        mean = float(arr.mean())
        std = float(arr.std()) if n > 1 else 0
        t_stat = mean / (std / np.sqrt(n)) if std > 0 and n > 1 else 0
        return {
            "count": n,
            "mean_return": round(mean, 4),
            "median_return": round(float(np.median(arr)), 4),
            "std": round(std, 4),
            "hit_rate": round(float((arr > 0).sum() / n * 100), 1),
            "t_statistic": round(t_stat, 3),
        }

    buy_stats = {}
    sell_stats = {}
    for hz in horizons:
        key = f"{hz}d"
        buy_stats[key] = compute_stats(buy_signals, key)
        sell_stats[key] = compute_stats(sell_signals, key)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "autocorrelation": autocorrelations,
        "buy_signal_threshold": 30,
        "sell_signal_threshold": 70,
        "buy_signals": {"count": len(buy_signals), "stats": buy_stats},
        "sell_signals": {"count": len(sell_signals), "stats": sell_stats},
    }

    out_path = data_dir / "backtest_results.json"
    out_path.write_text(json.dumps(_sanitize(output), indent=2, allow_nan=False))

    # Print summary
    print(f"Backtest Results:")
    print(f"  Buy signals (<30): {len(buy_signals)}")
    print(f"  Sell signals (>70): {len(sell_signals)}")
    print(f"  Autocorrelation: {autocorrelations}")
    for hz in horizons:
        key = f"{hz}d"
        bs = buy_stats.get(key)
        ss = sell_stats.get(key)
        if bs:
            print(f"  Buy {key}: mean={bs['mean_return']:.2f}%, hit={bs['hit_rate']:.0f}%, t={bs['t_statistic']:.2f}")
        if ss:
            print(f"  Sell {key}: mean={ss['mean_return']:.2f}%, hit={ss['hit_rate']:.0f}%, t={ss['t_statistic']:.2f}")


if __name__ == "__main__":
    main()
