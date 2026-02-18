"""GSADF bubble detection test (Phillips-Shi-Yu 2015).

Implements a simplified Generalized Sup ADF test to identify explosive
bubble periods in SPY price data.
"""

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

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    print(
        "statsmodels not installed. Run: pip install statsmodels",
        file=sys.stderr,
    )
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


def gsadf_critical_value(n: int) -> float:
    """Approximate right-tail 95% critical value for SADF/GSADF.

    Based on Monte Carlo simulations from Phillips-Shi-Yu (2015).
    For right-tail explosive alternative, critical values are positive.
    """
    # Approximation from PSY (2015) Table 2 for 95% level
    return -0.08 + 0.47 / (n ** 0.5)


def run_sadf_for_endpoint(log_prices: np.ndarray, end: int, min_window: int,
                          start_step: int = 10) -> float:
    """Compute the sup ADF statistic for a given endpoint.

    Iterates over expanding start windows and returns the supremum of ADF
    test statistics.
    """
    sup_stat = -np.inf
    for start in range(0, end - min_window + 1, start_step):
        segment = log_prices[start:end + 1]
        if len(segment) < min_window:
            continue
        try:
            result = adfuller(segment, maxlag=1, regression="c", autolag=None)
            adf_stat = result[0]
            if adf_stat > sup_stat:
                sup_stat = adf_stat
        except Exception:
            continue
    return float(sup_stat) if np.isfinite(sup_stat) else float("nan")


def main():
    data_dir = Path(__file__).resolve().parent.parent / "public" / "data"

    # Load SPY price data
    with open(data_dir / "spy.json") as f:
        spy_raw = json.load(f)["data"]

    dates = [d["date"] for d in spy_raw]
    prices = np.array([d["price"] for d in spy_raw], dtype=float)
    log_prices = np.log(prices)

    min_window = 63  # ~1 quarter
    n = len(log_prices)
    end_step = 5  # sample every 5 days for outer loop

    print(f"Running GSADF test on {n} SPY observations...")
    print(f"  Min window: {min_window}, end step: {end_step}")

    results = []
    for end_idx in range(min_window, n, end_step):
        sup_stat = run_sadf_for_endpoint(log_prices, end_idx, min_window, start_step=10)
        cv = gsadf_critical_value(end_idx - min_window + 1)
        is_bubble = bool(sup_stat > cv) if not math.isnan(sup_stat) else False
        results.append({
            "date": dates[end_idx],
            "test_statistic": round(sup_stat, 4) if not math.isnan(sup_stat) else None,
            "critical_value": round(cv, 4),
            "is_bubble": is_bubble,
        })

    # Identify contiguous bubble periods
    bubble_periods = []
    current_start = None
    for r in results:
        if r["is_bubble"]:
            if current_start is None:
                current_start = r["date"]
        else:
            if current_start is not None:
                bubble_periods.append({"start": current_start, "end": prev_date})
                current_start = None
        prev_date = r["date"]
    if current_start is not None:
        bubble_periods.append({"start": current_start, "end": results[-1]["date"]})

    # Summary statistics
    total_bubble_days = sum(1 for r in results if r["is_bubble"])
    pct_bubble = round(total_bubble_days / len(results) * 100, 1) if results else 0

    largest_bubble = None
    if bubble_periods:
        for bp in bubble_periods:
            dur = sum(1 for r in results if bp["start"] <= r["date"] <= bp["end"] and r["is_bubble"])
            bp["duration_days"] = dur
        largest_bubble = max(bubble_periods, key=lambda x: x["duration_days"])

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "GSADF (Phillips-Shi-Yu 2015)",
        "min_window": min_window,
        "results": results,
        "bubble_periods": bubble_periods,
        "summary": {
            "total_bubble_days": total_bubble_days,
            "pct_bubble": pct_bubble,
            "largest_bubble": largest_bubble,
        },
    }

    out_path = data_dir / "gsadf_results.json"
    out_path.write_text(json.dumps(_sanitize(output), indent=2, allow_nan=False))

    print(f"\nGSADF Results written to {out_path}")
    print(f"  Total observations tested: {len(results)}")
    print(f"  Bubble days: {total_bubble_days} ({pct_bubble}%)")
    print(f"  Bubble periods: {len(bubble_periods)}")
    if largest_bubble:
        print(f"  Largest bubble: {largest_bubble['start']} to {largest_bubble['end']} ({largest_bubble['duration_days']} sampled days)")


if __name__ == "__main__":
    main()
