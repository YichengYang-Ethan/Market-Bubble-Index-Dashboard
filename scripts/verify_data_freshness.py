"""Verify all data JSON files are fresh (run after update-data workflow steps)."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

# Max age for generated_at (run in same workflow, files just written)
MAX_AGE_HOURS = 3
# For date-only (e.g. calibration_date), allow 36h (workflow runs daily)
MAX_AGE_HOURS_DATE_ONLY = 36

# Files that must have generated_at
REQUIRED_FILES = [
    "bubble_index.json",
    "bubble_history.json",
    "qqq.json",
    "spy.json",
    "tqqq.json",
    "iwm.json",
    "backtest_results.json",
    "gsadf_results.json",
    "markov_regimes.json",
    "drawdown_model.json",  # uses calibration_date
]


def get_generated_at(path: Path) -> str | None:
    """Extract generated_at or calibration_date from JSON."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        return data.get("generated_at") or data.get("calibration_date")
    return None


def get_last_data_date(path: Path) -> str | None:
    """Get last date in history or data array."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        if "history" in data and data["history"]:
            return data["history"][-1].get("date")
        if "data" in data and data["data"]:
            return data["data"][-1].get("date")
        dq = data.get("data_quality", {}).get("data_end_dates", {})
        if dq:
            return max(dq.values())
    return None


def main() -> int:
    now = datetime.now(timezone.utc)
    failed = []

    for name in REQUIRED_FILES:
        path = DATA_DIR / name
        if not path.exists():
            failed.append(f"{name}: missing")
            continue

        ga = get_generated_at(path)
        if not ga:
            failed.append(f"{name}: no generated_at")
            continue

        try:
            # Parse ISO format (handle Z and +00:00)
            ga_parsed = ga.replace("Z", "+00:00")
            ts = datetime.fromisoformat(ga_parsed)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            delta = (now - ts).total_seconds() / 3600
            max_hours = MAX_AGE_HOURS_DATE_ONLY if len(ga) <= 10 else MAX_AGE_HOURS
            if delta > max_hours:
                failed.append(f"{name}: generated_at {ga[:19] if len(ga) > 19 else ga} is {delta:.1f}h old (> {max_hours}h)")
        except (ValueError, TypeError):
            failed.append(f"{name}: invalid generated_at format")

    # Check bubble_history last date is within 5 trading days (allow weekends/holidays)
    bh = DATA_DIR / "bubble_history.json"
    if bh.exists():
        last = get_last_data_date(bh)
        if last:
            try:
                last_dt = datetime.strptime(last, "%Y-%m-%d").date()
                days_old = (now.date() - last_dt).days
                if days_old > 5:
                    failed.append(f"bubble_history: last date {last} is {days_old} days old")
            except (ValueError, TypeError):
                pass

    # Check GSADF timeline is aligned with SPY timeline (allow at most 1-day lag)
    gsadf = DATA_DIR / "gsadf_results.json"
    spy = DATA_DIR / "spy.json"
    if gsadf.exists() and spy.exists():
        try:
            gsadf_data = json.loads(gsadf.read_text())
            spy_data = json.loads(spy.read_text())
            gsadf_last = gsadf_data.get("results", [])[-1].get("date") if gsadf_data.get("results") else None
            spy_last = spy_data.get("data", [])[-1].get("date") if spy_data.get("data") else None
            if gsadf_last and spy_last:
                gsadf_dt = datetime.strptime(gsadf_last, "%Y-%m-%d").date()
                spy_dt = datetime.strptime(spy_last, "%Y-%m-%d").date()
                if (spy_dt - gsadf_dt).days > 1:
                    failed.append(f"gsadf_results: last result date {gsadf_last} lags SPY {spy_last}")
        except Exception:
            failed.append("gsadf_results: failed to validate timeline alignment")

    if failed:
        for msg in failed:
            print(f"ERROR: {msg}", file=sys.stderr)
        print("\nData freshness check failed. Ensure update-data workflow ran successfully.", file=sys.stderr)
        return 1

    print("All data files are fresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
