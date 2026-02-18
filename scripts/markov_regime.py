"""Markov Regime-Switching model (Hamilton 1989) for bubble regime classification.

Fits a 3-regime Markov-switching model to the bubble composite score series
and extracts smoothed probabilities and transition dynamics.
"""

import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy not installed. Run: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("pandas not installed. Run: pip install pandas", file=sys.stderr)
    sys.exit(1)

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except ImportError:
    print(
        "statsmodels not installed or too old. Run: pip install statsmodels>=0.13",
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


def main():
    data_dir = Path(__file__).resolve().parent.parent / "public" / "data"

    # Load bubble history
    with open(data_dir / "bubble_history.json") as f:
        history = json.load(f)["history"]

    dates = [h["date"] for h in history]
    scores = [h["composite_score"] for h in history]

    series = pd.Series(scores, index=pd.DatetimeIndex(dates), name="composite_score")
    series = series.dropna()

    # Suppress statsmodels convergence warnings (expected with EM estimation)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*regime transition probabilities.*")
    warnings.filterwarnings("ignore", message=".*date index.*frequency.*")

    print(f"Fitting Markov Regime-Switching model to {len(series)} observations...")

    # Fit 3-regime model with switching mean and variance
    # Try multiple starting parameters for robustness
    best_model = None
    best_llf = -np.inf

    for attempt in range(5):
        try:
            model = MarkovRegression(
                series,
                k_regimes=3,
                order=1,
                switching_variance=True,
            )
            result = model.fit(
                maxiter=200,
                disp=False,
                search_reps=20 if attempt == 0 else 5,
            )
            if result.llf > best_llf:
                best_llf = result.llf
                best_model = result
        except Exception as e:
            if attempt == 0:
                print(f"  Attempt {attempt + 1} failed: {e}", file=sys.stderr)
            continue

    if best_model is None:
        print("ERROR: Markov model failed to converge after 5 attempts.", file=sys.stderr)
        sys.exit(1)

    result = best_model
    print(f"  Model converged (log-likelihood: {result.llf:.1f})")

    # Extract smoothed probabilities
    smoothed_probs = result.smoothed_marginal_probabilities

    # Determine regime ordering by mean (lowest mean = normal, highest = bubble)
    regime_means = []
    for i in range(3):
        regime_mean = float(result.params[f"const[{i}]"])
        regime_means.append((i, regime_mean))
    regime_means.sort(key=lambda x: x[1])

    # Map: regime with lowest mean -> "normal", middle -> "elevated", highest -> "bubble"
    regime_map = {
        regime_means[0][0]: "normal",
        regime_means[1][0]: "elevated",
        regime_means[2][0]: "bubble",
    }
    labels = ["normal", "elevated", "bubble"]
    ordered_indices = [regime_means[0][0], regime_means[1][0], regime_means[2][0]]

    # Build results array
    # smoothed_probs is a DataFrame with columns 0, 1, 2
    results_list = []
    for i, date in enumerate(series.index):
        date_str = date.strftime("%Y-%m-%d")
        probs = {}
        for orig_idx, label in regime_map.items():
            probs[f"{label}_prob"] = round(float(smoothed_probs.iloc[i, orig_idx]), 4)
        results_list.append({"date": date_str, **probs})

    # Transition matrix (reordered to normal/elevated/bubble)
    raw_matrix = result.regime_transition
    # statsmodels stores as (k_regimes, k_regimes, 1) -- squeeze last dim
    if raw_matrix.ndim == 3:
        raw_matrix = raw_matrix[:, :, 0]
    # raw_matrix[j, i] = P(S_t=j | S_{t-1}=i)
    transition_matrix = []
    for from_regime in ordered_indices:
        row = []
        for to_regime in ordered_indices:
            row.append(round(float(raw_matrix[to_regime, from_regime]), 4))
        transition_matrix.append(row)

    # Regime means (reordered)
    ordered_means = [round(regime_means[i][1], 1) for i in range(3)]

    # Current regime
    last_probs = results_list[-1]
    current_probs = {
        "normal": last_probs["normal_prob"],
        "elevated": last_probs["elevated_prob"],
        "bubble": last_probs["bubble_prob"],
    }
    current_regime = max(current_probs, key=current_probs.get)
    current_regime_prob = current_probs[current_regime]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Markov Regime-Switching (Hamilton 1989)",
        "n_regimes": 3,
        "regime_labels": labels,
        "results": results_list,
        "transition_matrix": transition_matrix,
        "regime_means": ordered_means,
        "current_regime": current_regime,
        "current_regime_prob": round(current_regime_prob, 4),
    }

    out_path = data_dir / "markov_regimes.json"
    out_path.write_text(json.dumps(_sanitize(output), indent=2, allow_nan=False))

    print(f"\nMarkov Regime results written to {out_path}")
    print(f"  Regime means: normal={ordered_means[0]}, elevated={ordered_means[1]}, bubble={ordered_means[2]}")
    print(f"  Current regime: {current_regime} (prob={current_regime_prob:.2%})")
    print(f"  Transition matrix:")
    for i, label in enumerate(labels):
        print(f"    {label}: {transition_matrix[i]}")


if __name__ == "__main__":
    main()
