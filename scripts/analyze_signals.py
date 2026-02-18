#!/usr/bin/env python3
"""
Analyze QQQ deviation index as a predictor of forward returns.

Buckets the deviation index into quintiles and computes forward return
distributions for each bucket, then runs Welch's t-test comparing the
bottom quintile (0-20) vs the top quintile (80-100).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "public" / "data" / "qqq.json"
OUTPUT_DIR = ROOT / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

QUINTILE_BINS = [0, 20, 40, 60, 80, 100]
QUINTILE_LABELS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
FORWARD_WINDOWS = {"5d": 5, "21d": 21, "63d": 63, "126d": 126}


def load_data() -> pd.DataFrame:
    with open(DATA_PATH) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["data"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    for label, days in FORWARD_WINDOWS.items():
        df[f"fwd_{label}"] = df["price"].shift(-days) / df["price"] - 1
    return df


def bucket_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    df["quintile"] = pd.cut(
        df["index"], bins=QUINTILE_BINS, labels=QUINTILE_LABELS, include_lowest=True
    )
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in QUINTILE_LABELS:
        subset = df[df["quintile"] == q]
        row = {"Quintile": q, "N": len(subset)}
        for label in FORWARD_WINDOWS:
            col = f"fwd_{label}"
            vals = subset[col].dropna()
            row[f"{label} Mean"] = vals.mean()
            row[f"{label} Median"] = vals.median()
            row[f"{label} Std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows)


def run_t_tests(df: pd.DataFrame) -> pd.DataFrame:
    bottom = df[df["quintile"] == "0-20"]
    top = df[df["quintile"] == "80-100"]
    rows = []
    for label in FORWARD_WINDOWS:
        col = f"fwd_{label}"
        a = bottom[col].dropna()
        b = top[col].dropna()
        if len(a) < 2 or len(b) < 2:
            rows.append({
                "Horizon": label,
                "Bottom Mean": a.mean() if len(a) else float("nan"),
                "Top Mean": b.mean() if len(b) else float("nan"),
                "t-stat": float("nan"),
                "p-value": float("nan"),
                "Significant": "N/A (insufficient data)",
            })
            continue
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        rows.append({
            "Horizon": label,
            "Bottom Mean": a.mean(),
            "Top Mean": b.mean(),
            "t-stat": t_stat,
            "p-value": p_val,
            "Significant": "Yes" if p_val < 0.05 else "No",
        })
    return pd.DataFrame(rows)


def plot_forward_returns(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "QQQ Forward Returns by Deviation Index Quintile", fontsize=15, fontweight="bold"
    )

    colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"]

    for ax, (label, days) in zip(axes.flat, FORWARD_WINDOWS.items()):
        col = f"fwd_{label}"
        data_by_q = []
        for q in QUINTILE_LABELS:
            vals = df[df["quintile"] == q][col].dropna()
            data_by_q.append(vals.values * 100)

        bp = ax.boxplot(
            data_by_q,
            tick_labels=QUINTILE_LABELS,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # overlay means
        means = [np.mean(d) if len(d) else 0 for d in data_by_q]
        ax.plot(range(1, 6), means, "ko-", markersize=5, label="Mean")

        ax.set_title(f"{label} Forward Return", fontsize=12)
        ax.set_xlabel("Deviation Index Quintile")
        ax.set_ylabel("Return (%)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "forward_returns.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def main():
    print("Loading QQQ data...")
    df = load_data()
    print(f"  {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")

    df = compute_forward_returns(df)
    df = bucket_quintiles(df)

    print("\n=== Forward Return Summary by Quintile ===\n")
    summary = build_summary(df)
    # Format for display
    fmt = summary.copy()
    for c in fmt.columns:
        if "Mean" in c or "Median" in c or "Std" in c:
            fmt[c] = fmt[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    print(fmt.to_string(index=False))

    print("\n=== Welch's t-test: Bottom Quintile (0-20) vs Top Quintile (80-100) ===\n")
    ttest = run_t_tests(df)
    fmt_t = ttest.copy()
    fmt_t["Bottom Mean"] = fmt_t["Bottom Mean"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    fmt_t["Top Mean"] = fmt_t["Top Mean"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    fmt_t["t-stat"] = fmt_t["t-stat"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    fmt_t["p-value"] = fmt_t["p-value"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    print(fmt_t.to_string(index=False))

    print("\nGenerating chart...")
    path = plot_forward_returns(df)
    print(f"  Saved to {path}")

    # Return summary and ttest for README generation
    return summary, ttest


if __name__ == "__main__":
    main()
