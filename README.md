# Market Bubble Index Dashboard

[![Deploy](https://github.com/yichengyang-ethan/QQQ-200D-Deviation-Dashboard/actions/workflows/deploy.yml/badge.svg)](https://github.com/yichengyang-ethan/QQQ-200D-Deviation-Dashboard/actions/workflows/deploy.yml)
[![Update Data](https://github.com/yichengyang-ethan/QQQ-200D-Deviation-Dashboard/actions/workflows/update-data.yml/badge.svg)](https://github.com/yichengyang-ethan/QQQ-200D-Deviation-Dashboard/actions/workflows/update-data.yml)

**Live:** [yichengyang-ethan.github.io/QQQ-200D-Deviation-Dashboard](https://yichengyang-ethan.github.io/QQQ-200D-Deviation-Dashboard/)

A single-page financial dashboard that tracks market bubble risk through a composite index of **6 weighted indicators**. Designed as a portfolio showcase piece with professional data visualization.

## Indicators

| Indicator | Weight | Source | Signal |
|-----------|--------|--------|--------|
| QQQ Deviation | 20% | Yahoo Finance | 200-day SMA deviation, percentile-ranked |
| VIX Level | 18% | Yahoo Finance | Inverted VIX (low VIX = complacency) |
| Put/Call Ratio | 17% | FRED (PCCE) | Inverted equity put/call ratio |
| Sector Breadth | 15% | Yahoo Finance | Fraction of sectors above 50-day SMA |
| Credit Spread | 15% | Yahoo Finance | HYG/IEF ratio (tight spreads = risk-on) |
| Yield Curve | 15% | FRED (T10Y2Y) | 10Y-2Y spread (steepening = risk-on) |

## Regime Classification

| Regime | Score Range | Interpretation |
|--------|-------------|----------------|
| Low | 0–30 | Depressed sentiment, accumulation zone |
| Moderate | 30–50 | Normal conditions, balanced risk |
| Elevated | 50–70 | Rising euphoria, tighten risk mgmt |
| High | 70–85 | Frothy, consider reducing exposure |
| Extreme | 85–100 | Bubble territory, high correction risk |

## Dashboard Sections

- **Hero** — Large composite gauge with regime badge and sub-scores
- **Indicator Grid** — 2x3 cards with score, trend arrow, and 30-day sparkline
- **Composite History** — Time series with toggleable per-indicator overlays
- **Indicator Deep Dive** — Individual accordion charts for all 6 indicators
- **QQQ Deviation Tracker** — Detailed deviation analysis with backtesting
- **Methodology** — Regime guide and indicator explanations

## Tech Stack

- React 19 + TypeScript
- Recharts 3.6 for data visualization
- Tailwind CSS for styling
- Vite for build tooling
- GitHub Actions for daily data updates + deployment

## Data Pipeline

Data is refreshed daily via GitHub Actions at 9:30 PM UTC on weekdays:

1. `scripts/fetch_qqq_data.py` — Fetches QQQ/SPY/TQQQ/IWM deviation data from Yahoo Finance
2. `scripts/fetch_bubble.py` — Computes all 6 bubble indicators and generates composite index

Set `FRED_API_KEY` as a repository secret to enable put/call ratio and yield curve indicators.

## Quick Start

```bash
pnpm install
pnpm run dev      # Start dev server on port 3000
pnpm run check    # TypeScript type checking
pnpm run test     # Run tests
pnpm run build    # Production build
```

## License

MIT
