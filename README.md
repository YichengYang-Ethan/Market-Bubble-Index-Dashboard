# Market Bubble Index Dashboard

[![Deploy](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/deploy.yml/badge.svg)](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/deploy.yml)
[![Update Data](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/update-data.yml/badge.svg)](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/update-data.yml)
[![CI](https://github.com/YichengYang-Ethan/Market-Bubble-Index-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/YichengYang-Ethan/Market-Bubble-Index-Dashboard/actions/workflows/ci.yml)

**Live:** [yichengyang-ethan.github.io/Market-Bubble-Index-Dashboard](https://yichengyang-ethan.github.io/Market-Bubble-Index-Dashboard/)

A single-page financial dashboard that tracks market bubble risk through a composite index of **7 weighted indicators** across three dimensions — sentiment, liquidity, and valuation. Uses 10-year rolling data for robust percentile rankings. Designed as a portfolio showcase piece with professional data visualization.

## Indicators

| Indicator | Weight | Category | Source | Signal |
|-----------|--------|----------|--------|--------|
| QQQ Deviation | 17% | Sentiment | Yahoo Finance | 200-day SMA deviation, percentile-ranked (200-day lookback) |
| VIX Level | 15% | Sentiment | Yahoo Finance | Inverted VIX — low VIX = complacency (252-day lookback) |
| CAPE Valuation | 15% | Valuation | Yahoo Finance (^GSPC) | S&P 500 price / 10-year moving average, percentile-ranked |
| Tail Risk (SKEW) | 14% | Sentiment | Yahoo Finance (^SKEW) | CBOE SKEW index — high SKEW = complacency (252-day lookback) |
| Sector Breadth | 13% | Liquidity | Yahoo Finance | Fraction of 11 sector ETFs above 50-day SMA, percentile-ranked (50-day lookback) |
| Credit Spread | 13% | Liquidity | Yahoo Finance | HYG/IEF ratio — tight spreads = risk-on (252-day lookback) |
| Yield Curve | 13% | Liquidity | FRED (T10Y2Y) | 10Y-2Y Treasury spread — steepening = risk-on (252-day lookback) |

**Normalization:** Each indicator is converted to a 0-100 percentile rank within its rolling lookback window. Composite score = weighted average of all available indicators. If any indicator is unavailable, weights are automatically redistributed.

## Sub-Scores

| Sub-Score | Indicators | Interpretation |
|-----------|-----------|----------------|
| Sentiment | QQQ Deviation, VIX, SKEW | Market emotion and momentum |
| Liquidity | Sector Breadth, Credit Spread, Yield Curve | Risk appetite and market breadth |
| Valuation | CAPE Valuation | Fundamental expense level |

## Regime Classification

| Regime | Score Range | Interpretation |
|--------|-------------|----------------|
| Low | 0–30 | Depressed sentiment, accumulation zone |
| Moderate | 30–50 | Normal conditions, balanced risk |
| Elevated | 50–70 | Rising euphoria, tighten risk mgmt |
| High | 70–85 | Frothy, consider reducing exposure |
| Extreme | 85–100 | Bubble territory, high correction risk |

## Dashboard Sections

- **Hero** — Large composite gauge with regime badge and 3 sub-scores (sentiment / liquidity / valuation)
- **Indicator Grid** — 7 cards with score, trend arrow, and 30-day sparkline
- **Composite History** — Full history time series with toggleable per-indicator overlays
- **Indicator Deep Dive** — Individual accordion charts for all 7 indicators
- **QQQ Deviation Tracker** — Detailed deviation analysis with interactive backtesting
- **Methodology** — Regime guide, indicator weights, data sources, normalization and backtest methodology

## Tech Stack

- React 19 + TypeScript
- Recharts 3.6 for data visualization
- Tailwind CSS for styling
- Vite for build tooling
- GitHub Actions for daily data updates + deployment

## Data Pipeline

Data is refreshed daily via GitHub Actions at 9:30 PM UTC on weekdays (after US market close):

1. `scripts/fetch_qqq_data.py` — Fetches 10-year QQQ/SPY/TQQQ/IWM deviation data from Yahoo Finance
2. `scripts/fetch_bubble.py` — Computes all 7 bubble indicators and generates composite index with full history

Set `FRED_API_KEY` as a repository secret to enable the yield curve indicator. All other indicators use Yahoo Finance (no key needed).

## Quick Start

```bash
npm install
npm run dev      # Start dev server on port 3000
npm run check    # TypeScript type checking
npm run test     # Run tests
npm run build    # Production build
```

## License

MIT
