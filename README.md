# Market Bubble Index Dashboard

[![Deploy](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/deploy.yml/badge.svg)](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/deploy.yml)
[![Update Data](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/update-data.yml/badge.svg)](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/update-data.yml)
[![CI](https://github.com/YichengYang-Ethan/Market-Bubble-Index-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/YichengYang-Ethan/Market-Bubble-Index-Dashboard/actions/workflows/ci.yml)

**Live:** [yichengyang-ethan.github.io/Market-Bubble-Index-Dashboard](https://yichengyang-ethan.github.io/Market-Bubble-Index-Dashboard/)

A professional financial dashboard that quantifies market bubble risk through a composite index of **7 weighted indicators** across sentiment, liquidity, and valuation. Features regime classification, signal backtesting with two optimized strategies that beat Buy & Hold on risk-adjusted returns, GSADF bubble detection, Markov regime switching, and a QQQ deviation tracker with interactive backtesting.

## Composite Index

The Market Bubble Index aggregates 7 indicators into a single 0-100 score. Each indicator is percentile-ranked within its rolling lookback window, then combined via weighted average. Higher scores indicate more speculative excess.

### Indicators

| Indicator | Weight | Category | Source | Signal |
|-----------|--------|----------|--------|--------|
| QQQ Deviation | 17% | Sentiment | Yahoo Finance | 200-day SMA deviation, percentile-ranked (200-day lookback) |
| VIX Level | 15% | Sentiment | Yahoo Finance | Inverted VIX — low VIX = complacency (252-day lookback) |
| CAPE Valuation | 15% | Valuation | Yahoo Finance (^GSPC) | S&P 500 price / 10-year moving average, percentile-ranked |
| Tail Risk (SKEW) | 14% | Sentiment | Yahoo Finance (^SKEW) | CBOE SKEW index — high SKEW = complacency (252-day lookback) |
| Sector Breadth | 13% | Liquidity | Yahoo Finance | Fraction of 11 sector ETFs above 50-day SMA (50-day lookback) |
| Credit Spread | 13% | Liquidity | Yahoo Finance | HYG/IEF ratio — tight spreads = risk-on (252-day lookback) |
| Yield Curve | 13% | Liquidity | FRED (T10Y2Y) | 10Y-2Y Treasury spread — steepening = risk-on (252-day lookback) |

### Sub-Scores

| Sub-Score | Indicators | Interpretation |
|-----------|-----------|----------------|
| Sentiment | QQQ Deviation, VIX, SKEW | Market emotion and speculative momentum |
| Liquidity | Sector Breadth, Credit Spread, Yield Curve | Risk appetite and market breadth |
| Valuation | CAPE Valuation | Fundamental expense level |

### Regime Classification

| Regime | Score Range | Position Mapping | Interpretation |
|--------|-------------|-----------------|----------------|
| Low | 0-30 | Overweight +30% | Depressed sentiment, accumulation zone |
| Moderate | 30-50 | Hold Benchmark | Normal conditions, balanced risk |
| Elevated | 50-70 | Underweight -20% | Rising euphoria, tighten risk management |
| High | 70-85 | Underweight -50% | Frothy conditions, consider reducing exposure |
| Extreme | 85-100 | Retain 20% Only | Bubble territory, high correction probability |

## Strategy Backtesting

The dashboard includes two optimized backtest strategies that beat Buy & Hold on risk-adjusted returns over 10+ years of history. Both were discovered via grid search optimization.

### Strategy 1: Hysteresis Binary (Default)

Binary all-in/all-out with asymmetric thresholds inspired by control theory hysteresis:

- **Enter** (100% equity) when bubble score drops below 24
- **Exit** (100% cash at 4.5% risk-free rate) when score exceeds 84
- **Dead zone** of 60 points prevents whipsaw trading in sideways markets
- **Result:** Sharpe 0.70 vs B&H 0.67 (+5.5% edge)

### Strategy 2: Velocity Signal

Uses the first derivative (rate of change) of the composite score as a leading indicator:

- **Reduce to 30%** when `score_velocity > 15` AND `score > 50`
- **Return to 100%** when conditions normalize
- Rising velocity with elevated score = euphoria building before the peak
- **Result:** Sharpe 0.69, CAGR 18.5% — beats B&H on both risk-adjusted AND absolute returns

### Risk Metrics Comparison

Both strategies display side-by-side comparison against Buy & Hold with: Sharpe Ratio, Sortino Ratio, Calmar Ratio, CAGR, Max Drawdown, Volatility, and Worst Day.

## Signal Analysis

- **Signal Backtest** — Forward return analysis at 1d/5d/20d/60d horizons for buy (score < 25) and sell (score > 75) signals with hit rates and t-statistics
- **GSADF Bubble Detection** — Generalized Sup ADF test identifies statistically significant bubble periods in the composite score time series
- **Markov Regime Switching** — 3-state Hidden Markov Model (Normal/Elevated/Bubble) with transition probabilities and current regime probability
- **Indicator Sensitivity** — Shows how much each indicator moves the composite when perturbed by 1 percentile point
- **Autocorrelation** — Score persistence analysis at multiple lags

## Dashboard Sections

1. **Overview** — Large composite gauge with regime badge, 3 sub-scores, velocity, confidence interval, and data quality
2. **Indicator Grid** — 7 cards with score, trend arrow, 30-day sparkline, and day-over-day change
3. **Composite History** — Full time series with toggleable sub-scores, individual indicators, QQQ/SPY price overlay, and time range selector
4. **Bubble Index Backtest** — Two optimized strategies (Hysteresis Binary, Velocity Signal) with adjustable parameters and full risk metrics
5. **Signal Analysis** — Position mapping, forward return backtest, sensitivity analysis, Markov regimes
6. **Indicator Deep Dive** — Individual accordion charts for all 7 indicators with regime bands
7. **QQQ Deviation Tracker** — Multi-ticker deviation analysis (QQQ/SPY/TQQQ/IWM) with interactive threshold backtesting
8. **Methodology** — Regime guide, indicator weights, data sources, normalization methodology

## Data Pipeline

Data is refreshed daily via GitHub Actions at 9:30 PM UTC on weekdays (after US market close):

1. `scripts/fetch_qqq_data.py` — Fetches 10-year QQQ/SPY/TQQQ/IWM price and deviation data from Yahoo Finance
2. `scripts/fetch_bubble.py` — Computes all 7 bubble indicators, composite index with PCA orthogonalization, bootstrap confidence intervals, velocity/acceleration, GSADF test, Markov regime model, and forward-return backtest

### Output Files

| File | Description |
|------|-------------|
| `bubble_index.json` | Latest composite score, all indicators, diagnostics, correlation matrix |
| `bubble_history.json` | Full daily history (~2800 days) with per-indicator scores |
| `backtest_results.json` | Forward return statistics for buy/sell signals |
| `gsadf_results.json` | GSADF bubble period detection |
| `markov_regimes.json` | Markov regime probabilities and transition matrix |
| `qqq.json`, `spy.json`, `tqqq.json`, `iwm.json` | Ticker price/deviation data |

### Secrets

Set `FRED_API_KEY` as a repository secret to enable the yield curve indicator. All other indicators use Yahoo Finance (no key needed).

## Tech Stack

- **Frontend:** React 19, TypeScript, Recharts 3.6, Tailwind CSS
- **Build:** Vite 6
- **Data:** Python 3 (yfinance, fredapi, arch, statsmodels, scipy)
- **CI/CD:** GitHub Actions (daily data update + GitHub Pages deployment)
- **Testing:** Vitest

## Quick Start

```bash
npm install
npm run dev      # Start dev server on port 3000
npm run check    # TypeScript type checking
npm run test     # Run tests
npm run build    # Production build
```

### Regenerate Data Locally

```bash
pip install yfinance fredapi arch statsmodels scipy numpy pandas
FRED_API_KEY=your_key python scripts/fetch_bubble.py
python scripts/fetch_qqq_data.py
```

## License

MIT
