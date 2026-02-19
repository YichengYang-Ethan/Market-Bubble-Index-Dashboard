# Market Bubble Index Dashboard

[![Deploy](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/deploy.yml/badge.svg)](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/deploy.yml)
[![Update Data](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/update-data.yml/badge.svg)](https://github.com/yichengyang-ethan/Market-Bubble-Index-Dashboard/actions/workflows/update-data.yml)
[![CI](https://github.com/YichengYang-Ethan/Market-Bubble-Index-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/YichengYang-Ethan/Market-Bubble-Index-Dashboard/actions/workflows/ci.yml)

**Live:** [yichengyang-ethan.github.io/Market-Bubble-Index-Dashboard](https://yichengyang-ethan.github.io/Market-Bubble-Index-Dashboard/)

A professional financial dashboard that quantifies market bubble risk through a composite index of **7 weighted indicators** across sentiment, liquidity, and valuation. Features a **three-tab architecture** (Bubble Temperature / Crash Risk / Deviation Tracker), a **hybrid 3-layer drawdown probability model (v3.3)** rigorously validated via purged walk-forward cross-validation, two backtest strategies that beat Buy & Hold, GSADF bubble detection, and Markov regime switching.

## Three-Tab Dashboard Architecture

The dashboard is organized into three focused views, each with its own scroll-spy navigation:

| Tab | Story | Key Components |
|-----|-------|----------------|
| **Bubble Temperature** | "How euphoric is the market?" | Bubble gauge, 7 indicator cards, history chart, backtest, signals, deep dive, methodology |
| **Crash Risk** | "What is the probability of a crash?" | Risk gauge + logic chain, probability gauges (P >10/20/30/40%), risk indicators (3 inverted), risk history, regimes (GSADF + Markov), risk methodology |
| **Deviation Tracker** | "How far from the moving average?" | Multi-ticker deviation (QQQ/SPY/TQQQ/IWM), threshold backtesting, risk guide |

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

### Regime Classification

| Regime | Score Range | Position Mapping | Interpretation |
|--------|-------------|-----------------|----------------|
| Low | 0-30 | Overweight +30% | Depressed sentiment, accumulation zone |
| Moderate | 30-50 | Hold Benchmark | Normal conditions, balanced risk |
| Elevated | 50-70 | Underweight -20% | Rising euphoria, tighten risk management |
| High | 70-85 | Underweight -50% | Frothy conditions, consider reducing exposure |
| Extreme | 85-100 | Retain 20% Only | Bubble territory, high correction probability |

### Drawdown Risk Score

The dashboard introduces a **Drawdown Risk Score** alongside the Bubble Temperature, solving the logical gap between "market exuberance" and "crash probability":

| Score | Measures | VIX Direction | Monotonic? |
|-------|----------|---------------|-----------|
| Bubble Temperature | How exuberant the market is | Low VIX = complacent = high score | No (-0.6 Spearman) |
| **Drawdown Risk** | How dangerous conditions are | High VIX = stressed = high score | **Yes (+1.0 Spearman)** |

Both scores use the **same 7 indicators with the same weights** — the only difference is that the risk score inverts VIX, QQQ Deviation, and Yield Curve. This creates a monotonic relationship: higher risk score → higher drawdown probability (0% at score 20-40, 83% at score 80-100).

## QQQ Drawdown Probability Model (v3.3)

The dashboard features a **hybrid 3-layer drawdown probability model** that predicts the chance of QQQ experiencing significant drawdowns over the next ~9 months (180 trading days). The model has been rigorously validated through **purged walk-forward cross-validation** and systematic comparison against 6 model families.

### Model Architecture

| Layer | Method | Thresholds | Description |
|-------|--------|------------|-------------|
| 1 | Per-threshold penalized logistic + isotonic recalibration | 10%, 20% | Lasso (L1) for 10% DD with stability-selected features; L2 for 20% DD |
| 2 | Bayesian Beta-Binomial + PAVA monotonicity | 20%, 30% | Score-conditioned posteriors with Pool Adjacent Violators enforcement |
| 3 | EVT/GPD Tail Extrapolation | 40% | Generalized Pareto Distribution for extreme tail probabilities |

### Feature Selection: Stability Selection

Features for the 10% DD model were selected via **stability selection** — a bootstrap-based method that fits Lasso on 100 random 50% subsamples and retains features selected >60% of the time:

| Feature | Selection Rate | Description |
|---------|---------------|-------------|
| `risk_ema_20d` | 100% | 20-day EMA of drawdown risk score |
| `ind_qqq_deviation` | 100% | QQQ deviation from 200-day SMA |
| `ind_credit_spread` | 100% | Credit spread indicator (HYG/IEF) |
| `ind_yield_curve` | 99% | Yield curve indicator (10Y-2Y) |
| `ind_vix_level` | 96% | VIX percentile rank |
| `vix_x_credit` | 95% | VIX × credit spread interaction |
| `ind_vix_level_change_5d` | 82% | 5-day VIX momentum |
| `score_velocity` | 82% | Rate of change of composite score |

### Cross-Validation Methodology

All model evaluation uses **purged walk-forward cross-validation** — a time-series-aware CV method that prevents information leakage:

```
[======== TRAIN ========][-- 180d purge --][== TEST ==][- 20d embargo -]
                          ↑ prevents label    ↑ OOS       ↑ prevents
                            overlap            eval         contamination
```

- **5 folds**, expanding training window (minimum 35% of data)
- **180-day purge gap** = forward window length, ensures training labels don't overlap with test observations
- **20-day embargo** after each test fold prevents residual autocorrelation leakage
- **Degenerate fold filtering**: folds with base rate <2% or >98% are excluded (AUC/BSS unreliable)
- **Isotonic recalibration** within each fold: fit on train predictions, applied to test predictions

### Out-of-Sample Performance (Purged Walk-Forward CV)

| Threshold | AUC (CV) | Brier Skill Score | ECE | Folds | Primary Layer |
|-----------|----------|-------------------|-----|-------|---------------|
| >10% DD | **0.802 ± 0.076** | **+8.5% ± 7.7%** | 0.155 | 2 | Layer 1 (Logistic) |
| >20% DD | 0.552 ± 0.051 | -135% ± 184% | 0.358 | 3 | **Layer 2 (Bayesian)** |
| >30% DD | — | — | — | — | Layer 2 (Bayesian) |
| >40% DD | — | — | — | — | Layer 3 (EVT/GPD) |

The 10% DD model has **genuine predictive skill** (positive BSS = beats climatology baseline). The 20% DD logistic model struggles with the pre-COVID→COVID fold, so the **Bayesian Layer 2** serves as the primary calibrated estimate for 20%+ drawdowns:

| Risk Score Bin | P(>20% DD) | Sample Size |
|----------------|------------|-------------|
| 0-20 | N/A | 0 |
| 20-40 | 0.0% | 35 |
| 40-60 | 14.9% | 1,778 |
| 60-80 | 48.6% | 965 |
| 80-100 | 83.3% | 12 |

### Model Evolution: v3.0 → v3.3

| Version | Evaluation | 10% AUC | 10% BSS | 20% AUC | Key Change |
|---------|-----------|---------|---------|---------|------------|
| v3.0 | Single 70/30 split | 0.855 | +12.4% | 0.921 | Initial release — metrics inflated by favorable split point |
| v3.1 | Purged WF-CV (5 fold) | 0.455 | -251% | 0.719 | Honest evaluation exposed overfitting |
| v3.2 | Purged WF-CV + grid search | 0.692 | +11.3% | 0.650 | Automated feature/C selection across 28 combos |
| **v3.3** | **Purged WF-CV + stability selection** | **0.802** | **+8.5%** | **0.552** | **Lasso with bootstrap-stable features** |

### Multi-Model Comparison (6 Families, 100+ Configurations)

The v3.3 model was selected after exhaustive comparison across 6 model families, tested by 6 parallel agents:

| Model Family | Best 10% AUC | Best 10% BSS | Verdict |
|-------------|-------------|-------------|---------|
| **Penalized Logistic (Lasso/Ridge/ElasticNet)** | **0.802** | **+8.5%** | **Winner — only model with positive BSS** |
| Random Forest | 0.685 | -13.5% | Good ranking, poor calibration |
| XGBoost | 0.671 | -24.0% | Overfits on limited crash events |
| SVM (RBF) | 0.825 | -28.3% | Best AUC but terrible probability estimates |
| KNN | 0.781 | -22.2% | Cannot calibrate probabilities |
| MLP Neural Network | 0.617 | -34.8% | Dataset too small for neural nets |
| Ensemble/Stacking | various | all negative | Combining weak learners amplifies noise |

**Comparison scripts** (fully reproducible):

| Script | Method |
|--------|--------|
| `scripts/compare_penalized_linear.py` | Lasso, Ridge, ElasticNet (56 hyperparameter combos) |
| `scripts/compare_tree_models.py` | XGBoost, Random Forest (grid search) |
| `scripts/compare_nn_models.py` | MLP with 6 architectures |
| `scripts/compare_ensemble_models.py` | Simple averaging, weighted, stacking, SVM, KNN |
| `scripts/compare_aic_bic.py` | AIC/BIC forward/backward stepwise, exhaustive search |
| `scripts/compare_stability_selection.py` | Bootstrap Lasso, RFE, regularization path analysis |

### Why Logistic Regression Wins

With ~2,800 trading days and only 2-3 major drawdown events (2018 Q4, COVID 2020, 2022 bear market), the effective independent sample after 180-day purge is ~15 observations. Complex models (trees, neural nets, ensembles) overfit to the specific patterns of these few crashes. Logistic regression's strong inductive bias (linear decision boundary) acts as an implicit regularizer, and combined with:

1. **Stability selection** for robust feature identification
2. **Isotonic recalibration** for probability calibration
3. **Bayesian binning** (Layer 2) for non-parametric calibration at higher thresholds

...it produces the only model with genuinely positive out-of-sample Brier Skill Score.

## Strategy Backtesting

Two optimized backtest strategies that beat Buy & Hold on risk-adjusted returns over 10+ years:

### Strategy 1: Hysteresis Binary

Binary all-in/all-out with asymmetric thresholds (control theory hysteresis):
- **Enter** (100% equity) when bubble score < 24; **Exit** (100% cash) when score > 84
- Dead zone of 60 points prevents whipsaw trading
- **Result:** Sharpe 0.70 vs B&H 0.67 (+5.5% edge)

### Strategy 2: Velocity Signal

Uses the first derivative of the composite score as a leading indicator:
- **Reduce to 30%** when `score_velocity > 15` AND `score > 50`
- **Result:** Sharpe 0.69, CAGR 18.5% — beats B&H on both risk-adjusted AND absolute returns

## Signal Analysis

- **GSADF Bubble Detection** — Generalized Sup ADF test identifies statistically significant bubble periods
- **Markov Regime Switching** — 3-state HMM (Normal/Elevated/Bubble) with transition probabilities
- **Forward Return Backtest** — 1d/5d/20d/60d horizon analysis for buy/sell signals with hit rates
- **Indicator Sensitivity** — Per-indicator impact on composite when perturbed by 1 percentile point

## Data Pipeline

Data is refreshed daily via GitHub Actions at 9:30 PM UTC on weekdays:

| Script | Purpose |
|--------|---------|
| `scripts/fetch_qqq_data.py` | 10-year QQQ/SPY/TQQQ/IWM price and deviation data |
| `scripts/fetch_bubble.py` | 7 bubble indicators, composite index, GSADF, Markov, backtest |
| `scripts/fit_drawdown_model.py` | v3.3 hybrid 3-layer drawdown model with purged WF-CV |

### Output Files

| File | Description |
|------|-------------|
| `bubble_index.json` | Latest composite score, all indicators, diagnostics |
| `bubble_history.json` | Full daily history (~2,800 days) with per-indicator scores |
| `drawdown_model.json` | v3.3 model coefficients, CV metrics, Bayesian lookup tables, EVT parameters |
| `qqq_drawdown.json` | Rolling peak-to-trough drawdown series |
| `backtest_results.json` | Forward return statistics for buy/sell signals |
| `gsadf_results.json` | GSADF bubble period detection |
| `markov_regimes.json` | Markov regime probabilities and transition matrix |
| `qqq.json`, `spy.json`, `tqqq.json`, `iwm.json` | Ticker price/deviation data |

## Tech Stack

- **Frontend:** React 19, TypeScript, Recharts 3.6, Tailwind CSS
- **Build:** Vite 6
- **Data:** Python 3 (yfinance, fredapi, arch, statsmodels, scipy, scikit-learn)
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
pip install yfinance fredapi arch statsmodels scipy numpy pandas scikit-learn
FRED_API_KEY=your_key python scripts/fetch_bubble.py
python scripts/fetch_qqq_data.py
python scripts/fit_drawdown_model.py
```

### Run Model Comparison (Reproducibility)

```bash
python scripts/compare_penalized_linear.py      # Lasso/Ridge/ElasticNet
python scripts/compare_tree_models.py            # XGBoost/Random Forest
python scripts/compare_nn_models.py              # MLP Neural Networks
python scripts/compare_ensemble_models.py        # Ensembles/Stacking/SVM/KNN
python scripts/compare_aic_bic.py                # AIC/BIC Stepwise Selection
python scripts/compare_stability_selection.py    # Stability Selection/RFE
```

See [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) for the full strategy research report.

## License

MIT
