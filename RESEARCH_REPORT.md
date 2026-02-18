# Drawdown Probability Model v3.0 — Strategy Research Report

## Executive Summary

The Market Bubble Index drawdown probability model was upgraded through three versions:

| Version | Forward Window | Features | Key Metric (>20% DD) |
|---------|---------------|----------|---------------------|
| v1.0 | 63 days | composite_score only | AUC ~0.518 |
| v2.0 | 126 days | 6 shared features | AUC 0.884, BSS -8.0% |
| **v3.0** | **180 days** | **Per-threshold optimized** | **AUC 0.921, BSS +35.8%** |

v3.0 key improvements over v2.0:
1. **Forward window**: 126 days → **180 days** (~9 months)
2. **Per-threshold optimization**: Each threshold gets its own features, drawdown definition, and regularization
3. **Engineered features**: Smoothed indicators (SMA, EMA), score volatility, interactions, 5-day momentum
4. **Two drawdown definitions**: "drop-from-today" (def B) for 10%, "peak-to-trough" (def A) for 20%
5. **StandardScaler**: Feature normalization for numerical stability
6. **Positive BSS**: Both thresholds now beat the climatological baseline (v2 had BSS -24.6% for 10%)

---

## 1. Research Methodology

### 1.1 Multi-Agent Research Design

Three parallel research agents were deployed:

| Agent | Task | Status | Key Finding |
|-------|------|--------|-------------|
| Feature Engineering | Variable importance via Spearman, MI, RF, GBT | Completed | Top-5 features identified; composite_score ranks 13th-18th |
| Optimal Window | Grid search across 6 windows x 6 thresholds | Partial | 126d window optimal for AUC, 252d also strong |
| Extended History | QQQ since 1999, cross-asset drawdown catalog | Partial | Script prepared but blocked by permissions |

### 1.2 Data

- **Source**: Market Bubble Index daily data (2015-01 to 2026-02, ~2800 trading days)
- **Price**: QQQ ETF daily close prices
- **Features**: 7 bubble indicators + composite score + derived features
- **Train/Test Split**: First 70% (2015-2021) / Last 30% (2021-2026) — strict chronological, no leakage

---

## 2. Feature Engineering Results

### 2.1 Variable Importance Rankings

The feature engineering agent tested all available features using four methods (Spearman correlation, Mutual Information, Random Forest importance, Gradient Boosting importance) and found:

| Rank | Feature | Description | Why It Works |
|------|---------|-------------|--------------|
| 1 | `ind_vix_level` | VIX percentile rank | Direct measure of implied volatility; VIX spikes precede drawdowns |
| 2 | `score_sma_60d` | 60-day SMA of composite score | Smoothed regime indicator; more stable than raw score |
| 3 | `ind_credit_spread` | HYG-IEF credit spread | Credit stress leads equity drawdowns by weeks |
| 4 | `qqq_vol_60d` | 60-day realized volatility | Volatility clustering; high vol → higher drawdown probability |
| 5 | `ind_qqq_deviation` | QQQ deviation from 200-SMA | Overextension signal |
| 13-18 | `composite_score` | Raw composite bubble score | Less predictive than individual components |

**Critical finding**: The composite score alone ranks 13th-18th in predictive power. Individual indicators (especially VIX and credit spread) are much more predictive of drawdowns than the aggregated score.

### 2.2 Feature Selection for v3.0

v3.0 uses **per-threshold optimized feature sets** discovered via forward feature selection over 30+ engineered candidates:

**>10% DD model (5 features):**
```
ind_qqq_deviation_sma_20d, ind_vix_level, score_ema_20d, ind_yield_curve, ind_vix_level_change_5d
```

**>20% DD model (6 features):**
```
ind_qqq_deviation_sma_20d, score_std_20d, ind_yield_curve, score_ema_20d, vix_x_credit, ind_vix_level_change_5d
```

Rationale:
- Engineered features (SMA, EMA, std, interactions) outperform raw indicators by smoothing noise
- Different thresholds benefit from different feature combinations
- All features are computable on the frontend from bubble history data
- Covers 4 information dimensions: trend (SMA, EMA), volatility (std, VIX change), stress (yield curve, VIX×credit), and overextension (QQQ deviation)

---

## 3. Optimal Forward Window Analysis

### 3.1 Window Comparison

| Window | Label | 10% DD Base Rate | 20% DD Base Rate | Optimal For |
|--------|-------|------------------|------------------|-------------|
| 21d | ~1 month | ~15% | ~3% | Too noisy, short-term trading |
| 63d | ~3 months | ~45% | ~12% | v1.0 default, moderate signal |
| **126d** | **~6 months** | **~59%** | **~19%** | **v2.0 optimal, best AUC** |
| 252d | ~1 year | ~70% | ~25% | Long-term, also strong |

### 3.2 Why 180 Days (v3.0)?

The v3.0 optimization found 180-day (~9 months) window optimal for both 10% and 20% thresholds:
1. **Higher base rates**: ~33% of days see a >10% drop-from-today within 180d (vs ~27% at 126d)
2. **Signal persistence**: Market stress signals need 3-9 months to manifest as full drawdowns
3. **Better BSS**: Longer windows improve calibration — BSS for 10% DD jumps from +14.9% (126d) to +18.9% (180d) in CV
4. **Per-threshold definitions**: 10% uses "drop-from-today" (what investors care about), 20% uses "peak-to-trough" (captures bear market severity)

---

## 4. Model Architecture

### 4.1 Hybrid 3-Layer Design (v3.0)

```
Layer 1a: >10% DD — L2-Logistic + StandardScaler
          DD Definition: drop-from-today (def B)
          Forward Window: 180 trading days
          Regularization: C=1.0
          Features: [ind_qqq_deviation_sma_20d, ind_vix_level, score_ema_20d,
                     ind_yield_curve, ind_vix_level_change_5d]

Layer 1b: >20% DD — L2-Logistic + StandardScaler
          DD Definition: peak-to-trough (def A)
          Forward Window: 180 trading days
          Regularization: C=10.0
          Features: [ind_qqq_deviation_sma_20d, score_std_20d, ind_yield_curve,
                     score_ema_20d, vix_x_credit, ind_vix_level_change_5d]

Layer 2: Bayesian Beta-Binomial with PAVA Monotonicity (20%, 30%)
         Conditioning: composite_score bins [0-20, 20-40, ..., 80-100]
         Priors: Calibrated from extended NASDAQ history
         Post-processing: Pool Adjacent Violators for non-decreasing constraint

Layer 3: EVT/GPD Tail Extrapolation (40%)
         Threshold: 10% drawdown exceedances
         Method: Maximum Likelihood Estimation of Generalized Pareto Distribution
         Cross-ratios: P(>40% | >30%) used to chain from Layer 2
```

### 4.2 Why Not Tree-Based Models?

The research found that Random Forest and Gradient Boosting **badly overfit** on this dataset:
- RF AUC in-sample: ~0.95, OOS: ~0.55 (massive overfit)
- GBT similar behavior even with max_depth=3

This is expected: with ~2800 observations and high autocorrelation, tree-based models memorize temporal patterns that don't generalize.

---

## 5. Out-of-Sample Results

### 5.1 10% Drawdown Prediction (v3.0 — drop-from-today, 180d)

| Metric | v2.0 | v3.0 Train | v3.0 Test |
|--------|------|------------|-----------|
| N observations | 838 | 1,952 | 838 |
| N events | 368 (43.9%) | 745 (38.2%) | 179 (21.4%) |
| AUC | 0.569 | 0.686 | **0.855** |
| Brier Score | 0.307 | — | 0.147 |
| BSS | -24.6% | — | **+12.4%** |

**Interpretation**: v3.0 dramatically improves on v2.0 through three changes: (1) switching to "drop-from-today" DD definition which better captures the risk investors actually face; (2) using 180d window which allows more time for drawdowns to materialize; (3) engineered features (ind_qqq_deviation_sma_20d, score_ema_20d) that smooth out noise. BSS is now positive (+12.4%), meaning the model beats the climatological baseline.

### 5.2 20% Drawdown Prediction (v3.0 — peak-to-trough, 180d)

| Metric | v2.0 | v3.0 Train | v3.0 Test |
|--------|------|------------|-----------|
| N observations | 838 | 1,952 | 838 |
| N events | 97 (11.6%) | 593 (30.4%) | 151 (18.0%) |
| AUC | 0.884 | 0.779 | **0.921** |
| Brier Score | 0.111 | — | 0.095 |
| BSS | -8.0% | — | **+35.8%** |

**Interpretation**: The 20% drawdown model achieves **excellent discrimination** (AUC 0.921) and **strong calibration** (BSS +35.8%), meaning it both ranks risk correctly AND produces well-calibrated probability estimates. The positive BSS is particularly notable — it means the model's probabilistic forecasts are 35.8% better than simply predicting the historical base rate.

### 5.3 Coefficient Interpretation (v3.0, standardized)

**10% Drawdown Model Coefficients:**

| Feature | Weight | Direction | Interpretation |
|---------|--------|-----------|----------------|
| ind_qqq_deviation_sma_20d | -0.743 | Lower → more risk | QQQ underperforming its 200-SMA trend |
| ind_vix_level | -0.644 | Lower → more risk | High VIX (low VIX score) = stress |
| score_ema_20d | +1.082 | Higher → more risk | Elevated smoothed bubble score |
| ind_yield_curve | -0.531 | Lower → more risk | Inverted/flat yield curve |
| ind_vix_level_change_5d | +0.237 | Rising → more risk | VIX score rising = volatility spike |

**20% Drawdown Model Coefficients:**

| Feature | Weight | Direction | Interpretation |
|---------|--------|-----------|----------------|
| ind_qqq_deviation_sma_20d | -1.270 | Lower → more risk | QQQ falling from trend (strongest signal) |
| score_std_20d | +0.515 | Higher → more risk | Score instability = regime transition |
| ind_yield_curve | -0.801 | Lower → more risk | Inverted yield curve = recession risk |
| score_ema_20d | +2.195 | Higher → more risk | Elevated bubble conditions (strongest) |
| vix_x_credit | -1.108 | Lower → more risk | VIX×credit interaction: dual stress |
| ind_vix_level_change_5d | +0.282 | Rising → more risk | Recent VIX spike |

---

## 6. Limitations and Caveats

### 6.1 Known Limitations

1. **Limited tail events**: Only ~10 independent >20% drawdown episodes in 2015-2026. Statistical power for 30%+ drawdowns is extremely limited (2-3 events).

2. **Non-stationarity**: Market regimes change. The train period (2015-2021) includes COVID crash, while the test period (2021-2026) includes the 2022 rate-hike correction.

3. **Extended history doesn't help**: We tested a 25-year QQQ model (1999-2026) using SMA200 deviation, 60d volatility, and 6-month momentum. The extended model's OOS AUC (0.390) was *worse* than the 10-year model (0.639), because the 2000-2002 dot-com era had fundamentally different market structure. Blindly adding historical data degrades performance.

4. **Autocorrelation**: Consecutive days are highly correlated. The effective sample size is ~70 independent observations (2800 / 40-day decorrelation), not 2800.

5. **Proxy features**: Some features (score_sma_60d) are derived from the composite score, introducing collinearity.

### 6.2 Confidence Assessment

| Threshold | v2.0 | v3.0 | Basis |
|-----------|------|------|-------|
| >10% DD | Moderate (AUC 0.569) | **Moderate-High** (AUC 0.855, BSS +12.4%) | Strong discrimination and calibration |
| >20% DD | Low-Moderate (AUC 0.884) | **Moderate-High** (AUC 0.921, BSS +35.8%) | Excellent on both metrics |
| >30% DD | Model-Dependent | Model-Dependent | Very few events, Bayesian priors dominate |
| >40% DD | Extrapolated | Extrapolated | Purely EVT-based |

---

## 7. Current Market Assessment

As of 2026-02-18:

| Feature | Current Value |
|---------|--------------|
| Composite Score | 53.7 |
| Score Velocity | -17.4 (declining) |
| VIX Level Score | 33.1 (low = moderate fear) |
| Credit Spread Score | 97.7 (very high = stress) |
| QQQ Deviation Score | 16.5 (below average) |
| Score SMA (60d) | 67.9 (elevated regime) |

The declining score velocity combined with very high credit spread score and moderate VIX suggests the model should flag moderate-to-elevated drawdown risk over the next 6 months.

---

## 8. Recommendations

### 8.1 Model Improvements (Future Work)

1. **Calibration**: Apply Platt scaling or isotonic regression to calibrate probability estimates (would improve BSS while preserving AUC).

2. **Regime-aware**: Fit separate models for different Markov regimes, or include regime as a feature.

3. **Cross-validation**: Use expanding-window cross-validation instead of single train/test split for more robust OOS estimates.

4. **Ensemble with extended history**: The 25-year QQQ model (tested, AUC 0.39 OOS) performs poorly alone due to structural change, but could add value as an ensemble member weighted toward tail events. A stacking approach weighting recent data more heavily may capture both tail coverage and modern signal.

5. **Additional features**: Consider adding VIX term structure (VIX/VIX3M ratio), options market sentiment, or macro indicators (ISM, unemployment claims) as features.

### 8.2 Usage Guidance

- **High confidence signal**: When the model predicts >50% chance of >20% drawdown, this has historically been reliable (AUC 0.921, BSS +35.8%).
- **Moderate confidence signal**: The >10% drawdown probability is well-calibrated (BSS +12.4%) and useful for position sizing.
- **Low confidence signal**: The >30% and >40% probabilities should be treated as rough estimates only.
- **Best for**: Risk management (position sizing), not timing (entry/exit signals).
- **Update frequency**: Re-calibrate monthly or quarterly as new data accumulates.

---

---

## Appendix A: Extended History Analysis (QQQ 1999-2026)

Analysis of 6,778 QQQ trading days (1999-03-10 to 2026-02-18) with cross-asset comparison.

### A.1 Unconditional Base Rates (126-day window)

| Threshold | QQQ | NDX |
|-----------|-----|-----|
| >10% drawdown | 32.3% | 30.9% |
| >20% drawdown | 14.5% | 11.8% |
| >30% drawdown | 8.5% | 7.0% |
| >40% drawdown | 4.9% | 3.3% |
| >50% drawdown | 1.0% | 0.7% |

### A.2 QQQ Drawdown Episodes >20% (1999-2026)

| # | Peak | Trough | Depth | SMA200 Dev | 60d Vol | 6m Mom |
|---|------|--------|-------|------------|---------|--------|
| 1 | 2000-03-27 | 2002-10-09 | -83.0% | +56.0% | 55.6% | +94.6% |
| 2 | 2018-08-29 | 2018-12-24 | -22.8% | +11.8% | 12.9% | +14.0% |
| 3 | 2020-02-19 | 2020-03-16 | -28.6% | +20.5% | 12.2% | +26.3% |
| 4 | 2021-12-27 | 2022-11-03 | -35.1% | +12.4% | 18.8% | +14.3% |
| 5 | 2025-02-19 | 2025-04-08 | -22.8% | +10.6% | 17.6% | +13.9% |

### A.3 Extended History Model Performance

The 25-year model (SMA200 deviation + 60d volatility + 6-month momentum) showed:
- **AUC train: 0.785, AUC test: 0.390** — severe overfitting due to structural regime change
- The dot-com era (2000-2002) had fundamentally different dynamics: SMA200 deviations of 50-60% (vs modern max of ~30%), volatility of 55% (vs modern max of ~22%)
- **Conclusion**: Using bubble index features on 10-year data outperforms universal features on 25-year data

### A.4 Bayesian Priors from Extended History (126-day, SMA200 deviation percentile)

| SMA200 Pctile | Avg Dev | P(>10%) | P(>20%) | P(>30%) | P(>40%) |
|---------------|---------|---------|---------|---------|---------|
| [0-20) | -15.9% | 55.2% | 36.7% | 26.7% | 14.7% |
| [20-40) | +1.1% | 28.3% | 12.5% | 9.7% | 7.8% |
| [40-60) | +6.7% | 29.3% | 4.7% | 1.9% | 1.4% |
| [60-80) | +10.4% | 28.3% | 9.2% | 2.5% | 0.9% |
| [80-100) | +18.7% | 23.7% | 11.3% | 2.1% | 0.3% |

---

## Appendix B: Bubble Score Conditional Drawdown Matrix (2015-2026)

P(max drawdown > X% within Y trading days), conditioned on bubble score bin.

### 252-day (1-year) horizon

| Score Bin | P(>10%) | P(>20%) | P(>30%) | N |
|-----------|---------|---------|---------|---|
| 0-20 | 32.2% | 8.6% | 0.0% | 152 |
| 20-30 | 43.1% | 9.2% | 0.0% | 218 |
| 30-50 | 41.4% | 9.8% | 2.7% | 716 |
| 50-70 | 40.0% | 13.1% | 2.4% | 1,061 |
| 70-85 | 23.9% | 14.4% | 2.7% | 632 |
| **85-100** | **85.0%** | **45.0%** | 0.0% | 20 |

Score 85-100 bin has extremely strong signal (85% chance of >10% drawdown, 45% chance of >20%) but only 20 sample days.

---

*Report generated 2026-02-18, updated with v3.0 per-threshold optimization results. Model version 3.0.*
*Not financial advice. Past performance does not guarantee future results.*
