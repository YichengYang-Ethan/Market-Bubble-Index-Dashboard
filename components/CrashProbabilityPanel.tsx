import React, { useMemo } from 'react';
import { DrawdownModelData } from '../services/bubbleService';
import { BubbleIndicator, BubbleHistoryPoint } from '../types';

interface Props {
  model: DrawdownModelData;
  currentScore: number;
  scoreVelocity: number;
  drawdownRiskScore?: number;
  indicators?: Record<string, BubbleIndicator>;
  history?: BubbleHistoryPoint[];
}

// ---------------------------------------------------------------------------
// Probability computation — v3.0 per-threshold + v2.0/v1.0 fallback
// ---------------------------------------------------------------------------

function sigmoid(z: number): number {
  z = Math.max(-30, Math.min(30, z));
  return 1 / (1 + Math.exp(-z));
}

function interpolateLookup(score: number, centers: number[], probs: number[]): number {
  if (score <= centers[0]) return probs[0];
  if (score >= centers[centers.length - 1]) return probs[probs.length - 1];
  for (let i = 0; i < centers.length - 1; i++) {
    if (score >= centers[i] && score <= centers[i + 1]) {
      const t = (score - centers[i]) / (centers[i + 1] - centers[i]);
      return probs[i] + t * (probs[i + 1] - probs[i]);
    }
  }
  return probs[probs.length - 1];
}

/** Build current feature values from available data */
function buildFeatureValues(
  model: DrawdownModelData,
  currentScore: number,
  scoreVelocity: number,
  indicators?: Record<string, BubbleIndicator>,
  history?: BubbleHistoryPoint[],
): Record<string, number> {
  // drawdown_risk_score is passed from the parent or fallback to model's precomputed value
  const drawdownRisk = model.current_features?.drawdown_risk_score ?? currentScore;

  const values: Record<string, number> = {
    composite_score: currentScore,
    drawdown_risk_score: drawdownRisk,
    score_velocity: scoreVelocity,
  };

  // Direct indicator scores
  if (indicators) {
    values.ind_vix_level = indicators.vix_level?.score ?? 50;
    values.ind_credit_spread = indicators.credit_spread?.score ?? 50;
    values.ind_qqq_deviation = indicators.qqq_deviation?.score ?? 50;
    values.ind_yield_curve = indicators.yield_curve?.score ?? 50;
    values.ind_sector_breadth = indicators.sector_breadth?.score ?? 50;
    values.ind_cape_ratio = indicators.cape_ratio?.score ?? 50;
    values.ind_put_call_ratio = indicators.put_call_ratio?.score ?? 50;
  }

  // Derived features from history
  if (history && history.length >= 20) {
    const recent20 = history.slice(-20);
    const recent60 = history.slice(-60);

    // score_ema_20d: exponential moving average of composite score
    let ema = recent20[0].composite_score;
    const alpha = 2 / (20 + 1);
    for (let i = 1; i < recent20.length; i++) {
      ema = alpha * recent20[i].composite_score + (1 - alpha) * ema;
    }
    values.score_ema_20d = ema;

    // score_std_20d: rolling standard deviation of composite score
    const scores20 = recent20.map(h => h.composite_score);
    const mean20 = scores20.reduce((a, b) => a + b, 0) / scores20.length;
    const variance20 = scores20.reduce((a, v) => a + (v - mean20) ** 2, 0) / (scores20.length - 1);
    values.score_std_20d = Math.sqrt(Math.max(0, variance20));

    // score_sma_60d: 60-day SMA of composite score
    const sum60 = recent60.reduce((acc, h) => acc + h.composite_score, 0);
    values.score_sma_60d = sum60 / recent60.length;

    // risk_ema_20d: EMA of drawdown risk score
    const riskScores20 = recent20.map(h => h.drawdown_risk_score ?? 50);
    let riskEma = riskScores20[0];
    for (let i = 1; i < riskScores20.length; i++) {
      riskEma = alpha * riskScores20[i] + (1 - alpha) * riskEma;
    }
    values.risk_ema_20d = riskEma;
    values.drawdown_risk_score = riskScores20[riskScores20.length - 1];

    // ind_qqq_deviation_sma_20d: 20-day SMA of QQQ deviation indicator
    const devScores = recent20.map(h => h.indicators?.qqq_deviation ?? 50);
    values.ind_qqq_deviation_sma_20d = devScores.reduce((a, b) => a + (b ?? 50), 0) / devScores.length;

    // ind_vix_level_change_5d: 5-day change in VIX level indicator
    if (history.length >= 5) {
      const todayVix = indicators?.vix_level?.score ?? 50;
      const fiveDaysAgo = history[history.length - 5]?.indicators?.vix_level ?? 50;
      values.ind_vix_level_change_5d = todayVix - (fiveDaysAgo ?? 50);
    }
  }

  // Interaction features
  values.vix_x_credit = (values.ind_vix_level ?? 50) * (values.ind_credit_spread ?? 50) / 100;
  values.is_elevated = currentScore > 60 ? 1 : 0;

  // Fallback to model's pre-computed current features for any missing values
  if (model.current_features) {
    for (const [key, val] of Object.entries(model.current_features)) {
      if (!(key in values)) {
        values[key] = val;
      }
    }
  }

  return values;
}

interface DrawdownProb {
  threshold: number;
  label: string;
  probability: number;
  confidence: string;
  color: string;
  bgColor: string;
  ciLower?: number;
  ciUpper?: number;
}

/** Apply logistic model with optional StandardScaler (v3.0) */
function applyLogistic(
  coefs: DrawdownModelData['logistic_coefficients'][string],
  featureValues: Record<string, number>,
): number {
  if (!coefs?.weights || coefs?.intercept === undefined) return 0;

  let z = coefs.intercept;
  for (const [feat, w] of Object.entries(coefs.weights)) {
    let x = featureValues[feat] ?? 0;
    // Apply StandardScaler if present (v3.0)
    if (coefs.scaler_mean && coefs.scaler_std) {
      const mean = coefs.scaler_mean[feat] ?? 0;
      const std = coefs.scaler_std[feat] ?? 1;
      x = std > 0 ? (x - mean) / std : 0;
    }
    z += w * x;
  }
  return sigmoid(z);
}

function computeDrawdownProbabilities(
  model: DrawdownModelData,
  featureValues: Record<string, number>,
): DrawdownProb[] {
  const { logistic_coefficients: logCoefs, bayesian_lookup: bayesian, evt_parameters: evt, probability_ci: ci } = model;

  // --- Layer 1+2: Blend for 10% (logistic + Bayesian) ---
  const riskScore = featureValues.drawdown_risk_score ?? featureValues.composite_score ?? 0;
  const blendWeights10 = model.blend_weights?.['10pct'] ?? { logistic: 0.5, bayesian: 0.5 };

  const c10 = logCoefs['drawdown_10pct'];
  let p10_logistic = 0;
  if (c10?.weights && c10?.intercept !== undefined) {
    p10_logistic = applyLogistic(c10, featureValues);
  } else if (c10) {
    // v1.0 fallback
    const score = featureValues.composite_score ?? 0;
    const vel = featureValues.score_velocity ?? 0;
    p10_logistic = sigmoid(
      (c10.a_score_with_velocity ?? 0) * score +
      (c10.a_velocity ?? 0) * vel +
      (c10.b_with_velocity ?? 0)
    );
  }
  const b10 = bayesian['drawdown_10pct'];
  const p10_bayesian = b10
    ? interpolateLookup(riskScore, b10.bin_centers, b10.probabilities)
    : p10_logistic;
  const p10 = blendWeights10.logistic * p10_logistic + blendWeights10.bayesian * p10_bayesian;

  // --- Layer 1 + 2: Blend for 20% (Bayesian-dominated) ---
  const blendWeights = model.blend_weights?.['20pct'] ?? { logistic: 0.0, bayesian: 1.0 };

  const c20 = logCoefs['drawdown_20pct'];
  let p20_logistic = 0;
  if (c20?.weights && c20?.intercept !== undefined) {
    p20_logistic = applyLogistic(c20, featureValues);
  } else if (c20) {
    const score = featureValues.composite_score ?? 0;
    const vel = featureValues.score_velocity ?? 0;
    p20_logistic = sigmoid(
      (c20.a_score_with_velocity ?? 0) * score +
      (c20.a_velocity ?? 0) * vel +
      (c20.b_with_velocity ?? 0)
    );
  }
  const b20 = bayesian['drawdown_20pct'];
  const p20_bayesian = b20
    ? interpolateLookup(riskScore, b20.bin_centers, b20.probabilities)
    : p20_logistic;
  const p20 = blendWeights.logistic * p20_logistic + blendWeights.bayesian * p20_bayesian;

  // --- Layer 2: Pure Bayesian for 30% (100% Bayesian) ---
  const b30 = bayesian['drawdown_30pct'];
  const p30 = b30
    ? interpolateLookup(riskScore, b30.bin_centers, b30.probabilities)
    : 0;

  // --- Layer 3: EVT extrapolation for 40% ---
  const r40_30 = evt.cross_ratios?.['40pct_given_30pct'] ?? 0.35;
  const p40 = p30 * r40_30;

  return [
    {
      threshold: 10,
      label: '10%+',
      probability: Math.min(p10, 0.99),
      confidence: model.confidence_tiers['10pct'] ?? 'moderate',
      color: '#eab308',
      bgColor: 'bg-yellow-500/10 border-yellow-500/20',
      ciLower: ci?.['10pct']?.lower,
      ciUpper: ci?.['10pct']?.upper,
    },
    {
      threshold: 20,
      label: '20%+',
      probability: Math.min(p20, 0.99),
      confidence: model.confidence_tiers['20pct'] ?? 'low',
      color: '#f97316',
      bgColor: 'bg-orange-500/10 border-orange-500/20',
      ciLower: ci?.['20pct']?.lower,
      ciUpper: ci?.['20pct']?.upper,
    },
    {
      threshold: 30,
      label: '30%+',
      probability: Math.min(p30, 0.99),
      confidence: model.confidence_tiers['30pct'] ?? 'model_dependent',
      color: '#ef4444',
      bgColor: 'bg-red-500/10 border-red-500/20',
      ciLower: ci?.['30pct']?.lower,
      ciUpper: ci?.['30pct']?.upper,
    },
    {
      threshold: 40,
      label: '40%+',
      probability: Math.min(p40, 0.99),
      confidence: model.confidence_tiers['40pct'] ?? 'extrapolated',
      color: '#dc2626',
      bgColor: 'bg-red-600/10 border-red-600/20',
    },
  ];
}

// ---------------------------------------------------------------------------
// Confidence badge
// ---------------------------------------------------------------------------

const CONFIDENCE_DISPLAY: Record<string, { label: string; color: string }> = {
  'moderate-high': { label: 'Mod-High', color: 'text-emerald-400' },
  moderate: { label: 'Moderate', color: 'text-emerald-400' },
  low: { label: 'Low', color: 'text-yellow-400' },
  model_dependent: { label: 'Model-Dep.', color: 'text-orange-400' },
  extrapolated: { label: 'Extrapolated', color: 'text-red-400' },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const CrashProbabilityPanel: React.FC<Props> = ({ model, currentScore, scoreVelocity, drawdownRiskScore, indicators, history }) => {
  const featureValues = useMemo(
    () => buildFeatureValues(model, currentScore, scoreVelocity, indicators, history),
    [model, currentScore, scoreVelocity, indicators, history],
  );

  const probs = useMemo(
    () => computeDrawdownProbabilities(model, featureValues),
    [model, featureValues],
  );

  // Find the current score bin for empirical stats
  const binKey = useMemo(() => {
    if (currentScore < 20) return '0-20';
    if (currentScore < 40) return '20-40';
    if (currentScore < 60) return '40-60';
    if (currentScore < 80) return '60-80';
    return '80-100';
  }, [currentScore]);

  const empirical = model.empirical_stats?.[binKey];

  // OOS metrics from logistic model
  const oosMetrics = useMemo(() => {
    const c10 = model.logistic_coefficients['drawdown_10pct'];
    const c20 = model.logistic_coefficients['drawdown_20pct'];
    if (!c10?.auc_test) return null;
    return { c10, c20 };
  }, [model]);

  const modelVersion = model.model_version ?? '1.0';
  const isV4 = modelVersion.startsWith('4');
  const isV3 = modelVersion.startsWith('3');
  const isV2Plus = modelVersion >= '2.0';

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
          </svg>
          QQQ Drawdown Probability
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
            v{modelVersion}
          </span>
          <span className="text-xs text-slate-500 bg-slate-800 px-3 py-1 rounded-full">
            Next {model.forward_window_label}
          </span>
        </div>
      </div>

      {/* Probability gauges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {probs.map((p) => {
          const pct = (p.probability * 100);
          const conf = CONFIDENCE_DISPLAY[p.confidence] ?? CONFIDENCE_DISPLAY.low;
          const baseRateKey = `${p.threshold}pct`;
          const baseRate = model.unconditional_base_rates?.[baseRateKey];
          const baseRatePct = baseRate != null ? baseRate * 100 : null;
          // Model lift: how much the conditional probability exceeds the unconditional
          const lift = baseRatePct != null ? pct - baseRatePct : null;
          return (
            <div key={p.threshold} className={`border rounded-xl p-4 ${p.bgColor}`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-400 uppercase">Drop {p.label}</span>
                <span className={`text-[10px] ${conf.color}`}>{conf.label}</span>
              </div>

              {/* Circular gauge */}
              <div className="flex justify-center my-3">
                <div className="relative w-20 h-20">
                  <svg className="w-20 h-20 transform -rotate-90" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r="34" fill="transparent" stroke="#1e293b" strokeWidth="6" />
                    {/* Base rate reference arc (gray) */}
                    {baseRatePct != null && (
                      <circle
                        cx="40" cy="40" r="34" fill="transparent"
                        stroke="#475569"
                        strokeWidth="2"
                        strokeDasharray={`${baseRatePct * 2.136} 213.6`}
                        opacity={0.5}
                      />
                    )}
                    <circle
                      cx="40" cy="40" r="34" fill="transparent"
                      stroke={p.color}
                      strokeWidth="6"
                      strokeLinecap="round"
                      strokeDasharray={`${pct * 2.136} 213.6`}
                      style={{ filter: pct > 30 ? `drop-shadow(0 0 4px ${p.color})` : 'none' }}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-lg font-black text-white">{pct.toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* CI range */}
              {p.ciLower !== undefined && p.ciUpper !== undefined && (
                <p className="text-[10px] text-slate-500 text-center mt-1">
                  [{(p.ciLower * 100).toFixed(0)}%, {(p.ciUpper * 100).toFixed(0)}%]
                </p>
              )}

              {/* Bar with base rate marker */}
              <div className="relative h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: p.color }}
                />
                {baseRatePct != null && (
                  <div
                    className="absolute top-0 h-full w-0.5 bg-slate-400"
                    style={{ left: `${Math.min(baseRatePct, 100)}%` }}
                    title={`Base rate: ${baseRatePct.toFixed(1)}%`}
                  />
                )}
              </div>

              {/* Base rate annotation */}
              {baseRatePct != null && (
                <div className="flex items-center justify-between mt-1.5">
                  <span className="text-[9px] text-slate-500">Base: {baseRatePct.toFixed(1)}%</span>
                  {lift != null && (
                    <span className={`text-[9px] font-semibold ${lift > 0 ? 'text-red-400' : lift < -2 ? 'text-emerald-400' : 'text-slate-500'}`}>
                      {lift > 0 ? '+' : ''}{lift.toFixed(1)}pp
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Dual-Score Logic Chain */}
      {drawdownRiskScore !== undefined && (
        <div className="bg-slate-800/30 rounded-xl p-4 mb-4">
          <div className="flex items-center gap-6">
            {/* Bubble Temperature */}
            <div className="flex-1 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Bubble Temperature</p>
              <p className="text-2xl font-black text-blue-400">{currentScore.toFixed(0)}</p>
              <p className="text-[10px] text-slate-600">Market exuberance</p>
            </div>

            {/* Arrow */}
            <div className="flex flex-col items-center gap-0.5">
              <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
              <span className="text-[9px] text-slate-600">+ stress</span>
            </div>

            {/* Drawdown Risk */}
            <div className="flex-1 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Drawdown Risk</p>
              <p className={`text-2xl font-black ${drawdownRiskScore > 65 ? 'text-red-400' : drawdownRiskScore > 45 ? 'text-orange-400' : 'text-emerald-400'}`}>
                {drawdownRiskScore.toFixed(0)}
              </p>
              <p className="text-[10px] text-slate-600">Crash probability driver</p>
            </div>

            {/* Arrow */}
            <div className="flex flex-col items-center gap-0.5">
              <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
              <span className="text-[9px] text-slate-600">= risk</span>
            </div>

            {/* Drawdown Probability summary */}
            <div className="flex-1 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">P(Drawdown)</p>
              <p className={`text-2xl font-black ${probs[1].probability > 0.3 ? 'text-red-400' : probs[1].probability > 0.15 ? 'text-orange-400' : 'text-emerald-400'}`}>
                {(probs[1].probability * 100).toFixed(0)}%
              </p>
              <p className="text-[10px] text-slate-600">&gt;20% in {model.forward_window_label}</p>
            </div>
          </div>
          <p className="text-[10px] text-slate-600 mt-2 text-center">
            Same 7 indicators, different perspective: Bubble Temperature measures exuberance (high VIX score = complacent),
            Drawdown Risk inverts stress indicators (high VIX = danger). High risk + high temperature = bubble breaking.
          </p>
        </div>
      )}

      {/* OOS Performance Metrics — Logistic component only */}
      {oosMetrics && (
        <div className="bg-slate-800/30 rounded-xl p-4 mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Logistic Component OOS ({(() => {
              const folds10 = model.actual_folds_used?.['drawdown_10pct'];
              const folds20 = model.actual_folds_used?.['drawdown_20pct'];
              if (folds10 != null || folds20 != null) {
                return `${folds10 ?? '?'}/${folds20 ?? '?'} folds, purge=${model.purge_days ?? '?'}d`;
              }
              return model.train_test_split ?? '70/30 split';
            })()})
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-slate-500 text-xs">AUC (10% DD)</span>
              <p className={`font-semibold ${(oosMetrics.c10?.auc_test ?? 0) > 0.7 ? 'text-emerald-400' : (oosMetrics.c10?.auc_test ?? 0) > 0.6 ? 'text-yellow-400' : 'text-red-400'}`}>
                {oosMetrics.c10?.auc_test?.toFixed(3) ?? 'N/A'}
              </p>
            </div>
            <div>
              <span className="text-slate-500 text-xs">BSS (10% DD)</span>
              <p className={`font-semibold ${(oosMetrics.c10?.bss_test ?? 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {oosMetrics.c10?.bss_test !== undefined ? `${oosMetrics.c10.bss_test > 0 ? '+' : ''}${(oosMetrics.c10.bss_test * 100).toFixed(1)}%` : 'N/A'}
              </p>
            </div>
            <div>
              <span className="text-slate-500 text-xs">AUC (20% DD)</span>
              <p className={`font-semibold ${(oosMetrics.c20?.auc_test ?? 0) > 0.7 ? 'text-emerald-400' : (oosMetrics.c20?.auc_test ?? 0) > 0.6 ? 'text-yellow-400' : 'text-red-400'}`}>
                {oosMetrics.c20?.auc_test?.toFixed(3) ?? 'N/A'}
              </p>
            </div>
            <div>
              <span className="text-slate-500 text-xs">BSS (20% DD)</span>
              <p className={`font-semibold ${(oosMetrics.c20?.bss_test ?? 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {oosMetrics.c20?.bss_test !== undefined ? `${oosMetrics.c20.bss_test > 0 ? '+' : ''}${(oosMetrics.c20.bss_test * 100).toFixed(1)}%` : 'N/A'}
              </p>
            </div>
          </div>
          {/* Honest interpretation */}
          {((oosMetrics.c10?.bss_test ?? 0) < 0 || (oosMetrics.c20?.bss_test ?? 0) < 0) && (
            <div className="mt-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3">
              <p className="text-[11px] text-yellow-400 font-semibold mb-1">Model Limitations</p>
              <p className="text-[10px] text-slate-400 leading-relaxed">
                Negative BSS means the logistic regression alone is worse than always predicting the base rate.
                The displayed probabilities use <span className="text-white font-semibold">100% Bayesian Beta-Binomial</span> for all thresholds (10-30% DD).
                Logistic regression is retained for diagnostics only (weight=0).
                The primary signal comes from the Risk Score&apos;s empirical relationship with drawdowns, not the logistic model.
                With ~{model.effective_sample_size ?? 50} effective independent observations, point-prediction models have limited statistical power.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Empirical context */}
      {empirical && (
        <div className="bg-slate-800/30 rounded-xl p-4 mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Historical Context (Score Bin {binKey})
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-slate-500 text-xs">Sample Size</span>
              <p className="text-white font-semibold">{empirical.count} days</p>
            </div>
            <div>
              <span className="text-slate-500 text-xs">Mean Max DD</span>
              <p className="text-white font-semibold">-{empirical.mean_dd?.toFixed(1) ?? 'N/A'}%</p>
            </div>
            <div>
              <span className="text-slate-500 text-xs">Median Max DD</span>
              <p className="text-white font-semibold">-{empirical.median_dd?.toFixed(1) ?? 'N/A'}%</p>
            </div>
            <div>
              <span className="text-slate-500 text-xs">95th Pctile DD</span>
              <p className="text-red-400 font-semibold">-{empirical.p95_dd?.toFixed(1) ?? 'N/A'}%</p>
            </div>
          </div>
        </div>
      )}

      {/* Model description */}
      <div className="bg-slate-800/30 rounded-xl p-4 space-y-1">
        <p className="text-sm text-slate-400">
          <span className="text-white font-semibold">Hybrid 3-layer model v{modelVersion}</span>:
          {isV4 ? (
            <> Extended history (1999+, incl. dot-com &amp; GFC), penalized logistic regression
            with stability-selected features, Bayesian Beta-Binomial + PAVA (10-30%),
            EVT/GPD tail extrapolation (40%), bootstrap 90% confidence intervals.
            Forward window: <span className="text-white">{model.forward_window_days} trading days</span> (~{model.forward_window_label}).</>
          ) : isV3 ? (
            <> Bayesian-dominated blend: 50/50 logistic+Bayesian for 10% DD,
            100% Bayesian Beta-Binomial + PAVA monotonicity for 20-30% DD (binned by Risk Score),
            EVT/GPD tail extrapolation for 40% DD.
            Logistic component uses per-threshold penalized regression with StandardScaler.
            Forward window: <span className="text-white">{model.forward_window_days} trading days</span> (~{model.forward_window_label}).</>
          ) : isV2Plus ? (
            <> Multi-feature L2-regularized logistic regression ({model.feature_names?.length ?? 6} features) for 10-20% thresholds,
            Bayesian Beta-Binomial with PAVA monotonicity (20-30%), EVT/GPD tail extrapolation (40%).
            Forward window: <span className="text-white">{model.forward_window_days} trading days</span> (~{model.forward_window_label}).</>
          ) : (
            <> Logistic regression (10-20% thresholds),
            Bayesian Beta-Binomial with monotonicity (20-30%), EVT/GPD tail extrapolation (40%).
            Forward window: <span className="text-white">{model.forward_window_days} trading days</span> (~{model.forward_window_label}).</>
          )}
        </p>
        <p className="text-[10px] text-slate-600">
          Calibrated on {model.calibration_date} with {model.forward_window_days}-day forward drawdowns.
          {isV2Plus && model.feature_names && (
            <> Features: {model.feature_names.join(', ')}. Train/test split: {model.train_test_split}.</>
          )}
          {' '}Probabilities are conditional on current market conditions (score {currentScore.toFixed(1)}, velocity {scoreVelocity.toFixed(1)}).
          Confidence decreases with drawdown severity due to fewer historical events. Not financial advice.
        </p>
      </div>
    </div>
  );
};

export default CrashProbabilityPanel;
