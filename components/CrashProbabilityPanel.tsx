import React, { useMemo } from 'react';
import { DrawdownModelData } from '../services/bubbleService';
import { BubbleIndicator, BubbleHistoryPoint } from '../types';

interface Props {
  model: DrawdownModelData;
  currentScore: number;
  scoreVelocity: number;
  indicators?: Record<string, BubbleIndicator>;
  history?: BubbleHistoryPoint[];
}

// ---------------------------------------------------------------------------
// Probability computation — v2.0 multi-feature + v1.0 fallback
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
  const values: Record<string, number> = {
    composite_score: currentScore,
    score_velocity: scoreVelocity,
  };

  // Map indicator names — extract .score from BubbleIndicator
  if (indicators) {
    values.ind_vix_level = indicators.vix_level?.score ?? 50;
    values.ind_credit_spread = indicators.credit_spread?.score ?? 50;
    values.ind_qqq_deviation = indicators.qqq_deviation?.score ?? 50;
  }

  // Compute score_sma_60d from history
  if (history && history.length >= 20) {
    const recent = history.slice(-60);
    const sum = recent.reduce((acc, h) => acc + h.composite_score, 0);
    values.score_sma_60d = sum / recent.length;
  } else {
    values.score_sma_60d = currentScore;
  }

  // Fallback to model's pre-computed current features
  if (model.current_features) {
    for (const [key, val] of Object.entries(model.current_features)) {
      if (!(key in values) || values[key] === 50) {
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
}

function computeDrawdownProbabilities(
  model: DrawdownModelData,
  featureValues: Record<string, number>,
): DrawdownProb[] {
  const { logistic_coefficients: logCoefs, bayesian_lookup: bayesian, evt_parameters: evt } = model;

  // --- Layer 1: Multi-feature logistic for 10% ---
  const c10 = logCoefs['drawdown_10pct'];
  let p10 = 0;
  if (c10?.weights && c10?.intercept !== undefined) {
    // v2.0 multi-feature model
    let z = c10.intercept;
    for (const [feat, w] of Object.entries(c10.weights)) {
      z += w * (featureValues[feat] ?? 0);
    }
    p10 = sigmoid(z);
  } else if (c10) {
    // v1.0 fallback
    const score = featureValues.composite_score ?? 0;
    const vel = featureValues.score_velocity ?? 0;
    p10 = sigmoid(
      (c10.a_score_with_velocity ?? 0) * score +
      (c10.a_velocity ?? 0) * vel +
      (c10.b_with_velocity ?? 0)
    );
  }

  // --- Layer 1 + 2: Blend for 20% ---
  const c20 = logCoefs['drawdown_20pct'];
  let p20_logistic = 0;
  if (c20?.weights && c20?.intercept !== undefined) {
    let z = c20.intercept;
    for (const [feat, w] of Object.entries(c20.weights)) {
      z += w * (featureValues[feat] ?? 0);
    }
    p20_logistic = sigmoid(z);
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
    ? interpolateLookup(featureValues.composite_score ?? 0, b20.bin_centers, b20.probabilities)
    : p20_logistic;
  const p20 = 0.5 * p20_logistic + 0.5 * p20_bayesian;

  // --- Layer 2: Bayesian for 30% ---
  const b30 = bayesian['drawdown_30pct'];
  const p30 = b30
    ? interpolateLookup(featureValues.composite_score ?? 0, b30.bin_centers, b30.probabilities)
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
    },
    {
      threshold: 20,
      label: '20%+',
      probability: Math.min(p20, 0.99),
      confidence: model.confidence_tiers['20pct'] ?? 'low',
      color: '#f97316',
      bgColor: 'bg-orange-500/10 border-orange-500/20',
    },
    {
      threshold: 30,
      label: '30%+',
      probability: Math.min(p30, 0.99),
      confidence: model.confidence_tiers['30pct'] ?? 'model_dependent',
      color: '#ef4444',
      bgColor: 'bg-red-500/10 border-red-500/20',
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
  moderate: { label: 'Moderate', color: 'text-emerald-400' },
  low: { label: 'Low', color: 'text-yellow-400' },
  model_dependent: { label: 'Model-Dep.', color: 'text-orange-400' },
  extrapolated: { label: 'Extrapolated', color: 'text-red-400' },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const CrashProbabilityPanel: React.FC<Props> = ({ model, currentScore, scoreVelocity, indicators, history }) => {
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

  const isV2 = model.model_version === '2.0';

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
          {isV2 && (
            <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
              v2.0
            </span>
          )}
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

              {/* Bar */}
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: p.color }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* OOS Performance Metrics (v2.0 only) */}
      {oosMetrics && (
        <div className="bg-slate-800/30 rounded-xl p-4 mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Out-of-Sample Performance ({model.train_test_split ?? '70/30'} split)
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-slate-500 text-xs">AUC (10% DD)</span>
              <p className={`font-semibold ${(oosMetrics.c10?.auc_test ?? 0) > 0.6 ? 'text-emerald-400' : 'text-yellow-400'}`}>
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
              <p className={`font-semibold ${(oosMetrics.c20?.auc_test ?? 0) > 0.6 ? 'text-emerald-400' : 'text-yellow-400'}`}>
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
          <span className="text-white font-semibold">Hybrid 3-layer model {isV2 ? 'v2.0' : ''}</span>:
          {isV2 ? (
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
          {isV2 && model.feature_names && (
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
