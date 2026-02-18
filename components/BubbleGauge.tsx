import React from 'react';

interface BubbleGaugeProps {
  compositeScore: number;
  regime: string;
  sentimentScore: number | null;
  liquidityScore: number | null;
  valuationScore?: number | null;
  generatedAt?: string;
  scoreVelocity?: number;
  confidenceInterval?: { lower: number; upper: number };
  dataQuality?: {
    indicators_available: number;
    indicators_total: number;
    completeness: number;
    staleness_warning: boolean;
  };
}

function getScoreColor(score: number): string {
  if (score < 30) return '#22c55e';   // green-500
  if (score < 50) return '#22c55e';   // green-500
  if (score < 70) return '#eab308';   // yellow-500
  if (score < 85) return '#f97316';   // orange-500
  return '#ef4444';                    // red-500
}

function getRegimeLabel(regime: string): string {
  switch (regime) {
    case 'LOW': return 'Low Risk';
    case 'MODERATE': return 'Moderate';
    case 'ELEVATED': return 'Elevated';
    case 'HIGH': return 'High Risk';
    case 'EXTREME': return 'Extreme';
    default: return regime;
  }
}

function getRegimeBadgeClass(regime: string): string {
  switch (regime) {
    case 'LOW': return 'bg-emerald-500/20 text-emerald-400';
    case 'MODERATE': return 'bg-yellow-500/20 text-yellow-400';
    case 'ELEVATED': return 'bg-orange-500/20 text-orange-400';
    case 'HIGH': return 'bg-red-500/20 text-red-400';
    case 'EXTREME': return 'bg-red-600/30 text-red-300';
    default: return 'bg-slate-500/20 text-slate-400';
  }
}

function getVelocityArrow(velocity: number): { arrow: string; color: string } {
  if (velocity < -5) return { arrow: '\u2193', color: '#22c55e' };   // strong falling
  if (velocity < -1) return { arrow: '\u2198', color: '#4ade80' };   // falling
  if (velocity <= 1) return { arrow: '\u2192', color: '#94a3b8' };   // flat
  if (velocity <= 5) return { arrow: '\u2197', color: '#fb923c' };   // rising
  return { arrow: '\u2191', color: '#ef4444' };                       // strong rising
}

function formatGeneratedAt(isoString: string): string {
  const d = new Date(isoString);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

const BubbleGauge: React.FC<BubbleGaugeProps> = ({ compositeScore, regime, sentimentScore, liquidityScore, valuationScore, generatedAt, scoreVelocity, confidenceInterval, dataQuality }) => {
  const color = getScoreColor(compositeScore);
  const radius = 112;
  const circumference = 2 * Math.PI * radius;
  const progress = (compositeScore / 100) * circumference;

  const velocityInfo = scoreVelocity != null ? getVelocityArrow(scoreVelocity) : null;

  return (
    <div>
      <div className="flex flex-col items-center relative">
        {/* Data quality badge */}
        {dataQuality && (
          <div className={`absolute top-0 right-0 px-2 py-1 rounded-lg text-xs font-bold ${
            dataQuality.staleness_warning
              ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
              : 'bg-slate-800 text-slate-400 border border-slate-700'
          }`}>
            {dataQuality.indicators_available}/{dataQuality.indicators_total}
          </div>
        )}

        {/* Circular gauge */}
        <div className="relative w-64 h-64 mb-4">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 260 260">
            {/* Background circle */}
            <circle
              cx="130" cy="130" r={radius}
              fill="transparent"
              stroke="#1e293b"
              strokeWidth="14"
            />
            {/* Progress arc */}
            <circle
              cx="130" cy="130" r={radius}
              fill="transparent"
              stroke={color}
              strokeWidth="14"
              strokeLinecap="round"
              strokeDasharray={`${progress} ${circumference}`}
              className="transition-all duration-1000"
              style={{ filter: `drop-shadow(0 0 8px ${color})` }}
            />
          </svg>
          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="flex items-center gap-2">
              <span className="text-5xl font-black text-white">{compositeScore.toFixed(0)}</span>
              {velocityInfo && (
                <span className="text-2xl font-bold" style={{ color: velocityInfo.color }}>
                  {velocityInfo.arrow}
                </span>
              )}
            </div>
            <span className="text-xs text-slate-500 font-bold uppercase">Composite</span>
            {confidenceInterval && (
              <span className="text-xs text-slate-500 mt-0.5">
                [{confidenceInterval.lower.toFixed(1)} - {confidenceInterval.upper.toFixed(1)}]
              </span>
            )}
          </div>
        </div>

        {/* Velocity label */}
        {scoreVelocity != null && (
          <p className="text-xs text-slate-500 mb-1">
            Velocity: <span style={{ color: velocityInfo?.color }} className="font-semibold">{scoreVelocity > 0 ? '+' : ''}{scoreVelocity.toFixed(1)}</span> pts/wk
          </p>
        )}

        {/* Updated timestamp */}
        {generatedAt && (
          <p className="text-sm text-slate-500 mb-3">Updated: {formatGeneratedAt(generatedAt)}</p>
        )}

        {/* Regime badge */}
        <span className={`px-4 py-1.5 rounded-full text-sm font-bold ${getRegimeBadgeClass(regime)}`}>
          {getRegimeLabel(regime)}
        </span>

        {/* Sub-scores */}
        <div className="grid grid-cols-3 gap-4 mt-6 w-full">
          <div className="bg-slate-800/50 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-500 font-semibold uppercase mb-1">Sentiment</p>
            <p className="text-xl font-bold" style={{ color: sentimentScore != null ? getScoreColor(sentimentScore) : '#64748b' }}>
              {sentimentScore != null ? sentimentScore.toFixed(0) : '--'}
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-500 font-semibold uppercase mb-1">Liquidity</p>
            <p className="text-xl font-bold" style={{ color: liquidityScore != null ? getScoreColor(liquidityScore) : '#64748b' }}>
              {liquidityScore != null ? liquidityScore.toFixed(0) : '--'}
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-500 font-semibold uppercase mb-1">Valuation</p>
            <p className="text-xl font-bold" style={{ color: valuationScore != null ? getScoreColor(valuationScore) : '#64748b' }}>
              {valuationScore != null ? valuationScore.toFixed(0) : '--'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BubbleGauge;
