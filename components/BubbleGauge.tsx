import React from 'react';

interface BubbleGaugeProps {
  compositeScore: number;
  regime: string;
  sentimentScore: number | null;
  liquidityScore: number | null;
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

const BubbleGauge: React.FC<BubbleGaugeProps> = ({ compositeScore, regime, sentimentScore, liquidityScore }) => {
  const color = getScoreColor(compositeScore);
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const progress = (compositeScore / 100) * circumference;

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
      <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
        <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
        </svg>
        Bubble Index
      </h3>

      <div className="flex flex-col items-center">
        {/* Circular gauge */}
        <div className="relative w-48 h-48 mb-4">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
            {/* Background circle */}
            <circle
              cx="100" cy="100" r={radius}
              fill="transparent"
              stroke="#1e293b"
              strokeWidth="12"
            />
            {/* Progress arc */}
            <circle
              cx="100" cy="100" r={radius}
              fill="transparent"
              stroke={color}
              strokeWidth="12"
              strokeLinecap="round"
              strokeDasharray={`${progress} ${circumference}`}
              className="transition-all duration-1000"
              style={{ filter: `drop-shadow(0 0 8px ${color})` }}
            />
          </svg>
          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-black text-white">{compositeScore.toFixed(0)}</span>
            <span className="text-xs text-slate-500 font-bold uppercase">Composite</span>
          </div>
        </div>

        {/* Regime badge */}
        <span className={`px-4 py-1.5 rounded-full text-sm font-bold ${getRegimeBadgeClass(regime)}`}>
          {getRegimeLabel(regime)}
        </span>

        {/* Sub-scores */}
        <div className="grid grid-cols-2 gap-4 mt-6 w-full">
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
        </div>
      </div>
    </div>
  );
};

export default BubbleGauge;
