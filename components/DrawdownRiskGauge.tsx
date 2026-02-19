import React from 'react';

interface DrawdownRiskGaugeProps {
  riskScore: number;
  compositeScore: number;
  indicators: Record<string, { score: number; weight: number; label: string }>;
  generatedAt?: string;
}

// Risk score: higher = more dangerous → red
function getRiskColor(score: number): string {
  if (score < 30) return '#22c55e';   // green — low risk
  if (score < 50) return '#eab308';   // yellow
  if (score < 70) return '#f97316';   // orange
  if (score < 85) return '#ef4444';   // red
  return '#dc2626';                    // dark red — critical
}

function getRiskLevel(score: number): { label: string; badgeClass: string } {
  if (score < 30) return { label: 'Low Risk', badgeClass: 'bg-emerald-500/20 text-emerald-400' };
  if (score < 50) return { label: 'Moderate', badgeClass: 'bg-yellow-500/20 text-yellow-400' };
  if (score < 70) return { label: 'Elevated', badgeClass: 'bg-orange-500/20 text-orange-400' };
  if (score < 85) return { label: 'High Risk', badgeClass: 'bg-red-500/20 text-red-400' };
  return { label: 'Critical', badgeClass: 'bg-red-600/30 text-red-300' };
}

const RISK_INVERSIONS = new Set(['qqq_deviation', 'vix_level', 'yield_curve']);

const DrawdownRiskGauge: React.FC<DrawdownRiskGaugeProps> = ({ riskScore, compositeScore, indicators, generatedAt }) => {
  const color = getRiskColor(riskScore);
  const radius = 112;
  const circumference = 2 * Math.PI * radius;
  const progress = (riskScore / 100) * circumference;
  const riskLevel = getRiskLevel(riskScore);

  // Compute the 3 inverted indicator values for display
  const invertedIndicators = [
    { key: 'vix_level', label: 'VIX Stress', description: 'High VIX = stressed' },
    { key: 'qqq_deviation', label: 'QQQ Weakness', description: 'Low momentum = weak' },
    { key: 'yield_curve', label: 'Curve Risk', description: 'Inverted = recessionary' },
  ].map(({ key, label }) => {
    const ind = indicators[key];
    const invertedScore = ind ? 100 - ind.score : null;
    return { label, score: invertedScore };
  });

  return (
    <div>
      <div className="flex flex-col items-center relative">
        {/* Circular gauge */}
        <div className="relative w-64 h-64 mb-4">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 260 260">
            <circle
              cx="130" cy="130" r={radius}
              fill="transparent"
              stroke="#1e293b"
              strokeWidth="14"
            />
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
            <span className="text-5xl font-black text-white">{riskScore.toFixed(0)}</span>
            <span className="text-xs text-slate-500 font-bold uppercase">Risk Score</span>
            <span className="text-xs text-slate-600 mt-0.5">
              Bubble: {compositeScore.toFixed(0)}
            </span>
          </div>
        </div>

        {/* Relationship note */}
        <p className="text-xs text-slate-500 mb-1">
          Monotonicity: <span className="text-emerald-400 font-semibold">+1.0</span> with P(drawdown)
        </p>

        {/* Risk level badge */}
        <span className={`px-4 py-1.5 rounded-full text-sm font-bold ${riskLevel.badgeClass}`}>
          {riskLevel.label}
        </span>

        {/* Inverted indicators */}
        <div className="grid grid-cols-3 gap-4 mt-6 w-full">
          {invertedIndicators.map(({ label, score }) => (
            <div key={label} className="bg-slate-800/50 rounded-lg p-3 text-center">
              <p className="text-xs text-slate-500 font-semibold uppercase mb-1">{label}</p>
              <p className="text-xl font-bold" style={{ color: score != null ? getRiskColor(score) : '#64748b' }}>
                {score != null ? score.toFixed(0) : '--'}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DrawdownRiskGauge;
