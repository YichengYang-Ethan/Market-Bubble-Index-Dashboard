import React from 'react';
import { BubbleIndicator, IndicatorMeta } from '../types';
import { INDICATOR_META, RISK_INVERSIONS } from '../constants';

interface IndicatorCardProps {
  indicatorKey: string;
  indicator: BubbleIndicator;
  previousScore?: number | null;
  sparklineData: (number | null)[];
  perspective?: 'bubble' | 'risk';
}

function getScoreColor(score: number): string {
  if (score < 30) return '#22c55e';
  if (score < 70) return '#eab308';
  if (score < 85) return '#f97316';
  return '#ef4444';
}

function getScoreTextClass(score: number): string {
  if (score < 30) return 'text-emerald-400';
  if (score < 70) return 'text-yellow-400';
  if (score < 85) return 'text-orange-400';
  return 'text-red-400';
}

function buildSparklinePath(data: (number | null)[]): string {
  const filtered = data
    .map((v, i) => (v != null ? { x: i, y: v } : null))
    .filter((p): p is { x: number; y: number } => p !== null);

  if (filtered.length < 2) return '';

  const width = 120;
  const height = 32;
  const maxX = data.length - 1;
  const points = filtered.map(p => {
    const px = maxX > 0 ? (p.x / maxX) * width : 0;
    const py = height - (p.y / 100) * height;
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  });

  return points.join(' ');
}

const IndicatorCard: React.FC<IndicatorCardProps> = ({ indicatorKey, indicator, previousScore, sparklineData, perspective = 'bubble' }) => {
  const meta = INDICATOR_META.find((m: IndicatorMeta) => m.key === indicatorKey);
  const color = meta?.color ?? '#64748b';
  const label = meta?.label ?? indicatorKey;
  const weightPercent = Math.round(indicator.weight * 100);
  const isInverted = perspective === 'risk' && RISK_INVERSIONS.has(indicatorKey);
  const displayScore = isInverted ? 100 - indicator.score : indicator.score;
  const displayPrevious = isInverted && previousScore != null ? 100 - previousScore : previousScore;

  const trendDelta = displayPrevious != null ? displayScore - displayPrevious : 0;
  let trendArrow: string;
  let trendColor: string;
  if (trendDelta > 0.5) {
    trendArrow = '\u25B2'; // up triangle
    trendColor = '#ef4444'; // red — score increased (bad)
  } else if (trendDelta < -0.5) {
    trendArrow = '\u25BC'; // down triangle
    trendColor = '#22c55e'; // green — score decreased (good)
  } else {
    trendArrow = '\u25C6'; // diamond for flat
    trendColor = '#64748b'; // gray
  }

  const effectiveSparklineData = isInverted
    ? sparklineData.map(v => v != null ? 100 - v : null)
    : sparklineData;
  const sparklinePath = buildSparklinePath(effectiveSparklineData);

  return (
    <div
      className="bg-slate-900 rounded-xl border border-slate-800 p-4 shadow-lg"
      style={{ borderLeftWidth: '4px', borderLeftColor: color }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-semibold text-slate-300">{label}</h4>
          {isInverted && (
            <span className="text-[9px] font-bold text-red-400 bg-red-500/15 border border-red-500/20 rounded px-1.5 py-0.5 uppercase tracking-wider">
              Inverted
            </span>
          )}
        </div>
        <span className="text-[10px] font-bold text-slate-500 bg-slate-800 rounded-full px-2 py-0.5">
          {weightPercent}%
        </span>
      </div>

      {/* Score + trend */}
      <div className="flex items-baseline gap-2 mb-3">
        <span className={`text-3xl font-black ${getScoreTextClass(displayScore)}`}>
          {displayScore.toFixed(0)}
        </span>
        <span className="text-lg font-bold" style={{ color: trendColor }}>
          {trendArrow}
        </span>
      </div>

      {/* Mini sparkline */}
      {sparklinePath && (
        <svg
          width="120"
          height="32"
          viewBox="0 0 120 32"
          className="mt-1"
          style={{ overflow: 'visible' }}
        >
          <polyline
            points={sparklinePath}
            fill="none"
            stroke={color}
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </svg>
      )}
    </div>
  );
};

export default IndicatorCard;
