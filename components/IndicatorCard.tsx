import React from 'react';
import { BubbleIndicator, IndicatorMeta } from '../types';
import { INDICATOR_META } from '../constants';

interface IndicatorCardProps {
  indicatorKey: string;
  indicator: BubbleIndicator;
  previousScore?: number | null;
  sparklineData: (number | null)[];
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

const IndicatorCard: React.FC<IndicatorCardProps> = ({ indicatorKey, indicator, previousScore, sparklineData }) => {
  const meta = INDICATOR_META.find((m: IndicatorMeta) => m.key === indicatorKey);
  const color = meta?.color ?? '#64748b';
  const label = meta?.label ?? indicatorKey;
  const weightPercent = Math.round(indicator.weight * 100);

  const trendDelta = previousScore != null ? indicator.score - previousScore : 0;
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

  const sparklinePath = buildSparklinePath(sparklineData);

  return (
    <div
      className="bg-slate-900 rounded-xl border border-slate-800 p-4 shadow-lg"
      style={{ borderLeftWidth: '4px', borderLeftColor: color }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-slate-300">{label}</h4>
        <span className="text-[10px] font-bold text-slate-500 bg-slate-800 rounded-full px-2 py-0.5">
          {weightPercent}%
        </span>
      </div>

      {/* Score + trend */}
      <div className="flex items-baseline gap-2 mb-3">
        <span className={`text-3xl font-black ${getScoreTextClass(indicator.score)}`}>
          {indicator.score.toFixed(0)}
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
