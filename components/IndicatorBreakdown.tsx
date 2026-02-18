import React from 'react';
import { BubbleIndicator } from '../types';

interface IndicatorBreakdownProps {
  indicators: Record<string, BubbleIndicator>;
}

function getBarColor(score: number): string {
  if (score < 30) return 'bg-emerald-500';
  if (score < 50) return 'bg-emerald-400';
  if (score < 70) return 'bg-yellow-500';
  if (score < 85) return 'bg-orange-500';
  return 'bg-red-500';
}

function getTextColor(score: number): string {
  if (score < 30) return 'text-emerald-400';
  if (score < 50) return 'text-emerald-300';
  if (score < 70) return 'text-yellow-400';
  if (score < 85) return 'text-orange-400';
  return 'text-red-400';
}

const IndicatorBreakdown: React.FC<IndicatorBreakdownProps> = ({ indicators }) => {
  const sorted = Object.entries(indicators).sort(([, a], [, b]) => b.score - a.score);

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25H12" />
        </svg>
        Indicator Breakdown
      </h3>

      <div className="space-y-4">
        {sorted.map(([key, indicator]) => (
          <div key={key}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-sm font-medium text-slate-300">{indicator.label}</span>
              <div className="flex items-center gap-3">
                <span className="text-xs text-slate-500 font-mono">
                  w: {(indicator.weight * 100).toFixed(0)}%
                </span>
                <span className={`text-sm font-bold ${getTextColor(indicator.score)}`}>
                  {indicator.score.toFixed(0)}
                </span>
              </div>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ${getBarColor(indicator.score)}`}
                style={{ width: `${Math.max(2, indicator.score)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default IndicatorBreakdown;
