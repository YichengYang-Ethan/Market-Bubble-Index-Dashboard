import React, { useState } from 'react';
import { BubbleHistoryPoint } from '../types';
import { INDICATOR_META } from '../constants';
import IndicatorDetailChart from './IndicatorDetailChart';

interface IndicatorDeepDiveProps {
  history: BubbleHistoryPoint[];
}

function getScoreBadgeClass(score: number): string {
  if (score < 30) return 'bg-emerald-500/20 text-emerald-400';
  if (score < 70) return 'bg-yellow-500/20 text-yellow-400';
  if (score < 85) return 'bg-orange-500/20 text-orange-400';
  return 'bg-red-500/20 text-red-400';
}

const IndicatorDeepDive: React.FC<IndicatorDeepDiveProps> = ({ history }) => {
  const [openSections, setOpenSections] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    INDICATOR_META.forEach((meta, index) => {
      initial[meta.key] = index === 0;
    });
    return initial;
  });

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // For each indicator, find its most recent non-null score from history.
  // Different data sources (e.g. FRED, SKEW) may lag by a day or two,
  // so the absolute last history entry can have nulls for some indicators.
  const latestScores: Record<string, number | null> = {};
  for (const meta of INDICATOR_META) {
    let score: number | null = null;
    for (let i = history.length - 1; i >= 0; i--) {
      const val = history[i].indicators?.[meta.key];
      if (val != null) { score = val; break; }
    }
    latestScores[meta.key] = score;
  }

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5" />
        </svg>
        Indicator Deep Dive
      </h3>

      <div className="space-y-2">
        {INDICATOR_META.map((meta) => {
          const isOpen = openSections[meta.key] ?? false;
          const currentScore = latestScores[meta.key];

          return (
            <div key={meta.key} className="border border-slate-800 rounded-xl overflow-hidden">
              {/* Accordion header */}
              <button
                onClick={() => toggleSection(meta.key)}
                className="w-full flex items-center justify-between px-4 py-3 bg-slate-800/40 hover:bg-slate-800/70 transition-colors text-left"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: meta.color }}
                  />
                  <span className="text-sm font-semibold text-slate-200">{meta.label}</span>
                  {currentScore != null && (
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${getScoreBadgeClass(currentScore)}`}>
                      {currentScore.toFixed(0)}
                    </span>
                  )}
                </div>
                <svg
                  className={`w-4 h-4 text-slate-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                </svg>
              </button>

              {/* Accordion content */}
              {isOpen && (
                <div className="px-4 py-4 bg-slate-900/50">
                  <IndicatorDetailChart
                    indicatorKey={meta.key}
                    history={history}
                    color={meta.color}
                  />
                  <p className="text-xs text-slate-500 mt-3 leading-relaxed">
                    {meta.description}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default IndicatorDeepDive;
