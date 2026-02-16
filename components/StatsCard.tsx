
import React from 'react';
import { MarketSummary } from '../types';

interface Props {
  summary: MarketSummary;
}

const StatsCard: React.FC<Props> = ({ summary }) => {
  const isDanger = summary.currentIndex >= 80;

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shadow-sm">
        <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">Current QQQ Price</p>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-white">${summary.currentPrice.toFixed(2)}</span>
          <span className={`text-sm font-medium ${summary.change24h >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {summary.change24h >= 0 ? '+' : ''}{summary.change24h.toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shadow-sm">
        <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">200-Day SMA</p>
        <span className="text-2xl font-bold text-white">${summary.currentSMA.toFixed(2)}</span>
      </div>

      <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shadow-sm">
        <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">Deviation Index</p>
        <span className={`text-2xl font-bold ${isDanger ? 'text-red-400' : 'text-blue-400'}`}>
          {summary.currentIndex.toFixed(1)}
        </span>
      </div>

      <div className={`p-5 rounded-xl border shadow-sm flex flex-col justify-center ${
        isDanger ? 'bg-red-500/10 border-red-500/20' : 'bg-emerald-500/10 border-emerald-500/20'
      }`}>
        <p className={`text-xs font-semibold uppercase tracking-wider mb-1 ${
          isDanger ? 'text-red-400' : 'text-emerald-400'
        }`}>Risk Signal</p>
        <div className="flex items-center gap-2">
          <span className={`text-2xl font-black ${isDanger ? 'text-red-400' : 'text-emerald-400'}`}>
            {summary.riskLevel}
          </span>
          {isDanger && (
            <svg className="w-5 h-5 text-red-400 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
              <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
            </svg>
          )}
        </div>
      </div>
    </div>
  );
};

export default StatsCard;
