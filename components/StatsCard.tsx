
import React from 'react';
import { MarketSummary } from '../types';

interface Props {
  summary: MarketSummary;
}

const StatsCard: React.FC<Props> = ({ summary }) => {
  const isDanger = summary.currentIndex >= 80;

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <div className="bg-white p-5 rounded-xl border border-slate-100 shadow-sm">
        <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">Current QQQ Price</p>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-slate-800">${summary.currentPrice.toFixed(2)}</span>
          <span className={`text-sm font-medium ${summary.change24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {summary.change24h >= 0 ? '+' : ''}{summary.change24h.toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="bg-white p-5 rounded-xl border border-slate-100 shadow-sm">
        <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">200-Day SMA</p>
        <span className="text-2xl font-bold text-slate-800">${summary.currentSMA.toFixed(2)}</span>
      </div>

      <div className="bg-white p-5 rounded-xl border border-slate-100 shadow-sm">
        <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-1">Deviation Index</p>
        <span className={`text-2xl font-bold ${isDanger ? 'text-red-600' : 'text-blue-600'}`}>
          {summary.currentIndex.toFixed(1)}
        </span>
      </div>

      <div className={`p-5 rounded-xl border shadow-sm flex flex-col justify-center ${
        isDanger ? 'bg-red-50 border-red-100' : 'bg-green-50 border-green-100'
      }`}>
        <p className={`text-xs font-semibold uppercase tracking-wider mb-1 ${
          isDanger ? 'text-red-600' : 'text-green-600'
        }`}>Risk Signal</p>
        <div className="flex items-center gap-2">
          <span className={`text-2xl font-black ${isDanger ? 'text-red-700' : 'text-green-700'}`}>
            {summary.riskLevel}
          </span>
          {isDanger && (
            <i className="fa-solid fa-triangle-exclamation text-red-600 animate-pulse"></i>
          )}
        </div>
      </div>
    </div>
  );
};

export default StatsCard;
