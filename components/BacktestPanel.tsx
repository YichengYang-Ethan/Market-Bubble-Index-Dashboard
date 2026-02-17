
import React, { useState, useMemo } from 'react';
import { DataPoint, BacktestResult } from '../types';
import { runBacktest } from '../services/dataService';

interface Props {
  data: DataPoint[];
  ticker: string;
  onSignalsChange: (signals: BacktestResult['signals']) => void;
}

const BacktestPanel: React.FC<Props> = ({ data, ticker, onSignalsChange }) => {
  const [buyThreshold, setBuyThreshold] = useState(20);
  const [sellThreshold, setSellThreshold] = useState(80);
  const [initialInvestment, setInitialInvestment] = useState(10000);
  const [showOverlay, setShowOverlay] = useState(false);

  const result = useMemo(() => {
    if (data.length < 2) return null;
    return runBacktest(data, buyThreshold, sellThreshold, initialInvestment);
  }, [data, buyThreshold, sellThreshold, initialInvestment]);

  const handleToggleOverlay = () => {
    const next = !showOverlay;
    setShowOverlay(next);
    onSignalsChange(next && result ? result.signals : []);
  };

  // Update overlay when result changes and overlay is on
  React.useEffect(() => {
    if (showOverlay && result) {
      onSignalsChange(result.signals);
    }
  }, [result, showOverlay, onSignalsChange]);

  if (!result) return null;

  const fmt = (v: number) => (v * 100).toFixed(1) + '%';
  const fmtMoney = (v: number) => '$' + v.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 });

  const strategyBetter = result.strategyReturn > result.buyHoldReturn;

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-sm">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">Strategy Backtest</h2>
          <p className="text-sm text-slate-500">Buy low / sell high based on deviation index thresholds</p>
        </div>
        <span className="text-xs text-slate-500 bg-slate-800 px-3 py-1 rounded-full">{ticker}</span>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div>
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Buy Threshold: <span className="text-emerald-400">{buyThreshold}</span>
          </label>
          <input
            type="range"
            min={0}
            max={50}
            value={buyThreshold}
            onChange={e => setBuyThreshold(Number(e.target.value))}
            className="w-full accent-emerald-500"
          />
          <p className="text-[10px] text-slate-600 mt-1">Buy when index drops below this level</p>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Sell Threshold: <span className="text-red-400">{sellThreshold}</span>
          </label>
          <input
            type="range"
            min={50}
            max={100}
            value={sellThreshold}
            onChange={e => setSellThreshold(Number(e.target.value))}
            className="w-full accent-red-500"
          />
          <p className="text-[10px] text-slate-600 mt-1">Sell when index rises above this level</p>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Initial Investment
          </label>
          <input
            type="number"
            min={1000}
            step={1000}
            value={initialInvestment}
            onChange={e => setInitialInvestment(Math.max(1000, Number(e.target.value)))}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      {/* Results */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Trades</p>
          <p className="text-lg font-bold text-white">{result.numTrades}</p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Strategy Return</p>
          <p className={`text-lg font-bold ${result.strategyReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {fmt(result.strategyReturn)}
          </p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Strategy CAGR</p>
          <p className={`text-lg font-bold ${result.strategyAnnualizedReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {fmt(result.strategyAnnualizedReturn)}
          </p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Buy & Hold</p>
          <p className={`text-lg font-bold ${result.buyHoldReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {fmt(result.buyHoldReturn)}
          </p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">B&H CAGR</p>
          <p className={`text-lg font-bold ${result.buyHoldAnnualizedReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {fmt(result.buyHoldAnnualizedReturn)}
          </p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Max Drawdown</p>
          <p className="text-lg font-bold text-red-400">-{fmt(result.maxDrawdown)}</p>
        </div>
      </div>

      {/* Summary + overlay toggle */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          {strategyBetter ? (
            <span>Strategy <span className="text-emerald-400 font-semibold">outperformed</span> buy-and-hold by {fmt(result.strategyReturn - result.buyHoldReturn)}</span>
          ) : (
            <span>Strategy <span className="text-red-400 font-semibold">underperformed</span> buy-and-hold by {fmt(result.buyHoldReturn - result.strategyReturn)}</span>
          )}
          {' '}with {fmtMoney(initialInvestment)} initial investment.
        </p>
        <button
          onClick={handleToggleOverlay}
          className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
            showOverlay
              ? 'bg-blue-600 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
          }`}
        >
          {showOverlay ? 'Hide' : 'Show'} Signals on Chart
        </button>
      </div>
    </div>
  );
};

export default BacktestPanel;
