
import React, { useState, useMemo } from 'react';
import { BubbleHistoryPoint, DataPoint } from '../types';

interface BubbleBacktestResult {
  numTrades: number;
  strategyReturn: number;
  strategyAnnualizedReturn: number;
  buyHoldReturn: number;
  buyHoldAnnualizedReturn: number;
  maxDrawdown: number;
  signals: { date: string; type: 'buy' | 'sell'; price: number; score: number }[];
  finalValue: number;
}

interface Props {
  history: BubbleHistoryPoint[];
  priceData: DataPoint[];
  ticker: string;
}

function runBubbleBacktest(
  history: BubbleHistoryPoint[],
  priceData: DataPoint[],
  buyThreshold: number,
  sellThreshold: number,
  initialInvestment: number,
): BubbleBacktestResult | null {
  // Build price lookup by date
  const priceMap = new Map<string, number>();
  for (const p of priceData) {
    priceMap.set(p.date, p.price);
  }

  // Filter history to only dates with price data
  const merged = history
    .filter(h => priceMap.has(h.date))
    .map(h => ({ date: h.date, score: h.composite_score, price: priceMap.get(h.date)! }));

  if (merged.length < 2) return null;

  const signals: BubbleBacktestResult['signals'] = [];
  let cash = initialInvestment;
  let shares = 0;
  let inPosition = false;
  let peakValue = initialInvestment;
  let maxDrawdown = 0;

  for (const point of merged) {
    const portfolioValue = inPosition ? shares * point.price : cash;

    if (portfolioValue > peakValue) peakValue = portfolioValue;
    const drawdown = (peakValue - portfolioValue) / peakValue;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;

    // Buy when bubble score drops below threshold (market cheap / fear)
    if (!inPosition && point.score <= buyThreshold) {
      shares = cash / point.price;
      cash = 0;
      inPosition = true;
      signals.push({ date: point.date, type: 'buy', price: point.price, score: point.score });
    }
    // Sell when bubble score rises above threshold (market frothy)
    else if (inPosition && point.score >= sellThreshold) {
      cash = shares * point.price;
      shares = 0;
      inPosition = false;
      signals.push({ date: point.date, type: 'sell', price: point.price, score: point.score });
    }
  }

  const lastPrice = merged[merged.length - 1].price;
  const firstPrice = merged[0].price;
  const finalValue = inPosition ? shares * lastPrice : cash;
  const strategyReturn = (finalValue - initialInvestment) / initialInvestment;
  const buyHoldReturn = (lastPrice - firstPrice) / firstPrice;

  const firstDate = new Date(merged[0].date);
  const lastDate = new Date(merged[merged.length - 1].date);
  const years = (lastDate.getTime() - firstDate.getTime()) / (365.25 * 24 * 60 * 60 * 1000);
  const annualize = (r: number) => years > 0 ? Math.pow(1 + r, 1 / years) - 1 : 0;

  return {
    numTrades: signals.length,
    strategyReturn,
    strategyAnnualizedReturn: annualize(strategyReturn),
    buyHoldReturn,
    buyHoldAnnualizedReturn: annualize(buyHoldReturn),
    maxDrawdown,
    signals,
    finalValue,
  };
}

const BubbleBacktestPanel: React.FC<Props> = ({ history, priceData, ticker }) => {
  const [buyThreshold, setBuyThreshold] = useState(30);
  const [sellThreshold, setSellThreshold] = useState(70);
  const [initialInvestment, setInitialInvestment] = useState(10000);

  const result = useMemo(() => {
    return runBubbleBacktest(history, priceData, buyThreshold, sellThreshold, initialInvestment);
  }, [history, priceData, buyThreshold, sellThreshold, initialInvestment]);

  if (!result) return null;

  const fmt = (v: number) => (v * 100).toFixed(1) + '%';
  const fmtMoney = (v: number) => '$' + v.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 });

  const strategyBetter = result.strategyReturn > result.buyHoldReturn;

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-sm">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">Bubble Index Backtest</h2>
          <p className="text-sm text-slate-500">Buy low / sell high based on composite bubble score thresholds</p>
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
            min={5}
            max={50}
            value={buyThreshold}
            onChange={e => setBuyThreshold(Number(e.target.value))}
            className="w-full accent-emerald-500"
          />
          <p className="text-[10px] text-slate-600 mt-1">Buy {ticker} when bubble score drops below this level</p>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Sell Threshold: <span className="text-red-400">{sellThreshold}</span>
          </label>
          <input
            type="range"
            min={50}
            max={95}
            value={sellThreshold}
            onChange={e => setSellThreshold(Number(e.target.value))}
            className="w-full accent-red-500"
          />
          <p className="text-[10px] text-slate-600 mt-1">Sell {ticker} when bubble score rises above this level</p>
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

      {/* Trade log */}
      {result.signals.length > 0 && (
        <div className="mb-6">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Recent Trades</p>
          <div className="flex flex-wrap gap-2 max-h-24 overflow-y-auto">
            {result.signals.slice(-12).map((s, i) => (
              <span
                key={i}
                className={`text-xs px-2 py-1 rounded ${
                  s.type === 'buy'
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'bg-red-500/20 text-red-400'
                }`}
              >
                {s.type.toUpperCase()} {s.date} @ ${s.price.toFixed(0)} (score {s.score.toFixed(0)})
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          {strategyBetter ? (
            <span>Strategy <span className="text-emerald-400 font-semibold">outperformed</span> buy-and-hold by {fmt(result.strategyReturn - result.buyHoldReturn)}</span>
          ) : (
            <span>Strategy <span className="text-red-400 font-semibold">underperformed</span> buy-and-hold by {fmt(result.buyHoldReturn - result.strategyReturn)}</span>
          )}
          {' '}with {fmtMoney(initialInvestment)} initial investment.
          {' '}Final value: <span className="text-white font-semibold">{fmtMoney(result.finalValue)}</span>
        </p>
      </div>
    </div>
  );
};

export default BubbleBacktestPanel;
