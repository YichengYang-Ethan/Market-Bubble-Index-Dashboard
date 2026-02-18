
import React, { useState, useMemo } from 'react';
import { BubbleHistoryPoint, DataPoint } from '../types';

interface RiskMetrics {
  totalReturn: number;
  cagr: number;
  maxDrawdown: number;
  annualizedVol: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  worstDay: number;
  exposure: number;
}

interface BubbleBacktestResult {
  numTrades: number;
  signals: { date: string; type: 'buy' | 'sell'; price: number; score: number }[];
  finalValue: number;
  strategy: RiskMetrics;
  buyHold: RiskMetrics;
  winRate: number;       // % of round-trip trades that were profitable
  avgWin: number;        // avg return of winning round trips
  avgLoss: number;       // avg return of losing round trips
  profitFactor: number;  // total gains / total losses
}

interface Props {
  history: BubbleHistoryPoint[];
  priceData: DataPoint[];
  ticker: string;
}

const RISK_FREE_RATE = 0.045; // 4.5% annualized

function computeRiskMetrics(dailyValues: number[], totalDays: number): RiskMetrics {
  const dailyReturns: number[] = [];
  let peak = dailyValues[0];
  let maxDrawdown = 0;
  let daysExposed = 0;

  for (let i = 1; i < dailyValues.length; i++) {
    const prev = dailyValues[i - 1];
    const cur = dailyValues[i];
    dailyReturns.push((cur - prev) / prev);
    if (cur > peak) peak = cur;
    const dd = (peak - cur) / peak;
    if (dd > maxDrawdown) maxDrawdown = dd;
    if (Math.abs(cur - prev) > 0.001) daysExposed++;
  }

  const totalReturn = (dailyValues[dailyValues.length - 1] - dailyValues[0]) / dailyValues[0];
  const years = totalDays / 365.25;
  const cagr = years > 0 ? Math.pow(1 + totalReturn, 1 / years) - 1 : 0;

  const meanReturn = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
  const variance = dailyReturns.reduce((a, r) => a + (r - meanReturn) ** 2, 0) / dailyReturns.length;
  const dailyStd = Math.sqrt(variance);
  const annualizedVol = dailyStd * Math.sqrt(252);

  const rfDaily = RISK_FREE_RATE / 252;
  const sharpe = dailyStd > 0 ? ((meanReturn - rfDaily) / dailyStd) * Math.sqrt(252) : 0;

  const negReturns = dailyReturns.filter(r => r < 0);
  const downsideVar = negReturns.length > 0
    ? negReturns.reduce((a, r) => a + r * r, 0) / negReturns.length
    : 0.0001;
  const downsideStd = Math.sqrt(downsideVar) * Math.sqrt(252);
  const sortino = downsideStd > 0 ? (cagr - RISK_FREE_RATE) / downsideStd : 0;

  const calmar = maxDrawdown > 0 ? cagr / maxDrawdown : 0;
  const worstDay = dailyReturns.length > 0 ? Math.min(...dailyReturns) : 0;
  const exposure = daysExposed / Math.max(dailyReturns.length, 1);

  return { totalReturn, cagr, maxDrawdown, annualizedVol, sharpe, sortino, calmar, worstDay, exposure };
}

function runBubbleBacktest(
  history: BubbleHistoryPoint[],
  priceData: DataPoint[],
  buyThreshold: number,
  sellThreshold: number,
  initialInvestment: number,
): BubbleBacktestResult | null {
  const priceMap = new Map<string, number>();
  for (const p of priceData) priceMap.set(p.date, p.price);

  const merged = history
    .filter(h => priceMap.has(h.date))
    .map(h => ({ date: h.date, score: h.composite_score, price: priceMap.get(h.date)! }));

  if (merged.length < 20) return null;

  const signals: BubbleBacktestResult['signals'] = [];
  let cash = initialInvestment;
  let shares = 0;
  let inPosition = false;

  const strategyValues: number[] = [initialInvestment];
  const bhValues: number[] = [merged[0].price];

  // Track round trips for win rate
  let entryPrice = 0;
  const roundTrips: number[] = [];

  for (const point of merged) {
    const portfolioValue = inPosition ? shares * point.price : cash;
    strategyValues.push(portfolioValue);
    bhValues.push(point.price);

    if (!inPosition && point.score <= buyThreshold) {
      shares = cash / point.price;
      cash = 0;
      inPosition = true;
      entryPrice = point.price;
      signals.push({ date: point.date, type: 'buy', price: point.price, score: point.score });
    } else if (inPosition && point.score >= sellThreshold) {
      cash = shares * point.price;
      const tripReturn = (point.price - entryPrice) / entryPrice;
      roundTrips.push(tripReturn);
      shares = 0;
      inPosition = false;
      signals.push({ date: point.date, type: 'sell', price: point.price, score: point.score });
    }
  }

  // If still in position, count as open round trip
  if (inPosition) {
    const lastPrice = merged[merged.length - 1].price;
    roundTrips.push((lastPrice - entryPrice) / entryPrice);
  }

  const finalValue = inPosition ? shares * merged[merged.length - 1].price : cash;
  strategyValues[strategyValues.length - 1] = finalValue;

  const totalDays = (new Date(merged[merged.length - 1].date).getTime() - new Date(merged[0].date).getTime()) / (1000 * 60 * 60 * 24);

  // Normalize B&H values to same initial investment
  const bhScale = initialInvestment / bhValues[0];
  const bhNormalized = bhValues.map(v => v * bhScale);

  const strategy = computeRiskMetrics(strategyValues, totalDays);
  const buyHold = computeRiskMetrics(bhNormalized, totalDays);

  // Win rate & profit factor
  const wins = roundTrips.filter(r => r > 0);
  const losses = roundTrips.filter(r => r <= 0);
  const winRate = roundTrips.length > 0 ? wins.length / roundTrips.length : 0;
  const avgWin = wins.length > 0 ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / losses.length : 0;
  const totalGains = wins.reduce((a, b) => a + b, 0);
  const totalLosses = Math.abs(losses.reduce((a, b) => a + b, 0));
  const profitFactor = totalLosses > 0 ? totalGains / totalLosses : totalGains > 0 ? Infinity : 0;

  return {
    numTrades: signals.length,
    signals,
    finalValue,
    strategy,
    buyHold,
    winRate,
    avgWin,
    avgLoss,
    profitFactor,
  };
}

const BubbleBacktestPanel: React.FC<Props> = ({ history, priceData, ticker }) => {
  const [buyThreshold, setBuyThreshold] = useState(25);
  const [sellThreshold, setSellThreshold] = useState(85);
  const [initialInvestment, setInitialInvestment] = useState(10000);

  const result = useMemo(() => {
    return runBubbleBacktest(history, priceData, buyThreshold, sellThreshold, initialInvestment);
  }, [history, priceData, buyThreshold, sellThreshold, initialInvestment]);

  if (!result) return null;

  const fmt = (v: number) => (v * 100).toFixed(1) + '%';
  const fmtRatio = (v: number) => v === Infinity ? '--' : v.toFixed(2);
  const fmtMoney = (v: number) => '$' + v.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 });

  // Determine which metric is better for coloring
  const better = (strat: number, bh: number, higherIsBetter: boolean) => {
    if (higherIsBetter) return strat > bh + 0.001 ? 'text-emerald-400' : strat < bh - 0.001 ? 'text-red-400' : 'text-slate-300';
    return strat < bh - 0.001 ? 'text-emerald-400' : strat > bh + 0.001 ? 'text-red-400' : 'text-slate-300';
  };

  const { strategy: s, buyHold: bh } = result;

  const metrics: { label: string; strat: string; bh: string; higherBetter: boolean; stratVal: number; bhVal: number; tip: string }[] = [
    { label: 'Total Return', strat: fmt(s.totalReturn), bh: fmt(bh.totalReturn), higherBetter: true, stratVal: s.totalReturn, bhVal: bh.totalReturn, tip: 'Cumulative return over the full period' },
    { label: 'CAGR', strat: fmt(s.cagr), bh: fmt(bh.cagr), higherBetter: true, stratVal: s.cagr, bhVal: bh.cagr, tip: 'Compound Annual Growth Rate' },
    { label: 'Sharpe Ratio', strat: fmtRatio(s.sharpe), bh: fmtRatio(bh.sharpe), higherBetter: true, stratVal: s.sharpe, bhVal: bh.sharpe, tip: 'Risk-adjusted return (excess return / volatility)' },
    { label: 'Sortino Ratio', strat: fmtRatio(s.sortino), bh: fmtRatio(bh.sortino), higherBetter: true, stratVal: s.sortino, bhVal: bh.sortino, tip: 'Return per unit of downside risk only' },
    { label: 'Calmar Ratio', strat: fmtRatio(s.calmar), bh: fmtRatio(bh.calmar), higherBetter: true, stratVal: s.calmar, bhVal: bh.calmar, tip: 'CAGR / Max Drawdown' },
    { label: 'Max Drawdown', strat: fmt(s.maxDrawdown), bh: fmt(bh.maxDrawdown), higherBetter: false, stratVal: s.maxDrawdown, bhVal: bh.maxDrawdown, tip: 'Largest peak-to-trough decline' },
    { label: 'Volatility', strat: fmt(s.annualizedVol), bh: fmt(bh.annualizedVol), higherBetter: false, stratVal: s.annualizedVol, bhVal: bh.annualizedVol, tip: 'Annualized standard deviation of daily returns' },
    { label: 'Worst Day', strat: fmt(s.worstDay), bh: fmt(bh.worstDay), higherBetter: false, stratVal: Math.abs(s.worstDay), bhVal: Math.abs(bh.worstDay), tip: 'Largest single-day loss' },
  ];

  // Count how many metrics strategy wins
  const strategyWins = metrics.filter(m => m.higherBetter ? m.stratVal > m.bhVal + 0.001 : m.stratVal < m.bhVal - 0.001).length;
  const bhWins = metrics.filter(m => m.higherBetter ? m.bhVal > m.stratVal + 0.001 : m.bhVal < m.stratVal - 0.001).length;

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-sm">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">Bubble Index Backtest</h2>
          <p className="text-sm text-slate-500">Buy in fear, sell in euphoria — based on composite bubble score</p>
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

      {/* Side-by-side Comparison Table */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Strategy vs Buy & Hold</p>
          <div className="flex gap-3 text-xs">
            <span className="text-emerald-400">Strategy wins: {strategyWins}</span>
            <span className="text-slate-500">|</span>
            <span className="text-blue-400">B&H wins: {bhWins}</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left text-slate-400 py-2 pr-4">Metric</th>
                <th className="text-right text-emerald-400 py-2 px-3">Strategy</th>
                <th className="text-right text-blue-400 py-2 px-3">Buy & Hold</th>
                <th className="text-right text-slate-400 py-2 pl-3">Edge</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map(m => {
                const edge = m.higherBetter ? m.stratVal - m.bhVal : m.bhVal - m.stratVal;
                const edgeStr = m.label.includes('Ratio')
                  ? (edge > 0 ? '+' : '') + edge.toFixed(2)
                  : (edge > 0 ? '+' : '') + (edge * 100).toFixed(1) + '%';
                return (
                  <tr key={m.label} className="border-b border-slate-800" title={m.tip}>
                    <td className="text-slate-300 py-2 pr-4">{m.label}</td>
                    <td className={`text-right py-2 px-3 font-semibold ${better(m.stratVal, m.bhVal, m.higherBetter)}`}>
                      {m.strat}
                    </td>
                    <td className="text-right py-2 px-3 text-slate-300">{m.bh}</td>
                    <td className={`text-right py-2 pl-3 text-xs ${edge > 0.001 ? 'text-emerald-400' : edge < -0.001 ? 'text-red-400' : 'text-slate-500'}`}>
                      {edgeStr}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Trade Quality Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Trades</p>
          <p className="text-lg font-bold text-white">{result.numTrades}</p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Win Rate</p>
          <p className={`text-lg font-bold ${result.winRate >= 0.5 ? 'text-emerald-400' : 'text-red-400'}`}>
            {fmt(result.winRate)}
          </p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Avg Win</p>
          <p className="text-lg font-bold text-emerald-400">+{fmt(result.avgWin)}</p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Avg Loss</p>
          <p className="text-lg font-bold text-red-400">{fmt(result.avgLoss)}</p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Profit Factor</p>
          <p className={`text-lg font-bold ${result.profitFactor >= 1 ? 'text-emerald-400' : 'text-red-400'}`}>
            {result.profitFactor === Infinity ? '--' : result.profitFactor.toFixed(1)}x
          </p>
        </div>
      </div>

      {/* Trade log */}
      {result.signals.length > 0 && (
        <div className="mb-6">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Trade History</p>
          <div className="flex flex-wrap gap-2 max-h-24 overflow-y-auto">
            {result.signals.map((s, i) => (
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
      <div className="bg-slate-800/30 rounded-xl p-4">
        <p className="text-sm text-slate-400">
          {fmtMoney(initialInvestment)} invested
          {' '}&rarr;{' '}
          <span className="text-white font-semibold">{fmtMoney(result.finalValue)}</span>
          {' '}({fmt(result.strategy.totalReturn)})
          {' '}vs B&H{' '}
          <span className="text-slate-300">{fmtMoney(initialInvestment * (1 + result.buyHold.totalReturn))}</span>
          {' '}({fmt(result.buyHold.totalReturn)}).
          {' '}Market exposure:{' '}
          <span className="text-white font-semibold">{(result.strategy.exposure * 100).toFixed(0)}%</span>
          {' '}of the time.
        </p>
      </div>
    </div>
  );
};

export default BubbleBacktestPanel;
