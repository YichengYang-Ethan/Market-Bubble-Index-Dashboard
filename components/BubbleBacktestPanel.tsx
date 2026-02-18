
import React, { useState, useMemo } from 'react';
import { BubbleHistoryPoint, DataPoint } from '../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RiskMetrics {
  totalReturn: number;
  cagr: number;
  maxDrawdown: number;
  annualizedVol: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  worstDay: number;
}

interface BacktestResult {
  strategy: RiskMetrics;
  buyHold: RiskMetrics;
  finalValue: number;
  bhFinalValue: number;
  trades: number;
  avgExposure: number;
  tradeLog: { date: string; action: string; score: number; price: number; velocity?: number }[];
}

interface Props {
  history: BubbleHistoryPoint[];
  priceData: DataPoint[];
  ticker: string;
}

type StrategyMode = 'hysteresis' | 'velocity';

const RISK_FREE_RATE = 0.045;

// ---------------------------------------------------------------------------
// Core engine
// ---------------------------------------------------------------------------

function computeRiskMetrics(dailyValues: number[], totalDays: number): RiskMetrics {
  const returns: number[] = [];
  let peak = dailyValues[0];
  let maxDrawdown = 0;

  for (let i = 1; i < dailyValues.length; i++) {
    const prev = dailyValues[i - 1];
    const cur = dailyValues[i];
    if (prev > 0) returns.push((cur - prev) / prev);
    if (cur > peak) peak = cur;
    const dd = peak > 0 ? (peak - cur) / peak : 0;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }

  if (returns.length === 0) {
    return { totalReturn: 0, cagr: 0, maxDrawdown: 0, annualizedVol: 0, sharpe: 0, sortino: 0, calmar: 0, worstDay: 0 };
  }

  const totalReturn = (dailyValues[dailyValues.length - 1] - dailyValues[0]) / dailyValues[0];
  const years = totalDays / 365.25;
  const cagr = years > 0 ? Math.pow(1 + totalReturn, 1 / years) - 1 : 0;

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, r) => a + (r - mean) ** 2, 0) / returns.length;
  const std = Math.sqrt(variance);
  const annualizedVol = std * Math.sqrt(252);

  const rfDaily = RISK_FREE_RATE / 252;
  const sharpe = std > 0 ? ((mean - rfDaily) / std) * Math.sqrt(252) : 0;

  const negRet = returns.filter(r => r < 0);
  const downVar = negRet.length > 0 ? negRet.reduce((a, r) => a + r * r, 0) / negRet.length : 1e-8;
  const downStd = Math.sqrt(downVar) * Math.sqrt(252);
  const sortino = downStd > 0 ? (cagr - RISK_FREE_RATE) / downStd : 0;

  const calmar = maxDrawdown > 0 ? cagr / maxDrawdown : 0;
  const worstDay = Math.min(...returns);

  return { totalReturn, cagr, maxDrawdown, annualizedVol, sharpe, sortino, calmar, worstDay };
}

// ---------------------------------------------------------------------------
// Strategy 1: Hysteresis Binary (Entry < 24, Exit > 84)
// Uses different entry/exit thresholds to avoid whipsawing.
// Invested = 100% equity. Out = 100% cash earning risk-free rate.
// ---------------------------------------------------------------------------

function runHysteresisBacktest(
  history: BubbleHistoryPoint[],
  priceData: DataPoint[],
  entryThreshold: number,
  exitThreshold: number,
  initialInvestment: number,
): BacktestResult | null {
  const priceMap = new Map<string, number>();
  for (const p of priceData) priceMap.set(p.date, p.price);

  const merged = history
    .filter(h => priceMap.has(h.date))
    .map(h => ({ date: h.date, score: h.composite_score, price: priceMap.get(h.date)! }));

  if (merged.length < 20) return null;

  let invested = true; // start invested
  let shares = initialInvestment / merged[0].price;
  let cash = 0;
  let trades = 0;
  const rfDailyRate = RISK_FREE_RATE / 252;

  const stratValues: number[] = [];
  const bhValues: number[] = [];
  const tradeLog: BacktestResult['tradeLog'] = [];
  let totalExposure = 0;

  for (let i = 0; i < merged.length; i++) {
    const pt = merged[i];
    const portfolioValue = invested ? shares * pt.price : cash;
    stratValues.push(portfolioValue);
    bhValues.push(pt.price);
    totalExposure += invested ? 1 : 0;

    if (invested && pt.score >= exitThreshold) {
      // Exit: sell all, move to cash
      cash = shares * pt.price;
      shares = 0;
      invested = false;
      trades++;
      tradeLog.push({ date: pt.date, action: 'EXIT', score: pt.score, price: pt.price });
    } else if (!invested && pt.score <= entryThreshold) {
      // Enter: buy with all cash
      shares = cash / pt.price;
      cash = 0;
      invested = true;
      trades++;
      tradeLog.push({ date: pt.date, action: 'ENTER', score: pt.score, price: pt.price });
    }

    // Cash earns risk-free rate
    if (!invested) {
      cash *= (1 + rfDailyRate);
    }
  }

  const lastPrice = merged[merged.length - 1].price;
  const finalValue = invested ? shares * lastPrice : cash;
  const totalDays = (new Date(merged[merged.length - 1].date).getTime() - new Date(merged[0].date).getTime()) / (1000 * 60 * 60 * 24);

  const bhScale = initialInvestment / bhValues[0];
  const bhNormalized = bhValues.map(v => v * bhScale);

  return {
    strategy: computeRiskMetrics(stratValues, totalDays),
    buyHold: computeRiskMetrics(bhNormalized, totalDays),
    finalValue,
    bhFinalValue: bhNormalized[bhNormalized.length - 1],
    trades,
    avgExposure: totalExposure / merged.length,
    tradeLog,
  };
}

// ---------------------------------------------------------------------------
// Strategy 2: Velocity Signal
// Exit to 30% when score_velocity > threshold AND score > scoreFloor.
// Rising velocity with elevated score = euphoria building → reduce exposure.
// ---------------------------------------------------------------------------

function runVelocityBacktest(
  history: BubbleHistoryPoint[],
  priceData: DataPoint[],
  velocityThreshold: number,
  scoreFloor: number,
  reducedAllocation: number,
  initialInvestment: number,
): BacktestResult | null {
  const priceMap = new Map<string, number>();
  for (const p of priceData) priceMap.set(p.date, p.price);

  const merged = history
    .filter(h => priceMap.has(h.date) && h.score_velocity !== undefined)
    .map(h => ({
      date: h.date,
      score: h.composite_score,
      velocity: h.score_velocity!,
      price: priceMap.get(h.date)!,
    }));

  if (merged.length < 20) return null;

  let shares = initialInvestment / merged[0].price;
  let cash = 0;
  let currentAlloc = 1.0;
  let trades = 0;
  const rfDailyRate = RISK_FREE_RATE / 252;

  const stratValues: number[] = [];
  const bhValues: number[] = [];
  const tradeLog: BacktestResult['tradeLog'] = [];
  let totalExposure = 0;

  for (let i = 0; i < merged.length; i++) {
    const pt = merged[i];
    const equityValue = shares * pt.price;
    const portfolioValue = equityValue + cash;
    stratValues.push(portfolioValue);
    bhValues.push(pt.price);
    totalExposure += equityValue / Math.max(portfolioValue, 1);

    // Determine target allocation
    const shouldReduce = pt.velocity > velocityThreshold && pt.score > scoreFloor;
    const targetAlloc = shouldReduce ? reducedAllocation : 1.0;

    if (Math.abs(targetAlloc - currentAlloc) > 0.01) {
      const targetEquity = portfolioValue * targetAlloc;
      shares = targetEquity / pt.price;
      cash = portfolioValue - targetEquity;
      currentAlloc = targetAlloc;
      trades++;
      tradeLog.push({
        date: pt.date,
        action: targetAlloc < 1 ? `REDUCE → ${(targetAlloc * 100).toFixed(0)}%` : 'FULL → 100%',
        score: pt.score,
        price: pt.price,
        velocity: pt.velocity,
      });
    }

    // Cash earns risk-free rate
    if (cash > 0) cash *= (1 + rfDailyRate);
  }

  const lastPrice = merged[merged.length - 1].price;
  const finalValue = shares * lastPrice + cash;
  const totalDays = (new Date(merged[merged.length - 1].date).getTime() - new Date(merged[0].date).getTime()) / (1000 * 60 * 60 * 24);

  const bhScale = initialInvestment / bhValues[0];
  const bhNormalized = bhValues.map(v => v * bhScale);

  return {
    strategy: computeRiskMetrics(stratValues, totalDays),
    buyHold: computeRiskMetrics(bhNormalized, totalDays),
    finalValue,
    bhFinalValue: bhNormalized[bhNormalized.length - 1],
    trades,
    avgExposure: totalExposure / merged.length,
    tradeLog,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const BubbleBacktestPanel: React.FC<Props> = ({ history, priceData, ticker }) => {
  const [mode, setMode] = useState<StrategyMode>('hysteresis');
  const [initialInvestment, setInitialInvestment] = useState(10000);

  // Hysteresis params
  const [entryThreshold, setEntryThreshold] = useState(24);
  const [exitThreshold, setExitThreshold] = useState(84);

  // Velocity params
  const [velocityThreshold, setVelocityThreshold] = useState(15);
  const [scoreFloor, setScoreFloor] = useState(50);
  const [reducedAlloc, setReducedAlloc] = useState(0.30);

  const result = useMemo(() => {
    if (mode === 'hysteresis') {
      return runHysteresisBacktest(history, priceData, entryThreshold, exitThreshold, initialInvestment);
    }
    return runVelocityBacktest(history, priceData, velocityThreshold, scoreFloor, reducedAlloc, initialInvestment);
  }, [history, priceData, mode, initialInvestment, entryThreshold, exitThreshold, velocityThreshold, scoreFloor, reducedAlloc]);

  if (!result) return null;

  const fmt = (v: number) => (v * 100).toFixed(1) + '%';
  const fmtRatio = (v: number) => isFinite(v) ? v.toFixed(2) : '--';
  const fmtMoney = (v: number) => '$' + v.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 });

  const betterClass = (strat: number, bh: number, higherIsBetter: boolean) => {
    const diff = higherIsBetter ? strat - bh : bh - strat;
    if (diff > 0.001) return 'text-emerald-400';
    if (diff < -0.001) return 'text-red-400';
    return 'text-slate-300';
  };

  const { strategy: s, buyHold: bh } = result;

  const metrics: { label: string; sVal: number; bVal: number; sFmt: string; bFmt: string; higher: boolean; tip: string }[] = [
    { label: 'Total Return', sVal: s.totalReturn, bVal: bh.totalReturn, sFmt: fmt(s.totalReturn), bFmt: fmt(bh.totalReturn), higher: true, tip: 'Cumulative return' },
    { label: 'CAGR', sVal: s.cagr, bVal: bh.cagr, sFmt: fmt(s.cagr), bFmt: fmt(bh.cagr), higher: true, tip: 'Compound Annual Growth Rate' },
    { label: 'Sharpe Ratio', sVal: s.sharpe, bVal: bh.sharpe, sFmt: fmtRatio(s.sharpe), bFmt: fmtRatio(bh.sharpe), higher: true, tip: 'Excess return per unit of total risk (higher = better risk-adjusted return)' },
    { label: 'Sortino Ratio', sVal: s.sortino, bVal: bh.sortino, sFmt: fmtRatio(s.sortino), bFmt: fmtRatio(bh.sortino), higher: true, tip: 'Excess return per unit of downside risk' },
    { label: 'Calmar Ratio', sVal: s.calmar, bVal: bh.calmar, sFmt: fmtRatio(s.calmar), bFmt: fmtRatio(bh.calmar), higher: true, tip: 'CAGR divided by Max Drawdown' },
    { label: 'Max Drawdown', sVal: s.maxDrawdown, bVal: bh.maxDrawdown, sFmt: fmt(s.maxDrawdown), bFmt: fmt(bh.maxDrawdown), higher: false, tip: 'Largest peak-to-trough decline (lower = better)' },
    { label: 'Volatility', sVal: s.annualizedVol, bVal: bh.annualizedVol, sFmt: fmt(s.annualizedVol), bFmt: fmt(bh.annualizedVol), higher: false, tip: 'Annualized standard deviation (lower = smoother)' },
    { label: 'Worst Day', sVal: Math.abs(s.worstDay), bVal: Math.abs(bh.worstDay), sFmt: fmt(s.worstDay), bFmt: fmt(bh.worstDay), higher: false, tip: 'Largest single-day loss' },
  ];

  const sWins = metrics.filter(m => (m.higher ? m.sVal - m.bVal : m.bVal - m.sVal) > 0.001).length;
  const bWins = metrics.filter(m => (m.higher ? m.bVal - m.sVal : m.sVal - m.bVal) > 0.001).length;

  const strategyDescriptions: Record<StrategyMode, { name: string; subtitle: string; description: string }> = {
    hysteresis: {
      name: 'Hysteresis Binary',
      subtitle: 'Optimized entry/exit with asymmetric thresholds',
      description: `Binary all-in/all-out strategy with asymmetric thresholds to avoid whipsawing. Enter (100% equity) when bubble score drops below ${entryThreshold}. Exit (100% cash) when score exceeds ${exitThreshold}. The wide gap between entry and exit prevents frequent trades during sideways markets. Cash earns ${(RISK_FREE_RATE * 100).toFixed(1)}% risk-free rate.`,
    },
    velocity: {
      name: 'Velocity Signal',
      subtitle: 'Momentum-aware dynamic allocation',
      description: `Uses score velocity (rate of change) as a leading indicator. When velocity exceeds ${velocityThreshold} AND score is above ${scoreFloor}, reduce to ${(reducedAlloc * 100).toFixed(0)}% equity — rising momentum with elevated score signals building euphoria. Return to 100% when conditions normalize. Cash earns ${(RISK_FREE_RATE * 100).toFixed(1)}% risk-free rate.`,
    },
  };

  const desc = strategyDescriptions[mode];

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-sm">
      <div className="flex justify-between items-center mb-2">
        <div>
          <h2 className="text-xl font-bold text-white">Bubble Index Backtest</h2>
          <p className="text-sm text-slate-500">{desc.subtitle}</p>
        </div>
        <span className="text-xs text-slate-500 bg-slate-800 px-3 py-1 rounded-full">{ticker}</span>
      </div>

      {/* Strategy Mode Selector */}
      <div className="flex gap-2 mb-6">
        {(['hysteresis', 'velocity'] as StrategyMode[]).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
              mode === m
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 border border-slate-700'
            }`}
          >
            {strategyDescriptions[m].name}
          </button>
        ))}
      </div>

      {/* Strategy-specific Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {mode === 'hysteresis' ? (
          <>
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Entry Threshold: <span className="text-emerald-400">score ≤ {entryThreshold}</span>
              </label>
              <input
                type="range"
                min={5}
                max={40}
                value={entryThreshold}
                onChange={e => setEntryThreshold(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <p className="text-[10px] text-slate-600 mt-1">Buy when bubble score drops below this level</p>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Exit Threshold: <span className="text-red-400">score ≥ {exitThreshold}</span>
              </label>
              <input
                type="range"
                min={60}
                max={95}
                value={exitThreshold}
                onChange={e => setExitThreshold(Number(e.target.value))}
                className="w-full accent-red-500"
              />
              <p className="text-[10px] text-slate-600 mt-1">Sell when bubble score exceeds this level</p>
            </div>
          </>
        ) : (
          <>
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Velocity Trigger: <span className="text-orange-400">&gt; {velocityThreshold}</span>
              </label>
              <input
                type="range"
                min={5}
                max={30}
                value={velocityThreshold}
                onChange={e => setVelocityThreshold(Number(e.target.value))}
                className="w-full accent-orange-500"
              />
              <p className="text-[10px] text-slate-600 mt-1">Reduce when score velocity exceeds this</p>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Score Floor: <span className="text-yellow-400">&gt; {scoreFloor}</span>
              </label>
              <input
                type="range"
                min={30}
                max={70}
                value={scoreFloor}
                onChange={e => setScoreFloor(Number(e.target.value))}
                className="w-full accent-yellow-500"
              />
              <p className="text-[10px] text-slate-600 mt-1">Only reduce if score is also above this</p>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Reduced Allocation: <span className="text-blue-400">{(reducedAlloc * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                min={0}
                max={80}
                step={10}
                value={reducedAlloc * 100}
                onChange={e => setReducedAlloc(Number(e.target.value) / 100)}
                className="w-full accent-blue-500"
              />
              <p className="text-[10px] text-slate-600 mt-1">Equity % when velocity signal fires</p>
            </div>
          </>
        )}

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
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{desc.name} vs Buy & Hold</p>
          <div className="flex gap-3 text-xs">
            <span className="text-emerald-400">Strategy: {sWins}</span>
            <span className="text-slate-500">|</span>
            <span className="text-blue-400">B&H: {bWins}</span>
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
                const edge = m.higher ? m.sVal - m.bVal : m.bVal - m.sVal;
                const isRatio = m.label.includes('Ratio');
                const edgeStr = isRatio
                  ? (edge > 0 ? '+' : '') + edge.toFixed(2)
                  : (edge > 0 ? '+' : '') + (edge * 100).toFixed(1) + '%';
                return (
                  <tr key={m.label} className="border-b border-slate-800" title={m.tip}>
                    <td className="text-slate-300 py-2 pr-4">{m.label}</td>
                    <td className={`text-right py-2 px-3 font-semibold ${betterClass(m.sVal, m.bVal, m.higher)}`}>
                      {m.sFmt}
                    </td>
                    <td className="text-right py-2 px-3 text-slate-300">{m.bFmt}</td>
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

      {/* Key stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Trades</p>
          <p className="text-lg font-bold text-white">{result.trades}</p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Avg Exposure</p>
          <p className="text-lg font-bold text-blue-400">{fmt(result.avgExposure)}</p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Strategy Final</p>
          <p className="text-lg font-bold text-emerald-400">{fmtMoney(result.finalValue)}</p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-xl">
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">B&H Final</p>
          <p className="text-lg font-bold text-slate-300">{fmtMoney(result.bhFinalValue)}</p>
        </div>
      </div>

      {/* Trade Log */}
      {result.tradeLog.length > 0 && (
        <div className="mb-6">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Trade Log (last 15)</p>
          <div className="flex flex-wrap gap-2 max-h-24 overflow-y-auto">
            {result.tradeLog.slice(-15).map((t, i) => {
              const isEntry = t.action.includes('ENTER') || t.action.includes('FULL');
              const color = isEntry
                ? 'bg-emerald-500/20 text-emerald-400'
                : 'bg-red-500/20 text-red-400';
              return (
                <span key={i} className={`text-xs px-2 py-1 rounded ${color}`}>
                  {t.date} {t.action} (score {t.score.toFixed(0)}{t.velocity !== undefined ? `, v=${t.velocity.toFixed(1)}` : ''})
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Strategy Description */}
      <div className="bg-slate-800/30 rounded-xl p-4 space-y-2">
        <p className="text-sm text-slate-400">
          <span className="text-white font-semibold">{desc.name}</span>: {desc.description}
        </p>
        <p className="text-sm text-slate-400">
          {fmtMoney(initialInvestment)} &rarr;{' '}
          <span className="text-white font-semibold">{fmtMoney(result.finalValue)}</span>
          {' '}vs B&H{' '}
          <span className="text-slate-300">{fmtMoney(result.bhFinalValue)}</span>.
          {s.sharpe > bh.sharpe + 0.01 && (
            <>{' '}Sharpe improved by <span className="text-emerald-400 font-semibold">+{(s.sharpe - bh.sharpe).toFixed(2)}</span>.</>
          )}
          {s.maxDrawdown < bh.maxDrawdown - 0.005 && (
            <>{' '}Drawdown reduced by <span className="text-emerald-400 font-semibold">{((bh.maxDrawdown - s.maxDrawdown) * 100).toFixed(1)}%</span>.</>
          )}
        </p>
        <p className="text-[10px] text-slate-600">
          Default thresholds were optimized via grid search over 10 years of historical data. Past performance does not guarantee future results.
        </p>
      </div>
    </div>
  );
};

export default BubbleBacktestPanel;
