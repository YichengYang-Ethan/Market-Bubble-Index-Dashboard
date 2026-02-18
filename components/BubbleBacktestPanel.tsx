
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
  rebalances: number;
  avgExposure: number;
  allocationChanges: { date: string; score: number; allocation: number; price: number }[];
}

interface Props {
  history: BubbleHistoryPoint[];
  priceData: DataPoint[];
  ticker: string;
}

// ---------------------------------------------------------------------------
// Position sizing tiers (matches Position Mapping card in App.tsx)
// ---------------------------------------------------------------------------

interface Tier {
  maxScore: number;
  allocation: number;  // fraction of portfolio in equities
  label: string;
}

const DEFAULT_TIERS: Tier[] = [
  { maxScore: 30,  allocation: 1.00, label: 'Full 100%' },
  { maxScore: 50,  allocation: 1.00, label: 'Full 100%' },
  { maxScore: 70,  allocation: 0.80, label: 'Reduce 80%' },
  { maxScore: 85,  allocation: 0.50, label: 'Defensive 50%' },
  { maxScore: 100, allocation: 0.20, label: 'Minimal 20%' },
];

const LEVERAGED_TIERS: Tier[] = [
  { maxScore: 30,  allocation: 1.30, label: 'Overweight 130%' },
  { maxScore: 50,  allocation: 1.00, label: 'Full 100%' },
  { maxScore: 70,  allocation: 0.80, label: 'Reduce 80%' },
  { maxScore: 85,  allocation: 0.50, label: 'Defensive 50%' },
  { maxScore: 100, allocation: 0.20, label: 'Minimal 20%' },
];

const RISK_FREE_RATE = 0.045;

// ---------------------------------------------------------------------------
// Core engine
// ---------------------------------------------------------------------------

function getAllocation(score: number, tiers: Tier[]): number {
  for (const t of tiers) {
    if (score < t.maxScore) return t.allocation;
  }
  return tiers[tiers.length - 1].allocation;
}

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

function runGraduatedBacktest(
  history: BubbleHistoryPoint[],
  priceData: DataPoint[],
  tiers: Tier[],
  initialInvestment: number,
  rebalanceDays: number,
): BacktestResult | null {
  const priceMap = new Map<string, number>();
  for (const p of priceData) priceMap.set(p.date, p.price);

  const merged = history
    .filter(h => priceMap.has(h.date))
    .map(h => ({ date: h.date, score: h.composite_score, price: priceMap.get(h.date)! }));

  if (merged.length < 20) return null;

  // Strategy: graduated position sizing with periodic rebalancing
  // Portfolio = equity portion (in QQQ) + cash portion (earns risk-free rate)
  let equity = 0;
  let cash = initialInvestment;
  let shares = 0;
  let currentAllocation = 0;
  let daysSinceRebalance = rebalanceDays; // force rebalance on first day

  const stratValues: number[] = [];
  const bhValues: number[] = [];
  const changes: BacktestResult['allocationChanges'] = [];
  let totalExposure = 0;
  let rebalances = 0;

  const rfDailyRate = RISK_FREE_RATE / 252;

  for (let i = 0; i < merged.length; i++) {
    const pt = merged[i];
    const targetAlloc = getAllocation(pt.score, tiers);

    // Current portfolio value
    const equityValue = shares * pt.price;
    const portfolioValue = equityValue + cash;
    stratValues.push(portfolioValue);
    bhValues.push(pt.price);

    totalExposure += equityValue / Math.max(portfolioValue, 1);

    // Rebalance if allocation tier changed OR periodic rebalance
    daysSinceRebalance++;
    const allocChanged = Math.abs(targetAlloc - currentAllocation) > 0.01;

    if (allocChanged || daysSinceRebalance >= rebalanceDays) {
      const targetEquity = portfolioValue * Math.min(targetAlloc, 1.3);
      const targetCash = portfolioValue - targetEquity;

      // If leveraged (>100%), we allow negative cash (margin)
      shares = targetEquity / pt.price;
      cash = targetCash;
      currentAllocation = targetAlloc;
      daysSinceRebalance = 0;

      if (allocChanged) {
        rebalances++;
        changes.push({ date: pt.date, score: pt.score, allocation: targetAlloc, price: pt.price });
      }
    }

    // Cash earns risk-free rate daily
    if (cash > 0) {
      cash *= (1 + rfDailyRate);
    } else {
      // Margin interest on borrowed cash (same rate for simplicity)
      cash *= (1 + rfDailyRate);
    }
  }

  // Final values
  const lastPrice = merged[merged.length - 1].price;
  const finalValue = shares * lastPrice + cash;
  stratValues.push(finalValue);
  bhValues.push(lastPrice);

  const totalDays = (new Date(merged[merged.length - 1].date).getTime() - new Date(merged[0].date).getTime()) / (1000 * 60 * 60 * 24);

  const bhScale = initialInvestment / bhValues[0];
  const bhNormalized = bhValues.map(v => v * bhScale);

  const strategy = computeRiskMetrics(stratValues, totalDays);
  const buyHold = computeRiskMetrics(bhNormalized, totalDays);
  const avgExposure = totalExposure / merged.length;

  return {
    strategy,
    buyHold,
    finalValue,
    bhFinalValue: bhNormalized[bhNormalized.length - 1],
    rebalances,
    avgExposure,
    allocationChanges: changes,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const BubbleBacktestPanel: React.FC<Props> = ({ history, priceData, ticker }) => {
  const [initialInvestment, setInitialInvestment] = useState(10000);
  const [rebalanceDays, setRebalanceDays] = useState(5);
  const [useLeverage, setUseLeverage] = useState(false);

  const tiers = useMemo(() => {
    return useLeverage ? LEVERAGED_TIERS : DEFAULT_TIERS;
  }, [useLeverage]);

  const result = useMemo(() => {
    return runGraduatedBacktest(history, priceData, tiers, initialInvestment, rebalanceDays);
  }, [history, priceData, tiers, initialInvestment, rebalanceDays]);

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
    { label: 'Sharpe Ratio', sVal: s.sharpe, bVal: bh.sharpe, sFmt: fmtRatio(s.sharpe), bFmt: fmtRatio(bh.sharpe), higher: true, tip: 'Excess return per unit of total risk' },
    { label: 'Sortino Ratio', sVal: s.sortino, bVal: bh.sortino, sFmt: fmtRatio(s.sortino), bFmt: fmtRatio(bh.sortino), higher: true, tip: 'Excess return per unit of downside risk' },
    { label: 'Calmar Ratio', sVal: s.calmar, bVal: bh.calmar, sFmt: fmtRatio(s.calmar), bFmt: fmtRatio(bh.calmar), higher: true, tip: 'CAGR divided by Max Drawdown' },
    { label: 'Max Drawdown', sVal: s.maxDrawdown, bVal: bh.maxDrawdown, sFmt: fmt(s.maxDrawdown), bFmt: fmt(bh.maxDrawdown), higher: false, tip: 'Largest peak-to-trough decline' },
    { label: 'Volatility', sVal: s.annualizedVol, bVal: bh.annualizedVol, sFmt: fmt(s.annualizedVol), bFmt: fmt(bh.annualizedVol), higher: false, tip: 'Annualized standard deviation' },
    { label: 'Worst Day', sVal: Math.abs(s.worstDay), bVal: Math.abs(bh.worstDay), sFmt: fmt(s.worstDay), bFmt: fmt(bh.worstDay), higher: false, tip: 'Largest single-day loss' },
  ];

  const sWins = metrics.filter(m => (m.higher ? m.sVal - m.bVal : m.bVal - m.sVal) > 0.001).length;
  const bWins = metrics.filter(m => (m.higher ? m.bVal - m.sVal : m.sVal - m.bVal) > 0.001).length;

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-sm">
      <div className="flex justify-between items-center mb-2">
        <div>
          <h2 className="text-xl font-bold text-white">Bubble Index Backtest</h2>
          <p className="text-sm text-slate-500">Graduated position sizing based on composite bubble score</p>
        </div>
        <span className="text-xs text-slate-500 bg-slate-800 px-3 py-1 rounded-full">{ticker}</span>
      </div>

      {/* Allocation Tiers Visual */}
      <div className="mb-6">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Allocation Tiers</p>
        <div className="flex gap-1 h-8 rounded-lg overflow-hidden">
          {tiers.map((t, i) => {
            const prevMax = i > 0 ? tiers[i - 1].maxScore : 0;
            const width = t.maxScore - prevMax;
            const colors = ['bg-emerald-500/60', 'bg-slate-600/60', 'bg-yellow-500/50', 'bg-orange-500/50', 'bg-red-500/50'];
            return (
              <div
                key={t.maxScore}
                className={`${colors[i]} flex items-center justify-center text-xs font-semibold text-white`}
                style={{ width: `${width}%` }}
                title={`Score ${prevMax}-${t.maxScore}: ${(t.allocation * 100).toFixed(0)}% equity`}
              >
                {(t.allocation * 100).toFixed(0)}%
              </div>
            );
          })}
        </div>
        <div className="flex justify-between text-[10px] text-slate-500 mt-1 px-0.5">
          <span>0 (Fear)</span>
          <span>30</span>
          <span>50</span>
          <span>70</span>
          <span>85</span>
          <span>100 (Euphoria)</span>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div>
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Rebalance Every: <span className="text-blue-400">{rebalanceDays} days</span>
          </label>
          <input
            type="range"
            min={1}
            max={20}
            value={rebalanceDays}
            onChange={e => setRebalanceDays(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
          <p className="text-[10px] text-slate-600 mt-1">Also rebalances immediately on tier change</p>
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

        <div className="flex items-end pb-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useLeverage}
              onChange={e => setUseLeverage(e.target.checked)}
              className="w-4 h-4 accent-blue-500 rounded"
            />
            <span className="text-sm text-slate-300">Allow 130% leverage in fear zone</span>
          </label>
        </div>
      </div>

      {/* Side-by-side Comparison Table */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Graduated Strategy vs Buy & Hold</p>
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
          <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1">Rebalances</p>
          <p className="text-lg font-bold text-white">{result.rebalances}</p>
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

      {/* Recent allocation changes */}
      {result.allocationChanges.length > 0 && (
        <div className="mb-6">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Recent Allocation Changes</p>
          <div className="flex flex-wrap gap-2 max-h-24 overflow-y-auto">
            {result.allocationChanges.slice(-15).map((c, i) => {
              const pct = (c.allocation * 100).toFixed(0);
              const color = c.allocation >= 1.0 ? 'bg-emerald-500/20 text-emerald-400'
                : c.allocation >= 0.8 ? 'bg-slate-600/30 text-slate-300'
                : c.allocation >= 0.5 ? 'bg-yellow-500/20 text-yellow-400'
                : 'bg-red-500/20 text-red-400';
              return (
                <span key={i} className={`text-xs px-2 py-1 rounded ${color}`}>
                  {c.date} &rarr; {pct}% (score {c.score.toFixed(0)})
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Summary */}
      <div className="bg-slate-800/30 rounded-xl p-4 space-y-2">
        <p className="text-sm text-slate-400">
          <span className="text-white font-semibold">Risk management overlay</span>: stay fully invested in normal conditions,
          reduce to {tiers[2].allocation * 100}% in elevated zones, {tiers[3].allocation * 100}% in high-risk,
          and {tiers[4].allocation * 100}% in extreme bubble territory.
          Cash earns {(RISK_FREE_RATE * 100).toFixed(1)}% risk-free rate.
        </p>
        <p className="text-sm text-slate-400">
          {fmtMoney(initialInvestment)} &rarr;{' '}
          <span className="text-white font-semibold">{fmtMoney(result.finalValue)}</span>
          {' '}vs B&H{' '}
          <span className="text-slate-300">{fmtMoney(result.bhFinalValue)}</span>.
          {s.maxDrawdown < bh.maxDrawdown - 0.005 && (
            <>{' '}Drawdown reduced by <span className="text-emerald-400 font-semibold">{((bh.maxDrawdown - s.maxDrawdown) * 100).toFixed(1)}%</span>.</>
          )}
          {s.annualizedVol < bh.annualizedVol - 0.005 && (
            <>{' '}Volatility reduced by <span className="text-emerald-400 font-semibold">{((bh.annualizedVol - s.annualizedVol) * 100).toFixed(1)}%</span>.</>
          )}
        </p>
      </div>
    </div>
  );
};

export default BubbleBacktestPanel;
