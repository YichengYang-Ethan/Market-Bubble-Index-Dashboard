import { IndicatorMeta } from './types';

export const SUPPORTED_TICKERS = ['QQQ', 'SPY', 'TQQQ', 'IWM'] as const;
export type TickerSymbol = (typeof SUPPORTED_TICKERS)[number];

export const TICKER_LABELS: Record<TickerSymbol, string> = {
  QQQ: 'Nasdaq 100',
  SPY: 'S&P 500',
  TQQQ: 'Nasdaq 100 3x',
  IWM: 'Russell 2000',
};

export const DEVIATION_CONFIG = {
  SMA_PERIOD: 200,
  REFRESH_INTERVAL_MS: 300_000, // 5 minutes
  RISK_LEVELS: {
    LOW: 20,
    MODERATE: 50,
    HIGH: 80,
  },
  SIMULATION: {
    VOLATILITY: 0.002,
    MEAN_REVERSION: 0.0005,
    TREND_WEIGHT: 0.15,
    NOISE_WEIGHT: 0.40,
  },
} as const;

export type RiskLevel = "Low" | "Moderate" | "High" | "Danger";

export function getRiskLevel(index: number): RiskLevel {
  if (index < DEVIATION_CONFIG.RISK_LEVELS.LOW) return "Low";
  if (index < DEVIATION_CONFIG.RISK_LEVELS.MODERATE) return "Moderate";
  if (index < DEVIATION_CONFIG.RISK_LEVELS.HIGH) return "High";
  return "Danger";
}

export const INDICATOR_META: IndicatorMeta[] = [
  {
    key: 'qqq_deviation',
    label: 'QQQ Deviation',
    color: '#3b82f6',
    description: 'QQQ deviation from its 200-day moving average, percentile-ranked over a 200-day lookback.',
    category: 'sentiment',
  },
  {
    key: 'vix_level',
    label: 'VIX Level',
    color: '#f97316',
    description: 'Inverted VIX — low volatility signals complacency and higher bubble risk.',
    category: 'sentiment',
  },
  {
    key: 'sector_breadth',
    label: 'Sector Breadth',
    color: '#22c55e',
    description: 'Fraction of S&P sector ETFs trading above their 50-day SMA.',
    category: 'liquidity',
  },
  {
    key: 'credit_spread',
    label: 'Credit Spread',
    color: '#06b6d4',
    description: 'HYG/IEF ratio as a credit-spread proxy. Tight spreads signal risk-on behavior.',
    category: 'liquidity',
  },
  {
    key: 'put_call_ratio',
    label: 'Tail Risk (SKEW)',
    color: '#a855f7',
    description: 'CBOE SKEW index — measures tail-risk hedging demand. High SKEW with rising markets signals complacency.',
    category: 'sentiment',
  },
  {
    key: 'yield_curve',
    label: 'Yield Curve',
    color: '#eab308',
    description: '10Y-2Y Treasury spread. A steepening curve signals risk-on conditions.',
    category: 'liquidity',
  },
  {
    key: 'cape_ratio',
    label: 'CAPE Valuation',
    color: '#f43f5e',
    description: 'S&P 500 price relative to 10-year moving average, percentile-ranked. Higher values indicate expensive valuations typical of bubble conditions.',
    category: 'valuation',
  },
  {
    key: 'leverage_sentiment',
    label: 'Leverage Sentiment',
    color: '#ec4899',
    description: 'TQQQ/(TQQQ+SQQQ) volume ratio — measures retail leverage sentiment. High values indicate aggressive bullish positioning.',
    category: 'sentiment',
  },
];

export const BUBBLE_REGIME_CONFIG = [
  { threshold: 30, key: 'LOW', label: 'Low Risk', color: '#22c55e', bgClass: 'bg-emerald-500/10 border-emerald-500/20', textClass: 'text-emerald-400', description: 'Depressed sentiment. Potential accumulation zone.' },
  { threshold: 50, key: 'MODERATE', label: 'Moderate', color: '#eab308', bgClass: 'bg-yellow-500/10 border-yellow-500/20', textClass: 'text-yellow-400', description: 'Normal market conditions. Balanced risk/reward.' },
  { threshold: 70, key: 'ELEVATED', label: 'Elevated', color: '#f97316', bgClass: 'bg-orange-500/10 border-orange-500/20', textClass: 'text-orange-400', description: 'Rising euphoria. Tighten risk management.' },
  { threshold: 85, key: 'HIGH', label: 'High Risk', color: '#ef4444', bgClass: 'bg-red-500/10 border-red-500/20', textClass: 'text-red-400', description: 'Frothy conditions. Consider reducing exposure.' },
  { threshold: 100, key: 'EXTREME', label: 'Extreme', color: '#dc2626', bgClass: 'bg-red-600/10 border-red-600/20', textClass: 'text-red-300', description: 'Bubble territory. High correction probability.' },
] as const;
