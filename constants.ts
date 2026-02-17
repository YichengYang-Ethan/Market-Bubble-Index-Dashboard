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
