
export interface DataPoint {
  date: string;
  price: number;
  sma200: number;
  deviation: number;
  index: number; // The normalized 0-100 index
}

export interface MarketSummary {
  currentPrice: number;
  currentSMA: number;
  currentIndex: number;
  change24h: number;
  riskLevel: 'Low' | 'Moderate' | 'High' | 'Danger';
}

export interface BacktestResult {
  numTrades: number;
  strategyReturn: number;
  strategyAnnualizedReturn: number;
  buyHoldReturn: number;
  buyHoldAnnualizedReturn: number;
  maxDrawdown: number;
  signals: BacktestSignal[];
}

export interface BacktestSignal {
  date: string;
  type: 'buy' | 'sell';
  price: number;
  index: number;
}

export interface HistoricalSignal {
  date: string;
  type: 'buy' | 'sell';
  index: number;
  price: number;
}

export interface BubbleIndicator {
  score: number;
  raw_value: number;
  weight: number;
  label: string;
}

export interface BubbleIndexData {
  generated_at: string;
  composite_score: number;
  sentiment_score: number | null;
  liquidity_score: number | null;
  regime: string;
  indicators: Record<string, BubbleIndicator>;
}

export interface BubbleHistoryPoint {
  date: string;
  composite_score: number;
  sentiment_score: number | null;
  liquidity_score: number | null;
  regime: string;
}

export interface BubbleHistoryData {
  generated_at: string;
  history: BubbleHistoryPoint[];
}
