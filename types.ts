
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

export interface PreviousDay {
  composite_score: number;
  indicators: Record<string, number | null>;
}

export interface BubbleIndexData {
  generated_at: string;
  composite_score: number;
  sentiment_score: number | null;
  liquidity_score: number | null;
  valuation_score?: number | null;
  regime: string;
  indicators: Record<string, BubbleIndicator>;
  previous_day?: PreviousDay;
  score_velocity?: number;
  score_acceleration?: number;
  confidence_interval?: { lower: number; upper: number };
  data_quality?: {
    indicators_available: number;
    indicators_total: number;
    completeness: number;
    data_end_dates: Record<string, string>;
    staleness_warning: boolean;
  };
  diagnostics?: {
    correlation_matrix: Record<string, Record<string, number>>;
    sensitivity: Record<string, number>;
  };
}

export interface BubbleHistoryPoint {
  date: string;
  composite_score: number;
  sentiment_score: number | null;
  liquidity_score: number | null;
  valuation_score?: number | null;
  regime: string;
  indicators?: Record<string, number | null>;
  score_velocity?: number;
  score_acceleration?: number;
}

export interface BubbleHistoryData {
  generated_at: string;
  history: BubbleHistoryPoint[];
}

export interface IndicatorMeta {
  key: string;
  label: string;
  color: string;
  description: string;
  category: 'sentiment' | 'liquidity' | 'valuation';
}

export interface SignalStats {
  count: number;
  mean_return: number;
  median_return: number;
  std: number;
  hit_rate: number;
  t_statistic: number;
}

export interface BacktestResults {
  generated_at: string;
  autocorrelation: Record<string, number>;
  buy_signal_threshold: number;
  sell_signal_threshold: number;
  buy_signals: { count: number; stats: Record<string, SignalStats | null> };
  sell_signals: { count: number; stats: Record<string, SignalStats | null> };
}

export interface GSADFBubblePeriod {
  start: string;
  end: string;
  duration_days: number;
}

export interface GSADFResults {
  generated_at: string;
  method: string;
  bubble_periods: GSADFBubblePeriod[];
  summary: { total_bubble_days: number; pct_bubble: number; largest_bubble: GSADFBubblePeriod };
}

export interface MarkovRegimePoint {
  date: string;
  normal_prob: number;
  elevated_prob: number;
  bubble_prob: number;
}

export interface MarkovRegimes {
  generated_at: string;
  method: string;
  n_regimes: number;
  regime_labels: string[];
  results: MarkovRegimePoint[];
  regime_means: number[];
  current_regime: string;
  current_regime_prob: number;
  transition_matrix: number[][];
}
