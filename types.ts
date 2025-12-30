
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
