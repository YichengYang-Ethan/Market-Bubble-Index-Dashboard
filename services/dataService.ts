
import { DataPoint, MarketSummary } from '../types';
import { DEVIATION_CONFIG, getRiskLevel } from '../constants';

interface QQQDataFile {
  generated_at: string;
  ticker: string;
  data: DataPoint[];
}

/**
 * Fetch real QQQ data from the pre-generated static JSON file.
 * Falls back to simulated data if the fetch fails.
 */
export const fetchRealData = async (): Promise<{ data: DataPoint[]; isDemo: boolean; generatedAt?: string }> => {
  try {
    const base = import.meta.env.BASE_URL || '/';
    const res = await fetch(`${base}data/qqq.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json: QQQDataFile = await res.json();
    if (!json.data || json.data.length === 0) throw new Error('Empty data');
    return { data: json.data, isDemo: false, generatedAt: json.generated_at };
  } catch {
    // Fallback to simulated data
    return { data: generateHistoricalData(2), isDemo: true };
  }
};

/**
 * Simulated data generator (Brownian motion). Used as fallback only.
 */
export const generateHistoricalData = (years: number = 10): DataPoint[] => {
  const data: DataPoint[] = [];
  const totalDays = years * 252;
  const today = new Date();

  let currentPrice = 450;
  let currentSMA = 400;
  const volatility = 0.012;

  for (let i = totalDays; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);

    const change = (Math.random() - 0.48) * volatility;
    currentPrice = currentPrice * (1 + change);
    currentSMA = currentSMA * 0.999 + currentPrice * 0.001;
    const rawDeviation = (currentPrice - currentSMA) / currentSMA;

    let indexValue = ((rawDeviation + 0.15) / 0.40) * 100;
    indexValue = Math.max(0, Math.min(100, indexValue));

    data.push({
      date: date.toISOString().split('T')[0],
      price: currentPrice,
      sma200: currentSMA,
      deviation: rawDeviation,
      index: indexValue
    });
  }

  return data;
};

export const getMarketSummary = (data: DataPoint[]): MarketSummary => {
  const latest = data[data.length - 1];
  const previous = data[data.length - 2];

  const change = ((latest.price - previous.price) / previous.price) * 100;

  return {
    currentPrice: latest.price,
    currentSMA: latest.sma200,
    currentIndex: latest.index,
    change24h: change,
    riskLevel: getRiskLevel(latest.index)
  };
};
