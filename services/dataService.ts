
import { DataPoint, MarketSummary, BacktestResult, BacktestSignal, HistoricalSignal } from '../types';
import { DEVIATION_CONFIG, getRiskLevel } from '../constants';

interface TickerDataFile {
  generated_at: string;
  ticker: string;
  data: DataPoint[];
}

/**
 * Fetch data for a specific ticker from the pre-generated static JSON file.
 * Falls back to simulated data if the fetch fails.
 */
export const fetchRealData = async (ticker: string = 'QQQ'): Promise<{ data: DataPoint[]; isDemo: boolean; generatedAt?: string }> => {
  try {
    const base = '/Market-Bubble-Index-Dashboard/';
    const res = await fetch(`${base}data/${ticker.toLowerCase()}.json`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json: TickerDataFile = await res.json();
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
  const volatility = 0.012;
  const priceBuffer: number[] = [];
  const smaWindow = 200;

  for (let i = totalDays; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);

    const change = (Math.random() - 0.48) * volatility;
    currentPrice = currentPrice * (1 + change);
    priceBuffer.push(currentPrice);
    if (priceBuffer.length > smaWindow) priceBuffer.shift();
    const currentSMA = priceBuffer.reduce((sum, p) => sum + p, 0) / priceBuffer.length;
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

/**
 * Run a simple threshold-based backtest on the data.
 * Buy when index < buyThreshold, sell when index > sellThreshold.
 */
export const runBacktest = (
  data: DataPoint[],
  buyThreshold: number,
  sellThreshold: number,
  initialInvestment: number
): BacktestResult => {
  const signals: BacktestSignal[] = [];
  let cash = initialInvestment;
  let shares = 0;
  let inPosition = false;
  let peakValue = initialInvestment;
  let maxDrawdown = 0;

  for (const point of data) {
    const portfolioValue = inPosition ? shares * point.price : cash;

    if (portfolioValue > peakValue) peakValue = portfolioValue;
    const drawdown = (peakValue - portfolioValue) / peakValue;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;

    if (!inPosition && point.index <= buyThreshold) {
      shares = cash / point.price;
      cash = 0;
      inPosition = true;
      signals.push({ date: point.date, type: 'buy', price: point.price, index: point.index });
    } else if (inPosition && point.index >= sellThreshold) {
      cash = shares * point.price;
      shares = 0;
      inPosition = false;
      signals.push({ date: point.date, type: 'sell', price: point.price, index: point.index });
    }
  }

  // Final value
  const lastPrice = data[data.length - 1].price;
  const finalValue = inPosition ? shares * lastPrice : cash;
  const strategyReturn = (finalValue - initialInvestment) / initialInvestment;

  // Buy-and-hold comparison
  const firstPrice = data[0].price;
  const buyHoldReturn = (lastPrice - firstPrice) / firstPrice;

  // Annualize
  const firstDate = new Date(data[0].date);
  const lastDate = new Date(data[data.length - 1].date);
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
  };
};

/**
 * Auto-detect historical signals from peaks and troughs in the deviation index.
 */
export const detectHistoricalSignals = (data: DataPoint[], highThreshold: number = 80, lowThreshold: number = 20): HistoricalSignal[] => {
  const signals: HistoricalSignal[] = [];
  const windowSize = 20; // backward window for local extremes (no look-ahead)

  for (let i = windowSize; i < data.length; i++) {
    const point = data[i];

    if (point.index >= highThreshold) {
      // Check if this is a local maximum (backward only, no look-ahead bias)
      let isMax = true;
      for (let j = i - windowSize; j < i; j++) {
        if (data[j].index > point.index) { isMax = false; break; }
      }
      if (isMax) {
        // Skip if too close to the last signal of the same type
        const lastSell = signals.filter(s => s.type === 'sell').pop();
        if (!lastSell || daysBetween(lastSell.date, point.date) > 30) {
          signals.push({ date: point.date, type: 'sell', index: point.index, price: point.price });
        }
      }
    }

    if (point.index <= lowThreshold) {
      // Check if this is a local minimum (backward only, no look-ahead bias)
      let isMin = true;
      for (let j = i - windowSize; j < i; j++) {
        if (data[j].index < point.index) { isMin = false; break; }
      }
      if (isMin) {
        const lastBuy = signals.filter(s => s.type === 'buy').pop();
        if (!lastBuy || daysBetween(lastBuy.date, point.date) > 30) {
          signals.push({ date: point.date, type: 'buy', index: point.index, price: point.price });
        }
      }
    }
  }

  return signals.sort((a, b) => b.date.localeCompare(a.date)); // newest first
};

function daysBetween(d1: string, d2: string): number {
  return Math.abs(new Date(d2).getTime() - new Date(d1).getTime()) / (24 * 60 * 60 * 1000);
}
