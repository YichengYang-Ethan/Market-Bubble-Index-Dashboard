
import { DataPoint, MarketSummary } from '../types';
import { DEVIATION_CONFIG, getRiskLevel } from '../constants';

/**
 * The 200-day deviation index is calculated as:
 * 1. Calculate the percentage difference between the price and its 200-day SMA.
 * 2. Normalize this percentage to a 0-100 scale using historical volatility bounds.
 */

export const generateHistoricalData = (years: number = 10): DataPoint[] => {
  const data: DataPoint[] = [];
  const totalDays = years * 252; // Roughly 252 trading days per year
  const today = new Date();
  
  let currentPrice = 450; // Starting point roughly near current QQQ
  let currentSMA = 400;
  const volatility = 0.012; // Typical daily volatility

  for (let i = totalDays; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    
    // Simulate Brownian motion with some trend
    const change = (Math.random() - 0.48) * volatility; 
    currentPrice = currentPrice * (1 + change);
    
    // Simple mock for SMA that lags the price
    currentSMA = currentSMA * 0.999 + currentPrice * 0.001;

    // Deviation calculation: (Price - SMA) / SMA
    const rawDeviation = (currentPrice - currentSMA) / currentSMA;
    
    // Normalize to 0-100 index based on typical QQQ deviation bounds (-15% to +25%)
    // These bounds map to the 0-100 scale shown in the image
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
