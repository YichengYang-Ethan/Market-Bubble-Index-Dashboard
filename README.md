# QQQ 200-Day Moving Average Deviation Dashboard

Real-time monitoring of QQQ's deviation from its 200-day moving average — a key technical indicator for market timing.

![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)

## Why This Matters

The 200-day moving average (200 DMA) is one of the most watched technical indicators:

- **When QQQ is significantly ABOVE the 200 DMA** (index > 80): Market may be overextended, historically signals potential pullback
- **When QQQ is significantly BELOW the 200 DMA** (index < 20): Market may be oversold, historically signals potential bounce

> *"Historically, when the deviation index surpasses 80, it signals a high-probability market pullback."*

## Features

- **Real-time Deviation Index**: Normalized 0-100 scale for easy interpretation
- **Interactive Chart**: Historical deviation with trend visualization
- **Risk Level Indicator**: Low / Moderate / High / Danger zones
- **Stats Dashboard**: Current price, SMA, 24h change
- **Auto-refresh**: Simulated real-time updates every 10 seconds

## Tech Stack

- React + TypeScript
- Vite
- Recharts for data visualization

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

## Data Structure

```typescript
interface DataPoint {
  date: string;
  price: number;      // Current QQQ price
  sma200: number;     // 200-day simple moving average
  deviation: number;  // Raw deviation percentage
  index: number;      // Normalized 0-100 index
}

interface MarketSummary {
  currentPrice: number;
  currentSMA: number;
  currentIndex: number;
  change24h: number;
  riskLevel: 'Low' | 'Moderate' | 'High' | 'Danger';
}
```

## Interpretation Guide

| Index Range | Risk Level | Market Condition | Historical Action |
|-------------|------------|------------------|-------------------|
| 0 - 20 | Low | Oversold | Potential buying opportunity |
| 20 - 50 | Moderate | Normal range | Hold / monitor |
| 50 - 80 | High | Extended | Reduce exposure |
| 80 - 100 | Danger | Overextended | High pullback probability |

## License

MIT
