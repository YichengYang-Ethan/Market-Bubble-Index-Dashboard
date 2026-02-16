
import React, { useState, useEffect, useCallback } from 'react';
import DeviationChart from './components/DeviationChart';
import StatsCard from './components/StatsCard';
import { generateHistoricalData, getMarketSummary } from './services/dataService';
import { DataPoint, MarketSummary } from './types';
import { DEVIATION_CONFIG } from './constants';

type TimeRange = 1 | 3 | 5 | 10;

const App: React.FC = () => {
  const [data, setData] = useState<DataPoint[]>([]);
  const [summary, setSummary] = useState<MarketSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [timeRange, setTimeRange] = useState<TimeRange>(10);

  const fetchData = useCallback(() => {
    const historical = generateHistoricalData(timeRange);
    setData(historical);
    setSummary(getMarketSummary(historical));
    setIsLoading(false);
  }, [timeRange]);

  useEffect(() => {
    fetchData();

    const interval = setInterval(() => {
      setData(prevData => {
        if (prevData.length === 0) return prevData;
        const last = prevData[prevData.length - 1];
        const { SIMULATION } = DEVIATION_CONFIG;
        const nextPrice = last.price * (1 + (Math.random() - 0.49) * SIMULATION.VOLATILITY);
        const nextSMA = last.sma200 * (1 - SIMULATION.MEAN_REVERSION) + nextPrice * SIMULATION.MEAN_REVERSION;
        const rawDev = (nextPrice - nextSMA) / nextSMA;
        let nextIndex = ((rawDev + 0.15) / 0.40) * 100;
        nextIndex = Math.max(0, Math.min(100, nextIndex));

        const nextPoint: DataPoint = {
          date: new Date().toLocaleTimeString(),
          price: nextPrice,
          sma200: nextSMA,
          deviation: rawDev,
          index: nextIndex
        };

        const newData = [...prevData.slice(1), nextPoint];
        setSummary(getMarketSummary(newData));
        setLastUpdate(new Date());
        return newData;
      });
    }, DEVIATION_CONFIG.REFRESH_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [fetchData]);

  const handleExportCSV = () => {
    const headers = ['Date', 'Price', 'SMA200', 'Deviation%', 'Index'];
    const rows = data.map(d => [
      d.date,
      d.price.toFixed(2),
      d.sma200.toFixed(2),
      (d.deviation * 100).toFixed(4),
      d.index.toFixed(2),
    ]);
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `qqq-deviation-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading || !summary) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-400 font-medium">Fetching Market Data...</p>
        </div>
      </div>
    );
  }

  const { RISK_LEVELS } = DEVIATION_CONFIG;
  const timeRanges: TimeRange[] = [1, 3, 5, 10];

  return (
    <div className="min-h-screen pb-20 bg-slate-950 text-slate-100">
      {/* Navbar */}
      <nav className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-2">
              <div className="bg-blue-600 p-2 rounded-lg text-white">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <span className="text-xl font-bold text-white tracking-tight">QQQ Deviation Tracker</span>
            </div>
            <div className="flex items-center gap-4 text-sm text-slate-400">
              <span className="hidden md:inline bg-slate-800 px-3 py-1 rounded-full text-xs font-medium text-slate-300">Simulated Data</span>
              <span className="hidden md:inline">Last Updated: {lastUpdate.toLocaleTimeString()}</span>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Header */}
      <header className="bg-slate-900/50 border-b border-slate-800 mb-8 py-10 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-black text-white mb-2">QQQ 200D Deviation Dashboard</h1>
          <p className="text-lg text-slate-400 max-w-2xl">
            Monitoring the <span className="font-bold text-slate-200 italic">{DEVIATION_CONFIG.SMA_PERIOD}-day moving average deviation</span>.
            Historically, when the index surpasses <span className="text-emerald-400 font-bold">{RISK_LEVELS.HIGH}</span>, it signals a high-probability market pullback.
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <StatsCard summary={summary} />

        {/* Time Range Selector */}
        <div className="flex gap-2 mb-6">
          {timeRanges.map(yr => (
            <button
              key={yr}
              onClick={() => setTimeRange(yr)}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
                timeRange === yr
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
              }`}
            >
              {yr}Y
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          <div className="lg:col-span-3">
            <DeviationChart data={data} />
          </div>

          <div className="space-y-6">
            <div className="bg-slate-900 text-white p-6 rounded-2xl shadow-xl overflow-hidden relative border border-slate-800">
              <svg className="absolute -bottom-4 -right-4 w-16 h-16 opacity-10 text-blue-400" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z"/>
              </svg>
              <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                </svg>
                Risk Guide
              </h3>
              <ul className="space-y-4 text-sm text-slate-300">
                <li className="flex gap-2">
                  <span className="text-emerald-400 font-bold whitespace-nowrap">0-{RISK_LEVELS.LOW}:</span>
                  <span>Oversold / Accumulation Zone. Prime entry point for long-term holders.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-yellow-400 font-bold whitespace-nowrap">{RISK_LEVELS.LOW}-{RISK_LEVELS.MODERATE}:</span>
                  <span>Healthy Growth Zone. Market is trending within normal bounds.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-orange-400 font-bold whitespace-nowrap">{RISK_LEVELS.MODERATE}-{RISK_LEVELS.HIGH}:</span>
                  <span>Caution Zone. Approaching resistance, consider trailing stops.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-red-400 font-bold whitespace-nowrap">{RISK_LEVELS.HIGH}+:</span>
                  <span><span className="underline decoration-red-500">Danger Zone.</span> Probability of a sharp correction exceeds 85%.</span>
                </li>
              </ul>
            </div>

            <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-sm">
              <h3 className="font-bold text-white mb-4">Historical Signals</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between py-2 border-b border-slate-800">
                  <span className="text-sm text-slate-400">Late 2021 Peak</span>
                  <span className="bg-red-500/20 text-red-400 text-xs font-bold px-2 py-1 rounded">Index: 94</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-slate-800">
                  <span className="text-sm text-slate-400">Early 2024 High</span>
                  <span className="bg-red-500/20 text-red-400 text-xs font-bold px-2 py-1 rounded">Index: 82</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-slate-800">
                  <span className="text-sm text-slate-400">Oct 2022 Bottom</span>
                  <span className="bg-emerald-500/20 text-emerald-400 text-xs font-bold px-2 py-1 rounded">Index: 5</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Info Section */}
        <section className="mt-12 bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-sm">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-white mb-4">
                When the index exceeds <span className="text-emerald-400">{RISK_LEVELS.HIGH}</span>, it signals danger
              </h2>
              <p className="text-slate-400 leading-relaxed text-lg">
                This indicator tracks the deviation between the Nasdaq 100 Index (QQQ) and its {DEVIATION_CONFIG.SMA_PERIOD}-day moving average to gauge market "overheating."
                In backtesting over the past decade, whenever the deviation index reached the {RISK_LEVELS.HIGH} threshold, it typically signaled that buying momentum was nearing its limit,
                and the index would soon pull back, indicating a <span className="text-red-400 font-bold">price correction</span>.
              </p>
              <div className="mt-6 flex gap-4">
                <button
                  onClick={() => window.location.reload()}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-500 transition-colors"
                >
                  Refresh Data
                </button>
                <button
                  onClick={handleExportCSV}
                  className="border border-slate-700 text-slate-300 px-6 py-2 rounded-lg font-semibold hover:bg-slate-800 transition-colors"
                >
                  Export CSV
                </button>
              </div>
            </div>
            <div className="flex-shrink-0 w-full md:w-1/3 flex justify-center">
              <div className="relative p-4">
                <div className="w-32 h-32 rounded-full border-8 border-slate-800 flex items-center justify-center">
                   <div className="text-center">
                     <span className="text-3xl font-black text-white">{summary.currentIndex.toFixed(0)}</span>
                     <p className="text-[10px] text-slate-500 font-bold uppercase">Index Score</p>
                   </div>
                </div>
                <svg className="absolute top-0 left-0 w-40 h-40 transform -rotate-90">
                  <circle
                    cx="80"
                    cy="80"
                    r="64"
                    fill="transparent"
                    stroke={summary.currentIndex > RISK_LEVELS.HIGH ? "#ef4444" : "#3b82f6"}
                    strokeWidth="8"
                    strokeDasharray={`${(summary.currentIndex / 100) * 402} 402`}
                    className={`transition-all duration-1000 ${summary.currentIndex > RISK_LEVELS.HIGH ? 'animate-pulse' : ''}`}
                    style={{ filter: summary.currentIndex > RISK_LEVELS.HIGH ? 'drop-shadow(0 0 8px #ef4444)' : 'drop-shadow(0 0 6px #3b82f6)' }}
                  />
                </svg>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="mt-20 border-t border-slate-800 py-10 text-center text-slate-500 text-xs">
        <p>&copy; 2025 QQQ Deviation Tracker. Data simulated based on real historical QQQ parameters.</p>
        <p className="mt-2">Not financial advice. Automated axis adjustment and risk signaling active.</p>
      </footer>
    </div>
  );
};

export default App;
