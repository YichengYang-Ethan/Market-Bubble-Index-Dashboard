
import React, { useState, useEffect, useCallback } from 'react';
import DeviationChart from './components/DeviationChart';
import StatsCard from './components/StatsCard';
import BacktestPanel from './components/BacktestPanel';
import { fetchRealData, getMarketSummary, detectHistoricalSignals } from './services/dataService';
import { DataPoint, MarketSummary, BacktestSignal, HistoricalSignal } from './types';
import { DEVIATION_CONFIG, SUPPORTED_TICKERS, TICKER_LABELS, TickerSymbol } from './constants';

type TimeRange = '1Y' | '2Y' | 'ALL';

const REFRESH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

function getInitialTicker(): TickerSymbol {
  try {
    const hash = window.location.hash.slice(1).toUpperCase();
    if ((SUPPORTED_TICKERS as readonly string[]).includes(hash)) return hash as TickerSymbol;
    const stored = localStorage.getItem('selectedTicker');
    if (stored && (SUPPORTED_TICKERS as readonly string[]).includes(stored)) return stored as TickerSymbol;
  } catch { /* ignore */ }
  return 'QQQ';
}

const App: React.FC = () => {
  const [selectedTicker, setSelectedTicker] = useState<TickerSymbol>(getInitialTicker);
  const [allData, setAllData] = useState<DataPoint[]>([]);
  const [data, setData] = useState<DataPoint[]>([]);
  const [summary, setSummary] = useState<MarketSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [timeRange, setTimeRange] = useState<TimeRange>('ALL');
  const [backtestSignals, setBacktestSignals] = useState<BacktestSignal[]>([]);
  const [historicalSignals, setHistoricalSignals] = useState<HistoricalSignal[]>([]);

  const loadData = useCallback(async (ticker: TickerSymbol) => {
    setIsLoading(true);
    const result = await fetchRealData(ticker);
    setAllData(result.data);
    setIsDemo(result.isDemo);
    setLastUpdate(new Date());
    setBacktestSignals([]);
    setHistoricalSignals(detectHistoricalSignals(result.data));
    setIsLoading(false);
  }, []);

  const handleTickerChange = useCallback((ticker: TickerSymbol) => {
    setSelectedTicker(ticker);
    window.location.hash = ticker;
    try { localStorage.setItem('selectedTicker', ticker); } catch { /* ignore */ }
    loadData(ticker);
  }, [loadData]);

  // Apply time range filter
  useEffect(() => {
    if (allData.length === 0) return;
    let filtered = allData;
    if (timeRange !== 'ALL') {
      const years = timeRange === '1Y' ? 1 : 2;
      const cutoff = new Date();
      cutoff.setFullYear(cutoff.getFullYear() - years);
      const cutoffStr = cutoff.toISOString().split('T')[0];
      filtered = allData.filter(d => d.date >= cutoffStr);
    }
    setData(filtered);
    if (filtered.length >= 2) {
      setSummary(getMarketSummary(filtered));
    }
  }, [allData, timeRange]);

  useEffect(() => {
    loadData(selectedTicker);
    const interval = setInterval(() => loadData(selectedTicker), REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [selectedTicker, loadData]);

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
    a.download = `${selectedTicker.toLowerCase()}-deviation-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading || !summary) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-400 font-medium">Fetching {selectedTicker} Data...</p>
        </div>
      </div>
    );
  }

  const { RISK_LEVELS } = DEVIATION_CONFIG;
  const timeRanges: TimeRange[] = ['1Y', '2Y', 'ALL'];

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
              <span className="text-xl font-bold text-white tracking-tight">Deviation Tracker</span>
            </div>

            {/* Ticker Selector */}
            <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-1">
              {SUPPORTED_TICKERS.map(ticker => (
                <button
                  key={ticker}
                  onClick={() => handleTickerChange(ticker)}
                  className={`px-3 py-1.5 rounded-md text-sm font-semibold transition-colors ${
                    selectedTicker === ticker
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-slate-400 hover:text-white hover:bg-slate-700'
                  }`}
                >
                  {ticker}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-4 text-sm text-slate-400">
              {isDemo ? (
                <span className="hidden md:inline bg-amber-600/20 text-amber-400 px-3 py-1 rounded-full text-xs font-medium">Demo Mode</span>
              ) : (
                <span className="hidden md:inline bg-emerald-600/20 text-emerald-400 px-3 py-1 rounded-full text-xs font-medium">Live Data</span>
              )}
              <span className="hidden md:inline">Last Updated: {lastUpdate.toLocaleTimeString()}</span>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Header */}
      <header className="bg-slate-900/50 border-b border-slate-800 mb-8 py-10 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-black text-white mb-2">{selectedTicker} 200D Deviation Dashboard</h1>
          <p className="text-lg text-slate-400 max-w-2xl">
            Monitoring <span className="text-white font-semibold">{TICKER_LABELS[selectedTicker]}</span> ({selectedTicker}) <span className="font-bold text-slate-200 italic">{DEVIATION_CONFIG.SMA_PERIOD}-day moving average deviation</span>.
            When the index surpasses <span className="text-emerald-400 font-bold">{RISK_LEVELS.HIGH}</span>, it signals a high-probability pullback.
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <StatsCard summary={summary} ticker={selectedTicker} />

        {/* Time Range Selector */}
        <div className="flex gap-2 mb-6">
          {timeRanges.map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
                timeRange === range
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
              }`}
            >
              {range}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          <div className="lg:col-span-3">
            <DeviationChart data={data} ticker={selectedTicker} signals={backtestSignals} />
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

            {/* Dynamic Historical Signals */}
            <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-sm">
              <h3 className="font-bold text-white mb-4">Historical Signals</h3>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {historicalSignals.length === 0 ? (
                  <p className="text-sm text-slate-500">No signals detected in this dataset.</p>
                ) : (
                  historicalSignals.map((signal, i) => (
                    <div key={i} className="flex items-center justify-between py-2 border-b border-slate-800">
                      <div>
                        <span className="text-sm text-slate-400">{signal.date}</span>
                        <span className={`ml-2 text-xs font-semibold ${signal.type === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>
                          {signal.type === 'buy' ? 'BUY' : 'SELL'}
                        </span>
                      </div>
                      <span className={`text-xs font-bold px-2 py-1 rounded ${
                        signal.type === 'buy'
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        Index: {signal.index.toFixed(0)}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Backtest Panel */}
        <section className="mt-8">
          <BacktestPanel data={data} ticker={selectedTicker} onSignalsChange={setBacktestSignals} />
        </section>

        {/* Info Section */}
        <section className="mt-12 bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-sm">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-white mb-4">
                When the index exceeds <span className="text-emerald-400">{RISK_LEVELS.HIGH}</span>, it signals danger
              </h2>
              <p className="text-slate-400 leading-relaxed text-lg">
                This indicator tracks the deviation between {TICKER_LABELS[selectedTicker]} ({selectedTicker}) and its {DEVIATION_CONFIG.SMA_PERIOD}-day moving average to gauge market &quot;overheating.&quot;
                Whenever the deviation index reaches the {RISK_LEVELS.HIGH} threshold, it typically signals that buying momentum is nearing its limit,
                and the index will soon pull back, indicating a <span className="text-red-400 font-bold">price correction</span>.
              </p>
              <div className="mt-6 flex gap-4">
                <button
                  onClick={() => loadData(selectedTicker)}
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
        <p>&copy; 2025 Deviation Tracker. Daily market data via Yahoo Finance, updated by GitHub Actions.</p>
        <p className="mt-2">Not financial advice. Automated axis adjustment and risk signaling active.</p>
      </footer>
    </div>
  );
};

export default App;
