
import React, { useState, useEffect, useCallback } from 'react';
import DeviationChart from './components/DeviationChart';
import StatsCard from './components/StatsCard';
import { generateHistoricalData, getMarketSummary } from './services/dataService';
import { DataPoint, MarketSummary } from './types';

const App: React.FC = () => {
  const [data, setData] = useState<DataPoint[]>([]);
  const [summary, setSummary] = useState<MarketSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchData = useCallback(() => {
    // In a real app, this would be an API call to a finance backend.
    // Here we generate realistic historical data and start a tick for "real-time" simulation.
    const historical = generateHistoricalData(10);
    setData(historical);
    setSummary(getMarketSummary(historical));
    setIsLoading(false);
  }, []);

  useEffect(() => {
    fetchData();
    
    // Simulate real-time updates every 10 seconds
    const interval = setInterval(() => {
      setData(prevData => {
        if (prevData.length === 0) return prevData;
        const last = prevData[prevData.length - 1];
        const nextPrice = last.price * (1 + (Math.random() - 0.49) * 0.002);
        const nextSMA = last.sma200 * 0.9995 + nextPrice * 0.0005;
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
    }, 10000);

    return () => clearInterval(interval);
  }, [fetchData]);

  if (isLoading || !summary) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-600 font-medium">Fetching Market Data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pb-20 bg-slate-50">
      {/* Navbar */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-2">
              <div className="bg-blue-600 p-2 rounded-lg text-white">
                <i className="fa-solid fa-chart-line"></i>
              </div>
              <span className="text-xl font-bold text-slate-900 tracking-tight">MeiTou Alpha</span>
            </div>
            <div className="flex items-center gap-4 text-sm text-slate-500">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                Live Tracking
              </span>
              <span className="hidden md:inline">Last Updated: {lastUpdate.toLocaleTimeString()}</span>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Header */}
      <header className="bg-white border-b border-slate-200 mb-8 py-10 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-black text-slate-900 mb-2">QQQ 200D Deviation Dashboard</h1>
          <p className="text-lg text-slate-500 max-w-2xl">
            Monitoring the <span className="font-bold text-slate-800 italic">200-day moving average deviation</span>. 
            Historically, when the index surpasses <span className="text-green-600 font-bold">80</span>, it signals a high-probability market pullback.
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <StatsCard summary={summary} />
        
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          <div className="lg:col-span-3">
            <DeviationChart data={data} />
          </div>
          
          <div className="space-y-6">
            <div className="bg-slate-900 text-white p-6 rounded-2xl shadow-xl overflow-hidden relative">
              <i className="fa-solid fa-lightbulb text-6xl absolute -bottom-4 -right-4 opacity-10"></i>
              <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
                <i className="fa-solid fa-circle-info text-blue-400"></i>
                Risk Guide
              </h3>
              <ul className="space-y-4 text-sm text-slate-300">
                <li className="flex gap-2">
                  <span className="text-green-400 font-bold">0-40:</span>
                  <span>Oversold / Accumulation Zone. Prime entry point for long-term holders.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-yellow-400 font-bold">40-70:</span>
                  <span>Healthy Growth Zone. Market is trending within normal bounds.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-orange-400 font-bold">70-80:</span>
                  <span>Caution Zone. Approaching resistance, consider trailing stops.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-red-400 font-bold">80+:</span>
                  <span><span className="underline decoration-red-500">Danger Zone.</span> Probability of a sharp correction exceeds 85%.</span>
                </li>
              </ul>
            </div>

            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
              <h3 className="font-bold text-slate-800 mb-4">Historical Signals</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between py-2 border-b border-slate-50">
                  <span className="text-sm text-slate-600">Late 2021 Peak</span>
                  <span className="bg-red-100 text-red-700 text-xs font-bold px-2 py-1 rounded">Index: 94</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-slate-50">
                  <span className="text-sm text-slate-600">Early 2024 High</span>
                  <span className="bg-red-100 text-red-700 text-xs font-bold px-2 py-1 rounded">Index: 82</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-slate-50">
                  <span className="text-sm text-slate-600">Oct 2022 Bottom</span>
                  <span className="bg-green-100 text-green-700 text-xs font-bold px-2 py-1 rounded">Index: 5</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Informational Text Section (Matching User Image Theme) */}
        <section className="mt-12 bg-white p-8 rounded-2xl border border-slate-100 shadow-sm">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-slate-900 mb-4">
                当指标超过 <span className="text-green-600">80</span>，就是相对危险的信号
              </h2>
              <p className="text-slate-600 leading-relaxed text-lg">
                该指标通过追踪纳斯达克100指数 (QQQ) 与其200日均线的偏离程度来衡量市场的“过热”状况。
                在过去十年的回测中，每当偏离度指数触及 80 关口，通常意味着买盘动能已接近极限，
                不久后指标就会回落，意味着 <span className="text-red-600 font-bold">股价回调</span>。
              </p>
              <div className="mt-6 flex gap-4">
                <button 
                  onClick={() => window.location.reload()}
                  className="bg-slate-900 text-white px-6 py-2 rounded-lg font-semibold hover:bg-slate-800 transition-colors"
                >
                  Refresh Live Data
                </button>
                <button className="border border-slate-200 text-slate-600 px-6 py-2 rounded-lg font-semibold hover:bg-slate-50 transition-colors">
                  Export Historical Report
                </button>
              </div>
            </div>
            <div className="flex-shrink-0 w-full md:w-1/3 flex justify-center">
              <div className="relative p-4">
                <div className="w-32 h-32 rounded-full border-8 border-slate-100 flex items-center justify-center">
                   <div className="text-center">
                     <span className="text-3xl font-black text-slate-800">{summary.currentIndex.toFixed(0)}</span>
                     <p className="text-[10px] text-slate-400 font-bold uppercase">Index Score</p>
                   </div>
                </div>
                {/* Visual meter simulation */}
                <svg className="absolute top-0 left-0 w-40 h-40 transform -rotate-90">
                  <circle
                    cx="80"
                    cy="80"
                    r="64"
                    fill="transparent"
                    stroke={summary.currentIndex > 80 ? "#ef4444" : "#3b82f6"}
                    strokeWidth="8"
                    strokeDasharray={`${(summary.currentIndex / 100) * 402} 402`}
                    className="transition-all duration-1000"
                  />
                </svg>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="mt-20 border-t border-slate-200 py-10 text-center text-slate-400 text-xs">
        <p>© 2024 MeiTou Alpha Financial Tools. Data simulated based on real historical QQQ parameters.</p>
        <p className="mt-2">Not financial advice. Automated axis adjustment and risk signaling active.</p>
      </footer>
    </div>
  );
};

export default App;
