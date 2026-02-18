
import React, { useState, useEffect, useCallback, useRef } from 'react';
import DeviationChart from './components/DeviationChart';
import StatsCard from './components/StatsCard';
import BacktestPanel from './components/BacktestPanel';
import BubbleGauge from './components/BubbleGauge';
import BubbleHistoryChart from './components/BubbleHistoryChart';
import BubbleBacktestPanel from './components/BubbleBacktestPanel';
import CrashProbabilityPanel from './components/CrashProbabilityPanel';
import IndicatorGrid from './components/IndicatorGrid';
import IndicatorDeepDive from './components/IndicatorDeepDive';
import { fetchRealData, getMarketSummary, detectHistoricalSignals } from './services/dataService';
import { fetchBubbleIndex, fetchBubbleHistory, fetchTickerPrice, fetchBacktestResults, fetchGSADFResults, fetchMarkovRegimes, fetchDrawdownModel, fetchQQQDrawdown, DrawdownModelData, DrawdownPoint } from './services/bubbleService';
import { DataPoint, MarketSummary, BacktestSignal, HistoricalSignal, BubbleIndexData, BubbleHistoryPoint, BacktestResults, GSADFResults, MarkovRegimes } from './types';
import { DEVIATION_CONFIG, SUPPORTED_TICKERS, TICKER_LABELS, TickerSymbol, BUBBLE_REGIME_CONFIG, INDICATOR_META } from './constants';

type TimeRange = '1Y' | '2Y' | 'ALL';

const REFRESH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

const NAV_SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'indicators', label: 'Indicators' },
  { id: 'history', label: 'History' },
  { id: 'signal-analysis', label: 'Signal Analysis' },
  { id: 'deep-dive', label: 'Deep Dive' },
  { id: 'deviation', label: 'Deviation Tracker' },
  { id: 'methodology', label: 'Methodology' },
] as const;

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
  const [activeSection, setActiveSection] = useState<string>('overview');

  // Deviation state
  const [selectedTicker, setSelectedTicker] = useState<TickerSymbol>(getInitialTicker);
  const [allData, setAllData] = useState<DataPoint[]>([]);
  const [data, setData] = useState<DataPoint[]>([]);
  const [summary, setSummary] = useState<MarketSummary | null>(null);
  const [deviationLoading, setDeviationLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [timeRange, setTimeRange] = useState<TimeRange>('ALL');
  const [backtestSignals, setBacktestSignals] = useState<BacktestSignal[]>([]);
  const [historicalSignals, setHistoricalSignals] = useState<HistoricalSignal[]>([]);

  // Bubble state
  const [bubbleData, setBubbleData] = useState<BubbleIndexData | null>(null);
  const [bubbleHistory, setBubbleHistory] = useState<BubbleHistoryPoint[]>([]);
  const [bubbleLoading, setBubbleLoading] = useState(true);
  const [bubbleError, setBubbleError] = useState<string | null>(null);
  const [priceData, setPriceData] = useState<{ qqq: DataPoint[]; spy: DataPoint[] }>({ qqq: [], spy: [] });
  const [backtestResults, setBacktestResults] = useState<BacktestResults | null>(null);
  const [gsadfResults, setGsadfResults] = useState<GSADFResults | null>(null);
  const [markovRegimes, setMarkovRegimes] = useState<MarkovRegimes | null>(null);
  const [drawdownModel, setDrawdownModel] = useState<DrawdownModelData | null>(null);
  const [qqqDrawdown, setQqqDrawdown] = useState<DrawdownPoint[]>([]);

  // Refs for scroll-spy
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});

  const setSectionRef = useCallback((id: string) => (el: HTMLElement | null) => {
    sectionRefs.current[id] = el;
  }, []);

  // Scroll-spy via IntersectionObserver
  useEffect(() => {
    const observers: IntersectionObserver[] = [];
    const visibilityMap = new Map<string, number>();

    const handleIntersect = (entries: IntersectionObserverEntry[]) => {
      entries.forEach(entry => {
        const id = entry.target.getAttribute('id');
        if (id) {
          visibilityMap.set(id, entry.intersectionRatio);
        }
      });

      // Find the section with the highest intersection ratio
      let maxRatio = 0;
      let maxId = 'overview';
      visibilityMap.forEach((ratio, id) => {
        if (ratio > maxRatio) {
          maxRatio = ratio;
          maxId = id;
        }
      });

      if (maxRatio > 0) {
        setActiveSection(maxId);
      }
    };

    const observer = new IntersectionObserver(handleIntersect, {
      rootMargin: '-80px 0px -40% 0px',
      threshold: [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    });

    NAV_SECTIONS.forEach(({ id }) => {
      const el = sectionRefs.current[id];
      if (el) observer.observe(el);
    });

    observers.push(observer);

    return () => {
      observers.forEach(obs => obs.disconnect());
    };
  }, [bubbleData, summary]); // Re-attach after data loads

  const loadData = useCallback(async (ticker: TickerSymbol) => {
    setDeviationLoading(true);
    const result = await fetchRealData(ticker);
    setAllData(result.data);
    setIsDemo(result.isDemo);
    setLastUpdate(new Date());
    setBacktestSignals([]);
    setHistoricalSignals(detectHistoricalSignals(result.data));
    setDeviationLoading(false);
  }, []);

  const loadBubbleData = useCallback(async () => {
    setBubbleLoading(true);
    setBubbleError(null);
    try {
      const [indexData, historyData, qqqData, spyData, btResults, gsadf, markov, ddModel, ddSeries] = await Promise.all([
        fetchBubbleIndex(),
        fetchBubbleHistory(),
        fetchTickerPrice('qqq'),
        fetchTickerPrice('spy'),
        fetchBacktestResults(),
        fetchGSADFResults(),
        fetchMarkovRegimes(),
        fetchDrawdownModel(),
        fetchQQQDrawdown(),
      ]);
      setBubbleData(indexData);
      setBubbleHistory(historyData.history);
      setPriceData({ qqq: qqqData, spy: spyData });
      setBacktestResults(btResults);
      setGsadfResults(gsadf);
      setMarkovRegimes(markov);
      setDrawdownModel(ddModel);
      setQqqDrawdown(ddSeries);
    } catch (e) {
      setBubbleError(e instanceof Error ? e.message : 'Failed to load bubble data');
    } finally {
      setBubbleLoading(false);
    }
  }, []);

  const handleTickerChange = useCallback((ticker: TickerSymbol) => {
    setSelectedTicker(ticker);
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

  // Load bubble data once on mount
  useEffect(() => {
    loadBubbleData();
  }, [loadBubbleData]);

  // Load deviation data on mount and ticker change
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

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, sectionId: string) => {
    e.preventDefault();
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Initial loading: show spinner if bubble data is still loading
  if (bubbleLoading && !bubbleData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-400 font-medium">Loading Market Bubble Index...</p>
        </div>
      </div>
    );
  }

  const { RISK_LEVELS } = DEVIATION_CONFIG;
  const timeRanges: TimeRange[] = ['1Y', '2Y', 'ALL'];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Sticky Navbar with scroll-spy */}
      <nav className="bg-slate-900/95 backdrop-blur-sm border-b border-slate-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center h-14 gap-6">
            {/* Brand */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <div className="bg-blue-600 p-1.5 rounded-lg text-white">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <span className="text-base font-bold text-white tracking-tight hidden sm:inline">Market Bubble Index</span>
            </div>

            {/* Scroll-spy nav links */}
            <div className="flex items-center gap-1 overflow-x-auto scrollbar-hide">
              {NAV_SECTIONS.map(({ id, label }) => (
                <a
                  key={id}
                  href={`#${id}`}
                  onClick={(e) => handleNavClick(e, id)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium whitespace-nowrap transition-colors ${
                    activeSection === id
                      ? 'bg-blue-600/20 text-blue-400'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800'
                  }`}
                >
                  {label}
                </a>
              ))}
            </div>

            {/* Last updated */}
            <div className="ml-auto flex-shrink-0 hidden md:block">
              <span className="text-xs text-slate-500">Updated: {lastUpdate.toLocaleTimeString()}</span>
            </div>
          </div>
        </div>
      </nav>

      {/* ========== HERO SECTION ========== */}
      <section
        id="overview"
        ref={setSectionRef('overview')}
        className="bg-slate-900/50 border-b border-slate-800 py-12 px-4 sm:px-6 lg:px-8"
      >
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-black text-white mb-3">Market Bubble Index</h1>
            <p className="text-lg text-slate-400 max-w-3xl mx-auto">
              A composite measure of market euphoria across <span className="text-white font-semibold">7 indicators</span> including
              QQQ deviation, VIX, sector breadth, credit spreads, put/call ratio, yield curve, and CAPE valuation.
            </p>
          </div>

          {bubbleError ? (
            <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-8 text-center max-w-xl mx-auto">
              <p className="text-red-400 font-medium mb-4">Failed to load bubble index data</p>
              <p className="text-slate-500 text-sm mb-4">{bubbleError}</p>
              <button
                onClick={loadBubbleData}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-500 transition-colors"
              >
                Retry
              </button>
            </div>
          ) : bubbleData ? (
            <div className="flex justify-center">
              <BubbleGauge
                compositeScore={bubbleData.composite_score}
                regime={bubbleData.regime}
                sentimentScore={bubbleData.sentiment_score}
                liquidityScore={bubbleData.liquidity_score}
                valuationScore={bubbleData.valuation_score}
                generatedAt={bubbleData.generated_at}
                scoreVelocity={bubbleData.score_velocity}
                confidenceInterval={bubbleData.confidence_interval}
                dataQuality={bubbleData.data_quality}
              />
            </div>
          ) : null}
        </div>
      </section>

      {/* ========== INDICATOR GRID ========== */}
      {bubbleData && (
        <section
          id="indicators"
          ref={setSectionRef('indicators')}
          className="py-12 px-4 sm:px-6 lg:px-8"
        >
          <div className="max-w-7xl mx-auto">
            <h2 className="text-2xl font-bold text-white mb-6">Indicator Scores</h2>
            <IndicatorGrid
              indicators={bubbleData.indicators}
              previousDay={bubbleData.previous_day}
              history={bubbleHistory}
            />
          </div>
        </section>
      )}

      {/* ========== COMPOSITE HISTORY ========== */}
      {bubbleHistory.length > 0 && (
        <section
          id="history"
          ref={setSectionRef('history')}
          className="py-12 px-4 sm:px-6 lg:px-8 bg-slate-900/30"
        >
          <div className="max-w-7xl mx-auto">
            <h2 className="text-2xl font-bold text-white mb-6">Composite History</h2>
            <BubbleHistoryChart history={bubbleHistory} priceData={priceData} gsadfResults={gsadfResults} markovRegimes={markovRegimes} qqqDrawdown={qqqDrawdown} />
            {priceData.qqq.length > 0 && (
              <div className="mt-8">
                <BubbleBacktestPanel history={bubbleHistory} priceData={priceData.qqq} ticker="QQQ" />
              </div>
            )}
            {drawdownModel && bubbleData && (
              <div className="mt-8">
                <CrashProbabilityPanel
                  model={drawdownModel}
                  currentScore={bubbleData.composite_score}
                  scoreVelocity={bubbleData.score_velocity ?? 0}
                />
              </div>
            )}
          </div>
        </section>
      )}

      {/* ========== SIGNAL ANALYSIS ========== */}
      {(backtestResults || bubbleData?.diagnostics) && (
        <section
          id="signal-analysis"
          ref={setSectionRef('signal-analysis')}
          className="py-12 px-4 sm:px-6 lg:px-8"
        >
          <div className="max-w-7xl mx-auto">
            <h2 className="text-2xl font-bold text-white mb-6">Signal Analysis</h2>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

              {/* Position Mapping Card */}
              {bubbleData && (
                <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
                  <h3 className="text-lg font-bold text-white mb-4">Position Mapping</h3>
                  {(() => {
                    const score = bubbleData.composite_score;
                    const mapping = score < 30
                      ? { label: 'Overweight +30%', bg: 'bg-emerald-500/20', border: 'border-emerald-500/30', text: 'text-emerald-400' }
                      : score < 50
                      ? { label: 'Hold Benchmark', bg: 'bg-slate-700/30', border: 'border-slate-600', text: 'text-slate-300' }
                      : score < 70
                      ? { label: 'Underweight -20%', bg: 'bg-yellow-500/20', border: 'border-yellow-500/30', text: 'text-yellow-400' }
                      : score < 85
                      ? { label: 'Underweight -50%', bg: 'bg-orange-500/20', border: 'border-orange-500/30', text: 'text-orange-400' }
                      : { label: 'Retain 20% Only', bg: 'bg-red-500/20', border: 'border-red-500/30', text: 'text-red-400' };
                    return (
                      <div className={`${mapping.bg} border ${mapping.border} rounded-xl p-6 text-center`}>
                        <p className={`text-3xl font-black ${mapping.text}`}>{mapping.label}</p>
                        <p className="text-sm text-slate-400 mt-2">
                          Based on composite score: <span className="text-white font-semibold">{score.toFixed(1)}</span>
                        </p>
                        {markovRegimes && (
                          <p className="text-xs text-slate-500 mt-2">
                            Markov regime: <span className="text-slate-300 capitalize">{markovRegimes.current_regime}</span> ({(markovRegimes.current_regime_prob * 100).toFixed(0)}%)
                          </p>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}

              {/* Backtest Results Card */}
              {backtestResults && (
                <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl lg:col-span-2">
                  <h3 className="text-lg font-bold text-white mb-4">Signal Backtest</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-700">
                          <th className="text-left text-slate-400 py-2 pr-4">Horizon</th>
                          <th className="text-right text-emerald-400 py-2 px-3">Buy Avg</th>
                          <th className="text-right text-emerald-400 py-2 px-3">Buy Hit%</th>
                          <th className="text-right text-red-400 py-2 px-3">Sell Avg</th>
                          <th className="text-right text-red-400 py-2 px-3">Sell Hit%</th>
                        </tr>
                      </thead>
                      <tbody>
                        {['1d', '5d', '20d', '60d'].map((horizon) => {
                          const buy = backtestResults.buy_signals.stats[horizon];
                          const sell = backtestResults.sell_signals.stats[horizon];
                          return (
                            <tr key={horizon} className="border-b border-slate-800">
                              <td className="text-slate-300 py-2 pr-4 font-medium">{horizon}</td>
                              <td className="text-right py-2 px-3">
                                <span className={buy && buy.mean_return > 0 ? 'text-emerald-400' : 'text-red-400'}>
                                  {buy ? `${buy.mean_return > 0 ? '+' : ''}${buy.mean_return.toFixed(2)}%` : '--'}
                                </span>
                              </td>
                              <td className="text-right py-2 px-3 text-slate-300">
                                {buy ? `${buy.hit_rate.toFixed(0)}%` : '--'}
                              </td>
                              <td className="text-right py-2 px-3">
                                <span className={sell && sell.mean_return > 0 ? 'text-emerald-400' : 'text-red-400'}>
                                  {sell ? `${sell.mean_return > 0 ? '+' : ''}${sell.mean_return.toFixed(2)}%` : '--'}
                                </span>
                              </td>
                              <td className="text-right py-2 px-3 text-slate-300">
                                {sell ? `${sell.hit_rate.toFixed(0)}%` : '--'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-4 flex gap-4 text-xs text-slate-500">
                    <span>Buy signals: {backtestResults.buy_signals.count} (score &lt; {backtestResults.buy_signal_threshold})</span>
                    <span>Sell signals: {backtestResults.sell_signals.count} (score &gt; {backtestResults.sell_signal_threshold})</span>
                  </div>
                  <div className="mt-3">
                    <p className="text-xs text-slate-500 font-semibold uppercase mb-1">Autocorrelation</p>
                    <div className="flex gap-3 text-xs">
                      {Object.entries(backtestResults.autocorrelation).map(([lag, val]) => (
                        <span key={lag} className="text-slate-400">
                          {lag}: <span className="text-white font-semibold">{val.toFixed(3)}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Sensitivity Chart */}
            {bubbleData?.diagnostics?.sensitivity && (
              <div className="mt-6 bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
                <h3 className="text-lg font-bold text-white mb-4">Indicator Sensitivity</h3>
                <div className="space-y-3">
                  {Object.entries(bubbleData.diagnostics.sensitivity)
                    .sort(([, a], [, b]) => b - a)
                    .map(([key, value]) => {
                      const meta = INDICATOR_META.find((m) => m.key === key);
                      const maxSens = Math.max(...Object.values(bubbleData.diagnostics!.sensitivity));
                      return (
                        <div key={key} className="flex items-center gap-3">
                          <span className="text-sm text-slate-400 w-40 truncate">{meta?.label ?? key}</span>
                          <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{
                                width: `${(value / maxSens) * 100}%`,
                                backgroundColor: meta?.color ?? '#64748b',
                              }}
                            />
                          </div>
                          <span className="text-white font-semibold text-sm w-12 text-right">{value.toFixed(2)}</span>
                        </div>
                      );
                    })}
                </div>
                <p className="text-xs text-slate-500 mt-3">
                  Sensitivity = max score change when indicator moves 1 percentile point.
                </p>
              </div>
            )}
          </div>
        </section>
      )}

      {/* ========== INDICATOR DEEP DIVE ========== */}
      {bubbleHistory.length > 0 && (
        <section
          id="deep-dive"
          ref={setSectionRef('deep-dive')}
          className="py-12 px-4 sm:px-6 lg:px-8"
        >
          <div className="max-w-7xl mx-auto">
            <h2 className="text-2xl font-bold text-white mb-6">Indicator Deep Dive</h2>
            <IndicatorDeepDive history={bubbleHistory} />
          </div>
        </section>
      )}

      {/* ========== QQQ DEVIATION TRACKER ========== */}
      <section
        id="deviation"
        ref={setSectionRef('deviation')}
        className="py-12 px-4 sm:px-6 lg:px-8 bg-slate-900/30"
      >
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
            <h2 className="text-2xl font-bold text-white">QQQ Deviation Tracker</h2>

            <div className="flex items-center gap-3 flex-wrap">
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

              {/* Time Range Selector */}
              <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-1">
                {timeRanges.map(range => (
                  <button
                    key={range}
                    onClick={() => setTimeRange(range)}
                    className={`px-3 py-1.5 rounded-md text-sm font-semibold transition-colors ${
                      timeRange === range
                        ? 'bg-blue-600 text-white shadow-sm'
                        : 'text-slate-400 hover:text-white hover:bg-slate-700'
                    }`}
                  >
                    {range}
                  </button>
                ))}
              </div>

              {isDemo ? (
                <span className="bg-amber-600/20 text-amber-400 px-3 py-1 rounded-full text-xs font-medium">Demo Mode</span>
              ) : (
                <span className="bg-emerald-600/20 text-emerald-400 px-3 py-1 rounded-full text-xs font-medium">Live Data</span>
              )}
            </div>
          </div>

          {deviationLoading || !summary ? (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-slate-400 font-medium">Fetching {selectedTicker} Data...</p>
              </div>
            </div>
          ) : (
            <>
              <StatsCard summary={summary} ticker={selectedTicker} />

              <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 mt-6">
                <div className="lg:col-span-3">
                  <DeviationChart data={data} ticker={selectedTicker} signals={backtestSignals} />
                </div>

                <div className="space-y-6">
                  {/* Risk Guide */}
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

                  {/* Historical Signals */}
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
              <div className="mt-8">
                <BacktestPanel data={data} ticker={selectedTicker} onSignalsChange={setBacktestSignals} />
              </div>

              {/* Info Section with Export */}
              <div className="mt-12 bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-sm">
                <div className="flex flex-col md:flex-row items-center gap-8">
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold text-white mb-4">
                      When the index exceeds <span className="text-emerald-400">{RISK_LEVELS.HIGH}</span>, it signals danger
                    </h3>
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
              </div>
            </>
          )}
        </div>
      </section>

      {/* ========== METHODOLOGY ========== */}
      <section
        id="methodology"
        ref={setSectionRef('methodology')}
        className="py-12 px-4 sm:px-6 lg:px-8"
      >
        <div className="max-w-7xl mx-auto">
          <h2 className="text-2xl font-bold text-white mb-6">Methodology</h2>

          {/* Regime Guide Cards */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl mb-8">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
              </svg>
              Regime Guide
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {BUBBLE_REGIME_CONFIG.map(regime => (
                <div key={regime.key} className={`${regime.bgClass} border rounded-lg p-3`}>
                  <p className={`${regime.textClass} font-bold text-sm`}>
                    {regime.label} ({regime.key === 'LOW' ? '0' : BUBBLE_REGIME_CONFIG[BUBBLE_REGIME_CONFIG.indexOf(regime) - 1]?.threshold ?? 0}-{regime.threshold})
                  </p>
                  <p className="text-slate-400 text-xs mt-1">{regime.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Methodology Explanation */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl mb-8">
            <h3 className="text-lg font-bold text-white mb-4">How the Bubble Index Works</h3>
            <p className="text-slate-400 leading-relaxed mb-4">
              The Market Bubble Index is a composite score (0-100) that aggregates <span className="text-white font-semibold">7 independent market indicators</span> across
              three categories&mdash;sentiment, liquidity, and valuation&mdash;to measure the degree of speculative excess.
              Each indicator is converted to a percentile rank within a rolling lookback window (50-252 trading days depending on the indicator),
              then combined via weighted average into the composite score. If any indicator is unavailable, its weight is automatically redistributed among the remaining indicators.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {INDICATOR_META.map(meta => (
                <div key={meta.key} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: meta.color }} />
                    <span className="text-sm font-semibold text-white">{meta.label}</span>
                    <span className="text-xs text-slate-500 ml-auto capitalize">{meta.category}</span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">{meta.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Data Sources & Weights */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl mb-8">
            <h3 className="text-lg font-bold text-white mb-4">Data Sources & Indicator Weights</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-semibold text-slate-300 mb-3 uppercase tracking-wide">Data Sources</h4>
                <ul className="space-y-2 text-sm text-slate-400">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400 mt-0.5">&#x2022;</span>
                    <span><span className="text-white font-medium">Yahoo Finance</span> (yfinance) &mdash; QQQ, SPY, VIX, SKEW, 11 sector ETFs, HYG, IEF, S&amp;P 500 price data. 10-year history.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400 mt-0.5">&#x2022;</span>
                    <span><span className="text-white font-medium">FRED</span> (Federal Reserve) &mdash; 10Y-2Y Treasury yield spread (T10Y2Y) from 2015.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400 mt-0.5">&#x2022;</span>
                    <span><span className="text-white font-medium">Update Schedule</span> &mdash; Automated via GitHub Actions, weekdays at 9:30 PM UTC (after US market close).</span>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-slate-300 mb-3 uppercase tracking-wide">Indicator Weights</h4>
                <div className="space-y-2">
                  {[
                    { label: 'QQQ Deviation', weight: 17, cat: 'Sentiment' },
                    { label: 'CAPE Valuation', weight: 15, cat: 'Valuation' },
                    { label: 'VIX Level', weight: 15, cat: 'Sentiment' },
                    { label: 'Tail Risk (SKEW)', weight: 14, cat: 'Sentiment' },
                    { label: 'Sector Breadth', weight: 13, cat: 'Liquidity' },
                    { label: 'Credit Spread', weight: 13, cat: 'Liquidity' },
                    { label: 'Yield Curve', weight: 13, cat: 'Liquidity' },
                  ].map(item => (
                    <div key={item.label} className="flex items-center gap-3 text-sm">
                      <span className="text-slate-400 w-36 truncate">{item.label}</span>
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-blue-500/60 rounded-full" style={{ width: `${item.weight * 5}%` }} />
                      </div>
                      <span className="text-white font-semibold w-10 text-right">{item.weight}%</span>
                      <span className="text-xs text-slate-500 w-20">{item.cat}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Normalization & Backtest Methodology */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
            <h3 className="text-lg font-bold text-white mb-4">Normalization & Backtest</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 text-sm text-slate-400">
              <div>
                <h4 className="text-sm font-semibold text-slate-300 mb-2 uppercase tracking-wide">Percentile Ranking</h4>
                <p className="leading-relaxed">
                  Each raw indicator value is ranked against its own trailing history using an empirical percentile function:
                  for each value, we count how many preceding values in the lookback window are lower, divide by window size,
                  and scale to 0-100. This makes indicators comparable across different units and magnitudes. The lookback
                  window is 200 days for QQQ deviation, 50 days for sector breadth, and 252 days (one trading year) for all other indicators.
                </p>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-slate-300 mb-2 uppercase tracking-wide">Deviation Tracker Backtest</h4>
                <p className="leading-relaxed">
                  The Deviation Tracker section includes an interactive backtest engine. It uses a simple threshold strategy:
                  buy when the deviation index drops below the buy threshold (default 20), sell when it rises above the
                  sell threshold (default 80). The engine tracks cash, shares, and peak portfolio value to compute strategy
                  return, annualized CAGR, and maximum drawdown. Results are compared against buy-and-hold over the same period.
                  Historical signals use a backward-only 20-day window to avoid look-ahead bias.
                  Note: the backtest assumes full position sizing (all-in / all-out) with no transaction costs, slippage, or taxes.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ========== FOOTER ========== */}
      <footer className="border-t border-slate-800 py-10 text-center text-slate-500 text-xs">
        <p>&copy; 2025-2026 Market Bubble Index Dashboard. Daily market data via Yahoo Finance &amp; FRED, updated by GitHub Actions.</p>
        <p className="mt-2">Not financial advice. 7 indicators across sentiment, liquidity, and valuation dimensions. 10-year rolling data window.</p>
      </footer>
    </div>
  );
};

export default App;
