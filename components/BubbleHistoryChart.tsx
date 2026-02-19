import React, { useState, useMemo, useCallback } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
} from 'recharts';
import { BubbleHistoryPoint, DataPoint, GSADFResults, MarkovRegimes } from '../types';
import { DrawdownPoint } from '../services/bubbleService';
import { INDICATOR_META } from '../constants';

type TimeRange = '1Y' | '3Y' | '5Y' | 'ALL';

interface BubbleHistoryChartProps {
  history: BubbleHistoryPoint[];
  priceData?: { qqq: DataPoint[]; spy: DataPoint[] };
  gsadfResults?: GSADFResults | null;
  markovRegimes?: MarkovRegimes | null;
  qqqDrawdown?: DrawdownPoint[];
  mode?: 'composite' | 'risk';
}

const SUB_SCORES = [
  { key: 'sentiment_score', label: 'Sentiment', color: '#a78bfa' },
  { key: 'liquidity_score', label: 'Liquidity', color: '#34d399' },
  { key: 'valuation_score', label: 'Valuation', color: '#f43f5e' },
] as const;

const PRICE_OVERLAYS = [
  { key: 'qqq_price', label: 'QQQ Price', color: '#facc15', dataKey: 'qqq' },
  { key: 'spy_price', label: 'SPY Price', color: '#fb923c', dataKey: 'spy' },
] as const;

const REGIME_BANDS = [
  { y1: 0, y2: 30, fill: '#22c55e', opacity: 0.06 },
  { y1: 30, y2: 50, fill: '#eab308', opacity: 0.05 },
  { y1: 50, y2: 70, fill: '#f97316', opacity: 0.06 },
  { y1: 70, y2: 85, fill: '#ef4444', opacity: 0.06 },
  { y1: 85, y2: 100, fill: '#dc2626', opacity: 0.08 },
];

const MAX_POINTS = 500;

const BubbleHistoryChart: React.FC<BubbleHistoryChartProps> = ({ history, priceData, gsadfResults, markovRegimes, qqqDrawdown, mode = 'composite' }) => {
  const isRiskMode = mode === 'risk';
  const [timeRange, setTimeRange] = useState<TimeRange>('ALL');
  const [showSubScores, setShowSubScores] = useState<Record<string, boolean>>({
    sentiment_score: false,
    liquidity_score: false,
    valuation_score: false,
  });
  const [enabledIndicators, setEnabledIndicators] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    INDICATOR_META.forEach((ind) => { initial[ind.key] = false; });
    return initial;
  });
  const [priceOverlays, setPriceOverlays] = useState<Record<string, boolean>>({
    qqq_price: false,
    spy_price: false,
  });
  const [showComposite, setShowComposite] = useState(false);
  const [showGSADF, setShowGSADF] = useState(false);
  const [showVelocity, setShowVelocity] = useState(false);
  const [showDrawdown, setShowDrawdown] = useState(isRiskMode);

  const anyPriceActive = priceOverlays.qqq_price || priceOverlays.spy_price;

  const toggleSubScore = useCallback((key: string) => {
    setShowSubScores((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const toggleIndicator = useCallback((key: string) => {
    setEnabledIndicators((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const togglePrice = useCallback((key: string) => {
    setPriceOverlays((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  // Build price lookup maps
  const priceMaps = useMemo(() => {
    const qqqMap = new Map<string, number>();
    const spyMap = new Map<string, number>();
    if (priceData?.qqq) {
      for (const p of priceData.qqq) qqqMap.set(p.date, p.price);
    }
    if (priceData?.spy) {
      for (const p of priceData.spy) spyMap.set(p.date, p.price);
    }
    const ddMap = new Map<string, number>();
    if (qqqDrawdown) {
      for (const d of qqqDrawdown) ddMap.set(d.date, d.drawdown);
    }
    return { qqq: qqqMap, spy: spyMap, drawdown: ddMap };
  }, [priceData, qqqDrawdown]);

  const chartData = useMemo(() => {
    // 1. Filter by time range
    let filtered = history;
    if (timeRange !== 'ALL') {
      const years = timeRange === '1Y' ? 1 : timeRange === '3Y' ? 3 : 5;
      const cutoff = new Date();
      cutoff.setFullYear(cutoff.getFullYear() - years);
      const cutoffStr = cutoff.toISOString().split('T')[0];
      filtered = history.filter((d) => d.date >= cutoffStr);
    }

    // 2. Flatten indicators + merge price data + velocity
    let data = filtered.map((point) => {
      const flat: Record<string, unknown> = { ...point };
      INDICATOR_META.forEach((ind) => {
        flat[ind.key] = point.indicators?.[ind.key] ?? null;
      });
      flat.qqq_price = priceMaps.qqq.get(point.date) ?? null;
      flat.spy_price = priceMaps.spy.get(point.date) ?? null;
      flat.score_velocity = point.score_velocity ?? null;
      flat.qqq_drawdown = priceMaps.drawdown.get(point.date) ?? null;
      return flat;
    });

    // 3. Sample if too many points
    if (data.length > MAX_POINTS) {
      const step = Math.ceil(data.length / MAX_POINTS);
      data = data.filter((_, i) => i % step === 0 || i === data.length - 1);
    }

    return data;
  }, [history, timeRange, priceMaps]);

  // Compute price domain for right Y-axis
  const priceDomain = useMemo(() => {
    if (!anyPriceActive) return [0, 100];
    let min = Infinity;
    let max = -Infinity;
    for (const d of chartData) {
      if (priceOverlays.qqq_price && d.qqq_price != null) {
        const v = d.qqq_price as number;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      if (priceOverlays.spy_price && d.spy_price != null) {
        const v = d.spy_price as number;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min)) return [0, 100];
    const pad = (max - min) * 0.05;
    return [Math.floor(min - pad), Math.ceil(max + pad)];
  }, [chartData, anyPriceActive, priceOverlays]);

  // Custom tooltip
  const renderTooltip = useCallback(({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-xl text-xs">
        <p className="text-slate-400 mb-1.5 font-medium">{label}</p>
        {payload.map((entry: any) => {
          const isPrice = entry.dataKey === 'qqq_price' || entry.dataKey === 'spy_price';
          const isDrawdown = entry.dataKey === 'qqq_drawdown';
          const val = entry.value != null
            ? isPrice ? `$${Number(entry.value).toFixed(2)}`
            : isDrawdown ? `${Number(entry.value).toFixed(1)}%`
            : Number(entry.value).toFixed(1)
            : '--';
          return (
            <div key={entry.dataKey} className="flex items-center gap-2 py-0.5">
              <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: entry.color }} />
              <span className="text-slate-300">{entry.name}:</span>
              <span className="text-white font-semibold ml-auto">{val}</span>
            </div>
          );
        })}
      </div>
    );
  }, []);

  // Build tooltip name map
  const nameMap: Record<string, string> = useMemo(() => {
    const m: Record<string, string> = {
      drawdown_risk_score: isRiskMode ? 'Risk Score (predictor)' : 'Risk Score',
      composite_score: isRiskMode ? 'Bubble Temp' : 'Composite',
      sentiment_score: 'Sentiment',
      liquidity_score: 'Liquidity',
      valuation_score: 'Valuation',
      qqq_price: 'QQQ',
      spy_price: 'SPY',
      score_velocity: 'Velocity',
      qqq_drawdown: isRiskMode ? 'QQQ Drawdown (actual)' : 'QQQ Drawdown',
    };
    INDICATOR_META.forEach((ind) => { m[ind.key] = ind.label; });
    return m;
  }, [isRiskMode]);

  const pillClass = (active: boolean, color?: string) =>
    active
      ? 'text-white font-semibold shadow-sm'
      : 'bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-750 hover:text-slate-300';

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <svg className={`w-5 h-5 ${isRiskMode ? 'text-red-400' : 'text-blue-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
        </svg>
        {isRiskMode ? 'Risk Score History' : 'Bubble History'}
      </h3>

      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 5, right: anyPriceActive ? 10 : -10, left: -10, bottom: 5 }}>
            <defs>
              <linearGradient id="compositeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isRiskMode ? '#ef4444' : '#3b82f6'} stopOpacity={0.3} />
                <stop offset="100%" stopColor={isRiskMode ? '#ef4444' : '#3b82f6'} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="drawdownGradient" x1="0" y1="1" x2="0" y2="0">
                <stop offset="0%" stopColor="#f97316" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#f97316" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />

            {/* Regime background bands */}
            {REGIME_BANDS.map((band) => (
              <ReferenceArea
                key={band.y1}
                y1={band.y1}
                y2={band.y2}
                yAxisId="score"
                fill={band.fill}
                fillOpacity={band.opacity}
              />
            ))}

            <XAxis
              dataKey="date"
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#334155' }}
              axisLine={{ stroke: '#334155' }}
              tickFormatter={(val: string) => {
                const d = new Date(val);
                return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
              }}
              interval="preserveStartEnd"
              minTickGap={60}
            />

            {/* Left Y-axis: scores 0-100 */}
            <YAxis
              yAxisId="score"
              domain={[0, 100]}
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#334155' }}
              axisLine={{ stroke: '#334155' }}
            />

            {/* Right Y-axis: drawdown (only when active and no price) */}
            {showDrawdown && !anyPriceActive && (
              <YAxis
                yAxisId="drawdown"
                orientation="right"
                domain={[-60, 0]}
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={{ stroke: '#334155' }}
                axisLine={{ stroke: '#334155' }}
                tickFormatter={(v: number) => `${v}%`}
              />
            )}

            {/* Right Y-axis: price (only when active) */}
            {anyPriceActive && (
              <YAxis
                yAxisId="price"
                orientation="right"
                domain={priceDomain}
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={{ stroke: '#334155' }}
                axisLine={{ stroke: '#334155' }}
                tickFormatter={(v: number) => `$${v}`}
              />
            )}

            <Tooltip content={renderTooltip} />

            {/* Danger / warning reference lines */}
            <ReferenceLine
              y={85}
              yAxisId="score"
              stroke="#ef4444"
              strokeDasharray="6 4"
              strokeWidth={1.5}
              label={{ value: '85', fill: '#ef4444', fontSize: 10, position: 'insideTopRight' }}
            />
            <ReferenceLine
              y={70}
              yAxisId="score"
              stroke="#f97316"
              strokeDasharray="4 4"
              strokeWidth={1}
            />

            {/* Hero: primary area fill */}
            <Area
              type="monotone"
              dataKey={isRiskMode ? 'drawdown_risk_score' : 'composite_score'}
              yAxisId="score"
              fill="url(#compositeGradient)"
              stroke="none"
              name={nameMap[isRiskMode ? 'drawdown_risk_score' : 'composite_score']}
              dot={false}
              activeDot={false}
              isAnimationActive={false}
            />

            {/* Hero: primary line on top */}
            <Line
              type="monotone"
              dataKey={isRiskMode ? 'drawdown_risk_score' : 'composite_score'}
              yAxisId="score"
              stroke={isRiskMode ? '#ef4444' : '#3b82f6'}
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4, fill: isRiskMode ? '#ef4444' : '#3b82f6' }}
              name={nameMap[isRiskMode ? 'drawdown_risk_score' : 'composite_score']}
              isAnimationActive={false}
            />

            {/* In risk mode, composite is opt-in via toggle */}
            {isRiskMode && showComposite && (
              <Line
                type="monotone"
                dataKey="composite_score"
                yAxisId="score"
                stroke="#3b82f6"
                strokeWidth={1.2}
                strokeDasharray="6 3"
                dot={false}
                activeDot={{ r: 3, fill: '#3b82f6' }}
                name={nameMap.composite_score}
                isAnimationActive={false}
                opacity={0.6}
              />
            )}

            {/* Sub-score lines (dashed, opt-in) */}
            {SUB_SCORES.map((s) =>
              showSubScores[s.key] ? (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  yAxisId="score"
                  stroke={s.color}
                  strokeWidth={1.5}
                  strokeDasharray="6 3"
                  dot={false}
                  activeDot={{ r: 3, fill: s.color }}
                  name={nameMap[s.key]}
                  connectNulls
                  isAnimationActive={false}
                />
              ) : null
            )}

            {/* Individual indicator lines (dotted, opt-in) */}
            {INDICATOR_META.map((ind) =>
              enabledIndicators[ind.key] ? (
                <Line
                  key={ind.key}
                  type="monotone"
                  dataKey={ind.key}
                  yAxisId="score"
                  stroke={ind.color}
                  strokeWidth={1}
                  strokeDasharray="2 2"
                  dot={false}
                  activeDot={{ r: 2, fill: ind.color }}
                  name={nameMap[ind.key]}
                  connectNulls
                  isAnimationActive={false}
                />
              ) : null
            )}

            {/* Price overlay lines (right Y-axis) */}
            {PRICE_OVERLAYS.map((p) =>
              priceOverlays[p.key] ? (
                <Line
                  key={p.key}
                  type="monotone"
                  dataKey={p.key}
                  yAxisId="price"
                  stroke={p.color}
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{ r: 3, fill: p.color }}
                  name={nameMap[p.key]}
                  connectNulls
                  isAnimationActive={false}
                />
              ) : null
            )}

            {/* GSADF bubble period overlays */}
            {showGSADF && gsadfResults?.bubble_periods.map((period, i) => (
              <ReferenceArea
                key={`gsadf-${i}`}
                x1={period.start}
                x2={period.end}
                yAxisId="score"
                fill="#ef4444"
                fillOpacity={0.15}
              />
            ))}

            {/* Velocity line overlay */}
            {showVelocity && (
              <Line
                type="monotone"
                dataKey="score_velocity"
                yAxisId="score"
                stroke="#f472b6"
                strokeWidth={1}
                strokeDasharray="4 2"
                dot={false}
                activeDot={{ r: 2, fill: '#f472b6' }}
                name={nameMap.score_velocity}
                connectNulls
                isAnimationActive={false}
              />
            )}

            {/* QQQ Drawdown overlay — orange to distinguish from red risk score */}
            {showDrawdown && (
              <Area
                type="monotone"
                dataKey="qqq_drawdown"
                yAxisId={anyPriceActive ? 'price' : showDrawdown && !anyPriceActive ? 'drawdown' : 'score'}
                fill="url(#drawdownGradient)"
                stroke="#f97316"
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3, fill: '#f97316' }}
                name={nameMap.qqq_drawdown}
                connectNulls
                isAnimationActive={false}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend for risk mode */}
      {isRiskMode && (
        <div className="mt-4 flex items-center gap-6 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <div className="w-6 h-0.5 bg-red-500 rounded" />
            <span>Risk Score — <span className="text-slate-300">predictive indicator</span></span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-3 rounded-sm" style={{ background: 'linear-gradient(to top, rgba(249,115,22,0.35), rgba(249,115,22,0.05))' }} />
            <span>QQQ Drawdown — <span className="text-slate-300">actual outcome</span></span>
          </div>
          {showComposite && (
            <div className="flex items-center gap-2">
              <div className="w-6 h-0.5 rounded" style={{ background: '#3b82f6', opacity: 0.6 }} />
              <span>Bubble Temperature</span>
            </div>
          )}
        </div>
      )}

      {/* Toggle controls */}
      <div className="mt-4 space-y-3">
        {/* Time Range */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-500 font-semibold uppercase tracking-wide w-20">Range</span>
          <div className="flex gap-1.5">
            {(['1Y', '3Y', '5Y', 'ALL'] as TimeRange[]).map((r) => (
              <button
                key={r}
                onClick={() => setTimeRange(r)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  timeRange === r
                    ? 'bg-blue-600 text-white font-semibold shadow-sm'
                    : pillClass(false)
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        </div>

        {/* Sub-Scores */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-500 font-semibold uppercase tracking-wide w-20">Scores</span>
          <div className="flex gap-1.5">
            {isRiskMode && (
              <button
                onClick={() => setShowComposite((v) => !v)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  showComposite ? pillClass(true) : pillClass(false)
                }`}
                style={showComposite ? { backgroundColor: '#3b82f6' } : undefined}
              >
                Bubble Temp
              </button>
            )}
            {SUB_SCORES.map((s) => (
              <button
                key={s.key}
                onClick={() => toggleSubScore(s.key)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  showSubScores[s.key]
                    ? pillClass(true)
                    : pillClass(false)
                }`}
                style={showSubScores[s.key] ? { backgroundColor: s.color } : undefined}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {/* Overlays: Price + Indicators */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-500 font-semibold uppercase tracking-wide w-20">Overlays</span>
          <div className="flex gap-1.5 flex-wrap">
            {PRICE_OVERLAYS.map((p) => (
              <button
                key={p.key}
                onClick={() => togglePrice(p.key)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  priceOverlays[p.key]
                    ? pillClass(true)
                    : pillClass(false)
                }`}
                style={priceOverlays[p.key] ? { backgroundColor: p.color, color: '#000' } : undefined}
              >
                {p.label}
              </button>
            ))}
            {INDICATOR_META.map((ind) => (
              <button
                key={ind.key}
                onClick={() => toggleIndicator(ind.key)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  enabledIndicators[ind.key]
                    ? pillClass(true)
                    : pillClass(false)
                }`}
                style={enabledIndicators[ind.key] ? { backgroundColor: ind.color } : undefined}
              >
                {ind.label}
              </button>
            ))}
          </div>
        </div>

        {/* Analysis overlays */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-500 font-semibold uppercase tracking-wide w-20">Analysis</span>
          <div className="flex gap-1.5 flex-wrap">
            {gsadfResults && (
              <button
                onClick={() => setShowGSADF((v) => !v)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  showGSADF ? pillClass(true) : pillClass(false)
                }`}
                style={showGSADF ? { backgroundColor: '#ef4444' } : undefined}
              >
                GSADF Bubbles
              </button>
            )}
            <button
              onClick={() => setShowVelocity((v) => !v)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                showVelocity ? pillClass(true) : pillClass(false)
              }`}
              style={showVelocity ? { backgroundColor: '#f472b6' } : undefined}
            >
              Velocity
            </button>
            {qqqDrawdown && qqqDrawdown.length > 0 && (
              <button
                onClick={() => setShowDrawdown((v) => !v)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                  showDrawdown ? pillClass(true) : pillClass(false)
                }`}
                style={showDrawdown ? { backgroundColor: '#f97316' } : undefined}
              >
                QQQ Drawdown
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BubbleHistoryChart;
