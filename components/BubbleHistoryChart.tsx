import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { BubbleHistoryPoint } from '../types';
import { INDICATOR_META } from '../constants';

interface BubbleHistoryChartProps {
  history: BubbleHistoryPoint[];
}

const BubbleHistoryChart: React.FC<BubbleHistoryChartProps> = ({ history }) => {
  const [enabledIndicators, setEnabledIndicators] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    INDICATOR_META.forEach((ind) => {
      initial[ind.key] = false;
    });
    return initial;
  });

  const toggleIndicator = (key: string) => {
    setEnabledIndicators((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Transform history data to flatten indicator values to top-level keys for Recharts
  const chartData = useMemo(() => {
    return history.map((point) => {
      const flat: Record<string, unknown> = { ...point };
      INDICATOR_META.forEach((ind) => {
        flat[ind.key] = point.indicators?.[ind.key] ?? null;
      });
      return flat;
    });
  }, [history]);

  // Build tooltip labels including indicator names
  const tooltipLabels = useMemo(() => {
    const labels: Record<string, string> = {
      composite_score: 'Composite',
      sentiment_score: 'Sentiment',
      liquidity_score: 'Liquidity',
    };
    INDICATOR_META.forEach((ind) => {
      labels[ind.key] = ind.label;
    });
    return labels;
  }, []);

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl">
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
        </svg>
        Bubble History
      </h3>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#334155' }}
              axisLine={{ stroke: '#334155' }}
              tickFormatter={(val: string) => {
                const d = new Date(val);
                return `${d.getMonth() + 1}/${d.getDate()}`;
              }}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#334155' }}
              axisLine={{ stroke: '#334155' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0f172a',
                border: '1px solid #334155',
                borderRadius: '0.75rem',
                color: '#e2e8f0',
                fontSize: '0.8rem',
              }}
              labelFormatter={(label) => `Date: ${label}`}
              formatter={(value?: number, name?: string) => {
                return [value != null ? Number(value).toFixed(1) : '--', tooltipLabels[name ?? ''] || name || ''];
              }}
            />
            {/* Danger line at 85 */}
            <ReferenceLine
              y={85}
              stroke="#ef4444"
              strokeDasharray="6 4"
              strokeWidth={2}
              label={{ value: 'Danger (85)', fill: '#ef4444', fontSize: 11, position: 'insideTopRight' }}
            />
            {/* Warning line at 70 */}
            <ReferenceLine
              y={70}
              stroke="#f97316"
              strokeDasharray="4 4"
              strokeWidth={1}
            />
            <Line
              type="monotone"
              dataKey="composite_score"
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4, fill: '#3b82f6' }}
            />
            <Line
              type="monotone"
              dataKey="sentiment_score"
              stroke="#a78bfa"
              strokeWidth={1.5}
              strokeDasharray="4 2"
              dot={false}
              activeDot={{ r: 3, fill: '#a78bfa' }}
            />
            <Line
              type="monotone"
              dataKey="liquidity_score"
              stroke="#34d399"
              strokeWidth={1.5}
              strokeDasharray="4 2"
              dot={false}
              activeDot={{ r: 3, fill: '#34d399' }}
            />
            {/* Per-indicator lines (toggled) */}
            {INDICATOR_META.map((ind) =>
              enabledIndicators[ind.key] ? (
                <Line
                  key={ind.key}
                  type="monotone"
                  dataKey={ind.key}
                  stroke={ind.color}
                  strokeWidth={1.5}
                  strokeDasharray="2 2"
                  dot={false}
                  activeDot={{ r: 3, fill: ind.color }}
                  connectNulls
                />
              ) : null
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend - core lines */}
      <div className="flex items-center justify-center gap-6 mt-4 text-xs text-slate-400">
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-0.5 bg-blue-500 rounded" />
          <span>Composite</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-0.5 bg-violet-400 rounded" style={{ borderTop: '2px dashed #a78bfa' }} />
          <span>Sentiment</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-0.5 bg-emerald-400 rounded" style={{ borderTop: '2px dashed #34d399' }} />
          <span>Liquidity</span>
        </div>
      </div>

      {/* Indicator toggles */}
      <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2 mt-3 text-xs text-slate-400">
        {INDICATOR_META.map((ind) => (
          <label key={ind.key} className="flex items-center gap-1.5 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={enabledIndicators[ind.key]}
              onChange={() => toggleIndicator(ind.key)}
              className="sr-only"
            />
            <span
              className={`w-3.5 h-3.5 rounded border flex items-center justify-center transition-colors ${
                enabledIndicators[ind.key]
                  ? 'border-transparent'
                  : 'border-slate-600 bg-slate-800'
              }`}
              style={enabledIndicators[ind.key] ? { backgroundColor: ind.color } : undefined}
            >
              {enabledIndicators[ind.key] && (
                <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={3}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              )}
            </span>
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: ind.color }}
            />
            <span>{ind.label}</span>
          </label>
        ))}
      </div>
    </div>
  );
};

export default BubbleHistoryChart;
