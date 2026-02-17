
import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  ReferenceDot
} from 'recharts';
import { DataPoint, BacktestSignal } from '../types';
import { DEVIATION_CONFIG } from '../constants';

const { RISK_LEVELS } = DEVIATION_CONFIG;

interface Props {
  data: DataPoint[];
  ticker: string;
  signals?: BacktestSignal[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 p-4 border border-slate-700 shadow-xl rounded-lg backdrop-blur-sm">
        <p className="text-slate-400 text-sm font-medium">{label}</p>
        <p className="text-blue-400 font-bold text-lg">Index: {payload[0].value.toFixed(2)}</p>
        <p className="text-slate-500 text-xs">Price: ${payload[0].payload.price.toFixed(2)}</p>
      </div>
    );
  }
  return null;
};

const DeviationChart: React.FC<Props> = ({ data, ticker, signals }) => {
  const sampledData = data.filter((_, idx) => idx % 5 === 0);

  // Build a set of signal dates for O(1) lookup in sampled data
  const signalsByDate = new Map<string, BacktestSignal>();
  if (signals) {
    for (const s of signals) {
      signalsByDate.set(s.date, s);
    }
  }

  return (
    <div className="w-full h-[500px] bg-slate-900 p-6 rounded-2xl shadow-sm border border-slate-800">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">{ticker} 200-Day Deviation Index</h2>
          <p className="text-sm text-slate-500">Historical trend analysis and risk signaling</p>
        </div>
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
            <span className="text-xs font-medium text-slate-400">Threshold ({RISK_LEVELS.HIGH})</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-xs font-medium text-slate-400">Overheated Zone</span>
          </div>
          {signals && signals.length > 0 && (
            <>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-400 rounded-full border border-green-300"></div>
                <span className="text-xs font-medium text-slate-400">Buy</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-400 rounded-full border border-red-300"></div>
                <span className="text-xs font-medium text-slate-400">Sell</span>
              </div>
            </>
          )}
        </div>
      </div>

      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={sampledData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="lineGradient" x1="0" y1="1" x2="0" y2="0">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="60%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
          <XAxis
            dataKey="date"
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#64748b', fontSize: 10 }}
            minTickGap={100}
          />
          <YAxis
            domain={[0, 100]}
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#64748b', fontSize: 11 }}
            width={40}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Colored reference zones */}
          <ReferenceArea y1={0} y2={RISK_LEVELS.LOW} fill="#22c55e" fillOpacity={0.05} />
          <ReferenceArea y1={RISK_LEVELS.LOW} y2={RISK_LEVELS.MODERATE} fill="#eab308" fillOpacity={0.05} />
          <ReferenceArea y1={RISK_LEVELS.MODERATE} y2={RISK_LEVELS.HIGH} fill="#f97316" fillOpacity={0.05} />
          <ReferenceArea y1={RISK_LEVELS.HIGH} y2={100} fill="#ef4444" fillOpacity={0.1} />

          {/* Signal Lines */}
          <ReferenceLine y={RISK_LEVELS.HIGH} stroke="#22c55e" strokeWidth={2} label={{ position: 'right', value: String(RISK_LEVELS.HIGH), fill: '#22c55e', fontSize: 12, fontWeight: 'bold' }} />
          <ReferenceLine y={95} stroke="#ef4444" strokeDasharray="5 5" strokeWidth={1} label={{ position: 'right', value: 'Critical', fill: '#ef4444', fontSize: 10 }} />

          <Line
            type="monotone"
            dataKey="index"
            stroke="#60a5fa"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 8, strokeWidth: 2, stroke: '#3b82f6', fill: '#1e40af', filter: 'drop-shadow(0 0 6px #3b82f6)' }}
            animationDuration={1500}
          />

          {/* Backtest signal markers */}
          {signals && signals.map((s, i) => {
            // Only render if this date exists in the sampled data
            const exists = sampledData.some(d => d.date === s.date);
            if (!exists) return null;
            return (
              <ReferenceDot
                key={`signal-${i}`}
                x={s.date}
                y={s.index}
                r={5}
                fill={s.type === 'buy' ? '#4ade80' : '#f87171'}
                stroke={s.type === 'buy' ? '#22c55e' : '#ef4444'}
                strokeWidth={2}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default DeviationChart;
