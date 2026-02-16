
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
  ReferenceArea
} from 'recharts';
import { DataPoint } from '../types';

interface Props {
  data: DataPoint[];
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

const DeviationChart: React.FC<Props> = ({ data }) => {
  const sampledData = data.filter((_, idx) => idx % 5 === 0);

  return (
    <div className="w-full h-[500px] bg-slate-900 p-6 rounded-2xl shadow-sm border border-slate-800">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">QQQ 200-Day Deviation Index</h2>
          <p className="text-sm text-slate-500">Historical trend analysis and risk signaling</p>
        </div>
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
            <span className="text-xs font-medium text-slate-400">Threshold (80)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-xs font-medium text-slate-400">Overheated Zone</span>
          </div>
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
          <ReferenceArea y1={0} y2={40} fill="#22c55e" fillOpacity={0.05} />
          <ReferenceArea y1={40} y2={65} fill="#eab308" fillOpacity={0.05} />
          <ReferenceArea y1={65} y2={80} fill="#f97316" fillOpacity={0.05} />
          <ReferenceArea y1={80} y2={100} fill="#ef4444" fillOpacity={0.1} />

          {/* Signal Lines */}
          <ReferenceLine y={80} stroke="#22c55e" strokeWidth={2} label={{ position: 'right', value: '80', fill: '#22c55e', fontSize: 12, fontWeight: 'bold' }} />
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
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default DeviationChart;
