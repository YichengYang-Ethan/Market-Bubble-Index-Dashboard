import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import { BubbleHistoryPoint, IndicatorMeta } from '../types';
import { INDICATOR_META } from '../constants';

interface IndicatorDetailChartProps {
  indicatorKey: string;
  history: BubbleHistoryPoint[];
  color: string;
}

const IndicatorDetailChart: React.FC<IndicatorDetailChartProps> = ({ indicatorKey, history, color }) => {
  const meta = INDICATOR_META.find((m: IndicatorMeta) => m.key === indicatorKey);
  const label = meta?.label ?? indicatorKey;

  const chartData = useMemo(() => {
    return history
      .map((point) => {
        const score = point.indicators?.[indicatorKey];
        return score != null ? { date: point.date, score } : null;
      })
      .filter((d): d is { date: string; score: number } => d !== null);
  }, [history, indicatorKey]);

  const gradientId = `gradient-${indicatorKey}`;

  return (
    <div>
      <h4 className="text-sm font-bold text-slate-300 mb-3">{label}</h4>
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.3} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            tick={{ fill: '#475569', fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            minTickGap={40}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: '#475569', fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            width={32}
          />
          <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="4 4" strokeOpacity={0.4} />
          <ReferenceLine y={50} stroke="#eab308" strokeDasharray="4 4" strokeOpacity={0.4} />
          <ReferenceLine y={70} stroke="#f97316" strokeDasharray="4 4" strokeOpacity={0.4} />
          <ReferenceLine y={85} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.4} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f172a',
              border: '1px solid #334155',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            labelStyle={{ color: '#94a3b8', fontWeight: 600 }}
            itemStyle={{ color }}
            formatter={(value?: number) => [value != null ? value.toFixed(1) : '--', 'Score']}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke={color}
            strokeWidth={2}
            fill={`url(#${gradientId})`}
            dot={false}
            activeDot={{ r: 4, fill: color, stroke: '#0f172a', strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default IndicatorDetailChart;
