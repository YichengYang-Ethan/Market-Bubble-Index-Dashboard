
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
  Area,
  AreaChart,
  ReferenceArea
} from 'recharts';
import { DataPoint } from '../types';

interface Props {
  data: DataPoint[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-4 border border-slate-200 shadow-xl rounded-lg">
        <p className="text-slate-500 text-sm font-medium">{label}</p>
        <p className="text-blue-600 font-bold text-lg">Index: {payload[0].value.toFixed(2)}</p>
        <p className="text-slate-400 text-xs">Price: ${payload[0].payload.price.toFixed(2)}</p>
      </div>
    );
  }
  return null;
};

const DeviationChart: React.FC<Props> = ({ data }) => {
  // Sampling for performance
  const sampledData = data.filter((_, idx) => idx % 5 === 0);

  return (
    <div className="w-full h-[500px] bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-slate-800">MeiTou QQQ 200 Days Deviation Index</h2>
          <p className="text-sm text-slate-500">Historical trend analysis and risk signaling</p>
        </div>
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-xs font-medium text-slate-600">Threshold (80)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-xs font-medium text-slate-600">Overheated Zone</span>
          </div>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={sampledData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis 
            dataKey="date" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#94a3b8', fontSize: 10 }} 
            minTickGap={100}
          />
          <YAxis 
            domain={[0, 100]} 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            width={40}
          />
          <Tooltip content={<CustomTooltip />} />
          
          {/* Risk Zones */}
          <ReferenceArea y1={80} y2={100} fill="#fee2e2" fillOpacity={0.3} />
          
          {/* Signal Lines */}
          <ReferenceLine y={80} stroke="#22c55e" strokeWidth={2} label={{ position: 'right', value: '80', fill: '#22c55e', fontSize: 12, fontWeight: 'bold' }} />
          <ReferenceLine y={95} stroke="#ef4444" strokeDasharray="5 5" strokeWidth={1} label={{ position: 'right', value: 'Critical', fill: '#ef4444', fontSize: 10 }} />

          <Line 
            type="monotone" 
            dataKey="index" 
            stroke="#1e293b" 
            strokeWidth={2} 
            dot={false}
            activeDot={{ r: 6, strokeWidth: 0, fill: '#3b82f6' }}
            animationDuration={1500}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default DeviationChart;
