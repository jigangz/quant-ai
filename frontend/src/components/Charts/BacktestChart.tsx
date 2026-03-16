import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { BacktestPoint } from '../../api';

interface Props {
  data: BacktestPoint[];
}

export default function BacktestChart({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600">
        <div className="text-center">
          <div className="text-4xl mb-2">📊</div>
          <div className="text-sm">Run a backtest to see results</div>
        </div>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
        <XAxis
          dataKey="date"
          tick={{ fill: '#666', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          stroke="#2a2d3a"
          tickFormatter={(v) => v.slice(5)}
        />
        <YAxis
          tick={{ fill: '#666', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          stroke="#2a2d3a"
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1a1d27',
            border: '1px solid #2a2d3a',
            borderRadius: '4px',
            fontFamily: 'JetBrains Mono',
            fontSize: '12px',
          }}
          labelStyle={{ color: '#999' }}
          formatter={(value: number, name: string) => [
            `${(value * 100).toFixed(2)}%`,
            name === 'strategy_return' ? 'Strategy' : 'Benchmark',
          ]}
        />
        <Legend
          wrapperStyle={{ fontSize: '12px', fontFamily: 'Inter' }}
          formatter={(value) => (value === 'strategy_return' ? 'Strategy' : 'Benchmark')}
        />
        <Line
          type="monotone"
          dataKey="strategy_return"
          stroke="#6366f1"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: '#6366f1' }}
        />
        <Line
          type="monotone"
          dataKey="benchmark_return"
          stroke="#4b5563"
          strokeWidth={1.5}
          dot={false}
          strokeDasharray="5 5"
          activeDot={{ r: 3, fill: '#4b5563' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
