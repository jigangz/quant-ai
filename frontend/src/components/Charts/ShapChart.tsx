import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { ShapFeature } from '../../api';

interface Props {
  data: ShapFeature[];
}

export default function ShapChart({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600">
        <div className="text-center">
          <div className="text-4xl mb-2">🔬</div>
          <div className="text-sm">No SHAP data available</div>
        </div>
      </div>
    );
  }

  // Sort by absolute shap_value descending
  const sorted = [...data].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)).slice(0, 12);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={sorted} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fill: '#666', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          stroke="#2a2d3a"
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: '#999', fontSize: 11, fontFamily: 'Inter' }}
          stroke="#2a2d3a"
          width={90}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1a1d27',
            border: '1px solid #2a2d3a',
            borderRadius: '4px',
            fontFamily: 'JetBrains Mono',
            fontSize: '12px',
          }}
          labelStyle={{ color: '#ccc' }}
          formatter={(value: number, _name: string, props: any) => [
            `SHAP: ${value.toFixed(4)} (val: ${props.payload.value.toFixed(3)})`,
            'Impact',
          ]}
        />
        <Bar dataKey="shap_value" radius={[0, 2, 2, 0]}>
          {sorted.map((entry, index) => (
            <Cell
              key={index}
              fill={entry.shap_value >= 0 ? '#10b981' : '#ef4444'}
              opacity={0.85}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
