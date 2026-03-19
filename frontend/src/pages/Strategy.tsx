import { useState, useEffect } from 'react';
import { Play, FlaskConical, TrendingUp, TrendingDown, Minus, Sparkles } from 'lucide-react';
import Card from '../components/UI/Card';
import Badge from '../components/UI/Badge';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import {
  fetchStrategies,
  generateSignals,
  type Strategy as StrategyType,
  type SignalResult,
  type Signal,
} from '../api/strategies';
import { runBacktest, type BacktestResult } from '../api';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
  CartesianGrid,
} from 'recharts';

interface Props {
  ticker: string;
}

interface ChartDataPoint {
  date: string;
  price: number;
  signal?: 'BUY' | 'SELL';
  confidence?: number;
}

function StrategyCard({
  strategy,
  selected,
  onClick,
}: {
  strategy: StrategyType;
  selected: boolean;
  onClick: () => void;
}) {
  const categoryColors: Record<string, string> = {
    momentum: 'bg-blue-500/20 text-blue-400',
    'mean-reversion': 'bg-purple-500/20 text-purple-400',
    'ml-based': 'bg-green-500/20 text-green-400',
    hybrid: 'bg-orange-500/20 text-orange-400',
  };

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-sm border transition-all ${
        selected
          ? 'border-accent bg-accent/10'
          : 'border-dark-border bg-dark-card hover:border-gray-600'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-gray-200">{strategy.name}</span>
        <span className={`text-xs px-2 py-0.5 rounded ${categoryColors[strategy.category] || 'bg-gray-500/20 text-gray-400'}`}>
          {strategy.category}
        </span>
      </div>
      <p className="text-xs text-gray-500 line-clamp-2">{strategy.description}</p>
    </button>
  );
}

function ParameterInput({
  param,
  value,
  onChange,
}: {
  param: StrategyType['parameters'][0];
  value: unknown;
  onChange: (name: string, val: unknown) => void;
}) {
  if (param.type === 'select' && param.options) {
    return (
      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500">{param.label}</label>
        <select
          value={String(value)}
          onChange={(e) => onChange(param.name, e.target.value)}
          className="h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent"
        >
          {param.options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (param.type === 'boolean') {
    return (
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(param.name, e.target.checked)}
          className="w-4 h-4 rounded bg-dark-bg border-dark-border text-accent focus:ring-accent"
        />
        <label className="text-sm text-gray-300">{param.label}</label>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-500">{param.label}</label>
      <input
        type="number"
        value={Number(value)}
        min={param.min}
        max={param.max}
        step={param.step || 1}
        onChange={(e) => onChange(param.name, parseFloat(e.target.value))}
        className="h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent font-mono"
      />
    </div>
  );
}

function MetricCard({ label, value, suffix = '', color }: { label: string; value: string; suffix?: string; color: string }) {
  return (
    <Card className="p-3">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-lg font-mono font-bold ${color}`}>
        {value}
        {suffix && <span className="text-sm ml-0.5">{suffix}</span>}
      </div>
    </Card>
  );
}

export default function Strategy({ ticker }: Props) {
  const [strategies, setStrategies] = useState<StrategyType[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyType | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [signals, setSignals] = useState<SignalResult | null>(null);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [loadingStrategies, setLoadingStrategies] = useState(true);
  const [loadingSignals, setLoadingSignals] = useState(false);
  const [loadingBacktest, setLoadingBacktest] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load strategies
  useEffect(() => {
    setLoadingStrategies(true);
    fetchStrategies()
      .then((data) => {
        setStrategies(data);
        if (data.length > 0) {
          setSelectedStrategy(data[0]);
          const defaultParams: Record<string, unknown> = {};
          data[0].parameters.forEach((p) => {
            defaultParams[p.name] = p.default;
          });
          setParams(defaultParams);
        }
      })
      .catch(() => setError('Failed to load strategies'))
      .finally(() => setLoadingStrategies(false));
  }, []);

  // Reset params when strategy changes
  useEffect(() => {
    if (selectedStrategy) {
      const defaultParams: Record<string, unknown> = {};
      selectedStrategy.parameters.forEach((p) => {
        defaultParams[p.name] = p.default;
      });
      setParams(defaultParams);
      setSignals(null);
      setBacktestResult(null);
    }
  }, [selectedStrategy]);

  const handleParamChange = (name: string, value: unknown) => {
    setParams((prev) => ({ ...prev, [name]: value }));
  };

  const handleGenerateSignals = async () => {
    if (!selectedStrategy) return;
    setLoadingSignals(true);
    setError(null);
    try {
      const result = await generateSignals(ticker, selectedStrategy.id, params);
      setSignals(result);
    } catch {
      setError('Failed to generate signals');
    } finally {
      setLoadingSignals(false);
    }
  };

  const handleRunBacktest = async () => {
    setLoadingBacktest(true);
    setError(null);
    try {
      const endDate = new Date().toISOString().slice(0, 10);
      const startDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
      const modelType = String(params.model_type || 'xgboost');
      const result = await runBacktest({
        ticker,
        start_date: startDate,
        end_date: endDate,
        model_type: modelType,
      });
      setBacktestResult(result);
    } catch {
      setError('Backtest failed');
    } finally {
      setLoadingBacktest(false);
    }
  };

  // Prepare chart data with signals
  const chartData: ChartDataPoint[] = signals?.signals.map((s: Signal) => ({
    date: s.date,
    price: s.price,
    signal: s.signal !== 'HOLD' ? s.signal : undefined,
    confidence: s.confidence,
  })) || [];

  if (loadingStrategies) {
    return <LoadingSpinner text="Loading strategies..." />;
  }

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      {/* Strategy selector */}
      <Card title="Select Strategy" className="p-4">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {strategies.map((s) => (
            <StrategyCard
              key={s.id}
              strategy={s}
              selected={selectedStrategy?.id === s.id}
              onClick={() => setSelectedStrategy(s)}
            />
          ))}
        </div>
      </Card>

      {/* Parameter config + actions */}
      {selectedStrategy && (
        <Card title={`Configure: ${selectedStrategy.name}`} className="p-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">Ticker</label>
              <span className="text-sm font-mono text-accent font-bold h-8 flex items-center">{ticker}</span>
            </div>
            {selectedStrategy.parameters.map((p) => (
              <ParameterInput
                key={p.name}
                param={p}
                value={params[p.name]}
                onChange={handleParamChange}
              />
            ))}
            <button
              onClick={handleGenerateSignals}
              disabled={loadingSignals}
              className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent/80 text-white text-sm font-medium rounded-sm transition-colors disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              {loadingSignals ? 'Generating...' : 'Generate Signals'}
            </button>
            <button
              onClick={handleRunBacktest}
              disabled={loadingBacktest}
              className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white text-sm font-medium rounded-sm transition-colors disabled:opacity-50"
            >
              <FlaskConical className="w-4 h-4" />
              {loadingBacktest ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
        </Card>
      )}

      {error && (
        <div className="p-3 bg-bear/10 border border-bear/30 rounded-sm text-sm text-bear">{error}</div>
      )}

      {/* Signals chart */}
      {signals && chartData.length > 0 && (
        <Card title="Signal Chart" className="p-4">
          <div className="mb-3 flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-bull" />
              <span className="text-gray-400">Buy: {signals.summary.buy_signals}</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-bear" />
              <span className="text-gray-400">Sell: {signals.summary.sell_signals}</span>
            </div>
            <div className="flex items-center gap-2">
              <Minus className="w-4 h-4 text-gray-500" />
              <span className="text-gray-400">
                Avg Confidence: {(signals.summary.avg_confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#9CA3AF', fontSize: 10 }}
                  tickLine={{ stroke: '#4B5563' }}
                />
                <YAxis
                  domain={['auto', 'auto']}
                  tick={{ fill: '#9CA3AF', fontSize: 10 }}
                  tickLine={{ stroke: '#4B5563' }}
                  tickFormatter={(v) => `$${v}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '4px',
                  }}
                  labelStyle={{ color: '#9CA3AF' }}
                  formatter={(value: number, name: string) => {
                    if (name === 'price') return [`$${value.toFixed(2)}`, 'Price'];
                    return [value, name];
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="price"
                  stroke="#60A5FA"
                  dot={false}
                  strokeWidth={2}
                />
                {chartData
                  .filter((d) => d.signal === 'BUY')
                  .map((d, i) => (
                    <ReferenceDot
                      key={`buy-${i}`}
                      x={d.date}
                      y={d.price}
                      r={6}
                      fill="#22C55E"
                      stroke="#fff"
                      strokeWidth={1}
                    />
                  ))}
                {chartData
                  .filter((d) => d.signal === 'SELL')
                  .map((d, i) => (
                    <ReferenceDot
                      key={`sell-${i}`}
                      x={d.date}
                      y={d.price}
                      r={6}
                      fill="#EF4444"
                      stroke="#fff"
                      strokeWidth={1}
                    />
                  ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Backtest results */}
      {backtestResult && (
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12 grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Sharpe Ratio"
              value={backtestResult.metrics.sharpe_ratio.toFixed(2)}
              color={
                backtestResult.metrics.sharpe_ratio > 1
                  ? 'text-bull'
                  : backtestResult.metrics.sharpe_ratio > 0
                  ? 'text-accent'
                  : 'text-bear'
              }
            />
            <MetricCard
              label="Max Drawdown"
              value={`${(backtestResult.metrics.max_drawdown * 100).toFixed(1)}`}
              suffix="%"
              color="text-bear"
            />
            <MetricCard
              label="Win Rate"
              value={`${(backtestResult.metrics.win_rate * 100).toFixed(1)}`}
              suffix="%"
              color={backtestResult.metrics.win_rate > 0.5 ? 'text-bull' : 'text-bear'}
            />
            <MetricCard
              label="Total Return"
              value={`${backtestResult.metrics.total_return >= 0 ? '+' : ''}${(
                backtestResult.metrics.total_return * 100
              ).toFixed(1)}`}
              suffix="%"
              color={backtestResult.metrics.total_return >= 0 ? 'text-bull' : 'text-bear'}
            />
          </div>
          <div className="col-span-12">
            <Card title={`Trade Log (${backtestResult.trades.length} trades)`}>
              <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-dark-card">
                    <tr className="border-b border-dark-border text-xs text-gray-500">
                      <th className="text-left px-4 py-2 font-medium">Date</th>
                      <th className="text-left px-4 py-2 font-medium">Direction</th>
                      <th className="text-right px-4 py-2 font-medium">Entry</th>
                      <th className="text-right px-4 py-2 font-medium">Exit</th>
                      <th className="text-right px-4 py-2 font-medium">Return</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtestResult.trades.slice(0, 20).map((t, i) => (
                      <tr key={i} className="border-b border-dark-border/50 hover:bg-dark-hover transition-colors">
                        <td className="px-4 py-2 font-mono text-gray-300">{t.date}</td>
                        <td className="px-4 py-2">
                          <Badge variant={t.direction === 'LONG' ? 'success' : 'danger'}>
                            {t.direction}
                          </Badge>
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-gray-300">
                          ${t.entry_price.toFixed(2)}
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-gray-300">
                          ${t.exit_price.toFixed(2)}
                        </td>
                        <td
                          className={`px-4 py-2 text-right font-mono font-semibold ${
                            t.return_pct >= 0 ? 'text-bull' : 'text-bear'
                          }`}
                        >
                          {t.return_pct >= 0 ? '+' : ''}
                          {(t.return_pct * 100).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!signals && !backtestResult && !loadingSignals && !loadingBacktest && selectedStrategy && (
        <div className="flex flex-col items-center justify-center py-16 text-gray-600">
          <Play className="w-12 h-12 mb-4 text-gray-600" />
          <div className="text-lg font-medium mb-1">Ready to Generate</div>
          <div className="text-sm text-center max-w-md">
            Configure parameters above and click "Generate Signals" to see trading signals,
            or "Run Backtest" to evaluate strategy performance.
          </div>
        </div>
      )}
    </div>
  );
}
