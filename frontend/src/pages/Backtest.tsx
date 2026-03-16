import { useState } from 'react';
import { Play, ChevronLeft, ChevronRight } from 'lucide-react';
import Card from '../components/UI/Card';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import BacktestChart from '../components/Charts/BacktestChart';
import Badge from '../components/UI/Badge';
import { runBacktest, type BacktestResult, type BacktestTrade } from '../api';

interface Props {
  ticker: string;
}

function MetricCard({ label, value, suffix = '', color }: { label: string; value: string; suffix?: string; color: string }) {
  return (
    <Card className="p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-xl font-mono font-bold ${color}`}>
        {value}
        {suffix && <span className="text-sm ml-0.5">{suffix}</span>}
      </div>
    </Card>
  );
}

const PAGE_SIZE = 10;

export default function Backtest({ ticker }: Props) {
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [modelType, setModelType] = useState('xgboost');
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tradePage, setTradePage] = useState(0);

  async function handleRun() {
    setLoading(true);
    setError(null);
    try {
      const res = await runBacktest({
        ticker,
        start_date: startDate,
        end_date: endDate,
        model_type: modelType,
      });
      setResult(res);
      setTradePage(0);
    } catch {
      setError('Backtest failed');
    } finally {
      setLoading(false);
    }
  }

  const trades = result?.trades ?? [];
  const totalPages = Math.ceil(trades.length / PAGE_SIZE);
  const pagedTrades = trades.slice(tradePage * PAGE_SIZE, (tradePage + 1) * PAGE_SIZE);

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      {/* Config bar */}
      <Card className="p-4">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Ticker</label>
            <span className="text-sm font-mono text-accent font-bold">{ticker}</span>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Model</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent"
            >
              <option value="xgboost">XGBoost</option>
              <option value="lightgbm">LightGBM</option>
              <option value="lstm">LSTM</option>
              <option value="ensemble">Ensemble</option>
            </select>
          </div>
          <button
            onClick={handleRun}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent/80 text-white text-sm font-medium rounded-sm transition-colors disabled:opacity-50 mt-4 sm:mt-0"
          >
            <Play className="w-4 h-4" />
            {loading ? 'Running...' : 'Run Backtest'}
          </button>
        </div>
      </Card>

      {error && (
        <div className="p-3 bg-bear/10 border border-bear/30 rounded-sm text-sm text-bear">{error}</div>
      )}

      {loading && <LoadingSpinner text="Running backtest..." />}

      {result && !loading && (
        <div className="grid grid-cols-12 gap-4">
          {/* Equity curve */}
          <div className="col-span-8">
            <Card title="Equity Curve" className="h-[400px]">
              <div className="p-2 h-[350px]">
                <BacktestChart data={result.equity_curve} />
              </div>
            </Card>
          </div>

          {/* Metrics */}
          <div className="col-span-4 grid grid-cols-2 gap-4">
            <MetricCard
              label="Sharpe Ratio"
              value={result.metrics.sharpe_ratio.toFixed(2)}
              color={result.metrics.sharpe_ratio > 1 ? 'text-bull' : result.metrics.sharpe_ratio > 0 ? 'text-accent' : 'text-bear'}
            />
            <MetricCard
              label="Max Drawdown"
              value={`${(result.metrics.max_drawdown * 100).toFixed(1)}`}
              suffix="%"
              color="text-bear"
            />
            <MetricCard
              label="Win Rate"
              value={`${(result.metrics.win_rate * 100).toFixed(1)}`}
              suffix="%"
              color={result.metrics.win_rate > 0.5 ? 'text-bull' : 'text-bear'}
            />
            <MetricCard
              label="Total Return"
              value={`${result.metrics.total_return >= 0 ? '+' : ''}${(result.metrics.total_return * 100).toFixed(1)}`}
              suffix="%"
              color={result.metrics.total_return >= 0 ? 'text-bull' : 'text-bear'}
            />
          </div>

          {/* Trades table */}
          <div className="col-span-12">
            <Card
              title={`Trade Log (${trades.length} trades)`}
              action={
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <button
                    onClick={() => setTradePage(Math.max(0, tradePage - 1))}
                    disabled={tradePage === 0}
                    className="p-1 hover:text-gray-300 disabled:opacity-30"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="font-mono">
                    {tradePage + 1}/{totalPages || 1}
                  </span>
                  <button
                    onClick={() => setTradePage(Math.min(totalPages - 1, tradePage + 1))}
                    disabled={tradePage >= totalPages - 1}
                    className="p-1 hover:text-gray-300 disabled:opacity-30"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              }
            >
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-dark-border text-xs text-gray-500">
                      <th className="text-left px-4 py-2 font-medium">Date</th>
                      <th className="text-left px-4 py-2 font-medium">Direction</th>
                      <th className="text-right px-4 py-2 font-medium">Entry</th>
                      <th className="text-right px-4 py-2 font-medium">Exit</th>
                      <th className="text-right px-4 py-2 font-medium">Return</th>
                      <th className="text-right px-4 py-2 font-medium">Days</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pagedTrades.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="text-center py-8 text-gray-600">
                          No trades
                        </td>
                      </tr>
                    ) : (
                      pagedTrades.map((t: BacktestTrade, i: number) => (
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
                          <td className="px-4 py-2 text-right font-mono text-gray-400">
                            {t.holding_days}d
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="flex flex-col items-center justify-center py-24 text-gray-600">
          <div className="text-6xl mb-4">🧪</div>
          <div className="text-lg font-medium mb-1">Configure & Run</div>
          <div className="text-sm">Set parameters above and run a backtest to see performance</div>
        </div>
      )}
    </div>
  );
}
