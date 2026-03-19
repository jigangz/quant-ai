import { useState, useEffect, useCallback } from 'react';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Send,
  X,
  RefreshCw,
  Briefcase,
  Clock,
  History,
} from 'lucide-react';
import Card from '../components/UI/Card';
import Badge from '../components/UI/Badge';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import {
  fetchPortfolio,
  fetchOpenOrders,
  fetchTradeHistory,
  fetchEquityCurve,
  fetchQuote,
  placeOrder,
  cancelOrder,
  type Portfolio,
  type Order,
  type Trade,
  type EquityPoint,
  type Quote,
} from '../api/trading';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';

interface Props {
  ticker: string;
}

function SummaryBar({ portfolio }: { portfolio: Portfolio }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      <Card className="p-4">
        <div className="flex items-center gap-2 text-xs text-gray-500 mb-1">
          <DollarSign className="w-4 h-4" />
          Cash
        </div>
        <div className="text-xl font-mono font-bold text-gray-200">
          ${portfolio.cash.toLocaleString('en-US', { minimumFractionDigits: 2 })}
        </div>
      </Card>
      <Card className="p-4">
        <div className="flex items-center gap-2 text-xs text-gray-500 mb-1">
          <Briefcase className="w-4 h-4" />
          Positions Value
        </div>
        <div className="text-xl font-mono font-bold text-gray-200">
          ${portfolio.positions_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
        </div>
      </Card>
      <Card className="p-4">
        <div className="flex items-center gap-2 text-xs text-gray-500 mb-1">
          <TrendingUp className="w-4 h-4" />
          Total Value
        </div>
        <div className="text-xl font-mono font-bold text-accent">
          ${portfolio.total_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
        </div>
      </Card>
      <Card className="p-4">
        <div className="flex items-center gap-2 text-xs text-gray-500 mb-1">
          {portfolio.total_pnl >= 0 ? (
            <TrendingUp className="w-4 h-4 text-bull" />
          ) : (
            <TrendingDown className="w-4 h-4 text-bear" />
          )}
          Unrealized P&L
        </div>
        <div
          className={`text-xl font-mono font-bold ${
            portfolio.total_pnl >= 0 ? 'text-bull' : 'text-bear'
          }`}
        >
          {portfolio.total_pnl >= 0 ? '+' : ''}$
          {portfolio.total_pnl.toLocaleString('en-US', { minimumFractionDigits: 2 })}
        </div>
      </Card>
      <Card className="p-4">
        <div className="text-xs text-gray-500 mb-1">P&L %</div>
        <div
          className={`text-xl font-mono font-bold ${
            portfolio.total_pnl_pct >= 0 ? 'text-bull' : 'text-bear'
          }`}
        >
          {portfolio.total_pnl_pct >= 0 ? '+' : ''}
          {portfolio.total_pnl_pct.toFixed(2)}%
        </div>
      </Card>
    </div>
  );
}

function PositionsTable({ positions }: { positions: Portfolio['positions'] }) {
  if (positions.length === 0) {
    return (
      <div className="text-center py-8 text-gray-600">
        <Briefcase className="w-8 h-8 mx-auto mb-2" />
        <div>No positions</div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-dark-border text-xs text-gray-500">
            <th className="text-left px-4 py-2 font-medium">Ticker</th>
            <th className="text-right px-4 py-2 font-medium">Qty</th>
            <th className="text-right px-4 py-2 font-medium">Avg Cost</th>
            <th className="text-right px-4 py-2 font-medium">Price</th>
            <th className="text-right px-4 py-2 font-medium">Value</th>
            <th className="text-right px-4 py-2 font-medium">P&L</th>
            <th className="text-right px-4 py-2 font-medium">%</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => (
            <tr
              key={p.ticker}
              className="border-b border-dark-border/50 hover:bg-dark-hover transition-colors"
            >
              <td className="px-4 py-2 font-mono font-bold text-accent">{p.ticker}</td>
              <td className="px-4 py-2 text-right font-mono text-gray-300">{p.quantity}</td>
              <td className="px-4 py-2 text-right font-mono text-gray-400">
                ${p.avg_cost.toFixed(2)}
              </td>
              <td className="px-4 py-2 text-right font-mono text-gray-300">
                ${p.current_price.toFixed(2)}
              </td>
              <td className="px-4 py-2 text-right font-mono text-gray-300">
                ${p.market_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </td>
              <td
                className={`px-4 py-2 text-right font-mono font-semibold ${
                  p.unrealized_pnl >= 0 ? 'text-bull' : 'text-bear'
                }`}
              >
                {p.unrealized_pnl >= 0 ? '+' : ''}$
                {p.unrealized_pnl.toFixed(2)}
              </td>
              <td
                className={`px-4 py-2 text-right font-mono ${
                  p.unrealized_pnl_pct >= 0 ? 'text-bull' : 'text-bear'
                }`}
              >
                {p.unrealized_pnl_pct >= 0 ? '+' : ''}
                {p.unrealized_pnl_pct.toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OrderForm({
  onSubmit,
  quote,
  onTickerChange,
}: {
  onSubmit: (ticker: string, side: 'BUY' | 'SELL', type: 'MARKET' | 'LIMIT', qty: number, price?: number) => void;
  quote: Quote | null;
  onTickerChange: (t: string) => void;
}) {
  const [formTicker, setFormTicker] = useState('');
  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT'>('MARKET');
  const [quantity, setQuantity] = useState('10');
  const [limitPrice, setLimitPrice] = useState('');

  const handleTickerBlur = () => {
    if (formTicker.trim()) {
      onTickerChange(formTicker.trim().toUpperCase());
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const ticker = formTicker.trim().toUpperCase();
    if (!ticker || !quantity) return;
    onSubmit(
      ticker,
      side,
      orderType,
      parseInt(quantity, 10),
      orderType === 'LIMIT' ? parseFloat(limitPrice) : undefined
    );
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3">
      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500">Ticker</label>
        <input
          type="text"
          value={formTicker}
          onChange={(e) => setFormTicker(e.target.value.toUpperCase())}
          onBlur={handleTickerBlur}
          placeholder="AAPL"
          className="w-24 h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent font-mono uppercase"
        />
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500">Side</label>
        <div className="flex gap-1">
          <button
            type="button"
            onClick={() => setSide('BUY')}
            className={`px-3 h-8 text-sm rounded-sm font-medium transition-colors ${
              side === 'BUY' ? 'bg-bull text-white' : 'bg-dark-bg border border-dark-border text-gray-400'
            }`}
          >
            BUY
          </button>
          <button
            type="button"
            onClick={() => setSide('SELL')}
            className={`px-3 h-8 text-sm rounded-sm font-medium transition-colors ${
              side === 'SELL' ? 'bg-bear text-white' : 'bg-dark-bg border border-dark-border text-gray-400'
            }`}
          >
            SELL
          </button>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500">Type</label>
        <select
          value={orderType}
          onChange={(e) => setOrderType(e.target.value as 'MARKET' | 'LIMIT')}
          className="h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent"
        >
          <option value="MARKET">Market</option>
          <option value="LIMIT">Limit</option>
        </select>
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500">Quantity</label>
        <input
          type="number"
          value={quantity}
          onChange={(e) => setQuantity(e.target.value)}
          min="1"
          className="w-20 h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent font-mono"
        />
      </div>
      {orderType === 'LIMIT' && (
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Limit Price</label>
          <input
            type="number"
            value={limitPrice}
            onChange={(e) => setLimitPrice(e.target.value)}
            step="0.01"
            placeholder={quote ? quote.price.toFixed(2) : '0.00'}
            className="w-24 h-8 px-2 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-300 focus:outline-none focus:border-accent font-mono"
          />
        </div>
      )}
      {quote && (
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Last Price</label>
          <span
            className={`h-8 flex items-center font-mono text-sm ${
              quote.change >= 0 ? 'text-bull' : 'text-bear'
            }`}
          >
            ${quote.price.toFixed(2)} ({quote.change >= 0 ? '+' : ''}{quote.change_pct.toFixed(2)}%)
          </span>
        </div>
      )}
      <button
        type="submit"
        className="flex items-center gap-2 px-4 py-2 h-8 bg-accent hover:bg-accent/80 text-white text-sm font-medium rounded-sm transition-colors"
      >
        <Send className="w-4 h-4" />
        Place Order
      </button>
    </form>
  );
}

function OpenOrdersTable({
  orders,
  onCancel,
}: {
  orders: Order[];
  onCancel: (id: string) => void;
}) {
  if (orders.length === 0) {
    return (
      <div className="text-center py-6 text-gray-600">
        <Clock className="w-6 h-6 mx-auto mb-2" />
        <div className="text-sm">No open orders</div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-dark-border text-xs text-gray-500">
            <th className="text-left px-4 py-2 font-medium">Ticker</th>
            <th className="text-left px-4 py-2 font-medium">Side</th>
            <th className="text-left px-4 py-2 font-medium">Type</th>
            <th className="text-right px-4 py-2 font-medium">Qty</th>
            <th className="text-right px-4 py-2 font-medium">Price</th>
            <th className="text-right px-4 py-2 font-medium">Created</th>
            <th className="text-center px-4 py-2 font-medium">Action</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((o) => (
            <tr
              key={o.id}
              className="border-b border-dark-border/50 hover:bg-dark-hover transition-colors"
            >
              <td className="px-4 py-2 font-mono font-bold text-accent">{o.ticker}</td>
              <td className="px-4 py-2">
                <Badge variant={o.side === 'BUY' ? 'success' : 'danger'}>{o.side}</Badge>
              </td>
              <td className="px-4 py-2 text-gray-400">{o.type}</td>
              <td className="px-4 py-2 text-right font-mono text-gray-300">{o.quantity}</td>
              <td className="px-4 py-2 text-right font-mono text-gray-300">
                {o.limit_price ? `$${o.limit_price.toFixed(2)}` : 'MKT'}
              </td>
              <td className="px-4 py-2 text-right text-xs text-gray-500">
                {new Date(o.created_at).toLocaleTimeString()}
              </td>
              <td className="px-4 py-2 text-center">
                <button
                  onClick={() => onCancel(o.id)}
                  className="p-1 text-gray-500 hover:text-bear transition-colors"
                  title="Cancel order"
                >
                  <X className="w-4 h-4" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TradeHistoryTable({ trades }: { trades: Trade[] }) {
  if (trades.length === 0) {
    return (
      <div className="text-center py-6 text-gray-600">
        <History className="w-6 h-6 mx-auto mb-2" />
        <div className="text-sm">No trade history</div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto max-h-[250px] overflow-y-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-dark-card">
          <tr className="border-b border-dark-border text-xs text-gray-500">
            <th className="text-left px-4 py-2 font-medium">Ticker</th>
            <th className="text-left px-4 py-2 font-medium">Side</th>
            <th className="text-right px-4 py-2 font-medium">Qty</th>
            <th className="text-right px-4 py-2 font-medium">Price</th>
            <th className="text-right px-4 py-2 font-medium">Total</th>
            <th className="text-right px-4 py-2 font-medium">Time</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr
              key={t.id}
              className="border-b border-dark-border/50 hover:bg-dark-hover transition-colors"
            >
              <td className="px-4 py-2 font-mono font-bold text-accent">{t.ticker}</td>
              <td className="px-4 py-2">
                <Badge variant={t.side === 'BUY' ? 'success' : 'danger'}>{t.side}</Badge>
              </td>
              <td className="px-4 py-2 text-right font-mono text-gray-300">{t.quantity}</td>
              <td className="px-4 py-2 text-right font-mono text-gray-300">
                ${t.price.toFixed(2)}
              </td>
              <td className="px-4 py-2 text-right font-mono text-gray-300">
                ${t.total.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </td>
              <td className="px-4 py-2 text-right text-xs text-gray-500">
                {new Date(t.executed_at).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EquityChart({ data }: { data: EquityPoint[] }) {
  return (
    <div className="h-[250px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="date"
            tick={{ fill: '#9CA3AF', fontSize: 10 }}
            tickLine={{ stroke: '#4B5563' }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={['auto', 'auto']}
            tick={{ fill: '#9CA3AF', fontSize: 10 }}
            tickLine={{ stroke: '#4B5563' }}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '4px',
            }}
            labelStyle={{ color: '#9CA3AF' }}
            formatter={(value: number) => [`$${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`, 'Value']}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#60A5FA"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Trading({ ticker }: Props) {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [openOrders, setOpenOrders] = useState<Order[]>([]);
  const [tradeHistory, setTradeHistory] = useState<Trade[]>([]);
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([]);
  const [quote, setQuote] = useState<Quote | null>(null);
  const [quoteTicker, setQuoteTicker] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initial load
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        const [p, o, t, e] = await Promise.all([
          fetchPortfolio(),
          fetchOpenOrders(),
          fetchTradeHistory(20),
          fetchEquityCurve(),
        ]);
        setPortfolio(p);
        setOpenOrders(o);
        setTradeHistory(t);
        setEquityCurve(e);
      } catch {
        setError('Failed to load trading data');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Mock WebSocket price updates
  useEffect(() => {
    if (!quoteTicker) return;
    
    // Initial fetch
    fetchQuote(quoteTicker).then(setQuote).catch(() => setQuote(null));
    
    // Simulate price updates every 3 seconds
    const interval = setInterval(() => {
      fetchQuote(quoteTicker).then(setQuote).catch(() => {});
    }, 3000);
    
    return () => clearInterval(interval);
  }, [quoteTicker]);

  const handleTickerChange = useCallback((t: string) => {
    setQuoteTicker(t);
  }, []);

  const handlePlaceOrder = useCallback(
    async (
      orderTicker: string,
      side: 'BUY' | 'SELL',
      type: 'MARKET' | 'LIMIT',
      qty: number,
      price?: number
    ) => {
      try {
        const newOrder = await placeOrder({
          ticker: orderTicker,
          side,
          type,
          quantity: qty,
          limit_price: price,
        });
        if (newOrder.status === 'PENDING') {
          setOpenOrders((prev) => [newOrder, ...prev]);
        } else if (newOrder.status === 'FILLED') {
          // Refresh all data
          const [p, t] = await Promise.all([fetchPortfolio(), fetchTradeHistory(20)]);
          setPortfolio(p);
          setTradeHistory(t);
        }
      } catch {
        setError('Failed to place order');
      }
    },
    []
  );

  const handleCancelOrder = useCallback(async (orderId: string) => {
    try {
      await cancelOrder(orderId);
      setOpenOrders((prev) => prev.filter((o) => o.id !== orderId));
    } catch {
      setError('Failed to cancel order');
    }
  }, []);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    try {
      const [p, o, t, e] = await Promise.all([
        fetchPortfolio(),
        fetchOpenOrders(),
        fetchTradeHistory(20),
        fetchEquityCurve(),
      ]);
      setPortfolio(p);
      setOpenOrders(o);
      setTradeHistory(t);
      setEquityCurve(e);
    } catch {
      setError('Failed to refresh data');
    } finally {
      setLoading(false);
    }
  }, []);

  if (loading && !portfolio) {
    return <LoadingSpinner text="Loading trading data..." />;
  }

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      {error && (
        <div className="p-3 bg-bear/10 border border-bear/30 rounded-sm text-sm text-bear flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-bear hover:text-red-400">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Portfolio summary */}
      {portfolio && <SummaryBar portfolio={portfolio} />}

      {/* Order form */}
      <Card
        title="Place Order"
        action={
          <button
            onClick={handleRefresh}
            className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
            title="Refresh data"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        }
      >
        <div className="p-4">
          <OrderForm onSubmit={handlePlaceOrder} quote={quote} onTickerChange={handleTickerChange} />
        </div>
      </Card>

      <div className="grid grid-cols-12 gap-4">
        {/* Positions */}
        <div className="col-span-12 lg:col-span-7">
          <Card title="Positions">
            <PositionsTable positions={portfolio?.positions || []} />
          </Card>
        </div>

        {/* Open orders */}
        <div className="col-span-12 lg:col-span-5">
          <Card title={`Open Orders (${openOrders.length})`}>
            <OpenOrdersTable orders={openOrders} onCancel={handleCancelOrder} />
          </Card>
        </div>

        {/* Equity chart */}
        <div className="col-span-12 lg:col-span-7">
          <Card title="Portfolio Equity">
            <div className="p-2">
              <EquityChart data={equityCurve} />
            </div>
          </Card>
        </div>

        {/* Trade history */}
        <div className="col-span-12 lg:col-span-5">
          <Card title="Trade History">
            <TradeHistoryTable trades={tradeHistory} />
          </Card>
        </div>
      </div>
    </div>
  );
}
