import { useState, useEffect, useRef } from "react";
import { placeOrder, listOrders, getPortfolio, resetPortfolio, cancelOrder, getTrades } from "../api/client";

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function Trading() {
  const [ticker, setTicker] = useState("AAPL");
  const [side, setSide] = useState("buy");
  const [orderType, setOrderType] = useState("market");
  const [quantity, setQuantity] = useState(10);
  const [price, setPrice] = useState("");
  const [orders, setOrders] = useState([]);
  const [portfolio, setPortfolio] = useState(null);
  const [trades, setTrades] = useState([]);
  const [livePrice, setLivePrice] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const wsRef = useRef(null);

  const refresh = () => {
    listOrders().then(setOrders).catch(() => {});
    getPortfolio().then(setPortfolio).catch(() => {});
    getTrades().then(setTrades).catch(() => {});
  };

  useEffect(() => { refresh(); }, []);

  // WebSocket price feed
  useEffect(() => {
    const wsUrl = BASE.replace("http", "ws") + "/api/trading/ws/prices";
    try {
      const ws = new WebSocket(wsUrl);
      ws.onmessage = (e) => {
        try { setLivePrice(JSON.parse(e.data)); } catch {}
      };
      ws.onerror = () => {};
      wsRef.current = ws;
    } catch {}
    return () => wsRef.current?.close();
  }, []);

  const handleOrder = async () => {
    setLoading(true);
    setError(null);
    try {
      await placeOrder({
        ticker: ticker.toUpperCase(),
        side,
        order_type: orderType,
        quantity: Number(quantity),
        ...(orderType === "limit" && price ? { price: Number(price) } : {}),
      });
      refresh();
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  const handleReset = async () => {
    if (!confirm("Reset portfolio?")) return;
    await resetPortfolio().catch(() => {});
    refresh();
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold text-white">Paper Trading</h1>
        {livePrice && (
          <span className="text-sm text-gray-400">
            Live: {livePrice.ticker} ${livePrice.price?.toFixed(2)}
          </span>
        )}
      </div>
      {error && <div className="bg-red-900/30 text-red-400 px-4 py-2 rounded mb-4">{error}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Order form */}
        <div className="bg-surface-card rounded-lg p-4">
          <h2 className="text-white font-medium mb-3">Place Order</h2>
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Ticker" className="w-full bg-surface border border-gray-700 rounded px-3 py-2 text-white mb-2" />
          <div className="flex gap-2 mb-2">
            <button onClick={() => setSide("buy")}
              className={`flex-1 py-2 rounded text-sm ${side === "buy" ? "bg-green-600 text-white" : "bg-surface text-gray-400"}`}>Buy</button>
            <button onClick={() => setSide("sell")}
              className={`flex-1 py-2 rounded text-sm ${side === "sell" ? "bg-red-600 text-white" : "bg-surface text-gray-400"}`}>Sell</button>
          </div>
          <select value={orderType} onChange={(e) => setOrderType(e.target.value)}
            className="w-full bg-surface border border-gray-700 rounded px-3 py-2 text-white mb-2">
            <option value="market">Market</option>
            <option value="limit">Limit</option>
          </select>
          <input type="number" value={quantity} onChange={(e) => setQuantity(e.target.value)}
            placeholder="Qty" className="w-full bg-surface border border-gray-700 rounded px-3 py-2 text-white mb-2" />
          {orderType === "limit" && (
            <input type="number" value={price} onChange={(e) => setPrice(e.target.value)}
              placeholder="Limit Price" className="w-full bg-surface border border-gray-700 rounded px-3 py-2 text-white mb-2" />
          )}
          <button onClick={handleOrder} disabled={loading}
            className="w-full bg-accent hover:bg-accent-dim text-white py-2 rounded text-sm disabled:opacity-50">
            {loading ? "Placing..." : `${side.toUpperCase()} ${ticker}`}
          </button>
          <button onClick={handleReset} className="w-full mt-2 bg-gray-700 hover:bg-gray-600 text-gray-300 py-2 rounded text-sm">
            Reset Portfolio
          </button>
        </div>

        {/* Portfolio */}
        <div className="bg-surface-card rounded-lg p-4">
          <h2 className="text-white font-medium mb-3">Portfolio</h2>
          {portfolio ? (
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Cash</span>
                <span className="text-white">${portfolio.cash?.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-sm mb-3">
                <span className="text-gray-400">Total Value</span>
                <span className="text-white">${portfolio.total_value?.toFixed(2)}</span>
              </div>
              <h3 className="text-gray-400 text-xs mb-1">Positions</h3>
              {(portfolio.positions || []).map((p) => (
                <div key={p.ticker} className="flex justify-between text-sm py-1 border-b border-gray-800">
                  <span className="text-white">{p.ticker} x{p.quantity}</span>
                  <span className={p.unrealized_pnl >= 0 ? "text-up" : "text-down"}>
                    ${p.unrealized_pnl?.toFixed(2)}
                  </span>
                </div>
              ))}
              {(!portfolio.positions || portfolio.positions.length === 0) && (
                <p className="text-gray-500 text-sm">No positions</p>
              )}
            </div>
          ) : <p className="text-gray-500 text-sm">Loading...</p>}
        </div>

        {/* Orders */}
        <div className="bg-surface-card rounded-lg p-4">
          <h2 className="text-white font-medium mb-3">Orders</h2>
          <div className="space-y-1 max-h-80 overflow-y-auto">
            {(orders.orders || orders || []).slice(0, 20).map((o) => (
              <div key={o.order_id || o.id} className="flex justify-between items-center text-sm py-1 border-b border-gray-800">
                <div>
                  <span className={o.side === "buy" ? "text-up" : "text-down"}>{o.side?.toUpperCase()}</span>
                  {" "}<span className="text-white">{o.ticker}</span>
                  {" "}<span className="text-gray-400">x{o.quantity}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs">{o.status}</span>
                  {o.status === "pending" && (
                    <button onClick={() => { cancelOrder(o.order_id || o.id).then(refresh); }}
                      className="text-red-400 text-xs hover:text-red-300">Cancel</button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent trades */}
        <div className="bg-surface-card rounded-lg p-4">
          <h2 className="text-white font-medium mb-3">Recent Trades</h2>
          <div className="space-y-1 max-h-80 overflow-y-auto">
            {(trades.trades || trades || []).slice(0, 20).map((t, i) => (
              <div key={i} className="flex justify-between text-sm py-1 border-b border-gray-800">
                <span className={t.side === "buy" ? "text-up" : "text-down"}>
                  {t.side?.toUpperCase()} {t.ticker}
                </span>
                <span className="text-gray-400">{t.quantity} @ ${t.fill_price?.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
