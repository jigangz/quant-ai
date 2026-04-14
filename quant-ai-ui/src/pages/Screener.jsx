import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getMarketMulti } from "../api/client";

const TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "AMD", "INTC"];

export default function Screener() {
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState("change");
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    setLoading(true);
    getMarketMulti(TICKERS)
      .then((results) => {
        const parsed = results
          .filter(Boolean)
          .map((rows) => {
            if (!rows.length) return null;
            const latest = rows[0];
            const prev = rows[1] || rows[0];
            const change = prev.close ? ((latest.close - prev.close) / prev.close) * 100 : 0;
            return { ...latest, change };
          })
          .filter(Boolean);
        setStocks(parsed);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const sorted = [...stocks].sort((a, b) =>
    sortKey === "change" ? b.change - a.change : b.volume - a.volume
  );

  if (loading) return <div className="text-gray-400 py-8">Loading screener...</div>;
  if (error) return <div className="text-red-400 py-8">Error: {error}</div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold text-white">Stock Screener</h1>
        <div className="flex gap-2">
          <button
            onClick={() => setSortKey("change")}
            className={`px-3 py-1 rounded text-sm ${sortKey === "change" ? "bg-accent text-white" : "bg-surface-card text-gray-400"}`}
          >
            Sort by Change %
          </button>
          <button
            onClick={() => setSortKey("volume")}
            className={`px-3 py-1 rounded text-sm ${sortKey === "volume" ? "bg-accent text-white" : "bg-surface-card text-gray-400"}`}
          >
            Sort by Volume
          </button>
        </div>
      </div>
      <div className="bg-surface-card rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400">
              <th className="text-left px-4 py-3">Ticker</th>
              <th className="text-right px-4 py-3">Last</th>
              <th className="text-right px-4 py-3">Change %</th>
              <th className="text-right px-4 py-3">Volume</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => (
              <tr
                key={s.ticker}
                onClick={() => navigate(`/dashboard?ticker=${s.ticker}`)}
                className="border-b border-gray-800 hover:bg-surface-hover cursor-pointer transition"
              >
                <td className="px-4 py-3 font-medium text-white">{s.ticker}</td>
                <td className="px-4 py-3 text-right">${s.close?.toFixed(2)}</td>
                <td className={`px-4 py-3 text-right font-medium ${s.change >= 0 ? "text-up" : "text-down"}`}>
                  {s.change >= 0 ? "+" : ""}{s.change?.toFixed(2)}%
                </td>
                <td className="px-4 py-3 text-right text-gray-400">
                  {s.volume?.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
