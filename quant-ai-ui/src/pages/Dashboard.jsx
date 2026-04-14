import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { getMarket, predict } from "../api/client";

export default function Dashboard() {
  const [searchParams] = useSearchParams();
  const initialTicker = searchParams.get("ticker") || "AAPL";
  const [ticker, setTicker] = useState(initialTicker);
  const [prices, setPrices] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // === Load market data ===
  useEffect(() => {
    async function loadMarket() {
      setLoading(true);
      setError(null);
      try {
        const data = await getMarket(ticker);
        setPrices(data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }

    loadMarket();
  }, [ticker]);

  // === Run prediction ===
  async function runPredict() {
    setLoading(true);
    setError(null);
    try {
      const result = await predict({
        ticker,
        horizons: [5],
        features: {
          technical: true,
        },
      });
      setPrediction(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold text-white mb-4">Dashboard</h2>

      {/* === Ticker selector === */}
      <div className="flex items-center gap-3 mb-6">
        <label className="text-gray-400 text-sm">Ticker:</label>
        <input
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          className="bg-surface-card border border-gray-700 text-white rounded px-3 py-1.5 text-sm focus:outline-none focus:border-accent"
        />
        <button
          onClick={runPredict}
          className="bg-accent hover:opacity-90 text-white px-4 py-1.5 rounded text-sm font-medium"
        >
          Predict (5d)
        </button>
      </div>

      {/* === Error / Loading === */}
      {loading && <p className="text-gray-400 text-sm mb-4">Loading...</p>}
      {error && <p className="text-down text-sm mb-4">{error}</p>}

      {/* === Price preview === */}
      <div className="bg-surface-card rounded-lg p-4 mb-4">
        <h3 className="text-white font-medium mb-3">Recent Prices</h3>
        <ul className="space-y-1">
          {prices.slice(0, 5).map((p, idx) => (
            <li key={idx} className="text-gray-300 text-sm">
              <span className="text-gray-500 mr-2">{p.date}</span>
              {p.close}
            </li>
          ))}
        </ul>
      </div>

      {/* === Prediction === */}
      {prediction && (
        <div className="bg-surface-card rounded-lg p-4 mt-4">
          <h3 className="text-white font-medium mb-2">Prediction</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-gray-400 text-sm">Signal</p>
              <p className={`text-lg font-bold ${
                prediction.signal === "LONG" ? "text-up" : prediction.signal === "SHORT" ? "text-down" : "text-gray-400"
              }`}>{prediction.signal || "N/A"}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm">Probability Up</p>
              <p className="text-lg font-bold text-white">{(prediction.prob_up * 100)?.toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm">Samples</p>
              <p className="text-lg font-bold text-white">{prediction.samples}</p>
            </div>
          </div>
        </div>
      )}
      {!prediction && !loading && (
        <div className="bg-surface-card rounded-lg p-4 mt-4">
          <p className="text-gray-500 text-sm">No prediction yet. Click Predict to run.</p>
        </div>
      )}
    </div>
  );
}
