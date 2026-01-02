import { useEffect, useState } from "react";
import { getMarket, predict } from "../api/client";

export default function Dashboard() {
  const [ticker, setTicker] = useState("AAPL");
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
        horizons: [5], // v1 固定 5，v2 可扩
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
    <div style={{ padding: 24 }}>
      <h2>📊 Dashboard</h2>

      {/* === Ticker selector === */}
      <div style={{ marginBottom: 16 }}>
        <label>
          Ticker:&nbsp;
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
          />
        </label>
        <button onClick={runPredict} style={{ marginLeft: 12 }}>
          Predict (5d)
        </button>
      </div>

      {/* === Error / Loading === */}
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* === Price preview === */}
      <div style={{ marginBottom: 24 }}>
        <h3>Recent Prices</h3>
        <ul>
          {prices.slice(0, 5).map((p, idx) => (
            <li key={idx}>
              {p.date}: {p.close}
            </li>
          ))}
        </ul>
      </div>

      {/* === Prediction === */}
      <div>
        <h3>Prediction</h3>
        {prediction ? (
          <pre>{JSON.stringify(prediction, null, 2)}</pre>
        ) : (
          <p>No prediction yet.</p>
        )}
      </div>
    </div>
  );
}
