import { useEffect, useState } from "react";
import { explain, search } from "../api/client";
import DisabledPanel from "../components/DisabledPanel";

export default function Explain() {
  const [ticker, setTicker] = useState("AAPL");
  const [shap, setShap] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function loadExplain() {
    setLoading(true);
    setError(null);
    try {
      const res = await explain(ticker);
      setShap(res);

      // 用一个“典型失败关键词”做向量检索示例
      const q = "high volatility rsi failed";
      const hits = await search(q);
      setSearchResults(hits);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadExplain();
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <h2>🔍 Explain</h2>

      {/* === Ticker selector === */}
      <div style={{ marginBottom: 16 }}>
        <label>
          Ticker:&nbsp;
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
          />
        </label>
        <button onClick={loadExplain} style={{ marginLeft: 12 }}>
          Reload
        </button>
      </div>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* === SHAP summary === */}
      <div style={{ marginBottom: 24 }}>
        <h3>SHAP Top Features</h3>
        {shap ? (
          <ul>
            {shap.top_features.map((f, idx) => (
              <li key={idx}>
                {f.feature}: {f.mean_abs_shap.toFixed(4)}
              </li>
            ))}
          </ul>
        ) : (
          <p>No SHAP data.</p>
        )}
      </div>

      {/* === Vector search results === */}
      <div style={{ marginBottom: 24 }}>
        <h3>Similar Historical Explanations</h3>
        {searchResults.length > 0 ? (
          <ul>
            {searchResults.map((r, idx) => (
              <li key={idx}>
                <strong>{r.score.toFixed(3)}</strong> — {r.text}
              </li>
            ))}
          </ul>
        ) : (
          <p>No similar records.</p>
        )}
      </div>

        <DisabledPanel
         title="Research Summary (v2)"
        description="LLM-generated summary over SHAP, signals, and similar historical cases."
        />

        <DisabledPanel
        title="Bull / Bear Narrative (v2)"
        description="Multi-horizon market narrative with confidence bands."
        />

        <DisabledPanel
        title="Agent Output (v3)"
        description="Autonomous trading agent reasoning and action plan."
        />

    </div>
  );
}
