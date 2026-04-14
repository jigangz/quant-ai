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
    <div className="p-6">
      <h2 className="text-xl font-bold text-white mb-4">Explain</h2>

      {/* === Ticker selector === */}
      <div className="flex items-center gap-3 mb-6">
        <label className="text-gray-400 text-sm">Ticker:</label>
        <input
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          className="bg-surface-card border border-gray-700 text-white rounded px-3 py-1.5 text-sm focus:outline-none focus:border-accent"
        />
        <button
          onClick={loadExplain}
          className="bg-accent hover:opacity-90 text-white px-4 py-1.5 rounded text-sm font-medium"
        >
          Reload
        </button>
      </div>

      {loading && <p className="text-gray-400 text-sm mb-4">Loading...</p>}
      {error && <p className="text-down text-sm mb-4">{error}</p>}

      {/* === SHAP summary === */}
      <div className="bg-surface-card rounded-lg p-4 mb-4">
        <h3 className="text-white font-medium mb-3">SHAP Top Features</h3>
        {shap ? (
          <ul className="space-y-1">
            {shap.top_features.map((f, idx) => (
              <li key={idx} className="text-gray-300 text-sm">
                <span className="text-white font-medium">{f.feature}</span>
                <span className="text-gray-500 mx-2">—</span>
                <span className="text-accent">{f.mean_abs_shap.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500 text-sm">No SHAP data.</p>
        )}
      </div>

      {/* === Vector search results === */}
      <div className="bg-surface-card rounded-lg p-4 mb-4">
        <h3 className="text-white font-medium mb-3">Similar Historical Explanations</h3>
        {searchResults.length > 0 ? (
          <ul className="space-y-2">
            {searchResults.map((r, idx) => (
              <li key={idx} className="text-gray-300 text-sm">
                <strong className="text-accent">{r.score.toFixed(3)}</strong>
                <span className="text-gray-500 mx-2">—</span>
                {r.text}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500 text-sm">No similar records.</p>
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
