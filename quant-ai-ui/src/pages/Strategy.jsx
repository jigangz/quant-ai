import { useState, useEffect } from "react";
import { listStrategies, getStrategy, generateSignals, runStrategyBacktest, optimizeStrategy } from "../api/client";

export default function Strategy() {
  const [strategies, setStrategies] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [params, setParams] = useState({});
  const [signals, setSignals] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [ticker, setTicker] = useState("AAPL");
  const [loading, setLoading] = useState("");
  const [error, setError] = useState(null);
  const [strategyOptResult, setStrategyOptResult] = useState(null);
  const [optimizingStrategy, setOptimizingStrategy] = useState(false);

  useEffect(() => {
    listStrategies()
      .then((data) => setStrategies(data.strategies || data))
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    getStrategy(selected)
      .then((d) => {
        setDetail(d);
        const defaults = {};
        (d.parameters || []).forEach((p) => { defaults[p.name] = p.default; });
        setParams(defaults);
      })
      .catch((e) => setError(e.message));
  }, [selected]);

  const handleSignals = () => {
    setLoading("signals");
    setError(null);
    generateSignals(selected, { ticker, parameters: params })
      .then(setSignals)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(""));
  };

  const handleOptimizeStrategy = () => {
    setOptimizingStrategy(true);
    setStrategyOptResult(null);
    setError(null);
    optimizeStrategy({ strategy_name: selected, ticker, n_trials: 50 })
      .then((result) => {
        setStrategyOptResult(result);
        if (result.best_params) {
          setParams((prev) => ({ ...prev, ...result.best_params }));
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setOptimizingStrategy(false));
  };

  const handleBacktest = () => {
    setLoading("backtest");
    setError(null);
    runStrategyBacktest(selected, { ticker, parameters: params })
      .then(setBacktest)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(""));
  };

  return (
    <div>
      <h1 className="text-xl font-bold text-white mb-4">Strategy Editor</h1>
      {error && <div className="bg-red-900/30 text-red-400 px-4 py-2 rounded mb-4">{error}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Strategy selector + params */}
        <div className="bg-surface-card rounded-lg p-4">
          <label className="block text-gray-400 text-sm mb-1">Strategy</label>
          <select
            value={selected || ""}
            onChange={(e) => { setSelected(e.target.value); setSignals(null); setBacktest(null); }}
            className="w-full bg-surface border border-gray-700 rounded px-3 py-2 text-white mb-3"
          >
            <option value="">Select...</option>
            {strategies.map((s) => (
              <option key={s.name || s} value={s.name || s}>{s.display_name || s.name || s}</option>
            ))}
          </select>

          <label className="block text-gray-400 text-sm mb-1">Ticker</label>
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            className="w-full bg-surface border border-gray-700 rounded px-3 py-2 text-white mb-3"
          />

          {detail?.parameters?.map((p) => (
            <div key={p.name} className="mb-2">
              <label className="block text-gray-400 text-sm mb-1">{p.name}</label>
              <input
                type="number"
                value={params[p.name] ?? ""}
                onChange={(e) => setParams({ ...params, [p.name]: Number(e.target.value) })}
                className="w-full bg-surface border border-gray-700 rounded px-3 py-2 text-white"
              />
            </div>
          ))}

          <div className="flex gap-2 mt-4">
            <button
              onClick={handleSignals}
              disabled={!selected || loading === "signals"}
              className="flex-1 bg-accent hover:bg-accent-dim text-white py-2 rounded text-sm disabled:opacity-50"
            >
              {loading === "signals" ? "Generating..." : "Generate Signals"}
            </button>
            <button
              onClick={handleBacktest}
              disabled={!selected || loading === "backtest"}
              className="flex-1 bg-green-600 hover:bg-green-700 text-white py-2 rounded text-sm disabled:opacity-50"
            >
              {loading === "backtest" ? "Running..." : "Run Backtest"}
            </button>
          </div>

          <button
            onClick={handleOptimizeStrategy}
            disabled={optimizingStrategy || !selected || !ticker}
            className="w-full mt-2 px-4 py-2 bg-accent text-white rounded hover:bg-accent/80 disabled:opacity-50 text-sm"
          >
            {optimizingStrategy ? "Optimizing..." : "Optimize Parameters"}
          </button>

          {strategyOptResult && (
            <div className="mt-4 p-4 bg-surface-card rounded-lg">
              <h4 className="text-sm font-medium text-gray-300 mb-2">Optimization Results</h4>
              <p className="text-xs text-gray-400">
                Best {strategyOptResult.best_metrics && Object.keys(strategyOptResult.best_metrics)[0]}
                = {strategyOptResult.best_metrics && Object.values(strategyOptResult.best_metrics)[0]?.toFixed(4)}
                ({strategyOptResult.n_trials} trials, {strategyOptResult.duration_seconds?.toFixed(1)}s)
              </p>
              <p className="text-xs text-accent mt-1">Parameters auto-filled above</p>
            </div>
          )}
        </div>

        {/* Signals */}
        <div className="bg-surface-card rounded-lg p-4">
          <h2 className="text-white font-medium mb-3">Signals</h2>
          {!signals ? (
            <p className="text-gray-500 text-sm">Generate signals to see results</p>
          ) : (
            <div className="space-y-1 max-h-96 overflow-y-auto">
              {(signals.signals || []).slice(0, 50).map((s, i) => (
                <div key={i} className={`flex justify-between text-sm px-2 py-1 rounded ${
                  s.signal === "BUY" ? "bg-green-900/20 text-up" : s.signal === "SELL" ? "bg-red-900/20 text-down" : "text-gray-400"
                }`}>
                  <span>{s.date}</span>
                  <span className="font-medium">{s.signal}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Backtest */}
        <div className="bg-surface-card rounded-lg p-4">
          <h2 className="text-white font-medium mb-3">Backtest Results</h2>
          {!backtest ? (
            <p className="text-gray-500 text-sm">Run backtest to see metrics</p>
          ) : (
            <div className="space-y-2">
              {Object.entries(backtest.metrics || backtest).map(([k, v]) => (
                <div key={k} className="flex justify-between text-sm">
                  <span className="text-gray-400">{k}</span>
                  <span className="text-white font-medium">
                    {typeof v === "number" ? v.toFixed(4) : String(v)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
