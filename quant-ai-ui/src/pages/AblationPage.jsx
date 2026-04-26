import { useState } from "react";
import { useAblationRun } from "@/api/leaderboardQueries";
import AblationMatrix from "@/features/ablation/AblationMatrix";

const DEFAULT_TARGETS = ["direction", "volatility", "meta_label"];
const DEFAULT_FEATURE_SETS = [
  { name: "ta_basic", groups: ["ta_basic"] },
  { name: "ta_basic + sentiment", groups: ["ta_basic", "sentiment"] },
];

export default function AblationPage() {
  const [ticker, setTicker] = useState("");
  const [result, setResult] = useState(null);
  const run = useAblationRun();

  const onRun = (e) => {
    e.preventDefault();
    if (!ticker) return;
    run.mutate(
      {
        ticker,
        targets: DEFAULT_TARGETS,
        feature_sets: DEFAULT_FEATURE_SETS,
        horizon_days: 5,
        model_type: "xgboost",
      },
      { onSuccess: setResult },
    );
  };

  return (
    <div className="p-6 space-y-4 max-w-7xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold">Ablation</h1>
        <p className="text-sm text-slate-400">
          Train 6 models (3 targets × 2 feature sets) with default params for fair comparison.
          Quantifies sentiment&apos;s contribution per target.
        </p>
      </header>

      <form onSubmit={onRun} className="flex items-end gap-3 p-4 bg-slate-900/40 rounded-lg">
        <div className="flex-1">
          <label className="block text-xs text-slate-400 mb-1" htmlFor="ablation-ticker">
            Ticker
          </label>
          <input
            id="ablation-ticker"
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="MSFT"
            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm"
          />
        </div>
        <button
          type="submit"
          disabled={!ticker || run.isPending}
          className="px-4 py-2 bg-emerald-600/20 border border-emerald-600/40 rounded text-sm hover:bg-emerald-600/30 disabled:opacity-50"
        >
          {run.isPending ? "Running..." : "Run ablation"}
        </button>
      </form>

      {run.isError && (
        <div className="p-3 bg-rose-500/15 text-rose-300 text-sm rounded">
          Error: {String(run.error?.message || "unknown")}
        </div>
      )}

      {result && (
        <>
          <AblationMatrix matrix={result.matrix} />
          {result.summary?.interpretation && (
            <div className="p-3 bg-slate-800/50 text-sm rounded">
              <div className="text-xs uppercase text-slate-400 mb-1">Interpretation</div>
              {result.summary.interpretation}
            </div>
          )}
          <div className="text-[10px] text-slate-500">
            Elapsed {result.elapsed_seconds}s · model_type {result.model_type}
          </div>
        </>
      )}
    </div>
  );
}
