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
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Ablation</h1>
        <p className="text-sm text-muted">
          Train 6 models (3 targets × 2 feature sets) with default params for fair comparison.
          Quantifies sentiment&apos;s contribution per target.
        </p>
      </header>

      <form onSubmit={onRun} className="flex items-end gap-3 p-4 bg-surface-card rounded-lg">
        <div className="flex-1">
          <label className="block text-xs text-muted mb-1" htmlFor="ablation-ticker">
            Ticker
          </label>
          <input
            id="ablation-ticker"
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="MSFT"
            className="w-full bg-surface-muted border border-surface-border rounded px-2 py-1 text-sm"
          />
        </div>
        <button
          type="submit"
          disabled={!ticker || run.isPending}
          className="px-4 py-2 bg-accent/15 border border-accent/40 rounded text-sm hover:bg-accent/25 disabled:opacity-50"
        >
          {run.isPending ? "Running..." : "Run ablation"}
        </button>
      </form>

      {run.isError && (
        <div className="p-3 bg-down/10 text-down text-sm rounded">
          Error: {String(run.error?.message || "unknown")}
        </div>
      )}

      {result && (
        <>
          <AblationMatrix matrix={result.matrix} />
          {result.summary?.interpretation && (
            <div className="p-3 bg-surface-card text-sm rounded">
              <div className="text-xs uppercase text-muted mb-1">Interpretation</div>
              {result.summary.interpretation}
            </div>
          )}
          {result.summary?.sentiment_note && (
            <div className="p-3 bg-surface-muted text-muted text-xs rounded">
              ℹ️ {result.summary.sentiment_note}
            </div>
          )}
          <div className="text-[10px] text-muted">
            Elapsed {result.elapsed_seconds}s · model_type {result.model_type}
          </div>
        </>
      )}
    </div>
  );
}
