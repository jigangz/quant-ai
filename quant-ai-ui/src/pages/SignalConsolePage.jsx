import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import TickerPicker from "@/features/signal-console/TickerPicker";
import StrategyMatrix from "@/features/signal-console/StrategyMatrix";
import SignalDetail from "@/features/signal-console/SignalDetail";
import { useMetaLabelTrain } from "@/api/signalQueries";

export default function SignalConsolePage() {
  const [params] = useSearchParams();
  const initialStrategy = params.get("strategy") || null;
  const [selectedTickers, setSelectedTickers] = useState(["AAPL", "MSFT", "GOOGL"]);
  const [selection, setSelection] = useState(null);
  const train = useMetaLabelTrain();

  const onTrain = ({ ticker, strategy }) => {
    train.mutate(
      {
        ticker,
        primary: { source: "strategy", strategy_name: strategy },
        barrier: { tp_k: 2.0, sl_k: 1.0, timeout_days: 5, vol_source: "realized_sigma" },
        cv: { n_splits: 5, embargo_pct: 0.01 },
        // V4 P4: random_forest is the prod-safe default (Render free tier
        // build doesn't ship xgboost). Users can override via Custom Train.
        model: { type: "random_forest" },
        window: { lookback_days: 730, feature_group: "ta_basic" },
      },
      {
        onSuccess: (data) => {
          setSelection({ ticker, strategy, model_id: data.model_id });
        },
      },
    );
  };

  return (
    <div className="p-6 space-y-4 max-w-7xl mx-auto">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold">Signal Console</h1>
        <p className="text-sm text-slate-400">
          Meta-label signal quality across strategies × tickers. Click a cell to preview its latest signal score.
          {initialStrategy && <span className="ml-2 text-emerald-400">· filtered: {initialStrategy}</span>}
        </p>
      </header>

      <TickerPicker selected={selectedTickers} onChange={setSelectedTickers} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <StrategyMatrix tickers={selectedTickers} onSelect={setSelection} onTrain={onTrain} />
          {train.isPending && (
            <div className="mt-2 text-xs text-amber-400">Training meta-model... (may take ~5s)</div>
          )}
          {train.isError && (
            <div className="mt-2 text-xs text-rose-400">
              Train failed: {String(train.error?.message || "unknown")}
            </div>
          )}
        </div>
        <div>
          <SignalDetail selection={selection} />
        </div>
      </div>
    </div>
  );
}
