import { usePredictedVolatility, useModelsForTickerByLabelType } from "@/api/queries";
import MetaSparkline from "@/features/signal-console/MetaSparkline";

// V4 Pivot · Phase 2 · FE-ENH-6
// Dashboard card showing forward-looking realized volatility prediction.
// - Calls POST /predict/volatility with promoted vol model.
// - Gracefully shows a "no vol model" empty state until Harry trains one.

function formatPct(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function bandForVol(vol) {
  // Rough bands for annualized realized vol (equities).
  if (vol == null) return { label: "—", color: "text-muted" };
  if (vol < 0.15) return { label: "Low volatility", color: "text-up" };
  if (vol < 0.30) return { label: "Moderate volatility", color: "text-foreground" };
  if (vol < 0.50) return { label: "High volatility", color: "text-amber-600" };
  return { label: "Very high volatility", color: "text-down" };
}

export function VolatilityCard({ ticker, horizonDays = 5 }) {
  // Check if any vol model exists for this ticker first — cheap registry call.
  const modelsQ = useModelsForTickerByLabelType(ticker, "volatility");
  const hasVolModel = (modelsQ.data ?? []).length > 0;
  const preferredVolModel = hasVolModel ? modelsQ.data[0] : null;

  // Only call /predict/volatility if a model exists (otherwise endpoint returns
  // graceful error, but we save a roundtrip and surface a clearer empty state).
  const predQ = usePredictedVolatility(
    ticker,
    preferredVolModel?.id ?? null,
    horizonDays,
  );

  const vol = predQ.data?.success ? predQ.data.predicted_volatility : null;
  const band = bandForVol(vol);

  return (
    <section className="rounded-lg border border-border bg-white p-4 mb-4">
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="text-sm font-bold text-foreground">
          📊 Predicted Volatility · {horizonDays}D annualized
        </h3>
        <span className="text-[10px] text-muted">V4 Phase 2</span>
      </div>

      {!hasVolModel ? (
        <EmptyState ticker={ticker} />
      ) : predQ.isLoading ? (
        <div className="text-sm text-muted py-4">Loading…</div>
      ) : predQ.data?.success ? (
        <ResultView
          vol={vol}
          band={band}
          modelName={preferredVolModel?.name}
          modelMetrics={preferredVolModel?.metrics}
        />
      ) : (
        <div className="text-xs text-down py-2">
          {predQ.data?.error ?? "Prediction failed"}
        </div>
      )}
      <MetaSparkline ticker={ticker} />
    </section>
  );
}

function EmptyState({ ticker }) {
  return (
    <div className="text-xs text-muted py-3 border border-dashed border-border rounded px-3">
      No volatility model trained yet for {ticker} (label_type=volatility).
      <br />
      Open <span className="font-medium">🧪 Modeling</span> and select "Prediction target → Volatility" to train one.
    </div>
  );
}

function ResultView({ vol, band, modelName, modelMetrics }) {
  const mae = modelMetrics?.val_mae ?? modelMetrics?.test_mae;
  return (
    <div>
      <div className="flex items-baseline gap-3">
        <span className="text-3xl font-bold text-foreground">{formatPct(vol)}</span>
        <span className={`text-sm font-medium ${band.color}`}>{band.label}</span>
      </div>
      <div className="text-[11px] text-muted mt-2 flex items-center gap-2 flex-wrap">
        {modelName && (
          <span>
            Source: <span className="font-mono">{modelName}</span>
          </span>
        )}
        {mae != null && (
          <span>· val MAE = {Number(mae).toFixed(4)}</span>
        )}
      </div>
    </div>
  );
}
