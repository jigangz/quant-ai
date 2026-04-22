import { usePredictedVolatility, useModelsForTickerByLabelType } from "@/api/queries";

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
  if (vol < 0.15) return { label: "低波动", color: "text-up" };
  if (vol < 0.30) return { label: "中等波动", color: "text-foreground" };
  if (vol < 0.50) return { label: "高波动", color: "text-amber-600" };
  return { label: "极高波动", color: "text-down" };
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
          📊 预测波动率 · {horizonDays} 日年化
        </h3>
        <span className="text-[10px] text-muted">V4 Phase 2</span>
      </div>

      {!hasVolModel ? (
        <EmptyState ticker={ticker} />
      ) : predQ.isLoading ? (
        <div className="text-sm text-muted py-4">加载中…</div>
      ) : predQ.data?.success ? (
        <ResultView
          vol={vol}
          band={band}
          modelName={preferredVolModel?.name}
          modelMetrics={preferredVolModel?.metrics}
        />
      ) : (
        <div className="text-xs text-down py-2">
          {predQ.data?.error ?? "预测失败"}
        </div>
      )}
    </section>
  );
}

function EmptyState({ ticker }) {
  return (
    <div className="text-xs text-muted py-3 border border-dashed border-border rounded px-3">
      还没有训练过 {ticker} 的波动率模型 (label_type=volatility)。
      <br />
      去 <span className="font-medium">🧪 建模</span> 选"预测目标 → 波动率"训练一个。
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
            来源: <span className="font-mono">{modelName}</span>
          </span>
        )}
        {mae != null && (
          <span>· val MAE = {Number(mae).toFixed(4)}</span>
        )}
      </div>
    </div>
  );
}
