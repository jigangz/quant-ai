import { PredictionCard } from "./PredictionCard";
import { AgentSummaryCard } from "./AgentSummaryCard";
import { ShapMiniCard } from "./ShapMiniCard";

export function AIInsightBand({ data, isLoading = false, error = null }) {
  if (isLoading) {
    return (
      <div data-testid="ai-band-skeleton" className="grid grid-cols-[1fr_1.5fr_1fr] gap-2.5 mb-4">
        <div className="h-32 bg-surface-muted rounded animate-pulse" />
        <div className="h-32 bg-surface-muted rounded animate-pulse" />
        <div className="h-32 bg-surface-muted rounded animate-pulse" />
      </div>
    );
  }
  if (error) {
    return (
      <div className="bg-down/10 border border-down/30 rounded p-4 mb-4 text-sm">
        AI 分析暂不可用。<button className="text-accent underline ml-2" onClick={() => window.location.reload()}>重试</button>
      </div>
    );
  }
  if (!data) return null;
  return (
    <div className="grid grid-cols-[1fr_1.5fr_1fr] gap-2.5 mb-4">
      <PredictionCard
        prediction={data.prediction}
        probability={data.probability}
        confidence={data.confidence}
        horizon={data.horizon}
      />
      <AgentSummaryCard summary={data.summary} />
      <ShapMiniCard features={data.top_features} />
    </div>
  );
}
