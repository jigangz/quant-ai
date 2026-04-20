export function PredictionCard({ prediction, probability, confidence, horizon }) {
  const isBull = prediction === 1;
  const isBear = prediction === 0;
  const color = isBull ? "text-up" : isBear ? "text-down" : "text-muted";
  const label = isBull ? "↗ 看涨" : isBear ? "↘ 看跌" : "→ 中性";
  const confLabel = confidence === "high" ? "高" : confidence === "medium" ? "中" : "低";
  const confBg = isBull ? "bg-up/10 text-up" : isBear ? "bg-down/10 text-down" : "bg-muted/10 text-muted";

  return (
    <div className="bg-surface border border-surface-border rounded-md p-3">
      <div className="text-[9px] uppercase tracking-wide text-muted">🤖 AI 预测</div>
      <div className={`text-xl font-bold my-1 ${color}`}>{label}</div>
      <div className="flex gap-1 items-center text-[10px]">
        <span className={`px-1.5 py-0.5 rounded ${confBg}`}>置信度 {confLabel}</span>
        <span className="px-1.5 py-0.5 bg-surface-muted rounded text-muted">{horizon ?? 5} 天</span>
      </div>
      <div className="font-mono text-xs mt-2 text-foreground">prob_up {probability?.up?.toFixed(2) ?? "—"}</div>
    </div>
  );
}
