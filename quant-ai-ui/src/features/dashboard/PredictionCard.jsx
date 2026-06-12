export function PredictionCard({ prediction, probability, confidence, horizon }) {
  const isBull = prediction === 1;
  const isBear = prediction === 0;
  const color = isBull ? "text-up" : isBear ? "text-down" : "text-muted";
  const label = isBull ? "↗ Bullish" : isBear ? "↘ Bearish" : "→ Neutral";
  const confLabel = confidence === "high" ? "High" : confidence === "medium" ? "Medium" : "Low";
  const confBg = isBull ? "bg-up/10 text-up" : isBear ? "bg-down/10 text-down" : "bg-muted/10 text-muted";

  // API returns `probability` as a float (P(up)); guard both shapes.
  const probUp =
    typeof probability === "number"
      ? probability
      : probability?.up ?? null;

  return (
    <div className="bg-surface border border-surface-border rounded-md p-3">
      <div className="text-[9px] uppercase tracking-wide text-muted">🤖 AI Prediction</div>
      <div className={`text-xl font-bold my-1 ${color}`}>{label}</div>
      <div className="flex gap-1 items-center text-[10px]">
        <span className={`px-1.5 py-0.5 rounded ${confBg}`}>{confLabel} confidence</span>
        <span className="px-1.5 py-0.5 bg-surface-muted rounded text-muted">{horizon ?? 5}d</span>
      </div>
      <div className="font-mono text-xs mt-2 text-foreground">
        P(up) {probUp != null ? Number(probUp).toFixed(2) : "—"}
      </div>
    </div>
  );
}
