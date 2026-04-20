export function Gauge({ label, score, scoreLabel, emphasized = false }) {
  const clamped = Math.max(-2, Math.min(2, score ?? 0));
  const normalized = (clamped + 2) / 4;
  const endAngle = Math.PI * normalized;
  const r = 40;
  const cx = 50;
  const cy = 55;
  const endX = cx - r * Math.cos(endAngle);
  const endY = cy - r * Math.sin(endAngle);
  const color = clamped >= 1.5 ? "rgb(var(--color-up))" : clamped >= 0.5 ? "rgb(5 150 105 / 0.65)" : clamped >= -0.5 ? "rgb(var(--color-text-muted))" : clamped >= -1.5 ? "rgb(225 29 72 / 0.65)" : "rgb(var(--color-down))";
  const labelColor = clamped >= 0.5 ? "text-up" : clamped <= -0.5 ? "text-down" : "text-muted";
  return (
    <div className={`text-center ${emphasized ? "bg-accent/5 rounded p-2" : ""}`}>
      <div className="text-[10px] text-muted mb-1.5">{label}</div>
      <svg viewBox="0 0 100 60" className="w-full max-w-[160px] mx-auto" role="meter" aria-valuemin={-2} aria-valuemax={2} aria-valuenow={clamped} aria-label={label}>
        <path d="M 10 55 A 40 40 0 0 1 90 55" stroke="rgb(var(--color-border))" strokeWidth="8" fill="none" />
        <path d={`M 10 55 A 40 40 0 0 1 ${endX.toFixed(2)} ${endY.toFixed(2)}`} stroke={color} strokeWidth="8" fill="none" />
        <line x1={cx} y1={cy} x2={endX} y2={endY} stroke="rgb(var(--color-text-primary))" strokeWidth="1.5" />
        <circle cx={cx} cy={cy} r="3" fill="rgb(var(--color-text-primary))" />
      </svg>
      <div className={`text-sm font-bold mt-1 ${labelColor}`}>{scoreLabel}</div>
    </div>
  );
}
