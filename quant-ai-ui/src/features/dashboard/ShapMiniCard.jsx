export function ShapMiniCard({ features }) {
  if (!features || features.length === 0) {
    return (
      <div className="bg-surface border border-surface-border rounded-md p-3">
        <div className="text-[9px] uppercase tracking-wide text-muted">📊 SHAP Top 3</div>
        <div className="text-xs text-muted mt-2">SHAP 未安装或不可用</div>
      </div>
    );
  }
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.contribution)), 0.01);
  return (
    <div className="bg-surface border border-surface-border rounded-md p-3">
      <div className="text-[9px] uppercase tracking-wide text-muted mb-2">📊 SHAP Top 3</div>
      <div className="flex flex-col gap-1">
        {features.slice(0, 3).map((f) => {
          const pct = Math.round((Math.abs(f.contribution) / maxAbs) * 100);
          const signed = (f.contribution >= 0 ? "+" : "") + Math.round(f.contribution * 100) + "%";
          const color = f.contribution >= 0 ? "bg-up" : "bg-down";
          return (
            <div key={f.name} className="flex items-center gap-1 text-[10px]">
              <span className="w-10 text-foreground truncate">{f.name}</span>
              <div className="flex-1 h-2.5 bg-surface-muted rounded">
                <div className={`h-full rounded ${color}`} style={{ width: `${pct}%` }} />
              </div>
              <span className="w-10 text-right font-mono text-foreground">{signed}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
