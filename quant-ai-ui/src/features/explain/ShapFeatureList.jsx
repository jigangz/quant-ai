export default function ShapFeatureList({ features }) {
  if (!features || features.length === 0) {
    return <p className="text-sm text-muted">No SHAP data available.</p>;
  }
  const max = features[0].mean_abs_shap;

  return (
    <div className="space-y-3">
      {features.map((f, i) => {
        const pct = (f.mean_abs_shap / max) * 100;
        return (
          <div key={i}>
            <div className="flex justify-between text-sm mb-1.5">
              <span className="font-mono font-medium text-foreground">{f.feature}</span>
              <span className="text-muted tabular-nums">{f.mean_abs_shap.toFixed(4)}</span>
            </div>
            <div className="h-2 bg-surface-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-accent transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
