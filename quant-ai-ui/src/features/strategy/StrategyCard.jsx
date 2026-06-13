import MetaLabelCoverageBadge from "@/components/MetaLabelCoverageBadge";

export default function StrategyCard({ strategy }) {
  if (!strategy) return null;
  return (
    <div className="p-4 bg-surface-muted rounded-lg border border-surface-border">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold text-foreground">{strategy.name}</h3>
        <MetaLabelCoverageBadge strategyName={strategy.name} />
      </div>
      {strategy.description && (
        <p className="text-sm text-muted">{strategy.description}</p>
      )}
      {strategy.version && (
        <div className="mt-2 text-xs text-muted">v{strategy.version}</div>
      )}
    </div>
  );
}
