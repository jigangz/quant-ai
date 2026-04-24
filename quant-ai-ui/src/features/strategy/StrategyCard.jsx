import MetaLabelCoverageBadge from "@/components/MetaLabelCoverageBadge";

export default function StrategyCard({ strategy }) {
  if (!strategy) return null;
  return (
    <div className="p-4 bg-slate-900/40 rounded-lg border border-slate-800">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold text-slate-100">{strategy.name}</h3>
        <MetaLabelCoverageBadge strategyName={strategy.name} />
      </div>
      {strategy.description && (
        <p className="text-sm text-slate-400">{strategy.description}</p>
      )}
      {strategy.version && (
        <div className="mt-2 text-xs text-slate-500">v{strategy.version}</div>
      )}
    </div>
  );
}
