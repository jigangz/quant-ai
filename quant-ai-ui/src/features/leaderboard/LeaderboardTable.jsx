import { useModelAccuracy } from "@/api/leaderboardQueries";

const PRIMARY_METRIC_KEY = {
  direction: "test_auc",
  volatility: "test_qlike",
  meta_label: "cv_auc_mean",
};

const PRIMARY_METRIC_LABEL = {
  direction: "AUC",
  volatility: "QLIKE",
  meta_label: "CV AUC",
};

function AccuracyCell({ modelId }) {
  const { data, isLoading } = useModelAccuracy(modelId);
  if (isLoading) return <span className="text-slate-500">…</span>;
  const hit = data?.stats?.hit_rate;
  if (hit === null || hit === undefined) return <span className="text-slate-600">—</span>;
  return <span className="text-slate-200">{(hit * 100).toFixed(0)}%</span>;
}

export default function LeaderboardTable({ models, labelType }) {
  if (!models || models.length === 0) {
    return (
      <div className="p-8 text-center text-sm text-slate-500 bg-slate-900/40 rounded-lg">
        No active {labelType} models trained yet.
      </div>
    );
  }
  const metricKey = PRIMARY_METRIC_KEY[labelType];
  const sorted = [...models].sort((a, b) => {
    const av = a.metrics?.[metricKey] ?? 0;
    const bv = b.metrics?.[metricKey] ?? 0;
    // QLIKE: lower is better
    return labelType === "volatility" ? av - bv : bv - av;
  });
  return (
    <div className="overflow-x-auto bg-slate-900/40 rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-slate-400 border-b border-slate-800">
            <th className="px-3 py-2 text-left">Model</th>
            <th className="px-3 py-2 text-left">Type</th>
            <th className="px-3 py-2 text-left">Tickers</th>
            <th className="px-3 py-2 text-right">{PRIMARY_METRIC_LABEL[labelType]}</th>
            <th className="px-3 py-2 text-right">Live hit rate (30d)</th>
            <th className="px-3 py-2 text-left">Created</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m) => (
            <tr key={m.id} className="border-b border-slate-800/50 hover:bg-slate-800/40">
              <td className="px-3 py-2 font-medium">{m.name || m.id}</td>
              <td className="px-3 py-2 text-slate-400">{m.model_type}</td>
              <td className="px-3 py-2 text-slate-400">{(m.tickers || []).join(", ")}</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {(m.metrics?.[metricKey] ?? 0).toFixed(3)}
              </td>
              <td className="px-3 py-2 text-right">
                <AccuracyCell modelId={m.id} />
              </td>
              <td className="px-3 py-2 text-slate-500 text-xs">
                {(m.created_at || "").slice(0, 10)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
