import { useNavigate } from "react-router-dom";
import { useMetaCoverage } from "@/api/signalQueries";

export default function MetaLabelCoverageBadge({ strategyName }) {
  const { data, isLoading, isError } = useMetaCoverage(strategyName);
  const navigate = useNavigate();

  if (isLoading || isError) return null;
  if (!data || data.count === 0) return null;

  const avg = data.avg_auc ?? 0;
  const variant = avg >= 0.60 ? "good" : avg >= 0.50 ? "neutral" : "warn";
  const warnMark = variant === "warn" ? " ⚠" : "";

  const onClick = () => navigate(`/signal-console?strategy=${encodeURIComponent(strategyName)}`);

  const bg = variant === "good" ? "bg-emerald-500/15 text-emerald-400"
    : variant === "warn" ? "bg-amber-500/15 text-amber-400"
    : "bg-surface-muted text-muted";

  return (
    <button
      type="button"
      onClick={onClick}
      data-variant={variant}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium ${bg}`}
      title={`${data.count} meta-model${data.count > 1 ? "s" : ""} · avg AUC ${avg.toFixed(2)} · tickers: ${data.tickers.join(", ")}`}
    >
      Meta ✓ {data.count} · AUC {data.max_auc.toFixed(2)}{warnMark}
    </button>
  );
}
