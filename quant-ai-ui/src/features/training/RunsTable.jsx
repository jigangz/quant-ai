import { useRuns } from "../../api/queries";
import { Badge } from "../../components/ui/badge";
import { LoadingOverlay } from "../../components/LoadingSpinner";
import EmptyState from "../../components/EmptyState";
import ErrorState from "../../components/ErrorState";
import { fmtDatetime } from "../../lib/formatters";

const STATUS_VARIANT = { success: "success", failed: "destructive", running: "info", pending: "warning" };

export default function RunsTable() {
  const { data, isLoading, error, refetch } = useRuns(20);
  if (isLoading) return <LoadingOverlay label="Loading runs..." />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;
  if (!data || data.length === 0) return <EmptyState title="No runs yet" description="Train a model to see runs here." />;

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase text-muted">
          <tr>
            <th className="px-4 py-3 text-left">Run ID</th>
            <th className="px-4 py-3 text-left">Model</th>
            <th className="px-4 py-3 text-left">Status</th>
            <th className="px-4 py-3 text-left">Started</th>
            <th className="px-4 py-3 text-right">Val AUC</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r) => (
            <tr key={r.run_id} className="border-t border-surface-border hover:bg-surface-hover">
              <td className="px-4 py-3 font-mono text-xs">{r.run_id}</td>
              <td className="px-4 py-3">{r.model_type || "—"}</td>
              <td className="px-4 py-3">
                <Badge variant={STATUS_VARIANT[r.status] || "secondary"}>{r.status}</Badge>
              </td>
              <td className="px-4 py-3 text-muted">{fmtDatetime(r.started_at || r.created_at)}</td>
              <td className="px-4 py-3 text-right font-mono">
                {r.metrics?.val_auc != null ? r.metrics.val_auc.toFixed(4) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
