import { Button } from "../../components/ui/button";
import { Badge } from "../../components/ui/badge";
import { LoadingOverlay } from "../../components/LoadingSpinner";
import EmptyState from "../../components/EmptyState";
import ErrorState from "../../components/ErrorState";
import ConfirmDialog from "../../components/ConfirmDialog";
import { useModels, usePromoteModel, usePromotedModel } from "../../api/queries";
import { fmtDate } from "../../lib/formatters";
import { Trophy } from "lucide-react";

export default function ModelsTable() {
  const { data, isLoading, error, refetch } = useModels();
  const { data: promoted } = usePromotedModel();
  const promote = usePromoteModel();

  if (isLoading) return <LoadingOverlay label="Loading models..." />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;

  const models = data?.models || data || [];
  if (models.length === 0) return <EmptyState title="No models registered" description="Finish a training run to register a model." />;

  const promotedId = promoted?.model_id;

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase text-muted">
          <tr>
            <th className="px-4 py-3 text-left">Model ID</th>
            <th className="px-4 py-3 text-left">Type</th>
            <th className="px-4 py-3 text-left">Trained</th>
            <th className="px-4 py-3 text-right">Val AUC</th>
            <th className="px-4 py-3 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const mid = m.model_id || m.id;
            const isPromoted = mid === promotedId;
            return (
              <tr key={mid} className={`border-t border-surface-border hover:bg-surface-hover ${isPromoted ? "bg-accent/5" : ""}`}>
                <td className="px-4 py-3 font-mono text-xs">
                  {mid}
                  {isPromoted && (
                    <Badge variant="success" className="ml-2">
                      <Trophy className="h-3 w-3 mr-1" /> Promoted
                    </Badge>
                  )}
                </td>
                <td className="px-4 py-3">{m.model_type || m.type || "—"}</td>
                <td className="px-4 py-3 text-muted">{fmtDate(m.created_at)}</td>
                <td className="px-4 py-3 text-right font-mono">
                  {m.metrics?.val_auc != null ? m.metrics.val_auc.toFixed(4) : "—"}
                </td>
                <td className="px-4 py-3 text-right">
                  {!isPromoted && (
                    <ConfirmDialog
                      trigger={<Button size="sm" variant="outline">Promote</Button>}
                      title="Promote this model?"
                      description="The promoted model is used by default for predictions. Any current promoted model will be demoted."
                      onConfirm={() => promote.mutate(mid)}
                    />
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
