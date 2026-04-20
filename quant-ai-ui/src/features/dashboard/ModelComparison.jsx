export function ModelComparison({ models = [], promotedId }) {
  if (!models.length) {
    return (
      <section className="mb-4">
        <h3 className="text-sm font-bold text-foreground">历史模型对此股的预测 ›</h3>
        <p className="text-[10px] text-muted mb-2">你训练过的 / 线上其他模型对此股的预测对比</p>
        <div className="border border-surface-border rounded-md p-6 text-center text-xs text-muted">
          首次训练后开启
        </div>
      </section>
    );
  }
  return (
    <section className="mb-4">
      <h3 className="text-sm font-bold text-foreground">历史模型对此股的预测 ›</h3>
      <p className="text-[10px] text-muted mb-2">你训练过的 / 线上其他模型对此股的预测对比</p>
      <div className="grid grid-cols-4 gap-2">
        {models.slice(0, 4).map((m) => {
          const isPromoted = m.id === promotedId;
          const auc = m.metrics?.val_auc ?? m.metrics?.test_auc;
          const accuracy = m.metrics?.accuracy ?? m.metrics?.val_accuracy;
          return (
            <div key={m.id} className="border border-surface-border rounded-md overflow-hidden">
              <div className="h-[50px] bg-gradient-to-r from-surface-muted via-up/10 to-down/10 p-2 text-[9px] text-muted">
                📈 sparkline
              </div>
              <div className="p-2">
                <div className="text-[10.5px] font-bold text-foreground">
                  {m.name ?? m.model_type} {isPromoted && <span className="text-warn">⭐ 当前</span>}
                </div>
                <div className="text-[9px] text-muted">AUC {auc?.toFixed(2) ?? "—"} · run #{m.training_run_id ?? "—"}</div>
                <div className={`text-[9px] mt-1 ${accuracy > 0.55 ? "text-up" : accuracy ? "text-down" : "text-muted"}`}>
                  {accuracy ? `${accuracy > 0.55 ? "✓" : ""} 准确率 ${Math.round(accuracy * 100)}%` : "数据积累中"}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
