const MONTH_LABELS = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"];

function bandColor(acc) {
  if (acc == null) return "bg-surface-muted text-muted";
  if (acc >= 0.6) return "bg-up/15 text-up";
  if (acc >= 0.5) return "bg-warn/15 text-warn";
  return "bg-down/15 text-down";
}

export function SeasonalityHeatmap({ monthly = null }) {
  if (monthly == null) {
    return (
      <section className="mb-4">
        <h3 className="text-sm font-bold text-foreground">季节性 ›</h3>
        <p className="text-[10px] text-muted mb-2">过去模型在这个月份的预测准确率</p>
        <div className="border border-surface-border rounded-md p-6 text-center text-xs text-muted">
          数据积累中 — 首次预测落实后开启
        </div>
      </section>
    );
  }
  return (
    <section className="mb-4">
      <h3 className="text-sm font-bold text-foreground">季节性 ›</h3>
      <p className="text-[10px] text-muted mb-2">过去模型在这个月份的预测准确率</p>
      <div className="grid grid-cols-12 gap-0.5 text-[9px] text-center">
        {MONTH_LABELS.map((label, i) => {
          const entry = monthly.find((m) => m.month === i + 1);
          const acc = entry?.accuracy;
          return (
            <div key={label} className={`py-2 rounded ${bandColor(acc)}`}>
              <div>{label}</div>
              <div>{acc != null ? Math.round(acc * 100) + "%" : "—"}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
