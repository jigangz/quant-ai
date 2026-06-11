const PRIMARY_METRIC = {
  direction: "auc",
  volatility: "qlike",
  meta_label: "auc_mean",
};

function cellColor(target, value, baseline) {
  // null = metric absent (backend reports honest n/a, never a fake 0.0)
  if (value == null || baseline == null) return "bg-surface-muted";
  const isLowerBetter = target === "volatility";
  const better = isLowerBetter ? value < baseline : value > baseline;
  return better ? "bg-up/15" : "bg-down/10";
}

const MOCK_SENTIMENT_HINT =
  "Sentiment features come from a mock provider in this build — deltas reflect pipeline wiring, not real news signal.";

export default function AblationMatrix({ matrix }) {
  if (!matrix || Object.keys(matrix).length === 0) {
    return (
      <div className="p-8 text-center text-sm text-muted bg-surface-card rounded-lg">
        Run an ablation to see the matrix.
      </div>
    );
  }
  const targets = Object.keys(matrix);
  const featureSetNames = Object.keys(matrix[targets[0]] || {});

  return (
    <div className="overflow-x-auto bg-surface-card rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-muted border-b border-surface-border">
            <th className="px-3 py-2 text-left">Target</th>
            {featureSetNames.map((fs) => (
              <th key={fs} className="px-3 py-2 text-left">
                {fs}
                {/sentiment/i.test(fs) && (
                  <span title={MOCK_SENTIMENT_HINT} aria-label="mock sentiment note" className="ml-1 cursor-help">
                    ℹ️
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {targets.map((target) => {
            const primary = PRIMARY_METRIC[target];
            const baseline = matrix[target][featureSetNames[0]]?.[primary];
            return (
              <tr key={target} className="border-b border-surface-border/50">
                <td className="px-3 py-2 font-medium uppercase">{target}</td>
                {featureSetNames.map((fs) => {
                  const cell = matrix[target][fs] || {};
                  if (cell.error) {
                    return (
                      <td key={fs} className="px-3 py-2 bg-down/10">
                        <div className="text-xs text-down">{cell.error}</div>
                      </td>
                    );
                  }
                  const v = cell[primary];
                  return (
                    <td key={fs} className={`px-3 py-2 ${cellColor(target, v, baseline)}`}>
                      <div className="font-semibold tabular-nums">
                        {v != null ? v.toFixed(3) : <span className="text-muted">n/a</span>}
                      </div>
                      {cell[`delta_${primary}`] != null && (
                        <div className="text-[10px] text-muted">
                          Δ {cell[`delta_${primary}`] >= 0 ? "+" : ""}
                          {cell[`delta_${primary}`].toFixed(3)}
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
