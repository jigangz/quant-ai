const PRIMARY_METRIC = {
  direction: "auc",
  volatility: "qlike",
  meta_label: "auc_mean",
};

function cellColor(target, value, baseline) {
  if (value === undefined || baseline === undefined) return "bg-slate-700/30";
  const isLowerBetter = target === "volatility";
  const better = isLowerBetter ? value < baseline : value > baseline;
  return better ? "bg-emerald-500/15" : "bg-amber-500/15";
}

export default function AblationMatrix({ matrix }) {
  if (!matrix || Object.keys(matrix).length === 0) {
    return (
      <div className="p-8 text-center text-sm text-slate-500 bg-slate-900/40 rounded-lg">
        Run an ablation to see the matrix.
      </div>
    );
  }
  const targets = Object.keys(matrix);
  const featureSetNames = Object.keys(matrix[targets[0]] || {});

  return (
    <div className="overflow-x-auto bg-slate-900/40 rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-slate-400 border-b border-slate-800">
            <th className="px-3 py-2 text-left">Target</th>
            {featureSetNames.map((fs) => (
              <th key={fs} className="px-3 py-2 text-left">{fs}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {targets.map((target) => {
            const primary = PRIMARY_METRIC[target];
            const baseline = matrix[target][featureSetNames[0]]?.[primary];
            return (
              <tr key={target} className="border-b border-slate-800/50">
                <td className="px-3 py-2 font-medium uppercase">{target}</td>
                {featureSetNames.map((fs) => {
                  const cell = matrix[target][fs] || {};
                  if (cell.error) {
                    return (
                      <td key={fs} className="px-3 py-2 bg-rose-500/15">
                        <div className="text-xs text-rose-300">{cell.error}</div>
                      </td>
                    );
                  }
                  const v = cell[primary];
                  return (
                    <td key={fs} className={`px-3 py-2 ${cellColor(target, v, baseline)}`}>
                      <div className="font-semibold tabular-nums">
                        {v !== undefined ? v.toFixed(3) : "—"}
                      </div>
                      {cell[`delta_${primary}`] !== undefined && (
                        <div className="text-[10px] text-slate-400">
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
