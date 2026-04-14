export default function RunsList({ runs = [], onRefresh }) {
  const statusEmoji = {
    pending: "⏳",
    running: "🔄",
    completed: "✅",
    failed: "❌",
  };

  const statusBadgeClass = {
    pending: "bg-yellow-600 text-white",
    running: "bg-blue-600 text-white",
    completed: "bg-green-700 text-white",
    failed: "bg-red-700 text-white",
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-white font-medium">Training Runs</h3>
        <button
          onClick={onRefresh}
          className="border border-gray-600 text-gray-300 hover:text-white px-3 py-1 rounded text-sm"
        >
          Refresh
        </button>
      </div>

      {runs.length === 0 ? (
        <p className="text-gray-500 italic text-sm">No training runs yet. Start a new training!</p>
      ) : (
        <div className="flex flex-col gap-3">
          {runs.map((run) => (
            <div
              key={run.job_id}
              className="bg-surface-card border border-gray-700 rounded-lg p-4 flex flex-wrap gap-4 items-start"
            >
              {/* Status Badge */}
              <span className={`px-2 py-1 rounded text-xs font-bold ${statusBadgeClass[run.status] || "bg-gray-600 text-white"}`}>
                {statusEmoji[run.status] || "?"} {run.status}
              </span>

              {/* Details */}
              <div className="flex-1 min-w-[200px]">
                <div className="text-gray-500 text-xs font-mono">
                  ID: {run.job_id?.substring(0, 8)}...
                </div>
                <div className="text-white text-sm mt-1">
                  <strong>{run.model_type}</strong> on{" "}
                  {run.tickers?.join(", ") || "?"}
                </div>
                <div className="text-gray-400 text-xs mt-1">
                  Features: {run.feature_groups?.join(", ") || "?"}
                </div>
              </div>

              {/* Metrics */}
              {run.status === "completed" && run.metrics && (
                <div className="flex gap-3">
                  {Object.entries(run.metrics)
                    .filter(([k]) => k.includes("val"))
                    .slice(0, 3)
                    .map(([k, v]) => (
                      <div key={k} className="bg-blue-900/30 px-2 py-1 rounded text-xs">
                        <span className="text-gray-400 mr-1">{k.replace("val_", "")}</span>
                        <span className="font-bold text-accent">
                          {typeof v === "number" ? v.toFixed(4) : v}
                        </span>
                      </div>
                    ))}
                </div>
              )}

              {/* Error */}
              {run.status === "failed" && run.error && (
                <div className="bg-red-900/30 text-red-300 px-3 py-2 rounded text-xs w-full">
                  {run.error}
                </div>
              )}

              {/* Timing */}
              {run.training_time_seconds > 0 && (
                <div className="text-gray-500 text-xs">
                  ⏱ {run.training_time_seconds.toFixed(1)}s
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
