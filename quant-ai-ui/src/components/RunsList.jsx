export default function RunsList({ runs = [], onRefresh }) {
  const statusEmoji = {
    pending: "⏳",
    running: "🔄",
    completed: "✅",
    failed: "❌",
  };

  const statusColor = {
    pending: "#ffc107",
    running: "#17a2b8",
    completed: "#28a745",
    failed: "#dc3545",
  };

  return (
    <div>
      <div style={styles.header}>
        <h3>Training Runs</h3>
        <button onClick={onRefresh} style={styles.refreshBtn}>
          🔄 Refresh
        </button>
      </div>

      {runs.length === 0 ? (
        <p style={styles.empty}>No training runs yet. Start a new training!</p>
      ) : (
        <div style={styles.list}>
          {runs.map((run) => (
            <div key={run.job_id} style={styles.run}>
              {/* Status Badge */}
              <div
                style={{
                  ...styles.status,
                  background: statusColor[run.status] || "#ccc",
                }}
              >
                {statusEmoji[run.status] || "?"} {run.status}
              </div>

              {/* Details */}
              <div style={styles.details}>
                <div style={styles.id}>
                  ID: {run.job_id?.substring(0, 8)}...
                </div>
                <div>
                  <strong>{run.model_type}</strong> on{" "}
                  {run.tickers?.join(", ") || "?"}
                </div>
                <div style={styles.meta}>
                  Features: {run.feature_groups?.join(", ") || "?"}
                </div>
              </div>

              {/* Metrics */}
              {run.status === "completed" && run.metrics && (
                <div style={styles.metrics}>
                  {Object.entries(run.metrics)
                    .filter(([k]) => k.includes("val"))
                    .slice(0, 3)
                    .map(([k, v]) => (
                      <div key={k} style={styles.metric}>
                        <span style={styles.metricLabel}>
                          {k.replace("val_", "")}
                        </span>
                        <span style={styles.metricValue}>
                          {typeof v === "number" ? v.toFixed(4) : v}
                        </span>
                      </div>
                    ))}
                </div>
              )}

              {/* Error */}
              {run.status === "failed" && run.error && (
                <div style={styles.error}>{run.error}</div>
              )}

              {/* Timing */}
              {run.training_time_seconds > 0 && (
                <div style={styles.time}>
                  ⏱️ {run.training_time_seconds.toFixed(1)}s
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  refreshBtn: {
    background: "none",
    border: "1px solid #ddd",
    padding: "4px 8px",
    borderRadius: 4,
    cursor: "pointer",
  },
  empty: {
    color: "#666",
    fontStyle: "italic",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  run: {
    background: "#fff",
    border: "1px solid #eee",
    borderRadius: 8,
    padding: 16,
    display: "flex",
    flexWrap: "wrap",
    gap: 16,
    alignItems: "flex-start",
  },
  status: {
    padding: "4px 8px",
    borderRadius: 4,
    color: "white",
    fontSize: 12,
    fontWeight: "bold",
  },
  details: {
    flex: 1,
    minWidth: 200,
  },
  id: {
    fontSize: 11,
    color: "#999",
    fontFamily: "monospace",
  },
  meta: {
    fontSize: 12,
    color: "#666",
    marginTop: 4,
  },
  metrics: {
    display: "flex",
    gap: 12,
  },
  metric: {
    background: "#f0f8ff",
    padding: "4px 8px",
    borderRadius: 4,
    fontSize: 12,
  },
  metricLabel: {
    color: "#666",
    marginRight: 4,
  },
  metricValue: {
    fontWeight: "bold",
    color: "#007bff",
  },
  error: {
    background: "#fee",
    color: "#c00",
    padding: 8,
    borderRadius: 4,
    fontSize: 12,
    width: "100%",
  },
  time: {
    fontSize: 12,
    color: "#666",
  },
};
