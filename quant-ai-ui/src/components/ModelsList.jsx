export default function ModelsList({
  models = [],
  promotedId,
  onPromote,
  onRefresh,
}) {
  return (
    <div>
      <div style={styles.header}>
        <h3>Registered Models</h3>
        <button onClick={onRefresh} style={styles.refreshBtn}>
          🔄 Refresh
        </button>
      </div>

      {/* Promoted Badge */}
      {promotedId && (
        <div style={styles.promoted}>
          🏆 <strong>Production Model:</strong> {promotedId.substring(0, 8)}...
        </div>
      )}

      {models.length === 0 ? (
        <p style={styles.empty}>No models registered. Train one first!</p>
      ) : (
        <table style={styles.table}>
          <thead>
            <tr>
              <th>Status</th>
              <th>Name</th>
              <th>Type</th>
              <th>Tickers</th>
              <th>Metrics</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => {
              const isPromoted = model.id === promotedId;
              return (
                <tr
                  key={model.id}
                  style={isPromoted ? styles.promotedRow : {}}
                >
                  {/* Status */}
                  <td>
                    {isPromoted ? (
                      <span style={styles.badge}>🏆 PROD</span>
                    ) : (
                      <span style={styles.activeBadge}>Active</span>
                    )}
                  </td>

                  {/* Name */}
                  <td>
                    <div style={styles.name}>{model.name}</div>
                    <div style={styles.id}>{model.id?.substring(0, 8)}...</div>
                  </td>

                  {/* Type */}
                  <td>
                    <span style={styles.modelType}>{model.model_type}</span>
                  </td>

                  {/* Tickers */}
                  <td>{model.tickers?.join(", ")}</td>

                  {/* Metrics */}
                  <td>
                    {model.metrics && (
                      <div style={styles.metrics}>
                        {model.metrics.val_auc && (
                          <div>
                            AUC: <strong>{model.metrics.val_auc}</strong>
                          </div>
                        )}
                        {model.metrics.val_f1 && (
                          <div>
                            F1: <strong>{model.metrics.val_f1}</strong>
                          </div>
                        )}
                      </div>
                    )}
                  </td>

                  {/* Actions */}
                  <td>
                    {!isPromoted && (
                      <button
                        onClick={() => onPromote(model.id)}
                        style={styles.promoteBtn}
                      >
                        ⬆️ Promote
                      </button>
                    )}
                    {isPromoted && (
                      <span style={styles.prodLabel}>In Production</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
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
  promoted: {
    background: "#fff3cd",
    border: "1px solid #ffc107",
    padding: 12,
    borderRadius: 4,
    marginBottom: 16,
  },
  empty: {
    color: "#666",
    fontStyle: "italic",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 14,
  },
  promotedRow: {
    background: "#fff8e1",
  },
  badge: {
    background: "#ffc107",
    color: "#000",
    padding: "2px 6px",
    borderRadius: 4,
    fontSize: 11,
    fontWeight: "bold",
  },
  activeBadge: {
    background: "#e8f5e9",
    color: "#2e7d32",
    padding: "2px 6px",
    borderRadius: 4,
    fontSize: 11,
  },
  name: {
    fontWeight: 500,
  },
  id: {
    fontSize: 10,
    color: "#999",
    fontFamily: "monospace",
  },
  modelType: {
    background: "#e3f2fd",
    padding: "2px 6px",
    borderRadius: 4,
    fontSize: 12,
  },
  metrics: {
    fontSize: 12,
  },
  promoteBtn: {
    background: "#28a745",
    color: "white",
    border: "none",
    padding: "4px 8px",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 12,
  },
  prodLabel: {
    color: "#666",
    fontSize: 11,
  },
};
