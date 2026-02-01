import { useState } from "react";

export default function TrainingForm({
  modelTypes = [],
  featureGroups = [],
  onSubmit,
  loading,
}) {
  const [formData, setFormData] = useState({
    tickers: "AAPL",
    start_date: "",
    end_date: "",
    model_type: "logistic",
    feature_groups: ["ta_basic", "momentum"],
    horizon_days: 5,
    search_mode: "none",
    search_trials: 20,
  });

  function handleChange(e) {
    const { name, value, type, checked } = e.target;

    if (name === "feature_groups") {
      // Multi-select for feature groups
      const newGroups = checked
        ? [...formData.feature_groups, value]
        : formData.feature_groups.filter((g) => g !== value);
      setFormData({ ...formData, feature_groups: newGroups });
    } else {
      setFormData({ ...formData, [name]: value });
    }
  }

  function handleSubmit(e) {
    e.preventDefault();

    // Parse tickers
    const tickers = formData.tickers
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t);

    const payload = {
      tickers,
      model_type: formData.model_type,
      feature_groups: formData.feature_groups,
      horizon_days: parseInt(formData.horizon_days),
      search_mode: formData.search_mode,
      search_trials: parseInt(formData.search_trials),
    };

    // Add dates if provided
    if (formData.start_date) payload.start_date = formData.start_date;
    if (formData.end_date) payload.end_date = formData.end_date;

    onSubmit(payload);
  }

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <h3>Train New Model</h3>

      {/* Tickers */}
      <div style={styles.field}>
        <label>Tickers (comma-separated)</label>
        <input
          type="text"
          name="tickers"
          value={formData.tickers}
          onChange={handleChange}
          placeholder="AAPL, MSFT, GOOGL"
          style={styles.input}
        />
      </div>

      {/* Date Range */}
      <div style={styles.row}>
        <div style={styles.field}>
          <label>Start Date (optional)</label>
          <input
            type="date"
            name="start_date"
            value={formData.start_date}
            onChange={handleChange}
            style={styles.input}
          />
        </div>
        <div style={styles.field}>
          <label>End Date (optional)</label>
          <input
            type="date"
            name="end_date"
            value={formData.end_date}
            onChange={handleChange}
            style={styles.input}
          />
        </div>
      </div>

      {/* Model Type */}
      <div style={styles.field}>
        <label>Model Type</label>
        <select
          name="model_type"
          value={formData.model_type}
          onChange={handleChange}
          style={styles.input}
        >
          {modelTypes.map((m) => (
            <option key={m.type} value={m.type}>
              {m.type} ({m.class_name})
            </option>
          ))}
        </select>
      </div>

      {/* Feature Groups */}
      <div style={styles.field}>
        <label>Feature Groups</label>
        <div style={styles.checkboxGroup}>
          {featureGroups.map((g) => (
            <label key={g.name} style={styles.checkbox}>
              <input
                type="checkbox"
                name="feature_groups"
                value={g.name}
                checked={formData.feature_groups.includes(g.name)}
                onChange={handleChange}
              />
              {g.name}
              <span style={styles.desc}> - {g.description}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Horizon */}
      <div style={styles.field}>
        <label>Prediction Horizon (days)</label>
        <input
          type="number"
          name="horizon_days"
          value={formData.horizon_days}
          onChange={handleChange}
          min={1}
          max={60}
          style={{ ...styles.input, width: 100 }}
        />
      </div>

      {/* Hyperparameter Search */}
      <div style={styles.field}>
        <label>Hyperparameter Search</label>
        <div style={styles.row}>
          <select
            name="search_mode"
            value={formData.search_mode}
            onChange={handleChange}
            style={{ ...styles.input, width: 120 }}
          >
            <option value="none">None</option>
            <option value="grid">Grid Search</option>
            <option value="optuna">Optuna</option>
          </select>
          {formData.search_mode !== "none" && (
            <input
              type="number"
              name="search_trials"
              value={formData.search_trials}
              onChange={handleChange}
              min={5}
              max={100}
              placeholder="Trials"
              style={{ ...styles.input, width: 80 }}
            />
          )}
        </div>
      </div>

      {/* Submit */}
      <button type="submit" disabled={loading} style={styles.submit}>
        {loading ? "Training..." : "🚀 Start Training"}
      </button>
    </form>
  );
}

const styles = {
  form: {
    background: "#f9f9f9",
    padding: 24,
    borderRadius: 8,
    maxWidth: 600,
  },
  field: {
    marginBottom: 16,
  },
  row: {
    display: "flex",
    gap: 16,
  },
  input: {
    display: "block",
    width: "100%",
    padding: 8,
    marginTop: 4,
    border: "1px solid #ddd",
    borderRadius: 4,
    fontSize: 14,
  },
  checkboxGroup: {
    marginTop: 8,
  },
  checkbox: {
    display: "block",
    marginBottom: 4,
    cursor: "pointer",
  },
  desc: {
    color: "#666",
    fontSize: 12,
  },
  submit: {
    background: "#007bff",
    color: "white",
    border: "none",
    padding: "12px 24px",
    borderRadius: 4,
    fontSize: 16,
    cursor: "pointer",
    marginTop: 16,
  },
};
