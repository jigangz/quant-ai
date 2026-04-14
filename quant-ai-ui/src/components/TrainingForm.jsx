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

    if (formData.start_date) payload.start_date = formData.start_date;
    if (formData.end_date) payload.end_date = formData.end_date;

    onSubmit(payload);
  }

  const inputClass = "block w-full mt-1 px-3 py-2 bg-surface-card border border-gray-700 text-white rounded text-sm focus:outline-none focus:border-accent";

  return (
    <form onSubmit={handleSubmit} className="bg-surface-card rounded-lg p-6 max-w-2xl">
      <h3 className="text-white font-medium mb-4">Train New Model</h3>

      {/* Tickers */}
      <div className="mb-4">
        <label className="text-gray-400 text-sm">Tickers (comma-separated)</label>
        <input
          type="text"
          name="tickers"
          value={formData.tickers}
          onChange={handleChange}
          placeholder="AAPL, MSFT, GOOGL"
          className={inputClass}
        />
      </div>

      {/* Date Range */}
      <div className="flex gap-4 mb-4">
        <div className="flex-1">
          <label className="text-gray-400 text-sm">Start Date (optional)</label>
          <input
            type="date"
            name="start_date"
            value={formData.start_date}
            onChange={handleChange}
            className={inputClass}
          />
        </div>
        <div className="flex-1">
          <label className="text-gray-400 text-sm">End Date (optional)</label>
          <input
            type="date"
            name="end_date"
            value={formData.end_date}
            onChange={handleChange}
            className={inputClass}
          />
        </div>
      </div>

      {/* Model Type */}
      <div className="mb-4">
        <label className="text-gray-400 text-sm">Model Type</label>
        <select
          name="model_type"
          value={formData.model_type}
          onChange={handleChange}
          className={inputClass}
        >
          {modelTypes.map((m) => (
            <option key={m.type} value={m.type}>
              {m.type} ({m.class_name})
            </option>
          ))}
        </select>
      </div>

      {/* Feature Groups */}
      <div className="mb-4">
        <label className="text-gray-400 text-sm">Feature Groups</label>
        <div className="mt-2 space-y-2">
          {featureGroups.map((g) => (
            <label key={g.name} className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                name="feature_groups"
                value={g.name}
                checked={formData.feature_groups.includes(g.name)}
                onChange={handleChange}
                className="accent-accent"
              />
              <span className="text-white text-sm">{g.name}</span>
              <span className="text-gray-500 text-xs">— {g.description}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Horizon */}
      <div className="mb-4">
        <label className="text-gray-400 text-sm">Prediction Horizon (days)</label>
        <input
          type="number"
          name="horizon_days"
          value={formData.horizon_days}
          onChange={handleChange}
          min={1}
          max={60}
          className="mt-1 w-24 px-3 py-2 bg-surface-card border border-gray-700 text-white rounded text-sm focus:outline-none focus:border-accent"
        />
      </div>

      {/* Hyperparameter Search */}
      <div className="mb-6">
        <label className="text-gray-400 text-sm">Hyperparameter Search</label>
        <div className="flex gap-3 mt-1">
          <select
            name="search_mode"
            value={formData.search_mode}
            onChange={handleChange}
            className="w-32 px-3 py-2 bg-surface-card border border-gray-700 text-white rounded text-sm focus:outline-none focus:border-accent"
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
              className="w-20 px-3 py-2 bg-surface-card border border-gray-700 text-white rounded text-sm focus:outline-none focus:border-accent"
            />
          )}
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className="bg-accent hover:opacity-90 disabled:opacity-50 text-white px-6 py-3 rounded text-base font-medium w-full"
      >
        {loading ? "Training..." : "Start Training"}
      </button>
    </form>
  );
}
