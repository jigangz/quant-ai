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

  const [ensembleConfig, setEnsembleConfig] = useState({
    mode: "voting_soft",
    base_models: ["logistic", "random_forest"],
    cv_folds: 5,
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
    if (formData.model_type === "ensemble") payload.ensemble_config = ensembleConfig;

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
          <option value="ensemble">ensemble (Ensemble — combine multiple models)</option>
        </select>
      </div>

      {/* Ensemble Config (shown only when ensemble selected) */}
      {formData.model_type === "ensemble" && (
        <div className="mb-4 p-4 rounded bg-surface-card border border-gray-700">
          <h3 className="text-sm font-semibold mb-3 text-accent">Ensemble Configuration</h3>

          {/* Mode */}
          <div className="mb-3">
            <label className="block text-sm font-medium text-gray-400 mb-1">Mode</label>
            <select
              className={inputClass}
              value={ensembleConfig.mode}
              onChange={(e) => setEnsembleConfig({ ...ensembleConfig, mode: e.target.value })}
            >
              <option value="voting_soft">Voting — Soft (average probabilities)</option>
              <option value="voting_hard">Voting — Hard (majority vote)</option>
              <option value="stacking_logistic">Stacking — Logistic meta-learner</option>
              <option value="stacking_xgboost">Stacking — XGBoost meta-learner</option>
            </select>
          </div>

          {/* Base Models */}
          <div className="mb-3">
            <label className="block text-sm font-medium text-gray-400 mb-1">Base Models (≥2)</label>
            <div className="flex flex-wrap gap-3 mt-1">
              {["logistic", "random_forest", "xgboost", "lightgbm", "catboost"].map((m) => (
                <label key={m} className="flex items-center gap-1 text-sm text-white cursor-pointer">
                  <input
                    type="checkbox"
                    className="accent-accent"
                    checked={ensembleConfig.base_models.includes(m)}
                    onChange={(e) => {
                      const next = e.target.checked
                        ? [...ensembleConfig.base_models, m]
                        : ensembleConfig.base_models.filter((x) => x !== m);
                      setEnsembleConfig({ ...ensembleConfig, base_models: next });
                    }}
                  />
                  {m}
                </label>
              ))}
            </div>
          </div>

          {/* CV Folds (stacking only) */}
          {ensembleConfig.mode.startsWith("stacking") && (
            <div className="mb-1">
              <label className="block text-sm font-medium text-gray-400 mb-1">CV Folds</label>
              <input
                type="number"
                min="2"
                max="10"
                className="w-24 px-3 py-2 bg-surface border border-gray-700 text-white rounded text-sm focus:outline-none focus:border-accent"
                value={ensembleConfig.cv_folds}
                onChange={(e) =>
                  setEnsembleConfig({ ...ensembleConfig, cv_folds: parseInt(e.target.value, 10) || 5 })
                }
              />
            </div>
          )}
        </div>
      )}

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
