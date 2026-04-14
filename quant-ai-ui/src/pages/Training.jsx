import { useState, useEffect } from "react";
import {
  train,
  listRuns,
  getRunStatus,
  listModels,
  promoteModel,
  getPromotedModel,
  listModelTypes,
  listFeatureGroups,
} from "../api/client";
import TrainingForm from "../components/TrainingForm";
import RunsList from "../components/RunsList";
import ModelsList from "../components/ModelsList";

export default function Training() {
  // State
  const [activeTab, setActiveTab] = useState("train");
  const [modelTypes, setModelTypes] = useState([]);
  const [featureGroups, setFeatureGroups] = useState([]);
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [promotedId, setPromotedId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load initial data
  useEffect(() => {
    loadModelTypes();
    loadFeatureGroups();
    loadRuns();
    loadModels();
    loadPromoted();
  }, []);

  async function loadModelTypes() {
    try {
      const data = await listModelTypes();
      setModelTypes(data.types || []);
    } catch (e) {
      // Fallback
      setModelTypes([
        { type: "logistic", class_name: "LogisticModel" },
        { type: "random_forest", class_name: "RandomForestModel" },
        { type: "xgboost", class_name: "XGBoostModel" },
      ]);
    }
  }

  async function loadFeatureGroups() {
    try {
      const data = await listFeatureGroups();
      setFeatureGroups(data.groups || []);
    } catch (e) {
      // Fallback
      setFeatureGroups([
        { name: "ta_basic", description: "Basic technical indicators" },
        { name: "momentum", description: "Momentum indicators" },
        { name: "volatility", description: "Volatility indicators" },
      ]);
    }
  }

  async function loadRuns() {
    try {
      const data = await listRuns(20);
      setRuns(data.runs || []);
    } catch (e) {
      // silent
    }
  }

  async function loadModels() {
    try {
      const data = await listModels("active", 50);
      setModels(data.models || []);
    } catch (e) {
      // silent
    }
  }

  async function loadPromoted() {
    try {
      const data = await getPromotedModel();
      setPromotedId(data.promoted_id);
    } catch (e) {
      // silent
    }
  }

  // Handle training submission
  async function handleTrain(formData) {
    setLoading(true);
    setError(null);

    try {
      const result = await train(formData);

      // If async, poll for status
      if (result.run_id) {
        pollRunStatus(result.run_id);
      }

      // Refresh lists
      setTimeout(loadRuns, 1000);
      setTimeout(loadModels, 5000);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  // Poll for run status
  async function pollRunStatus(runId) {
    const poll = async () => {
      try {
        const status = await getRunStatus(runId);

        // Update runs list
        setRuns((prev) => {
          const existing = prev.find((r) => r.job_id === runId);
          if (existing) {
            return prev.map((r) =>
              r.job_id === runId ? { ...r, ...status } : r
            );
          }
          return [status, ...prev];
        });

        // Continue polling if not done
        if (status.status === "pending" || status.status === "running") {
          setTimeout(poll, 2000);
        } else {
          // Done - refresh models
          loadModels();
        }
      } catch (e) {
        // silent
      }
    };

    poll();
  }

  // Handle promote
  async function handlePromote(modelId) {
    try {
      await promoteModel(modelId);
      setPromotedId(modelId);
      loadModels();
    } catch (e) {
      setError(e.message);
    }
  }

  const tabClass = (tab) =>
    activeTab === tab
      ? "px-4 py-2 bg-accent text-white rounded text-sm font-medium"
      : "px-4 py-2 text-gray-400 hover:text-white text-sm";

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold text-white mb-4">Training Panel</h1>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-700 pb-2">
        <button onClick={() => setActiveTab("train")} className={tabClass("train")}>
          Train New Model
        </button>
        <button onClick={() => setActiveTab("runs")} className={tabClass("runs")}>
          Training Runs ({runs.length})
        </button>
        <button onClick={() => setActiveTab("models")} className={tabClass("models")}>
          Models ({models.length})
        </button>
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-300 px-4 py-3 rounded mb-4 flex justify-between items-center">
          <span>{error}</span>
          <button
            onClick={() => setError(null)}
            className="text-red-300 hover:text-white text-xl leading-none ml-4"
          >
            &times;
          </button>
        </div>
      )}

      {/* Tab content */}
      {activeTab === "train" && (
        <TrainingForm
          modelTypes={modelTypes}
          featureGroups={featureGroups}
          onSubmit={handleTrain}
          loading={loading}
        />
      )}

      {activeTab === "runs" && (
        <RunsList runs={runs} onRefresh={loadRuns} />
      )}

      {activeTab === "models" && (
        <ModelsList
          models={models}
          promotedId={promotedId}
          onPromote={handlePromote}
          onRefresh={loadModels}
        />
      )}
    </div>
  );
}
