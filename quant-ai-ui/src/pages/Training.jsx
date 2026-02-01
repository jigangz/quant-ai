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
      console.error("Failed to load model types:", e);
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
      console.error("Failed to load feature groups:", e);
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
      console.error("Failed to load runs:", e);
    }
  }

  async function loadModels() {
    try {
      const data = await listModels("active", 50);
      setModels(data.models || []);
    } catch (e) {
      console.error("Failed to load models:", e);
    }
  }

  async function loadPromoted() {
    try {
      const data = await getPromotedModel();
      setPromotedId(data.promoted_id);
    } catch (e) {
      console.error("Failed to load promoted:", e);
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
        console.error("Poll failed:", e);
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

  return (
    <div>
      <h1>🎯 Training Panel</h1>

      {/* Tabs */}
      <div style={styles.tabs}>
        <button
          onClick={() => setActiveTab("train")}
          style={activeTab === "train" ? styles.activeTab : styles.tab}
        >
          Train New Model
        </button>
        <button
          onClick={() => setActiveTab("runs")}
          style={activeTab === "runs" ? styles.activeTab : styles.tab}
        >
          Training Runs ({runs.length})
        </button>
        <button
          onClick={() => setActiveTab("models")}
          style={activeTab === "models" ? styles.activeTab : styles.tab}
        >
          Models ({models.length})
        </button>
      </div>

      {/* Error display */}
      {error && (
        <div style={styles.error}>
          ❌ {error}
          <button onClick={() => setError(null)} style={styles.closeBtn}>
            ×
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

const styles = {
  tabs: {
    display: "flex",
    gap: 8,
    marginBottom: 24,
    borderBottom: "1px solid #ddd",
    paddingBottom: 8,
  },
  tab: {
    padding: "8px 16px",
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: 14,
  },
  activeTab: {
    padding: "8px 16px",
    background: "#007bff",
    color: "white",
    border: "none",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 14,
  },
  error: {
    background: "#fee",
    border: "1px solid #fcc",
    padding: 12,
    borderRadius: 4,
    marginBottom: 16,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  closeBtn: {
    background: "none",
    border: "none",
    fontSize: 20,
    cursor: "pointer",
  },
};
