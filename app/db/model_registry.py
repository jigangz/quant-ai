"""
Model Registry - Supabase operations for model versioning

Tables:
- model_registry: Model metadata (id, name, type, metrics, etc.)
- training_runs: Training history (run_id, model_id, params, metrics, etc.)
"""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.settings import settings

logger = logging.getLogger(__name__)


# ===================================
# Schemas
# ===================================
class ModelRecord(BaseModel):
    """Model metadata stored in registry."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    model_type: str
    version: int = 1

    # Training info
    tickers: list[str]
    feature_groups: list[str]
    feature_names: list[str] = []
    n_features: int = 0

    # Data info
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_date_range: tuple[str, str] | None = None

    # Metrics
    metrics: dict[str, float] = {}

    # Storage
    artifact_path: str | None = None
    storage_backend: str = "local"

    # Status
    status: str = "active"  # active, archived, deleted

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "ignore"


class TrainingRunRecord(BaseModel):
    """Training run history - complete experiment tracking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str | None = None  # Linked after successful training

    # === Reproducibility ===
    git_sha: str | None = None          # Git commit hash
    git_branch: str | None = None       # Git branch name
    git_dirty: bool = False             # Uncommitted changes?
    data_hash: str | None = None        # Hash of training data
    config_hash: str | None = None      # Hash of full config

    # === Request params ===
    tickers: list[str]
    model_type: str
    feature_groups: list[str]
    model_params: dict[str, Any] = {}
    horizon_days: int = 5
    label_type: str = "direction"
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # === Data info ===
    start_date: str | None = None
    end_date: str | None = None
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_date_range: tuple[str, str] | None = None
    val_date_range: tuple[str, str] | None = None
    test_date_range: tuple[str, str] | None = None

    # === Results ===
    success: bool = False
    error: str | None = None
    metrics: dict[str, float] = {}
    training_time_seconds: float = 0.0

    # === Environment ===
    python_version: str | None = None
    packages: dict[str, str] = {}  # key packages + versions

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    class Config:
        extra = "ignore"


# ===================================
# Supabase Client
# ===================================
def get_supabase_client():
    """Get Supabase client, or None if not configured."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        logger.warning("Supabase not configured, using local fallback")
        return None

    try:
        from supabase import create_client

        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    except ImportError:
        logger.warning("supabase-py not installed, using local fallback")
        return None
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None


# ===================================
# Local Fallback Storage
# ===================================
class LocalModelRegistry:
    """Local JSON-based registry for when Supabase is not available."""

    def __init__(self):
        import json
        from pathlib import Path

        self.storage_path = Path(settings.STORAGE_LOCAL_PATH) / "registry"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.models_file = self.storage_path / "models.json"
        self.runs_file = self.storage_path / "training_runs.json"
        self._json = json

    def _load_models(self) -> dict[str, dict]:
        if self.models_file.exists():
            return self._json.loads(self.models_file.read_text())
        return {}

    def _save_models(self, models: dict[str, dict]):
        self.models_file.write_text(self._json.dumps(models, indent=2, default=str))

    def _load_runs(self) -> dict[str, dict]:
        if self.runs_file.exists():
            return self._json.loads(self.runs_file.read_text())
        return {}

    def _save_runs(self, runs: dict[str, dict]):
        self.runs_file.write_text(self._json.dumps(runs, indent=2, default=str))

    def insert_model(self, record: ModelRecord) -> ModelRecord:
        models = self._load_models()
        models[record.id] = record.model_dump(mode="json")
        self._save_models(models)
        return record

    def get_model(self, model_id: str) -> ModelRecord | None:
        models = self._load_models()
        if model_id in models:
            return ModelRecord(**models[model_id])
        return None

    def list_models(
        self, status: str | None = "active", limit: int = 50
    ) -> list[ModelRecord]:
        models = self._load_models()
        result = []
        for data in models.values():
            if status and data.get("status") != status:
                continue
            result.append(ModelRecord(**data))
        # Sort by created_at desc
        result.sort(key=lambda m: m.created_at, reverse=True)
        return result[:limit]

    def update_model(self, model_id: str, updates: dict) -> ModelRecord | None:
        models = self._load_models()
        if model_id not in models:
            return None
        models[model_id].update(updates)
        models[model_id]["updated_at"] = datetime.utcnow().isoformat()
        self._save_models(models)
        return ModelRecord(**models[model_id])

    def insert_run(self, record: TrainingRunRecord) -> TrainingRunRecord:
        runs = self._load_runs()
        runs[record.id] = record.model_dump(mode="json")
        self._save_runs(runs)
        return record

    def get_run(self, run_id: str) -> TrainingRunRecord | None:
        runs = self._load_runs()
        if run_id in runs:
            return TrainingRunRecord(**runs[run_id])
        return None

    def list_runs(
        self, model_id: str | None = None, limit: int = 50
    ) -> list[TrainingRunRecord]:
        runs = self._load_runs()
        result = []
        for data in runs.values():
            if model_id and data.get("model_id") != model_id:
                continue
            result.append(TrainingRunRecord(**data))
        result.sort(key=lambda r: r.started_at, reverse=True)
        return result[:limit]


# ===================================
# Supabase Registry
# ===================================
class SupabaseModelRegistry:
    """Supabase-backed model registry."""

    def __init__(self, client):
        self.client = client
        self.models_table = "model_registry"
        self.runs_table = "training_runs"

    def insert_model(self, record: ModelRecord) -> ModelRecord:
        data = record.model_dump(mode="json")
        # Convert lists/dicts to JSON strings for Supabase
        data["tickers"] = data["tickers"]
        data["feature_groups"] = data["feature_groups"]
        data["feature_names"] = data["feature_names"]
        data["metrics"] = data["metrics"]

        result = self.client.table(self.models_table).insert(data).execute()
        if result.data:
            return ModelRecord(**result.data[0])
        raise Exception("Failed to insert model")

    def get_model(self, model_id: str) -> ModelRecord | None:
        result = (
            self.client.table(self.models_table)
            .select("*")
            .eq("id", model_id)
            .execute()
        )
        if result.data:
            return ModelRecord(**result.data[0])
        return None

    def list_models(
        self, status: str | None = "active", limit: int = 50
    ) -> list[ModelRecord]:
        query = self.client.table(self.models_table).select("*")
        if status:
            query = query.eq("status", status)
        result = query.order("created_at", desc=True).limit(limit).execute()
        return [ModelRecord(**row) for row in result.data]

    def update_model(self, model_id: str, updates: dict) -> ModelRecord | None:
        updates["updated_at"] = datetime.utcnow().isoformat()
        result = (
            self.client.table(self.models_table)
            .update(updates)
            .eq("id", model_id)
            .execute()
        )
        if result.data:
            return ModelRecord(**result.data[0])
        return None

    def insert_run(self, record: TrainingRunRecord) -> TrainingRunRecord:
        data = record.model_dump(mode="json")
        result = self.client.table(self.runs_table).insert(data).execute()
        if result.data:
            return TrainingRunRecord(**result.data[0])
        raise Exception("Failed to insert training run")

    def get_run(self, run_id: str) -> TrainingRunRecord | None:
        result = (
            self.client.table(self.runs_table)
            .select("*")
            .eq("id", run_id)
            .execute()
        )
        if result.data:
            return TrainingRunRecord(**result.data[0])
        return None

    def list_runs(
        self, model_id: str | None = None, limit: int = 50
    ) -> list[TrainingRunRecord]:
        query = self.client.table(self.runs_table).select("*")
        if model_id:
            query = query.eq("model_id", model_id)
        result = query.order("started_at", desc=True).limit(limit).execute()
        return [TrainingRunRecord(**row) for row in result.data]


# ===================================
# Factory
# ===================================
_registry_instance = None


def get_model_registry() -> LocalModelRegistry | SupabaseModelRegistry:
    """Get the model registry instance (singleton)."""
    global _registry_instance

    if _registry_instance is None:
        client = get_supabase_client()
        if client:
            logger.info("Using Supabase model registry")
            _registry_instance = SupabaseModelRegistry(client)
        else:
            logger.info("Using local model registry")
            _registry_instance = LocalModelRegistry()

    return _registry_instance
