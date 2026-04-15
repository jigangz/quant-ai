from __future__ import annotations

"""
Optimization Runs Repository

Persists optimization run results (model + strategy) to local JSON storage.
Follows the same pattern as LocalModelRegistry.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel as PydanticModel, ConfigDict, Field

from app.core.settings import settings

logger = logging.getLogger(__name__)


class OptimizationRun(PydanticModel):
    """A single optimization run record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "model" or "strategy"
    config: dict[str, Any]
    best_params: dict[str, Any]
    best_metrics: dict[str, Any]
    pareto_front: Optional[list[dict[str, Any]]] = None
    all_trials: list[dict[str, Any]] = Field(default_factory=list)
    n_trials: int
    duration_seconds: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="ignore")


class OptimizationRepo:
    """
    Local JSON-based repository for optimization runs.

    Stores data in STORAGE_LOCAL_PATH/registry/optimization_runs.json.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.STORAGE_LOCAL_PATH) / "registry"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_path / "optimization_runs.json"

    def _load(self) -> list[dict]:
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self, data: list[dict]) -> None:
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_run(self, run: OptimizationRun) -> str:
        """Save an optimization run. Returns the run ID."""
        data = self._load()
        data.append(run.model_dump(mode="json"))
        self._save(data)
        logger.info(f"Saved optimization run {run.id} (type={run.type})")
        return run.id

    def get_run(self, run_id: str) -> Optional[OptimizationRun]:
        """Get an optimization run by ID."""
        data = self._load()
        for item in data:
            if item.get("id") == run_id:
                return OptimizationRun(**item)
        return None

    def list_runs(
        self,
        type: Optional[str] = None,
        limit: int = 20,
    ) -> list[OptimizationRun]:
        """List optimization runs, optionally filtered by type."""
        data = self._load()

        if type is not None:
            data = [d for d in data if d.get("type") == type]

        # Sort by created_at desc
        data.sort(key=lambda d: d.get("created_at", ""), reverse=True)

        return [OptimizationRun(**d) for d in data[:limit]]
