"""
Prediction Log Repository — V4 P5 G1

Stores every /predict / /predict/volatility / /api/signal-score call so that
AccuracyService can later resolve them against actual market data and
compute live accuracy stats.

Backends:
- LocalPredictionLogRepo — JSON file under STORAGE_LOCAL_PATH/registry
- SupabasePredictionLogRepo — Supabase table 'prediction_log'

Factory `get_prediction_log_repo()` returns Supabase if configured else Local.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class PredictionLogRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str
    ticker: str
    label_type: Literal["direction", "volatility", "meta_label"]

    predicted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    horizon_days: int
    predicted_value: float
    predicted_signal: Optional[int] = None
    predicted_extras: dict[str, Any] = Field(default_factory=dict)

    resolve_at: datetime
    actual_value: Optional[float] = None
    actual_return: Optional[float] = None
    is_correct: Optional[bool] = None
    realized_R: Optional[float] = None
    resolved_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="ignore")


class LocalPredictionLogRepo:
    """JSON-on-disk repo. One file: registry/prediction_log.json keyed by id."""

    _lock = threading.Lock()

    def __init__(self, storage_dir: Optional[Any] = None):
        if storage_dir is None:
            from app.core.settings import settings
            storage_dir = Path(settings.STORAGE_LOCAL_PATH) / "registry"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.storage_dir / "prediction_log.json"

    def _load(self) -> dict[str, dict]:
        if not self.file.exists():
            return {}
        try:
            return json.loads(self.file.read_text())
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict[str, dict]) -> None:
        self.file.write_text(json.dumps(data, indent=2, default=str))

    def insert(self, record: PredictionLogRecord) -> PredictionLogRecord:
        with self._lock:
            data = self._load()
            data[record.id] = record.model_dump(mode="json")
            self._save(data)
        return record

    def list_unresolved(self, model_id: str, limit: int = 100) -> list[PredictionLogRecord]:
        data = self._load()
        now = datetime.now(timezone.utc)
        out = []
        for row in data.values():
            if row.get("model_id") != model_id:
                continue
            if row.get("resolved_at"):
                continue
            resolve_at = row.get("resolve_at")
            if isinstance(resolve_at, str):
                resolve_at_dt = datetime.fromisoformat(resolve_at.replace("Z", "+00:00"))
            else:
                resolve_at_dt = resolve_at
            if resolve_at_dt and resolve_at_dt < now:
                out.append(PredictionLogRecord(**row))
        out.sort(key=lambda r: r.resolve_at)
        return out[:limit]

    def list_by_model_id(
        self, model_id: str, since: Optional[datetime] = None, limit: int = 500
    ) -> list[PredictionLogRecord]:
        data = self._load()
        out = []
        for row in data.values():
            if row.get("model_id") != model_id:
                continue
            if since:
                created = row.get("created_at")
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                else:
                    created_dt = created
                if created_dt and created_dt < since:
                    continue
            out.append(PredictionLogRecord(**row))
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out[:limit]

    def update_resolution(
        self,
        record_id: str,
        *,
        actual_value: Optional[float] = None,
        actual_return: Optional[float] = None,
        is_correct: Optional[bool] = None,
        realized_R: Optional[float] = None,
    ) -> None:
        with self._lock:
            data = self._load()
            if record_id not in data:
                return
            row = data[record_id]
            if actual_value is not None:
                row["actual_value"] = actual_value
            if actual_return is not None:
                row["actual_return"] = actual_return
            if is_correct is not None:
                row["is_correct"] = is_correct
            if realized_R is not None:
                row["realized_R"] = realized_R
            row["resolved_at"] = datetime.now(timezone.utc).isoformat()
            self._save(data)


class SupabasePredictionLogRepo:
    """Supabase-backed repo."""

    table = "prediction_log"

    def __init__(self, client: Any):
        self.client = client

    def insert(self, record: PredictionLogRecord) -> PredictionLogRecord:
        data = record.model_dump(mode="json")
        result = self.client.table(self.table).insert(data).execute()
        if result.data:
            return PredictionLogRecord(**result.data[0])
        return record

    def list_unresolved(self, model_id: str, limit: int = 100) -> list[PredictionLogRecord]:
        now_iso = datetime.now(timezone.utc).isoformat()
        result = (
            self.client.table(self.table)
            .select("*")
            .eq("model_id", model_id)
            .is_("resolved_at", "null")
            .lt("resolve_at", now_iso)
            .order("resolve_at", desc=False)
            .limit(limit)
            .execute()
        )
        return [PredictionLogRecord(**row) for row in (result.data or [])]

    def list_by_model_id(
        self, model_id: str, since: Optional[datetime] = None, limit: int = 500
    ) -> list[PredictionLogRecord]:
        query = self.client.table(self.table).select("*").eq("model_id", model_id)
        if since:
            query = query.gte("created_at", since.isoformat())
        result = query.order("created_at", desc=True).limit(limit).execute()
        return [PredictionLogRecord(**row) for row in (result.data or [])]

    def update_resolution(
        self,
        record_id: str,
        *,
        actual_value: Optional[float] = None,
        actual_return: Optional[float] = None,
        is_correct: Optional[bool] = None,
        realized_R: Optional[float] = None,
    ) -> None:
        updates: dict[str, Any] = {"resolved_at": datetime.now(timezone.utc).isoformat()}
        if actual_value is not None:
            updates["actual_value"] = actual_value
        if actual_return is not None:
            updates["actual_return"] = actual_return
        if is_correct is not None:
            updates["is_correct"] = is_correct
        if realized_R is not None:
            updates["realized_R"] = realized_R
        self.client.table(self.table).update(updates).eq("id", record_id).execute()


def get_prediction_log_repo():
    """Return Supabase repo if configured, else Local repo."""
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        try:
            from supabase import create_client  # type: ignore
            client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
            return SupabasePredictionLogRepo(client)
        except Exception:
            pass
    return LocalPredictionLogRepo()
