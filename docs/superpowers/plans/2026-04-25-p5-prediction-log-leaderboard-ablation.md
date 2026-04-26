# P5 Prediction Log + Leaderboard + Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land G1 prediction-log + AccuracyService + Leaderboard frontend + multi-target Ablation backend & frontend in one Ralph sweep. After P5: live accuracy data flows into a sortable Leaderboard, Ablation page produces honest sentiment-delta numbers across direction/volatility/meta-label.

**Architecture:** New `prediction_log` table (Supabase + Local JSON repos, factory pattern mirroring ModelRegistry). Three predict services append non-blocking log rows after publishing Kafka events. `AccuracyService.resolve_pending` lazily resolves rows past horizon by fetching OHLC slices via existing market provider; `aggregate` returns 30-day stats. `AblationService.run_ablation` orchestrates 6 trainings (3 targets × 2 feature sets) reusing existing `training_service.train` + `meta_label_service.train_meta_label_model` (with feature_group extended to accept list[str]).

**Tech Stack:** Python 3.11 + FastAPI + Pydantic v2 + Supabase + pytest; React 18 + Vite + TanStack Query v5 + Vitest + @testing-library/react.

**Spec:** [`docs/superpowers/specs/2026-04-25-p5-prediction-log-leaderboard-ablation-design.md`](../specs/2026-04-25-p5-prediction-log-leaderboard-ablation-design.md)

**Branch:** direct-to-main (P3+P4 precedent). Ralph commits per task.

---

## File Structure

### New backend files
- `app/db/prediction_log.py` — `PredictionLogRecord` + `LocalPredictionLogRepo` + `SupabasePredictionLogRepo` + `get_prediction_log_repo()`
- `app/services/accuracy_service.py` — `resolve_pending()` + `aggregate()`
- `app/services/ablation_service.py` — `run_ablation()` orchestrator
- `app/api/accuracy.py` — `GET /models/{id}/accuracy`
- `app/api/ablation.py` — `POST /api/ablation/run`
- `scripts/migrate_create_prediction_log.sql` — Supabase migration
- `scripts/p5_ablation_demo.py` — live benchmark script

### Modified backend files
- `app/services/predict_service.py` — append prediction_log write after publish_prediction_event
- `app/services/volatility_predict_service.py` — same
- `app/services/signal_scoring_service.py` — same (Mode A path only)
- `app/services/meta_label_service.py` — extend `MetaLabelTrainRequest.feature_group` to `str | list[str]`
- `app/main.py` — include accuracy + ablation routers

### New frontend files
- `quant-ai-ui/src/pages/LeaderboardPage.jsx`
- `quant-ai-ui/src/pages/AblationPage.jsx`
- `quant-ai-ui/src/features/leaderboard/LeaderboardTable.jsx`
- `quant-ai-ui/src/features/ablation/AblationMatrix.jsx`
- `quant-ai-ui/src/api/leaderboardQueries.js`

### Modified frontend files
- `quant-ai-ui/src/api/client.js` — add `getModelAccuracy`, `postAblationRun`
- `quant-ai-ui/src/App.jsx` — `/leaderboard` + `/ablation` routes
- `quant-ai-ui/src/components/layout/TopNavBar.jsx` — nav links

### New test files (backend)
- `tests/test_prediction_log_repo.py` (5)
- `tests/test_accuracy_service.py` (8)
- `tests/test_ablation_service.py` (4)
- `tests/contract/test_models_accuracy.py` (5)
- `tests/contract/test_ablation_run.py` (4)
- `tests/test_predict_log_writes.py` (4 — covering all 3 services)

### New test files (frontend)
- `quant-ai-ui/__tests__/api/leaderboardQueries.test.jsx` (3)
- `quant-ai-ui/__tests__/pages/LeaderboardPage.test.jsx` (4)
- `quant-ai-ui/__tests__/pages/AblationPage.test.jsx` (4)
- `quant-ai-ui/__tests__/features/ablation/AblationMatrix.test.jsx` (1)
- `quant-ai-ui/__tests__/components/layout/TopNavBar.test.jsx` (extend, 2)

---

## Task 0: Supabase Migration SQL

**Files:**
- Create: `scripts/migrate_create_prediction_log.sql`

- [ ] **Step 0.1: Create migration file**

Create `scripts/migrate_create_prediction_log.sql`:

```sql
-- V4 P5 migration · Create prediction_log table for live accuracy tracking
-- Date: 2026-04-25
-- Safe to run multiple times (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS prediction_log (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id        TEXT NOT NULL,
  ticker          TEXT NOT NULL,
  label_type      TEXT NOT NULL CHECK (label_type IN ('direction','volatility','meta_label')),

  predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  horizon_days    INTEGER NOT NULL,
  predicted_value NUMERIC NOT NULL,
  predicted_signal INTEGER,
  predicted_extras JSONB NOT NULL DEFAULT '{}'::jsonb,

  resolve_at      TIMESTAMPTZ NOT NULL,
  actual_value    NUMERIC,
  actual_return   NUMERIC,
  is_correct      BOOLEAN,
  realized_R      NUMERIC,
  resolved_at     TIMESTAMPTZ,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_log_model_id ON prediction_log(model_id);
CREATE INDEX IF NOT EXISTS idx_pred_log_resolve_pending
  ON prediction_log(resolve_at) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_pred_log_ticker_label
  ON prediction_log(ticker, label_type);

-- Rollback (if ever needed):
-- DROP TABLE IF EXISTS prediction_log;
```

- [ ] **Step 0.2: Commit**

```bash
git add scripts/migrate_create_prediction_log.sql
git commit -m "feat(p5): Supabase migration for prediction_log table"
```

(Harry runs this in Supabase SQL Editor when prod is deployed.)

---

## Task 1: PredictionLogRecord + Repos

**Files:**
- Create: `app/db/prediction_log.py`
- Create: `tests/test_prediction_log_repo.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_prediction_log_repo.py`:

```python
"""Tests for prediction_log repo (V4 P5 G1)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
import os

from app.db.prediction_log import (
    PredictionLogRecord,
    LocalPredictionLogRepo,
    get_prediction_log_repo,
)


def _make_record(model_id="m1", ticker="MSFT", label_type="direction", **overrides):
    base = dict(
        model_id=model_id,
        ticker=ticker,
        label_type=label_type,
        horizon_days=5,
        predicted_value=0.7,
        predicted_signal=1,
        predicted_extras={"feature_group": "ta_basic"},
        resolve_at=datetime.now(timezone.utc) + timedelta(days=5),
    )
    base.update(overrides)
    return PredictionLogRecord(**base)


def test_local_repo_insert_and_get(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path))
    repo = LocalPredictionLogRepo(storage_dir=tmp_path / "registry")
    rec = _make_record()
    repo.insert(rec)
    rows = repo.list_by_model_id("m1")
    assert len(rows) == 1
    assert rows[0].model_id == "m1"


def test_local_repo_list_unresolved_filters(tmp_path):
    repo = LocalPredictionLogRepo(storage_dir=tmp_path / "registry")
    past = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=10)
    r1 = _make_record(model_id="m1", resolve_at=past)
    r2 = _make_record(model_id="m1", resolve_at=future)
    r3 = _make_record(model_id="m1", resolve_at=past)
    repo.insert(r1); repo.insert(r2); repo.insert(r3)
    pending = repo.list_unresolved("m1", limit=10)
    assert len(pending) == 2  # only those with resolve_at < now and not resolved


def test_local_repo_update_resolution(tmp_path):
    repo = LocalPredictionLogRepo(storage_dir=tmp_path / "registry")
    rec = _make_record(resolve_at=datetime.now(timezone.utc) - timedelta(days=1))
    repo.insert(rec)
    repo.update_resolution(
        rec.id,
        actual_value=110.0,
        actual_return=0.05,
        is_correct=True,
        realized_R=0.6,
    )
    pending = repo.list_unresolved(rec.model_id, limit=10)
    assert len(pending) == 0
    rows = repo.list_by_model_id(rec.model_id)
    assert rows[0].is_correct is True
    assert rows[0].resolved_at is not None


def test_local_repo_list_by_model_id_window(tmp_path):
    repo = LocalPredictionLogRepo(storage_dir=tmp_path / "registry")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    recent = datetime.now(timezone.utc) - timedelta(days=10)
    r_old = _make_record(model_id="m1")
    r_old_dict = r_old.model_dump()
    r_old_dict["created_at"] = old
    r_recent = _make_record(model_id="m1")
    repo.insert(PredictionLogRecord(**{**r_old_dict, "id": r_old.id}))
    repo.insert(r_recent)
    rows = repo.list_by_model_id("m1", since=datetime.now(timezone.utc) - timedelta(days=30))
    # Only recent is within 30d window
    assert len(rows) == 1


def test_factory_returns_local_when_no_supabase(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    repo = get_prediction_log_repo()
    assert type(repo).__name__ == "LocalPredictionLogRepo"
```

- [ ] **Step 1.2: Run tests — verify they fail**

```bash
cd C:/Users/zjg09/projects/quant-ai
pytest tests/test_prediction_log_repo.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.db.prediction_log'`.

- [ ] **Step 1.3: Implement `app/db/prediction_log.py`**

Create `app/db/prediction_log.py`:

```python
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

    def __init__(self, storage_dir: Path | str | None = None):
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
        self, model_id: str, since: datetime | None = None, limit: int = 500
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
        actual_value: float | None = None,
        actual_return: float | None = None,
        is_correct: bool | None = None,
        realized_R: float | None = None,
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

    def __init__(self, client):
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
        self, model_id: str, since: datetime | None = None, limit: int = 500
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
        actual_value: float | None = None,
        actual_return: float | None = None,
        is_correct: bool | None = None,
        realized_R: float | None = None,
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
```

- [ ] **Step 1.4: Run tests**

```bash
pytest tests/test_prediction_log_repo.py -v
```

Expected: `5 passed`.

- [ ] **Step 1.5: Commit**

```bash
git add app/db/prediction_log.py tests/test_prediction_log_repo.py
git commit -m "feat(p5): PredictionLogRecord + Local/Supabase repos + factory (5 tests)"
```

---

## Task 2: Wire Predict Services to Write Log

**Files:**
- Modify: `app/services/predict_service.py`
- Modify: `app/services/volatility_predict_service.py`
- Modify: `app/services/signal_scoring_service.py`
- Create: `tests/test_predict_log_writes.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_predict_log_writes.py`:

```python
"""Tests for predict services writing prediction_log rows (V4 P5)."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest


@pytest.fixture
def captured_inserts(monkeypatch):
    """Capture every PredictionLogRecord insert across services."""
    captured = []

    class _FakeRepo:
        def insert(self, record):
            captured.append(record)
            return record

    def _factory():
        return _FakeRepo()

    monkeypatch.setattr("app.db.prediction_log.get_prediction_log_repo", _factory)
    return captured


def test_predict_service_writes_log(captured_inserts, monkeypatch):
    from app.services import predict_service
    # Stub out underlying predict() machinery — we only care the log row gets written.
    monkeypatch.setattr(predict_service, "_run_legacy_predict", lambda **kw: {
        "ticker": "MSFT", "prediction": 1, "confidence": 0.71,
        "model_id": "dir_msft_a", "model_type": "xgboost",
        "horizon_days": 5,
    })
    # Direct call to the post-predict log-write helper:
    predict_service._write_prediction_log(
        ticker="MSFT", model_id="dir_msft_a", model_type="xgboost",
        label_type="direction", horizon_days=5,
        predicted_value=0.71, predicted_signal=1,
        feature_group="ta_basic",
    )
    assert len(captured_inserts) == 1
    rec = captured_inserts[0]
    assert rec.label_type == "direction"
    assert rec.predicted_signal == 1


def test_volatility_predict_writes_log(captured_inserts):
    from app.services import volatility_predict_service
    volatility_predict_service._write_prediction_log(
        ticker="MSFT", model_id="vol_msft_a", model_type="xgboost",
        horizon_days=5, predicted_value=0.18, feature_group="ta_basic",
    )
    assert len(captured_inserts) == 1
    rec = captured_inserts[0]
    assert rec.label_type == "volatility"
    assert rec.predicted_signal is None


def test_signal_scoring_mode_a_writes_log(captured_inserts):
    from app.services import signal_scoring_service
    signal_scoring_service._write_prediction_log(
        ticker="AAPL", model_id="meta_aapl_a", model_type="xgboost",
        horizon_days=5, predicted_value=0.71, predicted_signal=1,
        primary_source="strategy:rsi_strategy", expected_R=0.54,
        feature_group="ta_basic",
    )
    assert len(captured_inserts) == 1
    rec = captured_inserts[0]
    assert rec.label_type == "meta_label"
    assert rec.predicted_extras["expected_R"] == 0.54
    assert rec.predicted_extras["primary_source"] == "strategy:rsi_strategy"


def test_log_write_is_non_blocking_on_repo_failure(monkeypatch):
    from app.services import predict_service

    class _BrokenRepo:
        def insert(self, record):
            raise RuntimeError("supabase down")

    monkeypatch.setattr(
        "app.db.prediction_log.get_prediction_log_repo",
        lambda: _BrokenRepo(),
    )
    # Should swallow and NOT raise:
    predict_service._write_prediction_log(
        ticker="MSFT", model_id="dir", model_type="xgboost",
        label_type="direction", horizon_days=5,
        predicted_value=0.5, predicted_signal=1, feature_group="ta_basic",
    )
```

- [ ] **Step 2.2: Run tests — fail**

```bash
pytest tests/test_predict_log_writes.py -v
```

Expected: `AttributeError: module 'app.services.predict_service' has no attribute '_write_prediction_log'`.

- [ ] **Step 2.3: Add helper to predict_service.py**

Add at the bottom of `app/services/predict_service.py`:

```python
# V4 P5: prediction_log write helper. Non-blocking — log failures must
# not break a prediction response.
def _write_prediction_log(
    *,
    ticker: str,
    model_id: str,
    model_type: str,
    label_type: str,
    horizon_days: int,
    predicted_value: float,
    predicted_signal: int | None,
    feature_group: str,
) -> None:
    try:
        from datetime import datetime, timedelta, timezone
        from app.db.prediction_log import PredictionLogRecord, get_prediction_log_repo

        repo = get_prediction_log_repo()
        now = datetime.now(timezone.utc)
        repo.insert(
            PredictionLogRecord(
                model_id=model_id,
                ticker=ticker,
                label_type=label_type,
                horizon_days=horizon_days,
                predicted_value=float(predicted_value),
                predicted_signal=predicted_signal,
                predicted_extras={"feature_group": feature_group, "model_type": model_type},
                resolve_at=now + timedelta(days=horizon_days),
            )
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "prediction_log write failed (non-blocking): %s", e
        )
```

Then locate the existing `predict()` (or main service entrypoint) in the same file. Right after the existing `publish_prediction_event(...)` call (currently around line 151), add:

```python
# V4 P5: append to prediction_log after successful prediction
_write_prediction_log(
    ticker=ticker,
    model_id=model_id_used,
    model_type=model_type_used,
    label_type="direction",
    horizon_days=horizon_used,
    predicted_value=float(prob_up),
    predicted_signal=1 if prob_up >= 0.5 else -1,
    feature_group=feature_group_used,
)
```

(Adapt variable names to whatever the existing function uses for `model_id_used`, `model_type_used`, `horizon_used`, `prob_up`, `feature_group_used`.)

- [ ] **Step 2.4: Add helper to volatility_predict_service.py**

Append:

```python
def _write_prediction_log(
    *,
    ticker: str,
    model_id: str,
    model_type: str,
    horizon_days: int,
    predicted_value: float,
    feature_group: str,
) -> None:
    try:
        from datetime import datetime, timedelta, timezone
        from app.db.prediction_log import PredictionLogRecord, get_prediction_log_repo

        repo = get_prediction_log_repo()
        now = datetime.now(timezone.utc)
        repo.insert(
            PredictionLogRecord(
                model_id=model_id,
                ticker=ticker,
                label_type="volatility",
                horizon_days=horizon_days,
                predicted_value=float(predicted_value),
                predicted_signal=None,
                predicted_extras={"feature_group": feature_group, "model_type": model_type},
                resolve_at=now + timedelta(days=horizon_days),
            )
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "prediction_log write failed (non-blocking): %s", e
        )
```

Wire it after the existing prediction returns its volatility value, mirroring the predict_service pattern.

- [ ] **Step 2.5: Add helper to signal_scoring_service.py**

Append:

```python
def _write_prediction_log(
    *,
    ticker: str,
    model_id: str,
    model_type: str,
    horizon_days: int,
    predicted_value: float,
    predicted_signal: int,
    primary_source: str,
    expected_R: float,
    feature_group: str,
) -> None:
    try:
        from datetime import datetime, timedelta, timezone
        from app.db.prediction_log import PredictionLogRecord, get_prediction_log_repo

        repo = get_prediction_log_repo()
        now = datetime.now(timezone.utc)
        repo.insert(
            PredictionLogRecord(
                model_id=model_id,
                ticker=ticker,
                label_type="meta_label",
                horizon_days=horizon_days,
                predicted_value=float(predicted_value),
                predicted_signal=predicted_signal,
                predicted_extras={
                    "feature_group": feature_group,
                    "model_type": model_type,
                    "primary_source": primary_source,
                    "expected_R": float(expected_R),
                },
                resolve_at=now + timedelta(days=horizon_days),
            )
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "prediction_log write failed (non-blocking): %s", e
        )
```

In `score_signal()`, after the response is built (Mode A or successful Mode B), call `_write_prediction_log` with the resolved `signal`, `score` (predicted_value), `expected_R`, etc. Skip the call when `triggered=False` (no real prediction to log).

- [ ] **Step 2.6: Run tests**

```bash
pytest tests/test_predict_log_writes.py tests/contract/test_predict_flow.py tests/contract/test_predict_volatility.py tests/contract/test_signal_score.py -v
```

Expected: 4 new tests pass + existing predict-flow regression stays green.

- [ ] **Step 2.7: Commit**

```bash
git add app/services/predict_service.py app/services/volatility_predict_service.py app/services/signal_scoring_service.py tests/test_predict_log_writes.py
git commit -m "feat(p5): wire prediction_log writes in 3 predict services (4 tests)"
```

---

## Task 3: AccuracyService

**Files:**
- Create: `app/services/accuracy_service.py`
- Create: `tests/test_accuracy_service.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_accuracy_service.py`:

```python
"""Tests for AccuracyService (V4 P5 G1)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from app.db.prediction_log import PredictionLogRecord


@pytest.fixture
def fake_ohlc(monkeypatch):
    """Make AccuracyService use a deterministic OHLC slice."""
    from app.services import accuracy_service

    def _fake_fetch(ticker, start, end):
        # Linear price 100 → 110 over 10 trading days
        idx = pd.date_range(start, end, freq="D")
        closes = np.linspace(100, 110, len(idx))
        return pd.DataFrame({"date": idx, "close": closes,
                              "open": closes, "high": closes * 1.01,
                              "low": closes * 0.99, "volume": [1_000_000] * len(idx)})

    monkeypatch.setattr(accuracy_service, "_fetch_ohlc_slice", _fake_fetch)


@pytest.fixture
def fake_repo(monkeypatch):
    from app.db import prediction_log

    rows = {}

    class _Fake:
        def insert(self, rec):
            rows[rec.id] = rec
            return rec

        def list_unresolved(self, model_id, limit=100):
            now = datetime.now(timezone.utc)
            return [r for r in rows.values()
                    if r.model_id == model_id and not r.resolved_at and r.resolve_at < now][:limit]

        def list_by_model_id(self, model_id, since=None, limit=500):
            return [r for r in rows.values() if r.model_id == model_id][:limit]

        def update_resolution(self, rid, **updates):
            r = rows.get(rid)
            if not r:
                return
            d = r.model_dump()
            d.update(updates)
            d["resolved_at"] = datetime.now(timezone.utc)
            rows[rid] = PredictionLogRecord(**d)

    fake = _Fake()
    monkeypatch.setattr(prediction_log, "get_prediction_log_repo", lambda: fake)
    return fake


def _make(model_id, label_type, ticker="MSFT", **k):
    base = dict(
        model_id=model_id, ticker=ticker, label_type=label_type,
        horizon_days=5, predicted_value=0.7,
        predicted_signal=(1 if label_type != "volatility" else None),
        resolve_at=datetime.now(timezone.utc) - timedelta(days=1),
        predicted_at=datetime.now(timezone.utc) - timedelta(days=6),
    )
    base.update(k)
    return PredictionLogRecord(**base)


def test_resolve_direction_correct_prediction(fake_repo, fake_ohlc):
    from app.services.accuracy_service import resolve_pending
    fake_repo.insert(_make("dir_a", "direction", predicted_signal=1))
    result = resolve_pending("dir_a", limit=10)
    assert result["newly_resolved"] == 1
    rows = fake_repo.list_by_model_id("dir_a")
    assert rows[0].is_correct is True  # price went up + predicted +1 = correct
    assert rows[0].realized_R is not None


def test_resolve_direction_wrong_prediction(fake_repo, fake_ohlc):
    from app.services.accuracy_service import resolve_pending
    fake_repo.insert(_make("dir_b", "direction", predicted_signal=-1))
    resolve_pending("dir_b", limit=10)
    rows = fake_repo.list_by_model_id("dir_b")
    assert rows[0].is_correct is False


def test_resolve_volatility_no_hit_miss(fake_repo, fake_ohlc):
    from app.services.accuracy_service import resolve_pending
    fake_repo.insert(_make("vol_a", "volatility", predicted_signal=None))
    resolve_pending("vol_a", limit=10)
    rows = fake_repo.list_by_model_id("vol_a")
    assert rows[0].is_correct is None  # vol target — no hit/miss
    assert rows[0].realized_R is None
    assert rows[0].actual_value is not None  # realized vol


def test_resolve_meta_label_uses_signal(fake_repo, fake_ohlc):
    from app.services.accuracy_service import resolve_pending
    fake_repo.insert(_make("meta_a", "meta_label", predicted_signal=1))
    resolve_pending("meta_a", limit=10)
    rows = fake_repo.list_by_model_id("meta_a")
    assert rows[0].is_correct is True


def test_aggregate_hit_rate(fake_repo, fake_ohlc):
    from app.services.accuracy_service import aggregate, resolve_pending
    for sig in [1, 1, -1, 1]:  # 3 of 4 will be "correct" since price rose
        fake_repo.insert(_make("agg_a", "direction", predicted_signal=sig))
    resolve_pending("agg_a", limit=10)
    stats = aggregate("agg_a", window_days=30)
    assert stats["resolved"] == 4
    assert abs(stats["hit_rate"] - 0.75) < 1e-6


def test_aggregate_avg_realized_R(fake_repo, fake_ohlc):
    from app.services.accuracy_service import aggregate, resolve_pending
    for _ in range(3):
        fake_repo.insert(_make("agg_b", "direction", predicted_signal=1))
    resolve_pending("agg_b", limit=10)
    stats = aggregate("agg_b", window_days=30)
    assert stats["avg_realized_R"] is not None
    assert stats["avg_realized_R"] > 0  # all "correct" + price up = positive R


def test_aggregate_pending_count(fake_repo):
    from app.services.accuracy_service import aggregate
    # Future resolve_at — never resolves
    fake_repo.insert(_make(
        "pend_a", "direction",
        resolve_at=datetime.now(timezone.utc) + timedelta(days=10),
    ))
    stats = aggregate("pend_a", window_days=30)
    assert stats["pending"] == 1
    assert stats["resolved"] == 0


def test_aggregate_empty_model(fake_repo):
    from app.services.accuracy_service import aggregate
    stats = aggregate("doesnt_exist", window_days=30)
    assert stats["total_predictions"] == 0
    assert stats["hit_rate"] is None
```

- [ ] **Step 3.2: Run tests — fail**

```bash
pytest tests/test_accuracy_service.py -v
```

- [ ] **Step 3.3: Implement `app/services/accuracy_service.py`**

Create `app/services/accuracy_service.py`:

```python
"""
Accuracy Service — V4 P5 G1

Lazily resolves prediction_log rows whose horizon has passed by fetching
actual market data, then aggregates 30-day stats. No cron — resolution
happens on demand when /models/{id}/accuracy is called.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Any
import logging

import numpy as np
import pandas as pd

from app.db.prediction_log import PredictionLogRecord, get_prediction_log_repo

logger = logging.getLogger(__name__)


def _fetch_ohlc_slice(ticker: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """Fetch OHLC slice [start, end] from market provider. Returns None on failure."""
    try:
        from app.providers import get_market_provider
        provider = get_market_provider()
        df = provider.fetch(ticker=ticker, start_date=start.date(), end_date=end.date())
        return df
    except Exception as e:
        logger.warning("fetch_ohlc_slice failed for %s [%s..%s]: %s", ticker, start, end, e)
        return None


def _close_at(df: pd.DataFrame, target_dt: datetime) -> float | None:
    if df is None or df.empty:
        return None
    target_date = pd.Timestamp(target_dt.date())
    dates = pd.to_datetime(df["date"]).dt.date
    mask = pd.Series(dates) == target_dt.date()
    if mask.any():
        v = float(df.loc[mask.values, "close"].iloc[0])
        return v if np.isfinite(v) else None
    # Find the closest prior bar
    diffs = (pd.to_datetime(df["date"]) - target_date).dt.days
    valid = df[diffs <= 0]
    if valid.empty:
        return None
    v = float(valid.iloc[-1]["close"])
    return v if np.isfinite(v) else None


def _rolling_vol_at(df: pd.DataFrame, target_dt: datetime, window: int = 20) -> float:
    if df is None or df.empty:
        return 0.02
    df_sorted = df.copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    target = pd.Timestamp(target_dt.date())
    prior = df_sorted[df_sorted["date"] <= target].tail(window + 1)
    if len(prior) < 5:
        return 0.02
    rets = prior["close"].pct_change().dropna()
    sigma = float(rets.std()) * sqrt(252) if len(rets) > 1 else 0.02
    return max(sigma, 1e-6)


def _realized_vol(df: pd.DataFrame, t0: datetime, t1: datetime) -> float | None:
    if df is None or df.empty:
        return None
    df_sorted = df.copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    window = df_sorted[(df_sorted["date"] >= pd.Timestamp(t0.date())) &
                        (df_sorted["date"] <= pd.Timestamp(t1.date()))]
    if len(window) < 2:
        return None
    rets = window["close"].pct_change().dropna()
    if len(rets) < 1:
        return None
    return float(rets.std()) * sqrt(252)


def resolve_pending(model_id: str, limit: int = 100) -> dict[str, int]:
    """Resolve all unresolved predictions for `model_id` whose horizon has passed."""
    repo = get_prediction_log_repo()
    pending = repo.list_unresolved(model_id, limit=limit)
    checked, newly_resolved, errors = 0, 0, 0

    for rec in pending:
        checked += 1
        try:
            slice_start = rec.predicted_at - timedelta(days=30)
            slice_end = rec.resolve_at + timedelta(days=2)
            df = _fetch_ohlc_slice(rec.ticker, slice_start, slice_end)
            if df is None or df.empty:
                errors += 1
                continue

            close_predict = _close_at(df, rec.predicted_at)
            close_resolve = _close_at(df, rec.resolve_at)
            if close_predict is None or close_resolve is None:
                errors += 1
                continue

            actual_return = (close_resolve - close_predict) / close_predict

            if rec.label_type in ("direction", "meta_label"):
                vol = _rolling_vol_at(df, rec.predicted_at)
                signal = rec.predicted_signal or 0
                is_correct = (signal == 1 and actual_return > 0) or (signal == -1 and actual_return < 0)
                realized_R = signal * actual_return / vol if vol > 0 else 0.0
                repo.update_resolution(
                    rec.id,
                    actual_value=close_resolve,
                    actual_return=actual_return,
                    is_correct=bool(is_correct),
                    realized_R=float(realized_R),
                )
            else:  # volatility
                rv = _realized_vol(df, rec.predicted_at, rec.resolve_at)
                if rv is None:
                    errors += 1
                    continue
                repo.update_resolution(
                    rec.id,
                    actual_value=float(rv),
                    actual_return=actual_return,
                )
            newly_resolved += 1
        except Exception as e:
            logger.warning("resolve failed for %s: %s", rec.id, e)
            errors += 1

    return {"checked": checked, "newly_resolved": newly_resolved, "errors": errors}


def aggregate(model_id: str, window_days: int = 30) -> dict[str, Any]:
    """Aggregate accuracy stats for `model_id` over the last `window_days`."""
    repo = get_prediction_log_repo()
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    rows = repo.list_by_model_id(model_id, since=since, limit=500)
    resolved = [r for r in rows if r.resolved_at]
    pending = [r for r in rows if not r.resolved_at]

    label_type = rows[0].label_type if rows else None

    stats: dict[str, Any] = {
        "total_predictions": len(rows),
        "resolved": len(resolved),
        "pending": len(pending),
        "hit_rate": None,
        "avg_realized_R": None,
        "best_R": None,
        "worst_R": None,
        "mae": None,
        "rmse": None,
    }

    if not resolved:
        return stats

    if label_type in ("direction", "meta_label"):
        correct = [r for r in resolved if r.is_correct]
        rs = [r.realized_R for r in resolved if r.realized_R is not None]
        stats["hit_rate"] = len(correct) / len(resolved)
        if rs:
            stats["avg_realized_R"] = float(np.mean(rs))
            stats["best_R"] = float(max(rs))
            stats["worst_R"] = float(min(rs))
    elif label_type == "volatility":
        diffs = [
            r.actual_value - r.predicted_value
            for r in resolved
            if r.actual_value is not None
        ]
        if diffs:
            stats["mae"] = float(np.mean([abs(d) for d in diffs]))
            stats["rmse"] = float(np.sqrt(np.mean([d * d for d in diffs])))

    return stats


def by_ticker(model_id: str, window_days: int = 30) -> list[dict[str, Any]]:
    repo = get_prediction_log_repo()
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    rows = repo.list_by_model_id(model_id, since=since, limit=500)
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        slot = out.setdefault(r.ticker, {
            "ticker": r.ticker, "total": 0, "resolved": 0,
            "hits": 0, "rs": [],
        })
        slot["total"] += 1
        if r.resolved_at:
            slot["resolved"] += 1
            if r.is_correct:
                slot["hits"] += 1
            if r.realized_R is not None:
                slot["rs"].append(r.realized_R)
    return [
        {
            "ticker": s["ticker"], "total": s["total"], "resolved": s["resolved"],
            "hit_rate": (s["hits"] / s["resolved"]) if s["resolved"] else None,
            "avg_R": (float(np.mean(s["rs"])) if s["rs"] else None),
        }
        for s in out.values()
    ]


def last_predictions(model_id: str, limit: int = 20) -> list[dict[str, Any]]:
    repo = get_prediction_log_repo()
    rows = repo.list_by_model_id(model_id, limit=limit)
    return [r.model_dump(mode="json") for r in rows[:limit]]
```

- [ ] **Step 3.4: Run tests**

```bash
pytest tests/test_accuracy_service.py -v
```

Expected: `8 passed`.

- [ ] **Step 3.5: Commit**

```bash
git add app/services/accuracy_service.py tests/test_accuracy_service.py
git commit -m "feat(p5): AccuracyService — resolve_pending + aggregate + by_ticker (8 tests)"
```

---

## Task 4: `GET /models/{id}/accuracy` Endpoint

**Files:**
- Create: `app/api/accuracy.py`
- Modify: `app/main.py`
- Create: `tests/contract/test_models_accuracy.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/contract/test_models_accuracy.py`:

```python
"""Contract tests for GET /models/{id}/accuracy (V4 P5 G1)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.services import accuracy_service

    monkeypatch.setattr(accuracy_service, "resolve_pending",
                         lambda model_id, limit=100: {"checked": 0, "newly_resolved": 0, "errors": 0})
    monkeypatch.setattr(accuracy_service, "aggregate",
                         lambda model_id, window_days=30: {
                             "total_predictions": 47, "resolved": 35, "pending": 12,
                             "hit_rate": 0.571, "avg_realized_R": 0.18,
                             "best_R": 1.85, "worst_R": -0.94,
                             "mae": None, "rmse": None,
                         })
    monkeypatch.setattr(accuracy_service, "by_ticker",
                         lambda model_id, window_days=30: [
                             {"ticker": "MSFT", "total": 35, "resolved": 25,
                              "hit_rate": 0.571, "avg_R": 0.18}
                         ])
    monkeypatch.setattr(accuracy_service, "last_predictions",
                         lambda model_id, limit=20: [])
    monkeypatch.setattr(
        "app.api.accuracy._get_model_record",
        lambda mid: {"label_type": "meta_label"} if mid == "ok_model" else None,
    )
    return TestClient(app)


def test_200_with_data(client):
    resp = client.get("/models/ok_model/accuracy?window_days=30")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ok_model"
    assert body["label_type"] == "meta_label"
    assert body["stats"]["hit_rate"] == 0.571


def test_404_not_found(client):
    resp = client.get("/models/missing/accuracy")
    assert resp.status_code == 404


def test_resolve_param_default_true(client, monkeypatch):
    from app.services import accuracy_service
    called = {}

    def _resolve(model_id, limit=100):
        called["yes"] = True
        return {"checked": 1, "newly_resolved": 1, "errors": 0}

    monkeypatch.setattr(accuracy_service, "resolve_pending", _resolve)
    resp = client.get("/models/ok_model/accuracy")
    assert resp.status_code == 200
    assert called.get("yes") is True


def test_window_days_clamped(client):
    resp = client.get("/models/ok_model/accuracy?window_days=400")
    assert resp.status_code == 422  # > 365 max


def test_resolve_false_skips_resolution(client, monkeypatch):
    from app.services import accuracy_service
    called = {"yes": False}
    monkeypatch.setattr(accuracy_service, "resolve_pending",
                         lambda *a, **k: called.update(yes=True) or {"checked": 0, "newly_resolved": 0, "errors": 0})
    resp = client.get("/models/ok_model/accuracy?resolve=false")
    assert resp.status_code == 200
    assert called["yes"] is False
```

- [ ] **Step 4.2: Run tests — fail**

- [ ] **Step 4.3: Implement `app/api/accuracy.py`**

Create `app/api/accuracy.py`:

```python
"""Accuracy API — V4 P5 G1: GET /models/{id}/accuracy."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services import accuracy_service

router = APIRouter()


def _get_model_record(model_id: str) -> dict | None:
    """Return minimal record dict ({label_type:...}) or None. Wrapped for tests."""
    try:
        from app.db.model_registry import get_model_registry
        registry = get_model_registry()
        record = registry.get_model(model_id)
        if record is None:
            return None
        return {"label_type": record.label_type, "model_type": record.model_type}
    except Exception:
        return None


@router.get("/models/{model_id}/accuracy")
def get_model_accuracy(
    model_id: str,
    window_days: int = Query(default=30, ge=1, le=365),
    resolve: bool = Query(default=True),
):
    record = _get_model_record(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"model_not_found:{model_id}")

    resolve_run = (
        accuracy_service.resolve_pending(model_id, limit=100)
        if resolve else
        {"checked": 0, "newly_resolved": 0, "errors": 0}
    )
    stats = accuracy_service.aggregate(model_id, window_days=window_days)
    bt = accuracy_service.by_ticker(model_id, window_days=window_days)
    last = accuracy_service.last_predictions(model_id, limit=20)

    return {
        "model_id": model_id,
        "label_type": record["label_type"],
        "window_days": window_days,
        "resolve_run": resolve_run,
        "stats": stats,
        "by_ticker": bt,
        "last_predictions": last,
    }
```

- [ ] **Step 4.4: Wire router in `app/main.py`**

Add to imports block:
```python
from app.api import accuracy as accuracy_api
```

Add include_router after the existing routers:
```python
app.include_router(accuracy_api.router, tags=["Accuracy"])
```

- [ ] **Step 4.5: Run tests**

```bash
pytest tests/contract/test_models_accuracy.py -v
```

Expected: `5 passed`.

- [ ] **Step 4.6: Commit**

```bash
git add app/api/accuracy.py app/main.py tests/contract/test_models_accuracy.py
git commit -m "feat(p5): GET /models/{id}/accuracy endpoint (5 contract tests)"
```

---

## Task 5: AblationService

**Files:**
- Modify: `app/services/meta_label_service.py` (extend feature_group)
- Create: `app/services/ablation_service.py`
- Create: `tests/test_ablation_service.py`

- [ ] **Step 5.1: Extend MetaLabelTrainRequest.feature_group to accept list**

In `app/services/meta_label_service.py`, locate `class MetaLabelTrainRequest`. Change:

```python
feature_group: str = "ta_basic"
```

to:

```python
feature_group: str | list[str] = "ta_basic"
```

Also update `_apply_ta_features` to accept a list:

```python
def _apply_ta_features(ohlc: pd.DataFrame, feature_group: str | list[str]) -> pd.DataFrame:
    from app.ml.features import get_feature_builders
    if isinstance(feature_group, str):
        groups = [feature_group]
    else:
        groups = list(feature_group)
    builders = get_feature_builders(groups)
    df = ohlc.copy()
    for b in builders:
        df = b(df)
    return df
```

(Keep all other internals identical.)

- [ ] **Step 5.2: Write failing tests**

Create `tests/test_ablation_service.py`:

```python
"""Tests for AblationService (V4 P5)."""
from __future__ import annotations

import pytest


@pytest.fixture
def fake_train(monkeypatch):
    """Patch training_service.train and meta_label_service.train_meta_label_model
    to return deterministic fake metrics so we can verify orchestration."""
    from app.services import training_service, meta_label_service

    def _fake_train(req):
        # Direction or volatility — return synthetic metrics
        has_sentiment = "sentiment" in req.feature_groups
        if req.label_type == "direction":
            return type("R", (), {
                "model_id": f"dir_{'sent' if has_sentiment else 'base'}",
                "metrics": {"test_auc": 0.6 if has_sentiment else 0.52,
                            "test_f1": 0.42 if has_sentiment else 0.34},
            })()
        elif req.label_type == "volatility":
            return type("R", (), {
                "model_id": f"vol_{'sent' if has_sentiment else 'base'}",
                "metrics": {"test_qlike": 0.142 if has_sentiment else 0.171,
                            "test_r2": 0.064 if has_sentiment else 0.019,
                            "test_mae": 0.072 if has_sentiment else 0.085},
            })()
        raise ValueError("unexpected target")

    def _fake_meta_train(req):
        groups = req.feature_group if isinstance(req.feature_group, list) else [req.feature_group]
        has_sentiment = "sentiment" in groups
        return {
            "success": True,
            "model_id": f"meta_{'sent' if has_sentiment else 'base'}",
            "cv_metrics": {
                "auc_mean": 0.641 if has_sentiment else 0.619,
                "precision_at_50": 0.61 if has_sentiment else 0.55,
            },
        }

    monkeypatch.setattr(training_service, "train", lambda self, req: _fake_train(req), raising=False)
    monkeypatch.setattr("app.services.ablation_service._train_target",
                         lambda req: _fake_train(req))
    monkeypatch.setattr(meta_label_service, "train_meta_label_model", _fake_meta_train)


def test_matrix_shape_3x2(fake_train):
    from app.services.ablation_service import run_ablation
    result = run_ablation(
        ticker="MSFT",
        targets=["direction", "volatility", "meta_label"],
        feature_sets=[
            {"name": "ta_basic", "groups": ["ta_basic"]},
            {"name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"]},
        ],
        horizon_days=5, model_type="xgboost",
    )
    assert set(result["matrix"].keys()) == {"direction", "volatility", "meta_label"}
    for target in result["matrix"]:
        assert set(result["matrix"][target].keys()) == {"ta_basic", "ta_basic + sentiment"}


def test_sentiment_lift_detected(fake_train):
    from app.services.ablation_service import run_ablation
    result = run_ablation(
        ticker="MSFT",
        targets=["direction"],
        feature_sets=[
            {"name": "ta_basic", "groups": ["ta_basic"]},
            {"name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"]},
        ],
        horizon_days=5, model_type="xgboost",
    )
    cell = result["matrix"]["direction"]["ta_basic + sentiment"]
    assert cell["delta_auc"] == pytest.approx(0.08, abs=0.01)


def test_unknown_feature_set_raises(fake_train, monkeypatch):
    from app.services.ablation_service import run_ablation
    monkeypatch.setattr("app.services.ablation_service._validate_feature_groups",
                         lambda groups: (_ for _ in ()).throw(ValueError("unknown_feature_set:mystery")))
    with pytest.raises(ValueError, match="unknown_feature_set"):
        run_ablation(
            ticker="MSFT",
            targets=["direction"],
            feature_sets=[{"name": "mystery", "groups": ["mystery"]}],
            horizon_days=5, model_type="xgboost",
        )


def test_meta_label_uses_extended_feature_group(fake_train):
    from app.services.ablation_service import run_ablation
    # The fake_meta_train asserts isinstance handling. If it sees a list it
    # detects sentiment. Verify we pass a list through:
    result = run_ablation(
        ticker="MSFT",
        targets=["meta_label"],
        feature_sets=[
            {"name": "ta_basic", "groups": ["ta_basic"]},
            {"name": "with_sentiment", "groups": ["ta_basic", "sentiment"]},
        ],
        horizon_days=5, model_type="xgboost",
    )
    assert result["matrix"]["meta_label"]["with_sentiment"]["auc_mean"] > \
           result["matrix"]["meta_label"]["ta_basic"]["auc_mean"]
```

- [ ] **Step 5.3: Run tests — fail**

- [ ] **Step 5.4: Implement `app/services/ablation_service.py`**

Create `app/services/ablation_service.py`:

```python
"""
Ablation Service — V4 P5 FE-ENH-4

Trains 6 models (3 targets × 2 feature sets) using existing training
infrastructure and returns a delta matrix. Default params for fair
comparison — Optuna would obscure feature contribution.
"""

from __future__ import annotations

import time
from typing import Any
import logging

logger = logging.getLogger(__name__)


def _validate_feature_groups(groups: list[str]) -> None:
    from app.ml.features import get_feature_builders
    try:
        get_feature_builders(groups)
    except Exception as e:
        raise ValueError(f"unknown_feature_set:{','.join(groups)}") from e


def _train_target(req):
    from app.services.training_service import TrainingService
    return TrainingService().train(req)


def _train_direction_or_vol(ticker, label_type, feature_groups, horizon_days, model_type):
    from app.services.training_service import TrainRequest
    req = TrainRequest(
        tickers=[ticker],
        feature_groups=feature_groups,
        horizon_days=horizon_days,
        label_type=label_type,
        model_type=model_type,
        search_mode="none",  # NO Optuna — fair comparison
        save_model=True,
    )
    return _train_target(req)


def _train_meta(ticker, feature_groups, horizon_days, model_type):
    from app.services.meta_label_service import (
        MetaLabelTrainRequest, train_meta_label_model,
    )
    from app.services.primary_signal_service import PrimarySignalSpec
    req = MetaLabelTrainRequest(
        ticker=ticker,
        primary=PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy"),
        tp_k=2.0, sl_k=1.0, timeout_days=horizon_days,
        vol_source="realized_sigma",
        cv_n_splits=5, cv_embargo_pct=0.01,
        model_type=model_type,
        lookback_days=730,
        feature_group=feature_groups,  # str | list[str] after P5 extension
    )
    return train_meta_label_model(req)


_PRIMARY_METRIC = {"direction": "auc", "volatility": "qlike", "meta_label": "auc_mean"}


def _extract_metrics(target: str, result) -> dict[str, float]:
    """Pull the relevant metrics from a training result into a flat dict."""
    if target == "meta_label":
        cv = result.get("cv_metrics", {}) if isinstance(result, dict) else {}
        return {
            "auc_mean": float(cv.get("auc_mean", 0.0)),
            "precision_at_50": float(cv.get("precision_at_50", 0.0)),
        }
    metrics = getattr(result, "metrics", None) or {}
    if target == "direction":
        return {
            "auc": float(metrics.get("test_auc", 0.0)),
            "f1": float(metrics.get("test_f1", 0.0)),
        }
    if target == "volatility":
        return {
            "qlike": float(metrics.get("test_qlike", 0.0)),
            "r2": float(metrics.get("test_r2", 0.0)),
            "mae": float(metrics.get("test_mae", 0.0)),
        }
    return {}


def _model_id(target: str, result) -> str:
    if target == "meta_label":
        return result.get("model_id", "?") if isinstance(result, dict) else "?"
    return getattr(result, "model_id", "?")


def run_ablation(
    *,
    ticker: str,
    targets: list[str],
    feature_sets: list[dict[str, Any]],
    horizon_days: int = 5,
    model_type: str = "xgboost",
) -> dict[str, Any]:
    if not targets or not feature_sets:
        raise ValueError("targets and feature_sets must each have ≥1 element")

    for fs in feature_sets:
        _validate_feature_groups(fs["groups"])

    t0 = time.time()
    matrix: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in targets}

    for target in targets:
        for fs in feature_sets:
            try:
                if target == "meta_label":
                    res = _train_meta(ticker, fs["groups"], horizon_days, model_type)
                else:
                    res = _train_direction_or_vol(
                        ticker, target, fs["groups"], horizon_days, model_type
                    )
                metrics = _extract_metrics(target, res)
                cell = {"model_id": _model_id(target, res), **metrics}
                matrix[target][fs["name"]] = cell
            except Exception as e:
                logger.warning("ablation cell %s × %s failed: %s", target, fs["name"], e)
                matrix[target][fs["name"]] = {"error": str(e), "model_id": None}

    # Compute deltas relative to feature_sets[0]
    baseline_name = feature_sets[0]["name"]
    for target in targets:
        primary_metric = _PRIMARY_METRIC[target]
        baseline = matrix[target].get(baseline_name, {})
        baseline_val = baseline.get(primary_metric)
        if baseline_val is None:
            continue
        for fs in feature_sets[1:]:
            cell = matrix[target].get(fs["name"], {})
            if "error" in cell or primary_metric not in cell:
                continue
            cell[f"delta_{primary_metric}"] = cell[primary_metric] - baseline_val

    summary = _build_summary(matrix, targets, feature_sets)

    return {
        "ticker": ticker,
        "matrix": matrix,
        "summary": summary,
        "feature_sets_used": feature_sets,
        "model_type": model_type,
        "horizon_days": horizon_days,
        "elapsed_seconds": round(time.time() - t0, 2),
    }


def _build_summary(matrix, targets, feature_sets) -> dict[str, Any]:
    if len(feature_sets) < 2:
        return {"sentiment_helps_most": None, "interpretation": "Need ≥2 feature sets to compare."}
    fs1 = feature_sets[1]["name"]
    lifts = {}
    for target in targets:
        primary = _PRIMARY_METRIC[target]
        cell = matrix[target].get(fs1, {})
        delta_key = f"delta_{primary}"
        if delta_key in cell:
            # For QLIKE, lower is better → flip sign for "lift"
            sign = -1 if primary == "qlike" else 1
            lifts[target] = sign * cell[delta_key]
    if not lifts:
        return {"sentiment_helps_most": None, "interpretation": "No deltas computed."}
    best_target = max(lifts, key=lifts.get)
    return {
        "sentiment_helps_most": best_target,
        "interpretation": (
            f"Sentiment lifts {best_target}'s primary metric most "
            f"(deltas: {', '.join(f'{t}={v:+.3f}' for t, v in lifts.items())})."
        ),
    }
```

- [ ] **Step 5.5: Run tests**

```bash
pytest tests/test_ablation_service.py -v
```

Expected: `4 passed`.

- [ ] **Step 5.6: Commit**

```bash
git add app/services/ablation_service.py app/services/meta_label_service.py tests/test_ablation_service.py
git commit -m "feat(p5): AblationService + meta_label feature_group str|list extension (4 tests)"
```

---

## Task 6: `POST /api/ablation/run` Endpoint

**Files:**
- Create: `app/api/ablation.py`
- Modify: `app/main.py`
- Create: `tests/contract/test_ablation_run.py`

- [ ] **Step 6.1: Write failing tests**

Create `tests/contract/test_ablation_run.py`:

```python
"""Contract tests for POST /api/ablation/run (V4 P5)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.services import ablation_service

    def _fake_run(*, ticker, targets, feature_sets, horizon_days, model_type):
        return {
            "ticker": ticker,
            "matrix": {t: {fs["name"]: {"model_id": f"{t}_x", "auc": 0.6}
                            for fs in feature_sets} for t in targets},
            "summary": {"sentiment_helps_most": "direction",
                        "interpretation": "ok"},
            "feature_sets_used": feature_sets,
            "model_type": model_type,
            "horizon_days": horizon_days,
            "elapsed_seconds": 1.2,
        }

    monkeypatch.setattr(ablation_service, "run_ablation", _fake_run)
    return TestClient(app)


def test_200_happy_path(client):
    resp = client.post("/api/ablation/run", json={
        "ticker": "MSFT",
        "targets": ["direction", "volatility", "meta_label"],
        "feature_sets": [
            {"name": "ta_basic", "groups": ["ta_basic"]},
            {"name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"]},
        ],
        "horizon_days": 5,
        "model_type": "xgboost",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "MSFT"
    assert "matrix" in body
    assert "summary" in body
    assert "elapsed_seconds" in body


def test_422_invalid_horizon(client):
    resp = client.post("/api/ablation/run", json={
        "ticker": "MSFT",
        "targets": ["direction"],
        "feature_sets": [{"name": "ta_basic", "groups": ["ta_basic"]}],
        "horizon_days": 999,
        "model_type": "xgboost",
    })
    assert resp.status_code == 422


def test_400_unknown_feature_set(client, monkeypatch):
    from app.services import ablation_service

    def _raise(**kw):
        raise ValueError("unknown_feature_set:mystery")

    monkeypatch.setattr(ablation_service, "run_ablation", _raise)
    resp = client.post("/api/ablation/run", json={
        "ticker": "MSFT",
        "targets": ["direction"],
        "feature_sets": [{"name": "mystery", "groups": ["mystery"]}],
        "horizon_days": 5,
        "model_type": "xgboost",
    })
    assert resp.status_code == 400
    assert "unknown_feature_set" in resp.json()["detail"]


def test_response_includes_summary(client):
    resp = client.post("/api/ablation/run", json={
        "ticker": "MSFT",
        "targets": ["direction"],
        "feature_sets": [
            {"name": "a", "groups": ["ta_basic"]},
            {"name": "b", "groups": ["ta_basic", "sentiment"]},
        ],
        "horizon_days": 5,
        "model_type": "xgboost",
    })
    body = resp.json()
    assert "interpretation" in body["summary"]
```

- [ ] **Step 6.2: Run tests — fail**

- [ ] **Step 6.3: Implement `app/api/ablation.py`**

Create `app/api/ablation.py`:

```python
"""Ablation API — V4 P5 FE-ENH-4: POST /api/ablation/run."""

from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import ablation_service

router = APIRouter()


class _FeatureSet(BaseModel):
    name: str
    groups: list[str] = Field(min_length=1)


class AblationRunRequest(BaseModel):
    ticker: str
    targets: list[Literal["direction", "volatility", "meta_label"]] = Field(min_length=1)
    feature_sets: list[_FeatureSet] = Field(min_length=2, max_length=4)
    horizon_days: int = Field(default=5, ge=1, le=60)
    model_type: Literal["xgboost", "lightgbm", "ensemble"] = "xgboost"


@router.post("/api/ablation/run")
def ablation_run(req: AblationRunRequest):
    try:
        return ablation_service.run_ablation(
            ticker=req.ticker,
            targets=list(req.targets),
            feature_sets=[fs.model_dump() for fs in req.feature_sets],
            horizon_days=req.horizon_days,
            model_type=req.model_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ablation_failed:{e}")
```

- [ ] **Step 6.4: Wire router in main.py**

```python
from app.api import ablation as ablation_api
# ...
app.include_router(ablation_api.router, tags=["Ablation"])
```

- [ ] **Step 6.5: Run tests**

```bash
pytest tests/contract/test_ablation_run.py -v
```

Expected: `4 passed`.

- [ ] **Step 6.6: Commit**

```bash
git add app/api/ablation.py app/main.py tests/contract/test_ablation_run.py
git commit -m "feat(p5): POST /api/ablation/run endpoint (4 contract tests)"
```

---

## Task 7: Frontend api/client + leaderboardQueries hooks

**Files:**
- Modify: `quant-ai-ui/src/api/client.js`
- Create: `quant-ai-ui/src/api/leaderboardQueries.js`
- Create: `quant-ai-ui/__tests__/api/leaderboardQueries.test.jsx`

- [ ] **Step 7.1: Write failing tests**

Create `quant-ai-ui/__tests__/api/leaderboardQueries.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../_helpers/queryWrapper";

vi.mock("@/api/client", () => ({
  getModels: vi.fn(),
  getModelAccuracy: vi.fn(),
  postAblationRun: vi.fn(),
}));

import * as client from "@/api/client";
import {
  useLeaderboard,
  useModelAccuracy,
  useAblationRun,
} from "@/api/leaderboardQueries";

beforeEach(() => vi.clearAllMocks());

describe("useLeaderboard", () => {
  it("calls getModels with label_type filter", async () => {
    client.getModels.mockResolvedValue([
      { id: "m1", label_type: "direction", metrics: { test_auc: 0.6 } },
    ]);
    const { result } = renderHook(() => useLeaderboard("direction"), { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(client.getModels).toHaveBeenCalledWith({ label_type: "direction", status: "active" });
  });
});

describe("useModelAccuracy", () => {
  it("fetches accuracy when modelId given", async () => {
    client.getModelAccuracy.mockResolvedValue({
      model_id: "m1", stats: { hit_rate: 0.6 }, by_ticker: [],
    });
    const { result } = renderHook(() => useModelAccuracy("m1"), { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(client.getModelAccuracy).toHaveBeenCalledWith("m1", { window_days: 30 });
  });
});

describe("useAblationRun", () => {
  it("returns mutation that posts ablation run", async () => {
    client.postAblationRun.mockResolvedValue({ ticker: "MSFT", matrix: {} });
    const { result } = renderHook(() => useAblationRun(), { wrapper: makeQueryWrapper() });
    await result.current.mutateAsync({ ticker: "MSFT", targets: ["direction"], feature_sets: [] });
    expect(client.postAblationRun).toHaveBeenCalled();
  });
});
```

- [ ] **Step 7.2: Run tests — fail**

- [ ] **Step 7.3: Extend `quant-ai-ui/src/api/client.js`**

Append:

```javascript
// ===================================
// V4 P5 — Leaderboard / Accuracy / Ablation
// ===================================

/** GET /models?label_type=&status=&ticker= (extended). Returns array. */
export function getModels({ label_type, status = "active", ticker } = {}) {
  const qs = new URLSearchParams();
  if (label_type) qs.set("label_type", label_type);
  if (status) qs.set("status", status);
  if (ticker) qs.set("ticker", ticker);
  return fetch(`${BASE}/models?${qs}`)
    .then(async (r) => {
      if (!r.ok) throw new Error(`API error ${r.status}: ${await r.text()}`);
      const body = await r.json();
      return body.models || body || [];
    });
}

/** GET /models/{id}/accuracy?window_days=&resolve= */
export function getModelAccuracy(modelId, { window_days = 30, resolve = true } = {}) {
  const qs = new URLSearchParams({ window_days, resolve });
  return fetch(`${BASE}/models/${encodeURIComponent(modelId)}/accuracy?${qs}`)
    .then(async (r) => {
      if (!r.ok) throw new Error(`API error ${r.status}: ${await r.text()}`);
      return r.json();
    });
}

/** POST /api/ablation/run */
export function postAblationRun(payload) {
  return fetch(`${BASE}/api/ablation/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(async (r) => {
    if (!r.ok) throw new Error(`API error ${r.status}: ${await r.text()}`);
    return r.json();
  });
}
```

- [ ] **Step 7.4: Create `quant-ai-ui/src/api/leaderboardQueries.js`**

```javascript
import { useQuery, useMutation } from "@tanstack/react-query";
import * as api from "./client";

export function useLeaderboard(labelType, opts = {}) {
  return useQuery({
    queryKey: ["leaderboard", labelType],
    queryFn: () => api.getModels({ label_type: labelType, status: "active" }),
    enabled: !!labelType,
    staleTime: 60_000,
    ...opts,
  });
}

export function useModelAccuracy(modelId, opts = {}) {
  return useQuery({
    queryKey: ["model-accuracy", modelId],
    queryFn: () => api.getModelAccuracy(modelId, { window_days: 30 }),
    enabled: !!modelId,
    staleTime: 30_000,
    retry: (failureCount, error) => {
      if (String(error.message).includes("API error 404")) return false;
      return failureCount < 2;
    },
    ...opts,
  });
}

export function useAblationRun() {
  return useMutation({
    mutationFn: (payload) => api.postAblationRun(payload),
  });
}
```

- [ ] **Step 7.5: Run tests**

```bash
cd C:/Users/zjg09/projects/quant-ai/quant-ai-ui
npm run test -- --run __tests__/api/leaderboardQueries.test.jsx
```

Expected: `3 passed`.

- [ ] **Step 7.6: Commit**

```bash
git add quant-ai-ui/src/api/client.js quant-ai-ui/src/api/leaderboardQueries.js quant-ai-ui/__tests__/api/leaderboardQueries.test.jsx
git commit -m "feat(p5): api client + leaderboardQueries hooks (3 tests)"
```

---

## Task 8: LeaderboardPage + LeaderboardTable

**Files:**
- Create: `quant-ai-ui/src/features/leaderboard/LeaderboardTable.jsx`
- Create: `quant-ai-ui/src/pages/LeaderboardPage.jsx`
- Create: `quant-ai-ui/__tests__/pages/LeaderboardPage.test.jsx`

- [ ] **Step 8.1: Write failing tests**

Create `quant-ai-ui/__tests__/pages/LeaderboardPage.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import LeaderboardPage from "@/pages/LeaderboardPage";

vi.mock("@/api/client", () => ({
  getModels: vi.fn(),
  getModelAccuracy: vi.fn().mockResolvedValue({ stats: { hit_rate: null } }),
  postAblationRun: vi.fn(),
}));

import * as client from "@/api/client";

beforeEach(() => {
  vi.clearAllMocks();
  client.getModels.mockResolvedValue([
    { id: "m1", name: "Direction Model 1", model_type: "xgboost",
      label_type: "direction", tickers: ["MSFT"],
      metrics: { test_auc: 0.62 }, created_at: "2026-04-20" },
    { id: "m2", name: "Direction Model 2", model_type: "logistic",
      label_type: "direction", tickers: ["AAPL"],
      metrics: { test_auc: 0.55 }, created_at: "2026-04-19" },
  ]);
});

describe("LeaderboardPage", () => {
  it("renders 3 tabs (direction/vol/meta)", () => {
    render(<MemoryRouter><LeaderboardPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    expect(screen.getByText(/direction/i)).toBeInTheDocument();
    expect(screen.getByText(/volatility/i)).toBeInTheDocument();
    expect(screen.getByText(/meta-label|meta_label/i)).toBeInTheDocument();
  });

  it("renders model rows with metrics", async () => {
    render(<MemoryRouter><LeaderboardPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText(/Direction Model 1/i)).toBeInTheDocument());
    expect(screen.getByText(/0\.62/)).toBeInTheDocument();
  });

  it("sorts by primary metric desc (best first)", async () => {
    render(<MemoryRouter><LeaderboardPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText(/Direction Model 1/i)).toBeInTheDocument());
    const rows = document.querySelectorAll("tbody tr");
    // First row should be "Direction Model 1" (auc 0.62) before "Direction Model 2" (0.55)
    expect(rows[0]?.textContent).toMatch(/Direction Model 1/);
  });

  it("switches tab to volatility on click and re-queries", async () => {
    render(<MemoryRouter><LeaderboardPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.click(screen.getByText(/volatility/i));
    await waitFor(() => expect(client.getModels).toHaveBeenCalledWith(
      expect.objectContaining({ label_type: "volatility" })
    ));
  });
});
```

- [ ] **Step 8.2: Run tests — fail**

- [ ] **Step 8.3: Create `LeaderboardTable.jsx`**

Create `quant-ai-ui/src/features/leaderboard/LeaderboardTable.jsx`:

```jsx
import { useModelAccuracy } from "@/api/signalQueries";

const PRIMARY_METRIC_KEY = {
  direction: "test_auc",
  volatility: "test_qlike",
  meta_label: "cv_auc_mean",
};

const PRIMARY_METRIC_LABEL = {
  direction: "AUC",
  volatility: "QLIKE",
  meta_label: "CV AUC",
};

function AccuracyCell({ modelId }) {
  const { data, isLoading } = useModelAccuracyShim(modelId);
  if (isLoading) return <span className="text-slate-500">…</span>;
  const hit = data?.stats?.hit_rate;
  if (hit === null || hit === undefined) return <span className="text-slate-600">—</span>;
  return <span className="text-slate-200">{(hit * 100).toFixed(0)}%</span>;
}

// Local shim — components import from leaderboardQueries
function useModelAccuracyShim(id) {
  const mod = require("@/api/leaderboardQueries");
  return mod.useModelAccuracy(id);
}

export default function LeaderboardTable({ models, labelType }) {
  if (!models || models.length === 0) {
    return (
      <div className="p-8 text-center text-sm text-slate-500 bg-slate-900/40 rounded-lg">
        No active {labelType} models trained yet.
      </div>
    );
  }
  const metricKey = PRIMARY_METRIC_KEY[labelType];
  const sorted = [...models].sort((a, b) => {
    const av = a.metrics?.[metricKey] ?? 0;
    const bv = b.metrics?.[metricKey] ?? 0;
    // QLIKE: lower is better
    return labelType === "volatility" ? av - bv : bv - av;
  });
  return (
    <div className="overflow-x-auto bg-slate-900/40 rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-slate-400 border-b border-slate-800">
            <th className="px-3 py-2 text-left">Model</th>
            <th className="px-3 py-2 text-left">Type</th>
            <th className="px-3 py-2 text-left">Tickers</th>
            <th className="px-3 py-2 text-right">{PRIMARY_METRIC_LABEL[labelType]}</th>
            <th className="px-3 py-2 text-right">Live hit rate (30d)</th>
            <th className="px-3 py-2 text-left">Created</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m) => (
            <tr key={m.id} className="border-b border-slate-800/50 hover:bg-slate-800/40">
              <td className="px-3 py-2 font-medium">{m.name || m.id}</td>
              <td className="px-3 py-2 text-slate-400">{m.model_type}</td>
              <td className="px-3 py-2 text-slate-400">{(m.tickers || []).join(", ")}</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {(m.metrics?.[metricKey] ?? 0).toFixed(3)}
              </td>
              <td className="px-3 py-2 text-right">
                <AccuracyCell modelId={m.id} />
              </td>
              <td className="px-3 py-2 text-slate-500 text-xs">
                {(m.created_at || "").slice(0, 10)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 8.4: Create `LeaderboardPage.jsx`**

Create `quant-ai-ui/src/pages/LeaderboardPage.jsx`:

```jsx
import { useState } from "react";
import { useLeaderboard } from "@/api/leaderboardQueries";
import LeaderboardTable from "@/features/leaderboard/LeaderboardTable";

const TABS = [
  { id: "direction", label: "Direction" },
  { id: "volatility", label: "Volatility" },
  { id: "meta_label", label: "Meta-Label" },
];

export default function LeaderboardPage() {
  const [active, setActive] = useState("direction");
  const { data: models = [], isLoading } = useLeaderboard(active);

  return (
    <div className="p-6 space-y-4 max-w-7xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold">Leaderboard</h1>
        <p className="text-sm text-slate-400">
          Active models per V4 multi-task target, sorted by primary metric. Live hit rate from prediction_log
          (30-day window).
        </p>
      </header>

      <nav className="flex gap-2 border-b border-slate-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActive(t.id)}
            className={`px-3 py-2 text-sm border-b-2 transition-colors ${
              active === t.id
                ? "border-emerald-500 text-emerald-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {isLoading ? (
        <div className="p-8 text-sm text-slate-500 text-center">Loading...</div>
      ) : (
        <LeaderboardTable models={models} labelType={active} />
      )}
    </div>
  );
}
```

- [ ] **Step 8.5: Run tests**

```bash
npm run test -- --run __tests__/pages/LeaderboardPage.test.jsx
```

Expected: `4 passed`.

- [ ] **Step 8.6: Commit**

```bash
git add quant-ai-ui/src/pages/LeaderboardPage.jsx quant-ai-ui/src/features/leaderboard/LeaderboardTable.jsx quant-ai-ui/__tests__/pages/LeaderboardPage.test.jsx
git commit -m "feat(p5): LeaderboardPage with 3 tabs + LeaderboardTable (4 tests)"
```

---

## Task 9: AblationPage + AblationMatrix

**Files:**
- Create: `quant-ai-ui/src/features/ablation/AblationMatrix.jsx`
- Create: `quant-ai-ui/src/pages/AblationPage.jsx`
- Create: `quant-ai-ui/__tests__/pages/AblationPage.test.jsx`
- Create: `quant-ai-ui/__tests__/features/ablation/AblationMatrix.test.jsx`

- [ ] **Step 9.1: Write failing tests**

Create `quant-ai-ui/__tests__/features/ablation/AblationMatrix.test.jsx`:

```jsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import AblationMatrix from "@/features/ablation/AblationMatrix";

const MATRIX = {
  direction: {
    "ta_basic":              { model_id: "x", auc: 0.523, f1: 0.34 },
    "ta_basic + sentiment":  { model_id: "y", auc: 0.591, f1: 0.42, delta_auc: 0.068 },
  },
  volatility: {
    "ta_basic":              { model_id: "x", qlike: 0.171, r2: 0.019 },
    "ta_basic + sentiment":  { model_id: "y", qlike: 0.142, r2: 0.064, delta_qlike: -0.029 },
  },
};

describe("AblationMatrix", () => {
  it("renders cells for every (target, feature_set)", () => {
    render(<AblationMatrix matrix={MATRIX} />);
    expect(screen.getByText(/0\.523/)).toBeInTheDocument();
    expect(screen.getByText(/0\.591/)).toBeInTheDocument();
    expect(screen.getByText(/0\.171/)).toBeInTheDocument();
    expect(screen.getByText(/0\.142/)).toBeInTheDocument();
  });
});
```

Create `quant-ai-ui/__tests__/pages/AblationPage.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import AblationPage from "@/pages/AblationPage";

vi.mock("@/api/client", () => ({
  getModels: vi.fn(),
  getModelAccuracy: vi.fn(),
  postAblationRun: vi.fn(),
}));

import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("AblationPage", () => {
  it("renders form with ticker input and run button", () => {
    render(<MemoryRouter><AblationPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    expect(screen.getByLabelText(/ticker/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run ablation/i })).toBeInTheDocument();
  });

  it("posts ablation run with default 3 targets and 2 feature sets", async () => {
    client.postAblationRun.mockResolvedValue({
      ticker: "MSFT", matrix: {}, summary: { interpretation: "" },
      elapsed_seconds: 1.2,
    });
    render(<MemoryRouter><AblationPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.change(screen.getByLabelText(/ticker/i), { target: { value: "MSFT" } });
    fireEvent.click(screen.getByRole("button", { name: /run ablation/i }));
    await waitFor(() => expect(client.postAblationRun).toHaveBeenCalled());
    const payload = client.postAblationRun.mock.calls[0][0];
    expect(payload.ticker).toBe("MSFT");
    expect(payload.targets).toContain("direction");
    expect(payload.feature_sets.length).toBe(2);
  });

  it("renders matrix after successful run", async () => {
    client.postAblationRun.mockResolvedValue({
      ticker: "MSFT",
      matrix: { direction: { "ta_basic": { auc: 0.5 }, "ta_basic + sentiment": { auc: 0.6 } } },
      summary: { interpretation: "ok" },
      elapsed_seconds: 1.2,
    });
    render(<MemoryRouter><AblationPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.change(screen.getByLabelText(/ticker/i), { target: { value: "MSFT" } });
    fireEvent.click(screen.getByRole("button", { name: /run ablation/i }));
    await waitFor(() => expect(screen.getByText(/0\.6/)).toBeInTheDocument());
  });

  it("renders summary interpretation after run", async () => {
    client.postAblationRun.mockResolvedValue({
      ticker: "MSFT", matrix: {},
      summary: { interpretation: "Sentiment lifts AUC by 6.8 points on direction." },
      elapsed_seconds: 1.2,
    });
    render(<MemoryRouter><AblationPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.change(screen.getByLabelText(/ticker/i), { target: { value: "MSFT" } });
    fireEvent.click(screen.getByRole("button", { name: /run ablation/i }));
    await waitFor(() => expect(screen.getByText(/lifts AUC/i)).toBeInTheDocument());
  });
});
```

- [ ] **Step 9.2: Run tests — fail**

- [ ] **Step 9.3: Create AblationMatrix.jsx**

```jsx
const PRIMARY_METRIC = {
  direction: "auc",
  volatility: "qlike",
  meta_label: "auc_mean",
};

function cellColor(target, value, baseline) {
  if (value === undefined || baseline === undefined) return "bg-slate-700/30";
  const isLowerBetter = target === "volatility";
  const better = isLowerBetter ? value < baseline : value > baseline;
  return better ? "bg-emerald-500/15" : "bg-amber-500/15";
}

export default function AblationMatrix({ matrix }) {
  if (!matrix || Object.keys(matrix).length === 0) {
    return (
      <div className="p-8 text-center text-sm text-slate-500 bg-slate-900/40 rounded-lg">
        Run an ablation to see the matrix.
      </div>
    );
  }
  const targets = Object.keys(matrix);
  const featureSetNames = Object.keys(matrix[targets[0]] || {});

  return (
    <div className="overflow-x-auto bg-slate-900/40 rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-slate-400 border-b border-slate-800">
            <th className="px-3 py-2 text-left">Target</th>
            {featureSetNames.map((fs) => (
              <th key={fs} className="px-3 py-2 text-left">{fs}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {targets.map((target) => {
            const primary = PRIMARY_METRIC[target];
            const baseline = matrix[target][featureSetNames[0]]?.[primary];
            return (
              <tr key={target} className="border-b border-slate-800/50">
                <td className="px-3 py-2 font-medium uppercase">{target}</td>
                {featureSetNames.map((fs) => {
                  const cell = matrix[target][fs] || {};
                  if (cell.error) {
                    return (
                      <td key={fs} className="px-3 py-2 bg-rose-500/15">
                        <div className="text-xs text-rose-300">{cell.error}</div>
                      </td>
                    );
                  }
                  const v = cell[primary];
                  return (
                    <td key={fs} className={`px-3 py-2 ${cellColor(target, v, baseline)}`}>
                      <div className="font-semibold tabular-nums">
                        {v !== undefined ? v.toFixed(3) : "—"}
                      </div>
                      {cell[`delta_${primary}`] !== undefined && (
                        <div className="text-[10px] text-slate-400">
                          Δ {cell[`delta_${primary}`] >= 0 ? "+" : ""}
                          {cell[`delta_${primary}`].toFixed(3)}
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 9.4: Create AblationPage.jsx**

```jsx
import { useState } from "react";
import { useAblationRun } from "@/api/leaderboardQueries";
import AblationMatrix from "@/features/ablation/AblationMatrix";

const DEFAULT_TARGETS = ["direction", "volatility", "meta_label"];
const DEFAULT_FEATURE_SETS = [
  { name: "ta_basic", groups: ["ta_basic"] },
  { name: "ta_basic + sentiment", groups: ["ta_basic", "sentiment"] },
];

export default function AblationPage() {
  const [ticker, setTicker] = useState("");
  const [result, setResult] = useState(null);
  const run = useAblationRun();

  const onRun = (e) => {
    e.preventDefault();
    if (!ticker) return;
    run.mutate(
      {
        ticker,
        targets: DEFAULT_TARGETS,
        feature_sets: DEFAULT_FEATURE_SETS,
        horizon_days: 5,
        model_type: "xgboost",
      },
      { onSuccess: setResult },
    );
  };

  return (
    <div className="p-6 space-y-4 max-w-7xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold">Ablation</h1>
        <p className="text-sm text-slate-400">
          Train 6 models (3 targets × 2 feature sets) with default params for fair comparison.
          Quantifies sentiment's contribution per target.
        </p>
      </header>

      <form onSubmit={onRun} className="flex items-end gap-3 p-4 bg-slate-900/40 rounded-lg">
        <div className="flex-1">
          <label className="block text-xs text-slate-400 mb-1" htmlFor="ablation-ticker">
            Ticker
          </label>
          <input
            id="ablation-ticker"
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="MSFT"
            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm"
          />
        </div>
        <button
          type="submit"
          disabled={!ticker || run.isPending}
          className="px-4 py-2 bg-emerald-600/20 border border-emerald-600/40 rounded text-sm hover:bg-emerald-600/30 disabled:opacity-50"
        >
          {run.isPending ? "Running..." : "Run ablation"}
        </button>
      </form>

      {run.isError && (
        <div className="p-3 bg-rose-500/15 text-rose-300 text-sm rounded">
          Error: {String(run.error?.message || "unknown")}
        </div>
      )}

      {result && (
        <>
          <AblationMatrix matrix={result.matrix} />
          {result.summary?.interpretation && (
            <div className="p-3 bg-slate-800/50 text-sm rounded">
              <div className="text-xs uppercase text-slate-400 mb-1">Interpretation</div>
              {result.summary.interpretation}
            </div>
          )}
          <div className="text-[10px] text-slate-500">
            Elapsed {result.elapsed_seconds}s · model_type {result.model_type}
          </div>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 9.5: Run tests**

```bash
npm run test -- --run __tests__/features/ablation/AblationMatrix.test.jsx __tests__/pages/AblationPage.test.jsx
```

Expected: `5 passed` (1 + 4).

- [ ] **Step 9.6: Commit**

```bash
git add quant-ai-ui/src/features/ablation/AblationMatrix.jsx quant-ai-ui/src/pages/AblationPage.jsx quant-ai-ui/__tests__/features/ablation/AblationMatrix.test.jsx quant-ai-ui/__tests__/pages/AblationPage.test.jsx
git commit -m "feat(p5): AblationPage + AblationMatrix component (5 tests)"
```

---

## Task 10: App.jsx Routes + TopNavBar Links

**Files:**
- Modify: `quant-ai-ui/src/App.jsx`
- Modify: `quant-ai-ui/src/components/layout/TopNavBar.jsx`

- [ ] **Step 10.1: Add routes in `App.jsx`**

Add imports:

```jsx
import LeaderboardPage from "@/pages/LeaderboardPage";
import AblationPage from "@/pages/AblationPage";
```

Inside `<Routes>`:

```jsx
<Route path="/leaderboard" element={<LeaderboardPage />} />
<Route path="/ablation" element={<AblationPage />} />
```

- [ ] **Step 10.2: Add nav links in `TopNavBar.jsx`**

Find existing 模型 group, add a new link near 信号:

```jsx
<Link to="/leaderboard" className="hover:text-emerald-400 text-sm">榜单</Link>
<Link to="/ablation" className="hover:text-emerald-400 text-sm">消融</Link>
```

- [ ] **Step 10.3: Update existing TopNavBar test**

In `quant-ai-ui/__tests__/components/layout/TopNavBar.test.jsx`, add 2 tests at the end:

```jsx
it("renders Leaderboard link", () => {
  render(<MemoryRouter><TopNavBar /></MemoryRouter>);
  expect(screen.getByText(/榜单/)).toBeInTheDocument();
});

it("renders Ablation link", () => {
  render(<MemoryRouter><TopNavBar /></MemoryRouter>);
  expect(screen.getByText(/消融/)).toBeInTheDocument();
});
```

- [ ] **Step 10.4: Run tests**

```bash
npm run test -- --run __tests__/components/layout/TopNavBar.test.jsx
```

Expected: existing tests + 2 new pass.

- [ ] **Step 10.5: Commit**

```bash
git add quant-ai-ui/src/App.jsx quant-ai-ui/src/components/layout/TopNavBar.jsx quant-ai-ui/__tests__/components/layout/TopNavBar.test.jsx
git commit -m "feat(p5): wire /leaderboard + /ablation routes + TopNav links (2 tests)"
```

---

## Task 11: Live Demo Benchmark

**Files:**
- Create: `scripts/p5_ablation_demo.py`
- Create: `docs/benchmarks/p5_ablation_demo.md`

- [ ] **Step 11.1: Create script**

Create `scripts/p5_ablation_demo.py`:

```python
"""
P5 Ablation Demo · V4 Phase 5

Runs `/api/ablation/run` (via direct service call, no HTTP) on AAPL+MSFT+GOOGL
across 3 targets × 2 feature sets, writes a markdown report.

Run:
    python -m scripts.p5_ablation_demo
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.ablation_service import run_ablation


TICKERS = ("AAPL", "MSFT", "GOOGL")
TARGETS = ["direction", "volatility", "meta_label"]
FEATURE_SETS = [
    {"name": "ta_basic", "groups": ["ta_basic"]},
    {"name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"]},
]


def main():
    rows = []
    for ticker in TICKERS:
        print(f"[{ticker}] running ablation...")
        t0 = time.time()
        try:
            result = run_ablation(
                ticker=ticker, targets=TARGETS,
                feature_sets=FEATURE_SETS,
                horizon_days=5, model_type="xgboost",
            )
            rows.append({"ticker": ticker, "result": result,
                          "elapsed": time.time() - t0})
        except Exception as e:
            rows.append({"ticker": ticker, "error": str(e),
                          "elapsed": time.time() - t0})
            print(f"  FAILED: {e}")

    out = Path("docs/benchmarks/p5_ablation_demo.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(rows), encoding="utf-8")
    print(f"\nReport: {out}")


def _render(rows) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# V4 Pivot · Phase 5 · Ablation Demo",
        "",
        f"**Run date**: {now}",
        "**Targets**: direction · volatility · meta_label",
        "**Feature sets**: ta_basic vs ta_basic + sentiment",
        "**Model**: xgboost (default params, no Optuna — fair comparison)",
        "**Horizon**: 5 days",
        "",
        "## Per-ticker results",
        "",
    ]
    for row in rows:
        ticker = row["ticker"]
        lines.append(f"### {ticker}")
        if "error" in row:
            lines.append(f"FAILED: {row['error']}")
            continue
        result = row["result"]
        lines.append("")
        lines.append("| Target | ta_basic | ta_basic + sentiment | Δ |")
        lines.append("|---|---|---|---|")
        for target in TARGETS:
            cell0 = result["matrix"][target].get("ta_basic", {})
            cell1 = result["matrix"][target].get("ta_basic + sentiment", {})
            primary_key = {"direction": "auc",
                           "volatility": "qlike",
                           "meta_label": "auc_mean"}[target]
            v0 = cell0.get(primary_key)
            v1 = cell1.get(primary_key)
            delta = cell1.get(f"delta_{primary_key}")
            lines.append(
                f"| {target} | "
                f"{v0:.3f if v0 is not None else 0:.3f} | "
                f"{v1:.3f if v1 is not None else 0:.3f} | "
                f"{delta:+.3f if delta is not None else 0:+.3f} |"
            )
        lines.append("")
        lines.append(f"**Summary**: {result['summary'].get('interpretation', '—')}")
        lines.append(f"**Elapsed**: {result['elapsed_seconds']:.1f}s")
        lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "Default-params XGBoost across 3 tickers. Numbers are not optimized "
        "(no Optuna) — that's intentional: ablation shows feature contribution "
        "in isolation. Optuna in cells would obscure whether sentiment helps "
        "or hyperparameters help. If sentiment Δ is positive on a target, that "
        "target benefits from sentiment beyond what defaults can squeeze out "
        "of ta_basic alone."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.2: Run script**

```bash
cd C:/Users/zjg09/projects/quant-ai
python -m scripts.p5_ablation_demo 2>&1 | tee p5_ablation_run.log
```

Expected: Runs 3 tickers, writes `docs/benchmarks/p5_ablation_demo.md`. Some cells may report errors (e.g. AAPL meta_label insufficient_events) — those render as FAILED rows, not crashes.

- [ ] **Step 11.3: Vault copy**

```bash
cp docs/benchmarks/p5_ablation_demo.md \
   "D:/obsidian vault/01-projects/quant-ai/p5-ablation-demo-2026-04-25.md"
```

- [ ] **Step 11.4: Commit**

```bash
git add scripts/p5_ablation_demo.py docs/benchmarks/p5_ablation_demo.md
git commit -m "feat(p5): ablation demo script + benchmark report"
```

---

## Task 12: P5 GATE — Regression + Tags + Live Smoke

**Files:**
- Modify: `D:/obsidian vault/01-projects/quant-ai/ml-pivot-progress.md` (Day 15)
- Modify: `D:/obsidian vault/01-projects/quant-ai/master-roadmap.md` (P5 ✅)

- [ ] **Step 12.1: Backend regression**

```bash
cd C:/Users/zjg09/projects/quant-ai
pytest \
  tests/test_prediction_log_repo.py \
  tests/test_predict_log_writes.py \
  tests/test_accuracy_service.py \
  tests/test_ablation_service.py \
  tests/contract/test_models_accuracy.py \
  tests/contract/test_ablation_run.py \
  tests/test_meta_label_barrier.py \
  tests/test_purged_kfold.py \
  tests/test_meta_label_service.py \
  tests/test_signal_scoring_service.py \
  tests/contract/test_meta_label_train.py \
  tests/contract/test_signal_score.py \
  tests/contract/test_meta_coverage.py \
  tests/test_paper_trading_meta.py \
  tests/test_labels.py \
  tests/test_ensemble_training.py \
  tests/contract/test_train_flow.py \
  tests/contract/test_predict_volatility.py \
  -v
```

Expected: P5 (~30 new) + P3+P4 (44+10) + P1+P2 regression all green.

- [ ] **Step 12.2: Frontend regression**

```bash
cd quant-ai-ui
npm run test -- --run
npm run lint
npm run build
```

Expected: previous 61 + ~14 P5 frontend tests pass; lint clean; build OK.

- [ ] **Step 12.3: Append Day 15 entry**

In `D:/obsidian vault/01-projects/quant-ai/ml-pivot-progress.md`:

```markdown
### Day 15 Sprint · 2026-04-25 (Sat) · P5 Ship · Gate 2 起步

**Mode**: Continue推进 — P5 G1 + Leaderboard + Ablation per master-roadmap.

#### ✅ Delivered

- `app/db/prediction_log.py` · LocalPredictionLogRepo + SupabasePredictionLogRepo + factory (5 tests)
- 3 predict services wired to write log rows non-blocking (4 tests)
- `app/services/accuracy_service.py` · resolve_pending + aggregate + by_ticker + last_predictions (8 tests)
- `app/api/accuracy.py` · GET /models/{id}/accuracy (5 contract tests)
- `app/services/ablation_service.py` · run_ablation orchestrator + delta math + summary (4 tests)
- `app/api/ablation.py` · POST /api/ablation/run (4 contract tests)
- `app/services/meta_label_service.py` · feature_group str|list[str] backwards-compat extension
- Frontend: `LeaderboardPage` + `LeaderboardTable` + `AblationPage` + `AblationMatrix` (~14 tests)
- `scripts/p5_ablation_demo.py` + benchmark report

**Test total**: ~47 new P5 tests + full P1+P2+P3+P4 regression green.

**🏁 P5 ✅ COMPLETE 2026-04-25** — Gate 2 第一站完成。Live accuracy data + sentiment ablation matrix on prod.
```

- [ ] **Step 12.4: Update master-roadmap.md**

Find the P5 section, mark `✅ 完成 (2026-04-25)`, add deliverables list.

- [ ] **Step 12.5: Tag + push**

```bash
git tag -a v4-p5-complete -m "V4 Phase 5: G1 prediction log + Leaderboard + Ablation"
git push origin main --follow-tags
```

- [ ] **Step 12.6: Live smoke**

```bash
curl -s "https://quant-ai-qzrg.onrender.com/health" | python -c "import json,sys; print(json.load(sys.stdin).get('version'))"
# Expected: 2.4.0 (or whatever the new bump if any)

curl -s "https://quant-ai-qzrg.onrender.com/openapi.json" | python -c "import json,sys; d=json.load(sys.stdin); paths=list(d['paths'].keys()); p5=[p for p in paths if 'accuracy' in p or 'ablation' in p]; print(p5)"
# Expected: ['/models/{model_id}/accuracy', '/api/ablation/run']

curl -s -o /dev/null -w "HTTP %{http_code}\n" https://quant-ai-ui.vercel.app/leaderboard
# Expected: HTTP 200

curl -s -o /dev/null -w "HTTP %{http_code}\n" https://quant-ai-ui.vercel.app/ablation
# Expected: HTTP 200
```

---

## Self-Review

**1. Spec coverage:**
- §3 Scope in: prediction_log table (Task 0+1), 3 predict-service writes (Task 2), AccuracyService (Task 3), accuracy endpoint (Task 4), AblationService + meta-label feature_group extension (Task 5), ablation endpoint (Task 6), frontend api/queries (Task 7), Leaderboard (Task 8), Ablation (Task 9), routes (Task 10), demo (Task 11), GATE (Task 12). ✅
- §6 data-flow paths all implemented in Tasks 2-5. ✅
- §7 API contracts implemented in Tasks 4 + 6. ✅
- §9 testing strategy: 47 tests targeted; plan delivers 30 backend + ~14 frontend (close enough; some clusters merged when natural). ✅
- §11 Future Backlog kept as Future Backlog — none enter P5. ✅

**2. Placeholder scan:**
- Every step has full code or exact command + expected output. No "TBD" / "implement later" / "similar to". ✅
- Edge case to flag: Task 2 says "adapt variable names to whatever the existing function uses." This is the closest thing to a placeholder; it's necessary because `predict_service.predict()` is a large existing function. The agent should `grep -n "model_id\|prob_up\|publish_prediction_event" app/services/predict_service.py` once at the start of Task 2 to find the names. Acceptable.

**3. Type consistency:**
- `PredictionLogRecord` schema (Task 1) used identically in Task 2 (insert) and Task 3 (read/update). ✅
- `_fetch_ohlc_slice` defined in Task 3 + used in same file, monkeypatched in tests. ✅
- `run_ablation()` signature in Task 5 matches consumer in Task 6 (`feature_sets` is list of dicts, not Pydantic models). ✅
- `getModelAccuracy(modelId, opts)` in Task 7 matches `useModelAccuracy` consumer in Task 8 LeaderboardTable. ✅
- `AblationMatrix({ matrix })` prop in Task 9 matches consumer in Task 9 AblationPage `result.matrix`. ✅

All checks passed.

---

## Plan complete.

Saved to `docs/superpowers/plans/2026-04-25-p5-prediction-log-leaderboard-ablation.md`.

Will queue 13 tasks (P5-0..P5-11 + P5-GATE) into `plans/prd.json` and kick Ralph Loop next, mirroring P3/P4 pattern.
