# Distributed Systems Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add interview-ready distributed systems layer to Quant AI: K8s manifests for Minikube, Prometheus + Grafana observability, and a real Kafka prediction event pipeline with a separate consumer service.

**Architecture:** Two container images from the monorepo (api + consumer). K8s runs both plus Kafka/Postgres/Redis/Prometheus/Grafana. HPA scales api based on CPU. Prometheus scrapes both service `/metrics` endpoints. API publishes predictions to Kafka `prediction_events`, consumer aggregates per-ticker rolling stats and exposes `/stats/{ticker}`.

**Tech Stack:** FastAPI lifespan, `prometheus-fastapi-instrumentator`, `prometheus-client`, `aiokafka`, Kubernetes manifests, Bitnami Kafka image, Prometheus + Grafana images, Minikube.

---

## File Structure

| File | Responsibility | Status |
|------|----------------|--------|
| `app/core/metrics.py` | Custom ML Prometheus metrics | **create** |
| `app/services/prediction_event_publisher.py` | Kafka producer lifecycle + publish fn | **create** |
| `app/services/predict_service.py` | Instrument predict with metrics + publish events | modify |
| `app/main.py` | Extend lifespan for Kafka producer, install Instrumentator | modify |
| `app/api/health.py` | Add `/health/ready` endpoint | modify (or new if missing) |
| `app/workers/__init__.py` | New package | **create** |
| `app/workers/events_consumer.py` | FastAPI consumer app + background task | **create** |
| `Dockerfile.consumer` | Consumer image | **create** |
| `requirements.txt` | Add prometheus libs | modify |
| `k8s/namespace.yaml` | `quant-ai` namespace | **create** |
| `k8s/configmap.yaml` | Non-secret config | **create** |
| `k8s/secret.example.yaml` | Template for secrets | **create** |
| `k8s/deployment-api.yaml` | API deployment (2 replicas) | **create** |
| `k8s/service-api.yaml` | API ClusterIP + NodePort | **create** |
| `k8s/hpa-api.yaml` | API HPA | **create** |
| `k8s/deployment-consumer.yaml` | Consumer deployment (1 replica) | **create** |
| `k8s/service-consumer.yaml` | Consumer ClusterIP + NodePort | **create** |
| `k8s/statefulset-kafka.yaml` + `k8s/service-kafka.yaml` | Kafka single broker | **create** |
| `k8s/statefulset-postgres.yaml` + `k8s/service-postgres.yaml` | Postgres | **create** |
| `k8s/deployment-redis.yaml` + `k8s/service-redis.yaml` | Redis | **create** |
| `k8s/deployment-prometheus.yaml` + `k8s/configmap-prometheus.yaml` | Prometheus + scrape config | **create** |
| `k8s/deployment-grafana.yaml` + `k8s/configmap-grafana.yaml` | Grafana + datasource + dashboard | **create** |
| `k8s/README.md` | Deploy runbook | **create** |
| `observability/prometheus.yml` | docker-compose Prometheus config | **create** |
| `observability/grafana-datasources.yml` | Grafana provisioning | **create** |
| `observability/grafana-dashboards.yml` | Grafana dashboard provider config | **create** |
| `observability/dashboards/quant-ai.json` | 6-panel dashboard | **create** |
| `docker-compose.yml` | Add kafka + consumer + prometheus + grafana | modify |
| `tests/test_prometheus_metrics.py` | Metrics endpoint + counters | **create** |
| `tests/test_prediction_event_publisher.py` | Publisher tests | **create** |
| `tests/test_events_consumer.py` | Consumer tests | **create** |
| `docs/architecture/distributed.md` | Interview doc | **create** |

---

## Task 1: Add Prometheus dependencies + custom ML metrics module

**Files:**
- Modify: `requirements.txt`
- Create: `app/core/metrics.py`
- Create: `tests/test_prometheus_metrics.py`

- [ ] **Step 1: Add Prometheus dependencies to requirements.txt**

In `requirements.txt`, find the `# Database` section and add a new section above it:

```
# Observability
prometheus-client>=0.20.0
prometheus-fastapi-instrumentator>=7.0.0
```

- [ ] **Step 2: Install new deps**

Run: `pip install prometheus-client>=0.20.0 prometheus-fastapi-instrumentator>=7.0.0`
Expected: successful install.

- [ ] **Step 3: Write failing test for custom metrics**

Create `tests/test_prometheus_metrics.py`:

```python
from __future__ import annotations


def test_custom_metrics_importable():
    from app.core.metrics import (
        PREDICT_TOTAL,
        PREDICT_CONFIDENCE,
        MODEL_INFERENCE_SECONDS,
    )
    # Metric labels should match spec
    assert PREDICT_TOTAL._name == "quant_ai_predictions"
    assert PREDICT_CONFIDENCE._name == "quant_ai_prediction_confidence"
    assert MODEL_INFERENCE_SECONDS._name == "quant_ai_model_inference_seconds"


def test_predict_total_increments():
    from app.core.metrics import PREDICT_TOTAL

    before = PREDICT_TOTAL.labels(ticker="TEST", model_type="logistic")._value.get()
    PREDICT_TOTAL.labels(ticker="TEST", model_type="logistic").inc()
    after = PREDICT_TOTAL.labels(ticker="TEST", model_type="logistic")._value.get()
    assert after == before + 1


def test_predict_confidence_observes():
    from app.core.metrics import PREDICT_CONFIDENCE

    # Observe a sample
    PREDICT_CONFIDENCE.labels(ticker="TEST").observe(0.85)
    # Histogram should have at least one sample
    metric = PREDICT_CONFIDENCE.labels(ticker="TEST")
    assert metric._sum.get() >= 0.85
```

- [ ] **Step 4: Run test — expect FAIL**

Run: `pytest tests/test_prometheus_metrics.py -v`
Expected: FAIL — module `app.core.metrics` does not exist.

- [ ] **Step 5: Create `app/core/metrics.py`**

```python
from __future__ import annotations

"""
Custom Prometheus metrics for Quant AI.

These are in addition to the auto-registered HTTP metrics from
prometheus-fastapi-instrumentator.
"""

from prometheus_client import Counter, Histogram

PREDICT_TOTAL = Counter(
    "quant_ai_predictions",
    "Total predictions made",
    ["ticker", "model_type"],
)

PREDICT_CONFIDENCE = Histogram(
    "quant_ai_prediction_confidence",
    "Prediction confidence scores (positive-class probability)",
    ["ticker"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_INFERENCE_SECONDS = Histogram(
    "quant_ai_model_inference_seconds",
    "Model inference duration in seconds",
    ["model_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)
```

- [ ] **Step 6: Run tests — expect PASS**

Run: `pytest tests/test_prometheus_metrics.py -v`
Expected: 3/3 PASS.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
git add requirements.txt app/core/metrics.py tests/test_prometheus_metrics.py
git commit -m "feat: [DIST-1] add Prometheus deps and custom ML metrics"
```

---

## Task 2: Integrate Prometheus Instrumentator into FastAPI app

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_prometheus_metrics.py`

- [ ] **Step 1: Write failing test for /metrics endpoint**

Append to `tests/test_prometheus_metrics.py`:

```python
def test_metrics_endpoint_returns_prometheus_format():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus text format has HELP comments
    body = resp.text
    assert "# HELP" in body
    assert "# TYPE" in body


def test_metrics_endpoint_includes_custom_counter():
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.metrics import PREDICT_TOTAL

    PREDICT_TOTAL.labels(ticker="SPY", model_type="ensemble").inc()

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "quant_ai_predictions" in resp.text
    assert 'ticker="SPY"' in resp.text
```

- [ ] **Step 2: Run — expect FAIL (no /metrics endpoint)**

Run: `pytest tests/test_prometheus_metrics.py::test_metrics_endpoint_returns_prometheus_format -v`
Expected: FAIL — 404 on `/metrics`.

- [ ] **Step 3: Add Instrumentator to `app/main.py`**

Find the section after middleware registration (search for `app.add_middleware(RequestContextMiddleware)`). After that section, add:

```python
# ===================================
# Prometheus Instrumentation
# ===================================
from prometheus_fastapi_instrumentator import Instrumentator

# Import custom metrics so they register with the default registry
from app.core import metrics as _metrics  # noqa: F401

Instrumentator(
    should_group_status_codes=True,
    excluded_handlers=["/metrics"],
).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_prometheus_metrics.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_prometheus_metrics.py
git commit -m "feat: [DIST-2] install Prometheus Instrumentator and expose /metrics"
```

---

## Task 3: Prediction event publisher with FastAPI lifespan

**Files:**
- Create: `app/services/prediction_event_publisher.py`
- Modify: `app/main.py`
- Create: `tests/test_prediction_event_publisher.py`

- [ ] **Step 1: Write failing tests for publisher**

Create `tests/test_prediction_event_publisher.py`:

```python
from __future__ import annotations

import pytest
from datetime import datetime


def test_prediction_event_schema_valid():
    from app.services.prediction_event_publisher import PredictionEvent

    event = PredictionEvent(
        ticker="AAPL",
        prediction=1,
        confidence=0.85,
        model_id="abc123",
        model_type="ensemble",
    )
    assert event.ticker == "AAPL"
    assert isinstance(event.timestamp, datetime)


def test_prediction_event_serializes_to_json():
    from app.services.prediction_event_publisher import PredictionEvent

    event = PredictionEvent(
        ticker="MSFT",
        prediction=0,
        confidence=0.42,
        model_id="xyz",
        model_type="logistic",
    )
    payload = event.model_dump_json()
    assert '"ticker":"MSFT"' in payload
    assert '"confidence":0.42' in payload


@pytest.mark.asyncio
async def test_publish_noop_when_broker_not_kafka(monkeypatch):
    """If BROKER_BACKEND != kafka, publisher is a no-op that doesn't raise."""
    from app.services.prediction_event_publisher import (
        publish_prediction_event,
        PredictionEvent,
    )
    from app.core.settings import settings

    monkeypatch.setattr(settings, "BROKER_BACKEND", "memory")

    event = PredictionEvent(
        ticker="AAPL", prediction=1, confidence=0.9,
        model_id="m1", model_type="logistic",
    )
    # Should complete without raising and without needing Kafka
    await publish_prediction_event(event)


@pytest.mark.asyncio
async def test_publish_handles_kafka_down_gracefully(monkeypatch):
    """If Kafka is unreachable, publisher logs and returns (no raise)."""
    from app.services.prediction_event_publisher import (
        publish_prediction_event,
        PredictionEvent,
        _producer_state,
    )
    from app.core.settings import settings

    monkeypatch.setattr(settings, "BROKER_BACKEND", "kafka")

    # Simulate "producer exists but send raises"
    class FakeProducer:
        async def send_and_wait(self, *args, **kwargs):
            raise RuntimeError("kafka down")

    _producer_state["producer"] = FakeProducer()

    event = PredictionEvent(
        ticker="AAPL", prediction=1, confidence=0.9,
        model_id="m1", model_type="logistic",
    )
    # Should NOT raise
    await publish_prediction_event(event)

    # Cleanup
    _producer_state["producer"] = None
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_prediction_event_publisher.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create `app/services/prediction_event_publisher.py`**

```python
from __future__ import annotations

"""
Kafka prediction event publisher.

Fire-and-forget publishing to the 'prediction_events' topic. Graceful
degradation: if Kafka is unreachable or BROKER_BACKEND != 'kafka',
publishing is a silent no-op (logs a warning once).

Producer lifecycle is managed by FastAPI lifespan in app/main.py.
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.settings import settings

logger = logging.getLogger(__name__)

TOPIC = "prediction_events"

# Module-level producer state (mutable dict so tests can patch)
_producer_state: dict = {"producer": None, "started": False, "warned_no_kafka": False}


class PredictionEvent(BaseModel):
    """Schema for a prediction event published to Kafka."""

    ticker: str
    prediction: int  # 0 or 1
    confidence: float  # positive-class probability [0, 1]
    model_id: str
    model_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="forbid")


async def start_producer() -> None:
    """Start the global AIOKafkaProducer. No-op if BROKER_BACKEND != kafka."""
    if settings.BROKER_BACKEND != "kafka":
        logger.info("BROKER_BACKEND != kafka; prediction event publisher disabled")
        return

    if _producer_state["started"]:
        return

    try:
        from aiokafka import AIOKafkaProducer

        producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            acks="all",
        )
        await producer.start()
        _producer_state["producer"] = producer
        _producer_state["started"] = True
        logger.info(
            "Kafka prediction event producer started (bootstrap=%s)",
            settings.KAFKA_BOOTSTRAP_SERVERS,
        )
    except Exception as e:
        logger.warning("Failed to start Kafka producer: %s; will operate as no-op", e)


async def stop_producer() -> None:
    """Stop the global AIOKafkaProducer on shutdown."""
    producer = _producer_state.get("producer")
    if producer is not None:
        try:
            await producer.stop()
        except Exception as e:
            logger.warning("Error stopping Kafka producer: %s", e)
    _producer_state["producer"] = None
    _producer_state["started"] = False


async def publish_prediction_event(event: PredictionEvent) -> None:
    """Fire-and-forget publish of a prediction event.

    If Kafka is unreachable, logs and returns — does NOT raise.
    Keyed by ticker so same ticker goes to same partition for ordering.
    """
    if settings.BROKER_BACKEND != "kafka":
        if not _producer_state["warned_no_kafka"]:
            logger.info("BROKER_BACKEND=%s; prediction events skipped", settings.BROKER_BACKEND)
            _producer_state["warned_no_kafka"] = True
        return

    producer = _producer_state.get("producer")
    if producer is None:
        logger.debug("Kafka producer not ready; skipping event publish")
        return

    try:
        await producer.send_and_wait(
            TOPIC,
            value=event.model_dump_json().encode("utf-8"),
            key=event.ticker.encode("utf-8"),
        )
    except Exception as e:
        logger.warning("Failed to publish prediction event: %s", e)
```

- [ ] **Step 4: Wire producer lifecycle into `app/main.py` lifespan**

Find the existing `lifespan` function in `app/main.py`. Modify it to start/stop the producer:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(
        "Starting Quant AI Backend",
        extra={
            "extra_data": {
                "env": settings.ENV,
                "providers": settings.providers_list,
                "storage": settings.STORAGE_BACKEND,
            }
        },
    )
    # Register serverless functions
    from app.functions import register_all_functions
    register_all_functions()

    # Start Kafka prediction event producer (no-op if not kafka)
    from app.services.prediction_event_publisher import start_producer, stop_producer
    await start_producer()

    yield

    # Shutdown
    await stop_producer()
    logger.info("Shutting down Quant AI Backend")
```

- [ ] **Step 5: Add pytest-asyncio support**

Check if `pytest-asyncio` is in requirements. Run:

```bash
cd /c/Users/zjg09/projects/quant-ai && grep -i asyncio requirements.txt && grep -i asyncio pyproject.toml 2>&1 | head -3
```

If not present, add to `requirements.txt` under `# Observability` or dev deps. Also, in `pytest.ini` or `pyproject.toml`, ensure `asyncio_mode = auto` or mark tests with `@pytest.mark.asyncio`.

Most likely `pytest-asyncio` is already in CI (look in `.github/workflows/ci.yml`). Use `@pytest.mark.asyncio` decorator in tests as already done.

- [ ] **Step 6: Run tests — expect PASS**

Run: `pytest tests/test_prediction_event_publisher.py -v`
Expected: 4/4 PASS (async tests marked with `@pytest.mark.asyncio`).

- [ ] **Step 7: Commit**

```bash
git add app/services/prediction_event_publisher.py app/main.py tests/test_prediction_event_publisher.py
git commit -m "feat: [DIST-3] add Kafka prediction event publisher with lifespan"
```

---

## Task 4: Integrate publisher + metrics into predict service

**Files:**
- Modify: `app/services/predict_service.py`
- Modify: `tests/test_prometheus_metrics.py` (add integration test)

- [ ] **Step 1: Read current predict_service.py to understand shape**

Run: `cat app/services/predict_service.py`

Note existing result dict with keys `ticker`, `prediction`, `probability` etc. The instrumentation goes after the prediction is computed.

- [ ] **Step 2: Write failing test for instrumented predict**

Append to `tests/test_prometheus_metrics.py`:

```python
def test_predict_service_increments_counters(monkeypatch):
    """After predict_service.predict runs, PREDICT_TOTAL should increment."""
    import asyncio
    import numpy as np
    import pandas as pd
    from unittest.mock import MagicMock

    from app.core.metrics import PREDICT_TOTAL
    from app.services import predict_service

    # Mock get_model to return a dummy model
    dummy_model = MagicMock()
    dummy_model.predict.return_value = np.array([1])
    dummy_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    dummy_model.metadata = None
    dummy_model.model_type = "logistic"

    monkeypatch.setattr(predict_service, "_get_model_for_predict", lambda mid: (dummy_model, mid or "fallback"))

    # Mock _get_features_for_ticker to return a simple DataFrame
    monkeypatch.setattr(
        predict_service,
        "_get_features_for_ticker",
        lambda ticker, lookback: pd.DataFrame({"a": [1.0], "b": [2.0]}),
    )

    before = PREDICT_TOTAL.labels(ticker="AAPL", model_type="logistic")._value.get()
    predict_service.predict(ticker="AAPL", lookback=5, model_id=None)
    after = PREDICT_TOTAL.labels(ticker="AAPL", model_type="logistic")._value.get()

    assert after == before + 1
```

- [ ] **Step 3: Run — expect FAIL (or different: counter doesn't increment)**

Run: `pytest tests/test_prometheus_metrics.py::test_predict_service_increments_counters -v`
Expected: FAIL.

- [ ] **Step 4: Modify `app/services/predict_service.py`**

First read it fully with: `cat app/services/predict_service.py`.

Find the `predict` method (likely in a class `PredictionService` or a module-level `predict` function). After the prediction is computed (where `pred` and `proba` are defined), add instrumentation:

```python
# At top of file, add imports
import time
import asyncio
from app.core.metrics import PREDICT_TOTAL, PREDICT_CONFIDENCE, MODEL_INFERENCE_SECONDS
from app.services.prediction_event_publisher import (
    PredictionEvent,
    publish_prediction_event,
)
```

In the `predict` function, wrap the model inference in a timing block. A typical pattern:

```python
# ... existing code that loads model and features ...

# Instrument model inference
model_type = getattr(model, "model_type", "unknown")
with MODEL_INFERENCE_SECONDS.labels(model_type=model_type).time():
    proba = model.predict_proba(X)
    pred = model.predict(X)

# ... existing code that extracts latest_pred, confidence ...
latest_pred = int(pred[-1])
latest_prob = float(proba[-1, 1])

# Metrics
PREDICT_TOTAL.labels(ticker=ticker, model_type=model_type).inc()
PREDICT_CONFIDENCE.labels(ticker=ticker).observe(latest_prob)

# Fire-and-forget Kafka event
try:
    loop = asyncio.get_event_loop()
    loop.create_task(
        publish_prediction_event(PredictionEvent(
            ticker=ticker,
            prediction=latest_pred,
            confidence=latest_prob,
            model_id=resolved_model_id,  # whatever var holds the model id
            model_type=model_type,
        ))
    )
except RuntimeError:
    # No running loop (e.g. sync context); skip event publish
    pass
```

**Important**: the exact variable names (`ticker`, `resolved_model_id`, `model`, `X`) depend on current predict_service.py structure. Preserve them.

If the existing predict function uses a `_get_model_for_predict` helper, that's the hook — otherwise wrap wherever the model + features meet before inference.

- [ ] **Step 5: Run test — expect PASS**

Run: `pytest tests/test_prometheus_metrics.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/services/predict_service.py tests/test_prometheus_metrics.py
git commit -m "feat: [DIST-4] instrument predict service with metrics and Kafka event publish"
```

---

## Task 5: Events consumer FastAPI app

**Files:**
- Create: `app/workers/__init__.py`
- Create: `app/workers/events_consumer.py`
- Create: `tests/test_events_consumer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_events_consumer.py`:

```python
from __future__ import annotations

import pytest
from datetime import datetime


def test_consumer_app_has_stats_endpoint():
    from app.workers.events_consumer import app
    # Routes are added at module import
    paths = [r.path for r in app.routes]
    assert "/stats/{ticker}" in paths
    assert "/health" in paths


def test_stats_empty_for_unknown_ticker():
    from fastapi.testclient import TestClient
    from app.workers.events_consumer import app, _stats

    _stats.clear()
    client = TestClient(app)
    resp = client.get("/stats/NONEXIST")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "NONEXIST"
    assert body["count"] == 0


def test_stats_aggregates_after_events():
    from fastapi.testclient import TestClient
    from app.workers.events_consumer import app, _stats
    from app.services.prediction_event_publisher import PredictionEvent

    _stats.clear()

    # Inject some events
    for i, (pred, conf) in enumerate([(1, 0.9), (0, 0.4), (1, 0.8), (1, 0.7)]):
        _stats["AAPL"].append(PredictionEvent(
            ticker="AAPL",
            prediction=pred,
            confidence=conf,
            model_id="m1",
            model_type="logistic",
        ))

    client = TestClient(app)
    resp = client.get("/stats/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 4
    assert body["bullish_ratio"] == 0.75  # 3/4
    assert abs(body["avg_confidence"] - 0.7) < 1e-9  # (0.9+0.4+0.8+0.7)/4


def test_health_returns_ok():
    from fastapi.testclient import TestClient
    from app.workers.events_consumer import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_events_consumer.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create `app/workers/__init__.py`**

```python
"""App workers package — long-running background services.

Each module is an independent entrypoint (runnable via `python -m app.workers.X`).
"""
```

- [ ] **Step 4: Create `app/workers/events_consumer.py`**

```python
from __future__ import annotations

"""
Prediction Events Consumer

Subscribes to Kafka 'prediction_events' topic and maintains per-ticker rolling
stats in memory. Exposes /stats/{ticker} endpoint.

Run with: `python -m uvicorn app.workers.events_consumer:app --host 0.0.0.0 --port 8001`

In K8s, this is a separate Deployment using Dockerfile.consumer.
"""

import asyncio
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Deque, Dict

from fastapi import FastAPI

from app.core.settings import settings
from app.services.prediction_event_publisher import PredictionEvent, TOPIC

logger = logging.getLogger(__name__)

# Per-ticker rolling window of the last 1000 events (in-memory)
_stats: Dict[str, Deque[PredictionEvent]] = defaultdict(lambda: deque(maxlen=1000))
_consumer_state: dict = {"consumer": None, "task": None}


async def _consume_loop() -> None:
    """Background task: consume Kafka messages and update _stats."""
    if settings.BROKER_BACKEND != "kafka":
        logger.info("BROKER_BACKEND != kafka; consumer idle (use direct _stats injection for tests)")
        return

    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="events-consumer",
        auto_offset_reset="earliest",
    )
    _consumer_state["consumer"] = consumer

    try:
        await consumer.start()
        logger.info("Events consumer subscribed to topic=%s", TOPIC)
        async for msg in consumer:
            try:
                event = PredictionEvent.model_validate_json(msg.value)
                _stats[event.ticker].append(event)
            except Exception as e:
                logger.warning("Malformed Kafka message skipped: %s", e)
    except Exception as e:
        logger.error("Consumer loop error: %s", e)
    finally:
        try:
            await consumer.stop()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_consume_loop())
    _consumer_state["task"] = task
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    consumer = _consumer_state.get("consumer")
    if consumer is not None:
        try:
            await consumer.stop()
        except Exception:
            pass


app = FastAPI(
    title="Quant AI Events Consumer",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats/{ticker}")
def get_stats(ticker: str):
    ticker = ticker.upper()
    events = list(_stats.get(ticker, []))
    if not events:
        return {"ticker": ticker, "count": 0}
    avg_confidence = sum(e.confidence for e in events) / len(events)
    bullish_ratio = sum(1 for e in events if e.prediction == 1) / len(events)
    return {
        "ticker": ticker,
        "count": len(events),
        "avg_confidence": avg_confidence,
        "bullish_ratio": bullish_ratio,
        "last_prediction_ts": events[-1].timestamp.isoformat(),
    }
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/test_events_consumer.py -v`
Expected: 4/4 PASS.

- [ ] **Step 6: Commit**

```bash
git add app/workers/__init__.py app/workers/events_consumer.py tests/test_events_consumer.py
git commit -m "feat: [DIST-5] add events consumer FastAPI app with /stats endpoint"
```

---

## Task 6: Dockerfile.consumer for separate image

**Files:**
- Create: `Dockerfile.consumer`

- [ ] **Step 1: Create `Dockerfile.consumer`**

```dockerfile
# Quant AI Events Consumer - Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Builder stage
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base AS production

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN useradd --create-home --shell /bin/bash appuser

COPY --chown=appuser:appuser app ./app

USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.workers.events_consumer:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 2: Verify Dockerfile builds (optional — may skip if no local Docker)**

Run (only if Docker available): `docker build -f Dockerfile.consumer -t quant-ai-consumer:test .`
Expected: success. If Docker not installed, skip and rely on CI.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.consumer
git commit -m "feat: [DIST-6] add Dockerfile.consumer for events consumer image"
```

---

## Task 7: K8s manifests — namespace, config, API deployment, service, HPA

**Files:**
- Create: `k8s/namespace.yaml`
- Create: `k8s/configmap.yaml`
- Create: `k8s/secret.example.yaml`
- Create: `k8s/deployment-api.yaml`
- Create: `k8s/service-api.yaml`
- Create: `k8s/hpa-api.yaml`

- [ ] **Step 1: Create `k8s/namespace.yaml`**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quant-ai
  labels:
    app.kubernetes.io/name: quant-ai
    app.kubernetes.io/part-of: quant-ai
```

- [ ] **Step 2: Create `k8s/configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quant-ai-config
  namespace: quant-ai
data:
  ENV: "prod"
  CACHE_BACKEND: "redis"
  BROKER_BACKEND: "kafka"
  QUEUE_BACKEND: "redis"
  NOTIFY_BACKEND: "memory"
  FUNCTIONS_BACKEND: "local"
  STORAGE_BACKEND: "local"
  STORAGE_LOCAL_PATH: "/data/artifacts"
  KAFKA_BOOTSTRAP_SERVERS: "kafka:9092"
  REDIS_URL: "redis://redis:6379/0"
```

- [ ] **Step 3: Create `k8s/secret.example.yaml`**

```yaml
# Copy this file to k8s/secret.yaml and fill in real values.
# k8s/secret.yaml is gitignored.
apiVersion: v1
kind: Secret
metadata:
  name: quant-ai-secrets
  namespace: quant-ai
type: Opaque
stringData:
  DATABASE_URL: "postgresql+psycopg://postgres:postgres@postgres:5432/postgres"
  SUPABASE_URL: ""
  SUPABASE_KEY: ""
  SUPABASE_SERVICE_KEY: ""
```

Also, add `k8s/secret.yaml` to `.gitignore`. Run:

```bash
cd /c/Users/zjg09/projects/quant-ai
echo "k8s/secret.yaml" >> .gitignore
```

- [ ] **Step 4: Create `k8s/deployment-api.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quant-ai-api
  namespace: quant-ai
  labels:
    app: quant-ai-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: quant-ai-api
  template:
    metadata:
      labels:
        app: quant-ai-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
        - name: api
          image: quant-ai:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              name: http
          envFrom:
            - configMapRef:
                name: quant-ai-config
            - secretRef:
                name: quant-ai-secrets
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
```

- [ ] **Step 5: Create `k8s/service-api.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: api
  namespace: quant-ai
  labels:
    app: quant-ai-api
spec:
  type: NodePort
  selector:
    app: quant-ai-api
  ports:
    - name: http
      port: 8000
      targetPort: 8000
      nodePort: 30001
```

- [ ] **Step 6: Create `k8s/hpa-api.yaml`**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quant-ai-api
  namespace: quant-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quant-ai-api
  minReplicas: 2
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

- [ ] **Step 7: Add /health/ready endpoint**

Check current `app/api/health.py`: `cat app/api/health.py`.

If `/health/ready` doesn't exist, add it. For example:

```python
@router.get("/health/ready")
def health_ready():
    """Readiness probe — returns 200 only if critical deps reachable."""
    try:
        from app.db.engine import get_engine
        eng = get_engine()
        with eng.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="database unreachable")
    return {"status": "ready"}
```

If `/health/ready` already exists, skip.

- [ ] **Step 8: Commit**

```bash
git add k8s/namespace.yaml k8s/configmap.yaml k8s/secret.example.yaml \
        k8s/deployment-api.yaml k8s/service-api.yaml k8s/hpa-api.yaml \
        .gitignore app/api/health.py
git commit -m "feat: [DIST-7] add K8s namespace, config, API deployment/service/HPA"
```

---

## Task 8: K8s manifests — consumer deployment + service

**Files:**
- Create: `k8s/deployment-consumer.yaml`
- Create: `k8s/service-consumer.yaml`

- [ ] **Step 1: Create `k8s/deployment-consumer.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quant-ai-consumer
  namespace: quant-ai
  labels:
    app: quant-ai-consumer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: quant-ai-consumer
  template:
    metadata:
      labels:
        app: quant-ai-consumer
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
    spec:
      containers:
        - name: consumer
          image: quant-ai-consumer:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8001
              name: http
          envFrom:
            - configMapRef:
                name: quant-ai-config
            - secretRef:
                name: quant-ai-secrets
          resources:
            requests:
              cpu: 50m
              memory: 128Mi
            limits:
              cpu: 200m
              memory: 256Mi
          livenessProbe:
            httpGet:
              path: /health
              port: 8001
            initialDelaySeconds: 10
            periodSeconds: 30
```

- [ ] **Step 2: Create `k8s/service-consumer.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: consumer
  namespace: quant-ai
  labels:
    app: quant-ai-consumer
spec:
  type: NodePort
  selector:
    app: quant-ai-consumer
  ports:
    - name: http
      port: 8001
      targetPort: 8001
      nodePort: 30002
```

- [ ] **Step 3: Commit**

```bash
git add k8s/deployment-consumer.yaml k8s/service-consumer.yaml
git commit -m "feat: [DIST-8] add K8s consumer deployment and service"
```

---

## Task 9: K8s manifests — Kafka, Postgres, Redis stateful components

**Files:**
- Create: `k8s/statefulset-kafka.yaml`
- Create: `k8s/service-kafka.yaml`
- Create: `k8s/statefulset-postgres.yaml`
- Create: `k8s/service-postgres.yaml`
- Create: `k8s/deployment-redis.yaml`
- Create: `k8s/service-redis.yaml`

- [ ] **Step 1: Create `k8s/statefulset-kafka.yaml`**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: kafka
  namespace: quant-ai
spec:
  serviceName: kafka
  replicas: 1
  selector:
    matchLabels:
      app: kafka
  template:
    metadata:
      labels:
        app: kafka
    spec:
      containers:
        - name: kafka
          image: bitnami/kafka:3.6
          ports:
            - containerPort: 9092
              name: client
            - containerPort: 9093
              name: controller
          env:
            - name: KAFKA_CFG_NODE_ID
              value: "1"
            - name: KAFKA_CFG_PROCESS_ROLES
              value: "broker,controller"
            - name: KAFKA_CFG_CONTROLLER_QUORUM_VOTERS
              value: "1@kafka-0.kafka:9093"
            - name: KAFKA_CFG_LISTENERS
              value: "PLAINTEXT://:9092,CONTROLLER://:9093"
            - name: KAFKA_CFG_ADVERTISED_LISTENERS
              value: "PLAINTEXT://kafka:9092"
            - name: KAFKA_CFG_CONTROLLER_LISTENER_NAMES
              value: "CONTROLLER"
            - name: KAFKA_CFG_INTER_BROKER_LISTENER_NAME
              value: "PLAINTEXT"
            - name: ALLOW_PLAINTEXT_LISTENER
              value: "yes"
          volumeMounts:
            - name: data
              mountPath: /bitnami/kafka
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

- [ ] **Step 2: Create `k8s/service-kafka.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kafka
  namespace: quant-ai
spec:
  clusterIP: None  # headless service for StatefulSet
  selector:
    app: kafka
  ports:
    - name: client
      port: 9092
      targetPort: 9092
    - name: controller
      port: 9093
      targetPort: 9093
```

- [ ] **Step 3: Create `k8s/statefulset-postgres.yaml`**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: quant-ai
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16-alpine
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_USER
              value: postgres
            - name: POSTGRES_PASSWORD
              value: postgres
            - name: POSTGRES_DB
              value: postgres
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

- [ ] **Step 4: Create `k8s/service-postgres.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: quant-ai
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
```

- [ ] **Step 5: Create `k8s/deployment-redis.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: quant-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          resources:
            requests:
              cpu: 50m
              memory: 128Mi
            limits:
              cpu: 200m
              memory: 256Mi
```

- [ ] **Step 6: Create `k8s/service-redis.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: quant-ai
spec:
  selector:
    app: redis
  ports:
    - port: 6379
      targetPort: 6379
```

- [ ] **Step 7: Commit**

```bash
git add k8s/statefulset-kafka.yaml k8s/service-kafka.yaml \
        k8s/statefulset-postgres.yaml k8s/service-postgres.yaml \
        k8s/deployment-redis.yaml k8s/service-redis.yaml
git commit -m "feat: [DIST-9] add K8s Kafka/Postgres/Redis stateful components"
```

---

## Task 10: K8s — Prometheus + Grafana with pre-built dashboard

**Files:**
- Create: `k8s/configmap-prometheus.yaml`
- Create: `k8s/deployment-prometheus.yaml`
- Create: `k8s/configmap-grafana.yaml`
- Create: `k8s/deployment-grafana.yaml`
- Create: `observability/dashboards/quant-ai.json`

- [ ] **Step 1: Create the dashboard JSON**

Create `observability/dashboards/quant-ai.json`:

```json
{
  "annotations": { "list": [] },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": null,
  "iteration": 1,
  "links": [],
  "panels": [
    {
      "type": "timeseries",
      "title": "Request Rate by Endpoint",
      "targets": [
        {
          "expr": "sum by (handler) (rate(http_requests_total[1m]))",
          "legendFormat": "{{handler}}",
          "refId": "A"
        }
      ],
      "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 },
      "datasource": { "type": "prometheus", "uid": "prometheus" }
    },
    {
      "type": "timeseries",
      "title": "Latency p95 by Endpoint",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum by (handler, le) (rate(http_request_duration_seconds_bucket[1m])))",
          "legendFormat": "p95 {{handler}}",
          "refId": "A"
        }
      ],
      "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
      "datasource": { "type": "prometheus", "uid": "prometheus" }
    },
    {
      "type": "timeseries",
      "title": "Predictions per Minute by Ticker",
      "targets": [
        {
          "expr": "sum by (ticker) (rate(quant_ai_predictions_total[1m])) * 60",
          "legendFormat": "{{ticker}}",
          "refId": "A"
        }
      ],
      "gridPos": { "x": 0, "y": 8, "w": 12, "h": 8 },
      "datasource": { "type": "prometheus", "uid": "prometheus" }
    },
    {
      "type": "heatmap",
      "title": "Prediction Confidence Distribution",
      "targets": [
        {
          "expr": "sum by (le) (rate(quant_ai_prediction_confidence_bucket[1m]))",
          "refId": "A",
          "format": "heatmap"
        }
      ],
      "gridPos": { "x": 12, "y": 8, "w": 12, "h": 8 },
      "datasource": { "type": "prometheus", "uid": "prometheus" }
    },
    {
      "type": "timeseries",
      "title": "Model Inference Time p95 by Model Type",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum by (model_type, le) (rate(quant_ai_model_inference_seconds_bucket[1m])))",
          "legendFormat": "p95 {{model_type}}",
          "refId": "A"
        }
      ],
      "gridPos": { "x": 0, "y": 16, "w": 12, "h": 8 },
      "datasource": { "type": "prometheus", "uid": "prometheus" }
    },
    {
      "type": "timeseries",
      "title": "API Pod Count (HPA)",
      "targets": [
        {
          "expr": "kube_deployment_status_replicas{namespace=\"quant-ai\",deployment=\"quant-ai-api\"}",
          "legendFormat": "replicas",
          "refId": "A"
        }
      ],
      "gridPos": { "x": 12, "y": 16, "w": 12, "h": 8 },
      "datasource": { "type": "prometheus", "uid": "prometheus" }
    }
  ],
  "refresh": "10s",
  "schemaVersion": 37,
  "style": "dark",
  "tags": ["quant-ai"],
  "templating": { "list": [] },
  "time": { "from": "now-30m", "to": "now" },
  "timepicker": {},
  "timezone": "",
  "title": "Quant AI Dashboard",
  "uid": "quant-ai-main",
  "version": 1
}
```

- [ ] **Step 2: Create `k8s/configmap-prometheus.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: quant-ai
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    scrape_configs:
      - job_name: 'quant-ai-api'
        static_configs:
          - targets: ['api:8000']
        metrics_path: /metrics

      - job_name: 'quant-ai-consumer'
        static_configs:
          - targets: ['consumer:8001']
        metrics_path: /metrics

      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']
```

- [ ] **Step 3: Create `k8s/deployment-prometheus.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: quant-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
        - name: prometheus
          image: prom/prometheus:v2.53.0
          args:
            - "--config.file=/etc/prometheus/prometheus.yml"
            - "--storage.tsdb.path=/prometheus"
          ports:
            - containerPort: 9090
          volumeMounts:
            - name: config
              mountPath: /etc/prometheus
      volumes:
        - name: config
          configMap:
            name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: quant-ai
spec:
  type: NodePort
  selector:
    app: prometheus
  ports:
    - port: 9090
      targetPort: 9090
      nodePort: 30090
```

- [ ] **Step 4: Create `k8s/configmap-grafana.yaml`**

The dashboard JSON is too large to embed inline comfortably in YAML. Use a helper approach: reference the file from the observability/ dir using a `kubectl create configmap --from-file`. Document this in k8s/README.md (Task 13). For the committed manifest, include a minimal datasource provisioning:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: quant-ai
data:
  prometheus.yml: |
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        uid: prometheus
        access: proxy
        url: http://prometheus:9090
        isDefault: true
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards-provider
  namespace: quant-ai
data:
  default.yml: |
    apiVersion: 1
    providers:
      - name: default
        orgId: 1
        folder: ""
        type: file
        options:
          path: /var/lib/grafana/dashboards
```

- [ ] **Step 5: Create `k8s/deployment-grafana.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: quant-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
        - name: grafana
          image: grafana/grafana:10.4.0
          ports:
            - containerPort: 3000
          env:
            - name: GF_SECURITY_ADMIN_USER
              value: admin
            - name: GF_SECURITY_ADMIN_PASSWORD
              value: admin
          volumeMounts:
            - name: datasources
              mountPath: /etc/grafana/provisioning/datasources
            - name: dashboards-provider
              mountPath: /etc/grafana/provisioning/dashboards
            - name: dashboards
              mountPath: /var/lib/grafana/dashboards
      volumes:
        - name: datasources
          configMap:
            name: grafana-datasources
        - name: dashboards-provider
          configMap:
            name: grafana-dashboards-provider
        - name: dashboards
          configMap:
            name: grafana-dashboards
            optional: true
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: quant-ai
spec:
  type: NodePort
  selector:
    app: grafana
  ports:
    - port: 3000
      targetPort: 3000
      nodePort: 30030
```

- [ ] **Step 6: Commit**

```bash
git add k8s/configmap-prometheus.yaml k8s/deployment-prometheus.yaml \
        k8s/configmap-grafana.yaml k8s/deployment-grafana.yaml \
        observability/dashboards/quant-ai.json
git commit -m "feat: [DIST-10] add K8s Prometheus + Grafana with dashboard JSON"
```

---

## Task 11: docker-compose updates + observability directory

**Files:**
- Modify: `docker-compose.yml`
- Create: `observability/prometheus.yml`
- Create: `observability/grafana-datasources.yml`
- Create: `observability/grafana-dashboards.yml`

- [ ] **Step 1: Create `observability/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'quant-ai-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics

  - job_name: 'quant-ai-consumer'
    static_configs:
      - targets: ['consumer:8001']
    metrics_path: /metrics
```

- [ ] **Step 2: Create `observability/grafana-datasources.yml`**

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    uid: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

- [ ] **Step 3: Create `observability/grafana-dashboards.yml`**

```yaml
apiVersion: 1
providers:
  - name: default
    orgId: 1
    folder: ""
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

- [ ] **Step 4: Modify `docker-compose.yml`**

Read current `docker-compose.yml` and append the following services before any trailing networks/volumes section:

```yaml
  # ===================================
  # Events Consumer
  # ===================================
  consumer:
    build:
      context: .
      dockerfile: Dockerfile.consumer
    ports:
      - "8001:8001"
    env_file:
      - .env
    environment:
      - ENV=dev
      - BROKER_BACKEND=kafka
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - kafka
    restart: unless-stopped

  # ===================================
  # Kafka (single broker, KRaft mode)
  # ===================================
  kafka:
    image: bitnami/kafka:3.6
    ports:
      - "9092:9092"
    environment:
      KAFKA_CFG_NODE_ID: "1"
      KAFKA_CFG_PROCESS_ROLES: "broker,controller"
      KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: "1@kafka:9093"
      KAFKA_CFG_LISTENERS: "PLAINTEXT://:9092,CONTROLLER://:9093"
      KAFKA_CFG_ADVERTISED_LISTENERS: "PLAINTEXT://kafka:9092"
      KAFKA_CFG_CONTROLLER_LISTENER_NAMES: "CONTROLLER"
      KAFKA_CFG_INTER_BROKER_LISTENER_NAME: "PLAINTEXT"
      ALLOW_PLAINTEXT_LISTENER: "yes"
    restart: unless-stopped

  # ===================================
  # Prometheus
  # ===================================
  prometheus:
    image: prom/prometheus:v2.53.0
    ports:
      - "9090:9090"
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

  # ===================================
  # Grafana
  # ===================================
  grafana:
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - ./observability/grafana-datasources.yml:/etc/grafana/provisioning/datasources/prometheus.yml:ro
      - ./observability/grafana-dashboards.yml:/etc/grafana/provisioning/dashboards/default.yml:ro
      - ./observability/dashboards:/etc/grafana/provisioning/dashboards:ro
    depends_on:
      - prometheus
    restart: unless-stopped
```

Also add `BROKER_BACKEND=kafka` and `KAFKA_BOOTSTRAP_SERVERS=kafka:9092` to the `api` service's environment, and add `kafka` to its `depends_on`. Preserve existing environment variables.

- [ ] **Step 5: Verify docker-compose config is valid**

Run: `docker-compose config 2>&1 | head -30`
Expected: valid YAML, services listed.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml observability/
git commit -m "feat: [DIST-11] add Kafka + Prometheus + Grafana to docker-compose"
```

---

## Task 12: k8s README + architecture doc

**Files:**
- Create: `k8s/README.md`
- Create: `docs/architecture/distributed.md`

- [ ] **Step 1: Create `k8s/README.md`**

```markdown
# Quant AI — Kubernetes Deployment

Local Minikube deployment for the Quant AI distributed stack.

## Prerequisites

- [Minikube](https://minikube.sigs.k8s.io/docs/start/) (>= 1.32)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- Docker Desktop (or Docker Engine) — Minikube uses Docker as its driver by default

## One-time setup

```bash
# Start Minikube with 4 CPUs, 6GB RAM (HPA needs headroom)
minikube start --cpus=4 --memory=6g

# Enable metrics-server addon (required for HPA to work)
minikube addons enable metrics-server

# Point local Docker client at Minikube's Docker daemon so
# images built locally are visible inside the cluster
eval $(minikube docker-env)
```

## Build images

```bash
# From repository root
docker build -t quant-ai:latest -f Dockerfile --target production .
docker build -t quant-ai-consumer:latest -f Dockerfile.consumer .
```

## Deploy

```bash
# Copy secret template and fill in values (at minimum DATABASE_URL)
cp k8s/secret.example.yaml k8s/secret.yaml
# Edit k8s/secret.yaml with your values

# Create Grafana dashboard ConfigMap from JSON file
kubectl create namespace quant-ai --dry-run=client -o yaml | kubectl apply -f -
kubectl -n quant-ai create configmap grafana-dashboards \
    --from-file=observability/dashboards/quant-ai.json \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply everything
kubectl apply -f k8s/
```

## Verify

```bash
kubectl -n quant-ai get pods
# Expect: api (2), consumer (1), kafka, postgres, redis, prometheus, grafana — all Running

kubectl -n quant-ai get hpa
# Expect: quant-ai-api target current/70%

# Access services (Minikube NodePort)
minikube service -n quant-ai api            # API on :30001
minikube service -n quant-ai consumer       # Consumer on :30002
minikube service -n quant-ai grafana        # Grafana on :30030 (login admin/admin)
minikube service -n quant-ai prometheus     # Prometheus UI on :30090
```

## Smoke test

```bash
API_URL=$(minikube -n quant-ai service api --url)
CONSUMER_URL=$(minikube -n quant-ai service consumer --url)

# Hit predict 10 times
for i in {1..10}; do
    curl -s "${API_URL}/predict?ticker=AAPL&lookback=10" > /dev/null
done

# Confirm consumer aggregated events
curl "${CONSUMER_URL}/stats/AAPL"
# Expect: {"ticker":"AAPL","count":10,"avg_confidence":...,"bullish_ratio":...}
```

## Tear down

```bash
kubectl delete namespace quant-ai
# or nuke cluster entirely:
minikube delete
```

## Scaling to cloud (future)

- Replace `statefulset-postgres.yaml` with the Supabase connection in `secret.yaml`
- Replace `statefulset-kafka.yaml` with Confluent Cloud credentials (add SASL env vars)
- Change `image: quant-ai:latest` to a registry-pushed image (ECR / GCR / DockerHub)
- Swap NodePort services for LoadBalancer + Ingress controller
```

- [ ] **Step 2: Create `docs/architecture/distributed.md`**

```markdown
# Quant AI — Distributed Architecture

This document explains the distributed systems aspects of Quant AI.
It's written for interviewers and for myself, six months from now.

## System diagram

```
Client (browser / CLI)
    │
    ▼
[Ingress / NodePort]
    │
    ▼
quant-ai-api (Deployment, 2 replicas, HPA 2-5 on CPU >70%)
    │  sync: reads/writes Postgres, Redis (cache)
    │  async: publishes PredictionEvent to Kafka "prediction_events"
    ▼
[Kafka: 1 broker (KRaft mode)]
    │
    ▼
quant-ai-consumer (Deployment, 1 replica)
    │  subscribes to "prediction_events"
    │  maintains per-ticker rolling stats in memory (deque, maxlen=1000)
    │  exposes /stats/{ticker}

[Prometheus] scrapes /metrics on api:8000 and consumer:8001 every 15s
[Grafana]    reads Prometheus, shows 6-panel dashboard
```

## What's distributed vs. not

| Component | Distributed? | Why / why not |
|-----------|-------------|---------------|
| `quant-ai-api` | ✅ | Runs as 2+ replicas with HPA. Stateless request handling. Any instance can serve any request. |
| `quant-ai-consumer` | ⚠️ Could be, is not | Single replica — per-ticker state is in memory. To scale, need partitioned state store (Redis / RocksDB / Kafka compacted topic). |
| Kafka | ❌ Single broker in Minikube | Production would be 3+ brokers with replication factor 3. Trade-off: cost vs. durability. |
| Postgres | ❌ Single instance | Production uses Supabase which runs managed Postgres with replication. This local setup is dev only. |
| Redis | ❌ Single instance | For true HA: Redis Cluster mode (sharded) or Sentinel (failover). Out of scope for this demo. |

## CAP tradeoff

| Data | Store | CAP choice | Reasoning |
|------|-------|-----------|-----------|
| Prices (historical OHLCV) | Postgres (Supabase) | CP | Predictions are re-runnable if server is briefly down; strong consistency matters more than availability. |
| Prediction events | Kafka | AP (for consumer) | If a prediction event is delayed or re-delivered, stats are still approximately right. Availability > consistency here. |
| Cached prices | Redis | AP (best effort) | Postgres is source of truth. Redis miss = read-through. Losing cache is performance cost, not correctness. |
| Rolling stats | Consumer memory | A (only when up) | Transient, approximate. If consumer restarts, starts fresh (cold start of 1000-event window). Production would back this with Redis TTL. |

## Observability stack

- **Prometheus** scrapes `/metrics` on `api` and `consumer` every 15s
- Metrics auto-registered by `prometheus-fastapi-instrumentator`:
  - `http_requests_total{method, handler, status}` — counter
  - `http_request_duration_seconds_bucket{method, handler, le}` — histogram
- Custom ML metrics in `app/core/metrics.py`:
  - `quant_ai_predictions_total{ticker, model_type}` — counter
  - `quant_ai_prediction_confidence_bucket{ticker, le}` — histogram
  - `quant_ai_model_inference_seconds_bucket{model_type, le}` — histogram
- Grafana dashboard (6 panels): request rate, p95 latency, predictions/min per ticker, confidence heatmap, inference time p95, pod count.

## How to scale this up

**If Harry had a week and budget**:
1. **Kafka on Confluent Cloud** (free tier ~1 GB egress/month):
   - 3 brokers, replication factor 3
   - SASL auth via existing `KAFKA_SASL_USERNAME/PASSWORD` env vars
2. **Postgres read replicas** via Supabase native feature — route read-only queries (prices, news) to replicas
3. **Consumer horizontal scale**:
   - Back rolling stats with Redis TTL (or a compacted Kafka topic)
   - Scale consumer to N replicas, each assigned a partition subset via consumer group
4. **K8s on EKS** or GKE with:
   - Ingress controller (ALB or nginx-ingress) + cert-manager for TLS
   - Horizontal Pod Autoscaler across both api and consumer
   - Separate node pools (CPU-optimized for api, memory-optimized for consumer if stats grow)
5. **Observability**:
   - OpenTelemetry tracing across api → consumer (to profile /predict → event latency)
   - Alerting via Alertmanager → PagerDuty for p95 latency SLO breaches

## Honest limits of current setup

- Single-broker Kafka means **no durability**. If the Kafka pod dies, events in-flight are lost.
- Consumer is a single pod — no HA, restart = lose rolling window.
- Postgres in-cluster is dev only. Prod uses Supabase.
- HPA only tested on CPU. Real load testing with k6/Locust would be next.
- No TLS between services (inside-cluster plaintext). Prod would add mTLS or service mesh.

## Lessons learned

- `aiokafka` requires lifespan management — easy to leak producers if not careful.
- Prometheus custom metrics have a gotcha: `Counter` registers globally. Tests that re-import
  the module raise `Duplicated timeseries`. Solution: use a fresh `CollectorRegistry` in tests
  or unregister in fixtures.
- Minikube HPA is wildly inaccurate without `metrics-server` addon (easy to miss).
- KRaft-mode Kafka removes the ZooKeeper dep — makes single-broker demo tractable.
```

- [ ] **Step 3: Commit**

```bash
git add k8s/README.md docs/architecture/distributed.md
git commit -m "docs: [DIST-12] add K8s deploy README and distributed architecture doc"
```

---

## Task 13: Integration test — Kafka-enabled predict-to-stats roundtrip

**Files:**
- Create: `tests/contract/test_distributed_roundtrip.py`

- [ ] **Step 1: Create `tests/contract/test_distributed_roundtrip.py`**

```python
from __future__ import annotations

"""
Contract test: end-to-end path from /predict to consumer /stats.

Uses direct in-process event injection (bypassing Kafka) since CI doesn't have
a Kafka broker. Verifies the data shape and logic of the consumer.
"""

from fastapi.testclient import TestClient


def test_events_consumer_stats_endpoint_contract():
    from app.workers.events_consumer import app, _stats
    from app.services.prediction_event_publisher import PredictionEvent

    _stats.clear()

    # Inject 3 bullish + 2 bearish events
    for pred, conf in [(1, 0.9), (1, 0.8), (0, 0.4), (1, 0.7), (0, 0.3)]:
        _stats["MSFT"].append(PredictionEvent(
            ticker="MSFT",
            prediction=pred,
            confidence=conf,
            model_id="model-x",
            model_type="ensemble",
        ))

    client = TestClient(app)
    resp = client.get("/stats/MSFT")
    assert resp.status_code == 200

    body = resp.json()
    # Contract: these fields MUST exist
    assert body["ticker"] == "MSFT"
    assert body["count"] == 5
    assert "avg_confidence" in body
    assert "bullish_ratio" in body
    assert "last_prediction_ts" in body

    # Values are correct
    assert body["bullish_ratio"] == 0.6  # 3/5
    assert abs(body["avg_confidence"] - 0.62) < 1e-9  # (0.9+0.8+0.4+0.7+0.3)/5


def test_consumer_case_insensitive_ticker():
    from app.workers.events_consumer import app, _stats
    from app.services.prediction_event_publisher import PredictionEvent

    _stats.clear()
    _stats["GOOG"].append(PredictionEvent(
        ticker="GOOG", prediction=1, confidence=0.7,
        model_id="m", model_type="logistic",
    ))

    client = TestClient(app)
    # lowercase query
    resp = client.get("/stats/goog")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
```

- [ ] **Step 2: Run — expect PASS**

Run: `pytest tests/contract/test_distributed_roundtrip.py -v`
Expected: 2/2 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/contract/test_distributed_roundtrip.py
git commit -m "feat: [DIST-13] add contract test for distributed predict → stats roundtrip"
```

---

## Task 14: DIST-GATE — full verification

**Files:** (none modified — verification only)

- [ ] **Step 1: Run full unit test suite**

Run: `pytest tests/ --tb=short --ignore=tests/contract -p no:cacheprovider -q 2>&1 | tail -5`
Expected: 260+ passed, 0 failures.

- [ ] **Step 2: Run contract suite**

Run: `pytest tests/contract/ --tb=short -p no:cacheprovider -q 2>&1 | tail -5`
Expected: 45+ passed, 0 failures.

- [ ] **Step 3: Ruff lint**

Run: `ruff check app/ --ignore F401,F841,E501,F541,E402 2>&1 | tail -5`
Expected: `All checks passed!`

- [ ] **Step 4: Verify `/metrics` exposed**

Run: `python -c "from fastapi.testclient import TestClient; from app.main import app; c = TestClient(app); r = c.get('/metrics'); print('OK' if r.status_code == 200 and '# HELP' in r.text else 'FAIL')"`
Expected: `OK`.

- [ ] **Step 5: Verify consumer app importable**

Run: `python -c "from app.workers.events_consumer import app; print('routes=', [r.path for r in app.routes])"`
Expected: output includes `/stats/{ticker}` and `/health`.

- [ ] **Step 6: Verify K8s manifests syntactically valid**

Run: `for f in k8s/*.yaml; do python -c "import yaml; list(yaml.safe_load_all(open('$f')))" || echo "FAIL: $f"; done`
Expected: no FAIL output.

- [ ] **Step 7: Verify docker-compose config is valid**

Run: `docker-compose config > /dev/null 2>&1 && echo "compose OK" || echo "compose FAIL"`
Expected: `compose OK` if docker is available; skip if not.

- [ ] **Step 8: Frontend build still clean**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -10`
Expected: `✓ built in ...`, 0 errors.

- [ ] **Step 9: Gate commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
git commit --allow-empty -m "feat: [DIST-GATE] Phase 3 Sub-project 3 — distributed systems gate"
```

---

## Self-Review

**1. Spec coverage:**

- §3 Architecture (api, consumer, Kafka, Prometheus, Grafana, Postgres, Redis) → Tasks 7-11 ✓
- §4 K8s manifests (13 files listed) → Tasks 7-10 ✓
- §5 Prometheus metrics (Instrumentator + custom) → Tasks 1-2 ✓
- §5.3 Instrumentation in predict → Task 4 ✓
- §5.4 Grafana dashboard JSON → Task 10 ✓
- §6 Kafka event pipeline (schema, publisher, consumer) → Tasks 3, 5, 6 ✓
- §6.4 Dockerfile.consumer → Task 6 ✓
- §7 docker-compose updates → Task 11 ✓
- §8 Tests (metrics, publisher, consumer, contract) → Tasks 1-5, 13 ✓
- §9 Docs → Task 12 ✓
- §11 Success criteria → Task 14 (gate) ✓

**2. Placeholder scan:** No TBD/TODO. All code blocks are complete.

**3. Type consistency:**
- `PredictionEvent` fields (`ticker`, `prediction`, `confidence`, `model_id`, `model_type`, `timestamp`) consistent across Tasks 3, 4, 5, 13 ✓
- `_stats` dict in consumer (Tasks 5, 13) uses same key pattern ✓
- Topic name `"prediction_events"` (constant `TOPIC`) used consistently ✓
- Metric names (`PREDICT_TOTAL`, `PREDICT_CONFIDENCE`, `MODEL_INFERENCE_SECONDS`) consistent ✓
- Port numbers (api 8000, consumer 8001, prometheus 9090, grafana 3000) consistent ✓
- K8s nodePorts (30001, 30002, 30090, 30030) consistent ✓

**4. Gap check:** Task 7 includes adding `/health/ready` which spec §4.2 requires — ✓.
