# Distributed Systems Layer Design Spec

**Date**: 2026-04-16  
**Author**: Harry + Claude  
**Status**: Approved (pending implementation)  
**Phase**: 3 Sub-project 3 (after Ensemble Models)

## 1. Goal

Make Quant AI a legitimately distributed system so Harry can answer "do you have distributed systems experience?" with real code to point at:

- **Kubernetes**: Multi-pod deployment with HPA (auto-scaling), health probes, ConfigMap/Secret separation
- **Prometheus + Grafana**: Observability stack with custom ML metrics and pre-built dashboard
- **Kafka event stream**: Prediction events published by API, consumed by a separate service doing real-time aggregation

Target: interview-ready demo. Runs locally via Minikube — `minikube start` + `kubectl apply -f k8s/` brings up the full stack. No cloud cost.

## 2. Design Principles

- **Interview value first** — each component must be something Harry can point to in a `k8s/` directory or Grafana URL and explain in 30 seconds
- **Real integration, not theater** — Kafka event pipeline must actually run; predictions must actually flow through Kafka; consumer must actually expose aggregated stats. No mocks committed.
- **Monorepo-multi-service pattern** — one codebase, but two container images (API + consumer). Demonstrates real microservice separation.
- **Local first, cloud later** — all manifests target Minikube. Same YAMLs deploy to GKE/EKS with minimal changes (documented).
- **Don't break existing deployment** — Render backend and Vercel frontend continue working as-is. This adds a parallel K8s path.

## 3. Architecture

```
┌─────────────────────────────── Minikube Cluster ────────────────────────────┐
│                                                                              │
│  ┌─────────────────────────┐     ┌─────────────────────────────┐            │
│  │ quant-ai-api            │     │ quant-ai-consumer           │            │
│  │ Deployment (replicas=2) │     │ Deployment (replicas=1)     │            │
│  │                         │     │                             │            │
│  │  - /predict            ─┼──→ │  Subscribes to              │            │
│  │  - /metrics             │     │    "prediction_events"      │            │
│  │  - /health              │     │  - In-memory rolling stats  │            │
│  │  readiness/liveness     │     │  - /stats/{ticker}          │            │
│  └─────────────────────────┘     └─────────────────────────────┘            │
│         ↑                                  ↑                                 │
│         │                                  │                                 │
│   HPA 2-5 pods                                                              │
│   (CPU > 70%)                                                                │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │ Kafka        │  │ Prometheus   │  │ Grafana      │                      │
│  │ StatefulSet  │  │ (scraper)    │  │ (dashboard)  │                      │
│  │ 1 broker     │  │ scrapes      │  │ provisioned  │                      │
│  │              │  │  /metrics    │  │ datasource   │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐                                         │
│  │ Postgres     │  │ Redis        │                                         │
│  │ StatefulSet  │  │ Deployment   │                                         │
│  └──────────────┘  └──────────────┘                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Data flow** (predict path):
1. Client calls `POST /predict` on quant-ai-api
2. API computes prediction (via existing predict_service)
3. API fire-and-forget publishes `{ticker, prediction, confidence, model_id, ts}` to Kafka `prediction_events`
4. quant-ai-consumer subscribes to topic, updates per-ticker rolling stats in memory
5. Grafana/user queries `/stats/{ticker}` on consumer to see aggregates

**Observability flow**:
1. API and consumer both expose `/metrics` (Prometheus format)
2. Prometheus scrapes every 15s
3. Grafana reads from Prometheus; pre-built dashboard shows latency, RPS, predictions/min per ticker

## 4. Kubernetes Manifests

All in `k8s/` directory (new).

### 4.1 Files

| File | Purpose |
|------|---------|
| `namespace.yaml` | `quant-ai` namespace for isolation |
| `configmap.yaml` | Non-secret config: `CACHE_BACKEND=redis`, `BROKER_BACKEND=kafka`, `KAFKA_BOOTSTRAP_SERVERS=kafka:9092`, etc. |
| `secret.example.yaml` | Template for `DATABASE_URL`, `SUPABASE_KEY` — users copy to `secret.yaml` (gitignored) |
| `deployment-api.yaml` | 2 replicas FastAPI, uses `quant-ai:latest` image, `CMD: uvicorn app.main:app` |
| `service-api.yaml` | ClusterIP + NodePort `30001` for `minikube service` access |
| `hpa-api.yaml` | HorizontalPodAutoscaler: min=2, max=5, target CPU=70% |
| `deployment-consumer.yaml` | 1 replica consumer, uses `quant-ai-consumer:latest` (different image), `CMD: python -m app.workers.events_consumer` |
| `service-consumer.yaml` | ClusterIP + NodePort `30002` |
| `statefulset-kafka.yaml` | 1 broker Kafka (Bitnami image), PVC for logs |
| `service-kafka.yaml` | Headless service `kafka:9092` |
| `statefulset-postgres.yaml` | 1 replica Postgres, PVC for data |
| `service-postgres.yaml` | ClusterIP `postgres:5432` |
| `deployment-redis.yaml` | 1 replica Redis |
| `service-redis.yaml` | ClusterIP `redis:6379` |
| `deployment-prometheus.yaml` + `configmap-prometheus.yaml` | Prometheus with scrape config for api & consumer services |
| `deployment-grafana.yaml` + `configmap-grafana.yaml` | Grafana with pre-provisioned datasource (Prometheus) + dashboard (JSON) |
| `k8s/README.md` | Deploy guide: `minikube start` → `kubectl apply -f k8s/` → `minikube service api -n quant-ai` |

### 4.2 Liveness / Readiness

- **Liveness probe**: GET `/health` (returns 200 if process up), initialDelay=10s, period=30s
- **Readiness probe**: GET `/health/ready` (new endpoint — returns 200 only if DB + Redis reachable), initialDelay=5s, period=10s
- Consumer has liveness only (no /health/ready because it doesn't depend on Kafka being "ready" — it retries subscribe)

### 4.3 Resource Requests / Limits

API pod:
- requests: `cpu=100m, memory=256Mi`
- limits: `cpu=500m, memory=512Mi`

Consumer pod:
- requests: `cpu=50m, memory=128Mi`
- limits: `cpu=200m, memory=256Mi`

(Values chosen to fit Minikube's ~4GB default RAM easily.)

## 5. Prometheus Metrics

### 5.1 Automatic HTTP metrics

Use `prometheus-fastapi-instrumentator` (simple, battle-tested):

```python
# app/main.py (modify)
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Auto-exposes: `http_requests_total`, `http_request_duration_seconds` with `method`, `handler`, `status` labels.

### 5.2 Custom ML metrics

New file `app/core/metrics.py`:

```python
from prometheus_client import Counter, Histogram

PREDICT_TOTAL = Counter(
    "quant_ai_predictions_total",
    "Total predictions made",
    ["ticker", "model_type"],
)

PREDICT_CONFIDENCE = Histogram(
    "quant_ai_prediction_confidence",
    "Prediction confidence scores",
    ["ticker"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_INFERENCE_SECONDS = Histogram(
    "quant_ai_model_inference_seconds",
    "Model inference duration",
    ["model_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)
```

### 5.3 Instrumentation points

`app/services/predict_service.py` (modify):
- `PREDICT_TOTAL.labels(ticker=..., model_type=...).inc()` after successful predict
- `PREDICT_CONFIDENCE.labels(ticker=...).observe(prob)` with positive-class probability
- Wrap model inference in `MODEL_INFERENCE_SECONDS.labels(model_type=...).time()`

### 5.4 Grafana dashboard

`k8s/configmap-grafana.yaml` embeds a pre-built dashboard JSON with 6 panels:
1. Request rate (req/sec) per endpoint
2. p50 / p95 / p99 latency per endpoint
3. Predictions per minute per ticker (top 10)
4. Prediction confidence distribution (heatmap)
5. Model inference time by model type
6. API pod count (from `kube_deployment_status_replicas`) — shows HPA in action

Grafana admin: username `admin`, password `admin` (demo only, documented in README).

## 6. Kafka Prediction Event Stream

### 6.1 Event schema

Pydantic model in `app/services/prediction_event_publisher.py`:

```python
class PredictionEvent(BaseModel):
    ticker: str
    prediction: int          # 0 or 1
    confidence: float        # positive-class probability [0, 1]
    model_id: str
    model_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="forbid")
```

### 6.2 Publisher

`app/services/prediction_event_publisher.py`:

```python
async def publish_prediction_event(event: PredictionEvent) -> None:
    """Fire-and-forget publish to Kafka.
    
    If Kafka is unreachable, logs a warning and returns — does NOT block /predict.
    """
    # Uses global AIOKafkaProducer singleton managed via FastAPI lifespan:
    #   - started in app.main.lifespan() on startup (only if BROKER_BACKEND=kafka)
    #   - closed on shutdown
    #   - If settings.BROKER_BACKEND != "kafka", publisher becomes a no-op (logs once)
    # Topic: "prediction_events"
    # Key: event.ticker (so same ticker goes to same partition for ordering)
    # Value: event.model_dump_json().encode("utf-8")
```

**Lifecycle integration** (app/main.py):
```python
from contextlib import asynccontextmanager
from app.services.prediction_event_publisher import start_producer, stop_producer

@asynccontextmanager
async def lifespan(app):
    await start_producer()  # no-op if BROKER_BACKEND != kafka
    yield
    await stop_producer()

app = FastAPI(lifespan=lifespan, ...)
```

Integration: `app/services/predict_service.py` after prediction computed:

```python
import asyncio
from app.services.prediction_event_publisher import publish_prediction_event, PredictionEvent

# After prediction:
asyncio.create_task(publish_prediction_event(PredictionEvent(
    ticker=ticker, prediction=int(pred), confidence=float(conf),
    model_id=model_id, model_type=model_type,
)))
```

**Important**: `asyncio.create_task` = fire-and-forget, doesn't block the HTTP response.

### 6.3 Consumer

New file `app/workers/events_consumer.py` — a small FastAPI app with background task:

```python
from collections import defaultdict, deque
from fastapi import FastAPI
from aiokafka import AIOKafkaConsumer

app = FastAPI(title="quant-ai events consumer")

# In-memory rolling window (last 1000 predictions per ticker)
_stats: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))


@app.on_event("startup")
async def startup():
    # Spawn consumer background task
    asyncio.create_task(consume_loop())


async def consume_loop():
    consumer = AIOKafkaConsumer(
        "prediction_events",
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="events-consumer",
        auto_offset_reset="earliest",
    )
    await consumer.start()
    try:
        async for msg in consumer:
            event = PredictionEvent.model_validate_json(msg.value)
            _stats[event.ticker].append(event)
    finally:
        await consumer.stop()


@app.get("/stats/{ticker}")
def get_stats(ticker: str):
    events = list(_stats.get(ticker.upper(), []))
    if not events:
        return {"ticker": ticker.upper(), "count": 0}
    return {
        "ticker": ticker.upper(),
        "count": len(events),
        "avg_confidence": sum(e.confidence for e in events) / len(events),
        "bullish_ratio": sum(1 for e in events if e.prediction == 1) / len(events),
        "last_prediction_ts": events[-1].timestamp.isoformat(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
```

### 6.4 Dockerfile.consumer

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8001
CMD ["uvicorn", "app.workers.events_consumer:app", "--host", "0.0.0.0", "--port", "8001"]
```

## 7. docker-compose Updates (Parallel to K8s)

Update `docker-compose.yml` to add Kafka + Prometheus + Grafana + consumer. This gives devs a lightweight way to run the full stack without Minikube.

```yaml
# New services added:
  kafka:
    image: bitnami/kafka:3.6
    ports: ["9092:9092"]
    environment:
      KAFKA_CFG_NODE_ID: "1"
      KAFKA_CFG_PROCESS_ROLES: "broker,controller"
      KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: "1@kafka:9093"
      KAFKA_CFG_LISTENERS: "PLAINTEXT://:9092,CONTROLLER://:9093"
      KAFKA_CFG_ADVERTISED_LISTENERS: "PLAINTEXT://kafka:9092"
      KAFKA_CFG_CONTROLLER_LISTENER_NAMES: "CONTROLLER"

  consumer:
    build: { context: ., dockerfile: Dockerfile.consumer }
    ports: ["8001:8001"]
    environment:
      BROKER_BACKEND: kafka
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
    depends_on: [kafka]

  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes: ["./observability/prometheus.yml:/etc/prometheus/prometheus.yml:ro"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - "./observability/grafana-datasources.yml:/etc/grafana/provisioning/datasources/prometheus.yml:ro"
      - "./observability/dashboards:/etc/grafana/provisioning/dashboards:ro"
```

New directory `observability/`:
- `prometheus.yml` — scrape api + consumer
- `grafana-datasources.yml` — provision Prometheus datasource
- `dashboards/quant-ai.json` — the 6-panel dashboard

## 8. Testing

- `tests/test_prometheus_metrics.py` (3+ tests):
  - `/metrics` endpoint returns 200 with prometheus format
  - Custom metrics registered (PREDICT_TOTAL, PREDICT_CONFIDENCE, MODEL_INFERENCE_SECONDS)
  - Hitting `/predict` increments PREDICT_TOTAL counter

- `tests/test_prediction_event_publisher.py` (3+ tests):
  - PredictionEvent schema validation
  - Publisher logs warning and doesn't raise when Kafka unreachable (mocked AIOKafkaProducer raises `KafkaError`)
  - Publisher serializes event correctly (JSON shape matches schema)

- `tests/test_events_consumer.py` (3+ tests):
  - `/stats/{ticker}` returns count=0 for unknown ticker
  - After inserting mock events, `/stats/{ticker}` returns correct count / avg_confidence / bullish_ratio
  - Consumer gracefully handles malformed Kafka messages (logs error, doesn't crash)

- `tests/contract/test_api_stats.py` (2+ tests):
  - Full roundtrip: POST /predict → events_consumer /stats/{ticker} shows updated count
  - (Integration test; may skip if Kafka not available in CI, use a pytest marker)

## 9. Documentation

- `k8s/README.md`: Minikube deploy runbook
  - Prereqs: minikube, kubectl, docker
  - Steps: `minikube start` → `minikube addons enable metrics-server` (for HPA) → `eval $(minikube docker-env)` → `docker build` api + consumer images → `kubectl apply -f k8s/` → verify with `kubectl get pods -n quant-ai`
  - Port-forward commands for local access

- `docs/architecture/distributed.md`: **Interview doc**
  - System diagram (ASCII, not too fancy)
  - CAP tradeoff analysis for this system
  - What each component gives us (Kafka → decoupling, K8s → scaling, Prometheus → observability)
  - Honest limits: single-broker Kafka (no replication), HPA not tested at scale, etc.
  - "If I scaled this to production": bullet list of next steps (multi-broker Kafka on Confluent, Postgres read-replicas on Supabase, K8s on GKE/EKS, service mesh for tracing)

## 10. Constraints / Out of Scope

**Constraints**:
- Minikube must run on Harry's Windows machine (WSL2-backed)
- Total RAM budget: ~3GB (Minikube default); we use ~2.5GB across all pods
- No real cloud deployment in this sub-project (that's Phase B later)

**Out of scope** (defer to future sub-projects):
- Multi-broker Kafka with replication
- Postgres read replicas / sharding
- Service mesh (Istio/Linkerd) + distributed tracing (OpenTelemetry)
- Multi-zone / multi-region
- Distributed training (Ray) — covered in speculative Phase 3 Sub-project 4
- Ingress controller / TLS (use NodePort for demo)
- Helm chart (use plain YAMLs; Helm is a follow-up)

## 11. Success Criteria

1. `kubectl apply -f k8s/` brings up all pods healthy in Minikube within 3 minutes
2. `kubectl get hpa -n quant-ai` shows HPA active with current/target CPU
3. Hitting `/predict` causes PREDICT_TOTAL counter to increase (verified via `/metrics` scrape)
4. Grafana dashboard shows at least 4 panels populated with real data within 2 minutes of activity
5. POST `/predict` 10x for AAPL → `/stats/AAPL` on consumer returns `count=10` with correct aggregates
6. `pytest tests/` — all new tests pass, no existing tests break
7. `docker-compose up` stack also works (parallel validation)
8. `docs/architecture/distributed.md` readable in one sitting (<15 min), explains CAP + this system's place

## 12. Appendix: What this gives Harry at interviews

| Question | Answer (with code/URL proof) |
|----------|------------------------------|
| "Have you used Kubernetes?" | Yes — `k8s/` directory in my Quant AI repo. HPA auto-scales based on CPU, liveness/readiness probes, ConfigMap/Secret separation. |
| "Tell me about Kafka." | My Quant AI backend publishes prediction events to a `prediction_events` topic. A separate K8s deployment runs a consumer that aggregates rolling stats per ticker. Uses aiokafka with keyed partitioning for per-ticker ordering. |
| "How do you monitor services?" | Prometheus scrapes `/metrics` from both API and consumer. Custom ML metrics track predictions per ticker, confidence distributions, model inference time. Grafana dashboard visualizes. |
| "CAP tradeoff?" | Postgres is CP (strong consistency for prices). Kafka events are AP (prediction stream fine to lose one message). Redis cache is AP (best-effort caching, DB is source of truth). Documented in `docs/architecture/distributed.md`. |
| "How would you scale this to production?" | Multi-broker Kafka (Confluent Cloud), Postgres read replicas (Supabase native), K8s on EKS with Istio for tracing, separate worker pools per model type. Listed in the architecture doc. |
