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

## Component responsibilities

### quant-ai-api

The main FastAPI application. Handles all client requests:
- `/predict` — runs ML inference, publishes `PredictionEvent` to Kafka async (fire-and-forget)
- `/train` — queues model training jobs
- `/backtest` — runs strategy backtests
- `/metrics` — Prometheus scrape endpoint (auto-registered by prometheus-fastapi-instrumentator)

Runs 2 replicas by default. Stateless — any instance can handle any request. Session/cache state is in Redis. Model artifacts are in shared storage (local or Supabase in prod).

### quant-ai-consumer

A separate FastAPI application running on port 8001. Subscribes to the `prediction_events` Kafka topic and maintains an in-memory rolling window of up to 1000 events per ticker. Exposes:
- `/stats/{ticker}` — returns count, avg_confidence, bullish_ratio, last_prediction_ts
- `/health` — liveness probe

The consumer is stateless at startup (cold-starts fresh). In production this would be backed by Redis TTL or a compacted Kafka topic so stats survive restarts.

### Kafka (KRaft mode)

Single broker in Minikube. Uses KRaft (Kafka without ZooKeeper) introduced in Kafka 3.x, which simplifies the topology. In production: 3+ brokers, replication factor 3, min.insync.replicas=2.

Topic: `prediction_events` — keyed by ticker (ensures ordering per ticker across partitions).

### Postgres

Used for historical price data, model metadata, training runs, backtest results. In Minikube: single instance with 1Gi PVC. In production: Supabase managed Postgres with connection pooler (PgBouncer).

### Redis

Used for:
1. Price cache (TTL 5 minutes) — reduces Postgres reads during high-frequency prediction
2. Task queue (Redis Streams or list) for background training jobs
3. Rate limiting counters

### Prometheus

Scrapes `/metrics` on api:8000 and consumer:8001 every 15 seconds. Stores time-series data on a local volume.

### Grafana

Reads from Prometheus. Pre-provisioned with 6-panel dashboard (see `observability/dashboards/quant-ai.json`).

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

### Prometheus metrics auto-registered by `prometheus-fastapi-instrumentator`

- `http_requests_total{method, handler, status}` — counter
- `http_request_duration_seconds_bucket{method, handler, le}` — histogram

### Custom ML metrics in `app/core/metrics.py`

- `quant_ai_predictions_total{ticker, model_type}` — counter, increments on each prediction
- `quant_ai_prediction_confidence_bucket{ticker, le}` — histogram, positive-class probability
- `quant_ai_model_inference_seconds_bucket{model_type, le}` — histogram, model inference time

### Grafana dashboard panels (6)

1. **Request Rate by Endpoint** — `rate(http_requests_total[1m])` grouped by handler
2. **Latency p95 by Endpoint** — `histogram_quantile(0.95, ...)` of request duration
3. **Predictions per Minute by Ticker** — `rate(quant_ai_predictions_total[1m]) * 60`
4. **Prediction Confidence Distribution** — heatmap of confidence histogram
5. **Model Inference Time p95 by Model Type** — `histogram_quantile(0.95, ...)` of inference seconds
6. **API Pod Count (HPA)** — `kube_deployment_status_replicas{deployment="quant-ai-api"}`

## How to scale this up

**If time and budget were available**:

1. **Kafka on Confluent Cloud** (free tier ~1 GB egress/month):
   - 3 brokers, replication factor 3
   - SASL auth via `KAFKA_SASL_USERNAME/PASSWORD` env vars
   - Consumer group auto-rebalancing across multiple consumer replicas

2. **Postgres read replicas** via Supabase native feature:
   - Route read-only queries (prices, news) to replicas
   - Primary handles writes (model updates, training results)

3. **Consumer horizontal scale**:
   - Back rolling stats with Redis TTL (or a compacted Kafka topic)
   - Scale consumer to N replicas, each assigned a partition subset via consumer group
   - Requires redesigning `/stats` to aggregate from Redis instead of in-process deque

4. **K8s on EKS / GKE**:
   - Ingress controller (ALB or nginx-ingress) + cert-manager for TLS
   - HPA across both api and consumer
   - Separate node pools (CPU-optimized for api, memory-optimized for consumer)
   - `PodDisruptionBudget` for zero-downtime rolling deploys

5. **Observability improvements**:
   - OpenTelemetry distributed tracing: api → Kafka → consumer (to profile /predict → stats latency)
   - Alertmanager → PagerDuty for p95 latency SLO breaches
   - Loki for log aggregation (currently logs go to stdout)

## Honest limits of current setup

- **Single-broker Kafka means no durability**. If the Kafka pod dies, events in-flight are lost.
- **Consumer is a single pod** — no HA. Restart = lose rolling window.
- **Postgres in-cluster is dev only**. Prod uses Supabase.
- **HPA only tested on CPU**. Real load testing with k6/Locust would be the next step.
- **No TLS between services** (inside-cluster plaintext). Prod would add mTLS or service mesh (Istio/Linkerd).
- **No persistent storage for Prometheus**. Pod restart = loss of metric history. Prod: Thanos or remote_write to a TSDBon S3.

## Lessons learned

- `aiokafka` requires lifespan management — easy to leak producers if not careful. The `start_producer` / `stop_producer` pattern in FastAPI lifespan keeps this clean.
- Prometheus custom metrics have a gotcha: `Counter` and `Histogram` register globally on import. Tests that re-import the module raise `Duplicated timeseries`. Solution: register once in `app/core/metrics.py` and import from there everywhere; tests that need fresh state can use a dedicated `CollectorRegistry`.
- Minikube HPA is wildly inaccurate without the `metrics-server` addon enabled — easy to miss and very confusing.
- KRaft-mode Kafka (no ZooKeeper) makes single-broker demo tractable. The `KAFKA_CFG_CONTROLLER_QUORUM_VOTERS` env var replaces the ZooKeeper connection string.
- `prometheus-fastapi-instrumentator` wraps the app after all routers are registered. If you instrument before adding routers, new routes are not tracked — instrument AFTER `include_router` calls, or call `.instrument(app)` at module load time.
- Fire-and-forget `asyncio.create_task` for Kafka publish from a sync predict endpoint requires a running event loop. In sync FastAPI routes, use `asyncio.get_event_loop().create_task(...)` with a try/except for `RuntimeError` when no loop is running (e.g., in unit tests).
