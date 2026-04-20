# Quant AI

Production-grade ML platform for stock direction prediction, strategy backtesting, and paper trading. Full-stack (FastAPI + React), containerized, observable, deployable to Kubernetes.

**Live:** Frontend [quant-ai-ui.vercel.app](https://quant-ai-ui.vercel.app) · Backend [quant-ai-qzrg.onrender.com](https://quant-ai-qzrg.onrender.com) · [API docs](https://quant-ai-qzrg.onrender.com/docs)

---

## What it does

- **Predicts** stock direction with 6 model types (logistic, random forest, XGBoost, LightGBM, CatBoost, ensemble voting/stacking)
- **Optimizes** hyperparameters multi-objectively (NSGA-II via Optuna) and strategy parameters (TPE)
- **Backtests** both ML models and rule-based strategies with transaction costs and position sizing
- **Explains** predictions via SHAP feature importance and vector search across historical cases
- **Paper-trades** live with WebSocket price feed, order book, portfolio tracking
- **Observable**: Prometheus metrics, Grafana dashboard, Kafka event stream for per-ticker real-time stats
- **Handles time-series properly** — no look-ahead bias, deterministic train/val/test splits by date

---

## Frontend (React 19 + Tremor + shadcn/ui, dark theme)

| Page | Route | What you can do |
|------|-------|-----------------|
| Screener | `/screener` | 10 hot tickers with real Supabase prices, sort by change% or volume, click-through to Dashboard |
| Dashboard | `/dashboard?ticker=AAPL` | Lightweight Charts K-line + volume, 5-day prediction, SHAP explain panel |
| Training | `/training` | Train any of 6 model types, Auto-Optimize (Optuna), 3-tab layout (Train / Runs / Models promote) |
| Strategy | `/strategy` | 4 strategies (MA cross, RSI, Bollinger, Sentiment) with schema-driven params, signal viz, backtest, Optimize params |
| Trading | `/trading` | Paper-trade with market/limit orders, live WebSocket prices (Zustand store), portfolio P&L, order book |
| Explain | `/explain` | SHAP top features + similar historical cases via vector search, graceful fallback when optional deps missing |

Frontend stack: **React 19 + Vite + Tailwind v3 + Tremor** (charts/KPI) **+ shadcn/ui** (Radix primitives) **+ Lightweight Charts v4 + TanStack Query v5 + Zustand + react-hook-form + zod + Geist fonts + Vitest**. Page-level code splitting via `React.lazy()` keeps first-screen JS under 340KB.

---

## Backend API surface (~55 endpoints)

| Category | Representative endpoints |
|----------|-------------------------|
| **Health & Observability** | `/health`, `/health/ready`, `/metrics` (Prometheus) |
| **Market Data** | `/data/market`, `/data/sentiment`, `/data/news` |
| **ML Training** | `/train`, `/runs`, `/runs/{id}`, `/runs/{id}/reproduce` |
| **Models** | `/models`, `/models/{id}`, `/models/{id}/promote`, `/models/types` |
| **Prediction** | `/predict` (GET legacy + POST) |
| **Backtest** | `/backtest`, `/backtest/report` |
| **Features** | `/features/groups`, `/features/groups/{name}` |
| **Strategies** | `/api/strategies`, `/api/strategies/{name}/signals`, `/api/strategies/{name}/backtest` |
| **Paper Trading** | `/api/trading/orders`, `/api/trading/portfolio`, `/api/trading/trades`, `/api/trading/ws/prices` |
| **Hyperparameter Optimization** | `/api/optimize/model`, `/api/optimize/strategy`, `/api/optimize/runs` |
| **Explainability** | `/explain`, `/search` |
| **News** | `/news/{ticker}`, `/news/{ticker}/sentiment-summary`, `/news/{ticker}/similar-days` |
| **RAG** | `/rag/answer`, `/rag/search`, `/rag/index` |
| **Agents** | `/agents/technical`, `/agents/summary` |
| **Serverless Functions** | `/api/functions`, `/api/functions/{name}/invoke` |

Full OpenAPI docs: [https://quant-ai-qzrg.onrender.com/docs](https://quant-ai-qzrg.onrender.com/docs)

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI (Python 3.11) |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP |
| Database | PostgreSQL (Supabase) |
| Cache | Redis (fallback to in-memory) |
| Message broker | Kafka (aiokafka) / Redis / in-memory (pluggable) |
| Job queue | SQS / Redis / in-memory (pluggable) |
| Artifact storage | Local / Supabase / S3 (pluggable) |
| Observability | Prometheus metrics (`/metrics`) + Grafana dashboard |
| Frontend UI kit | Tremor + shadcn/ui (built on Radix) + Lightweight Charts v4 |
| Frontend | React 19, Vite, Tailwind CSS v3, React Router v7 |
| Frontend state | TanStack Query v5 (server state) + Zustand (WebSocket live store) |
| Frontend forms | react-hook-form + zod |
| Frontend fonts | Geist + Geist Mono (@font-face woff2) |
| Frontend tests | Vitest + @testing-library/react (6 smoke tests) |
| Container | Docker (multi-stage, separate api + consumer images) |
| Orchestration | Kubernetes (manifests in `k8s/`) + Horizontal Pod Autoscaler |
| CI | GitHub Actions (unit + contract + frontend test + docker build + post-deploy health check + keep-alive cron) |
| Deploy | Render (backend) + Vercel (frontend) + Supabase (prices + news + RLS) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│     React 19 + Tremor + shadcn/ui + Lightweight Charts (Vercel CDN)      │
│   Screener · Dashboard · Training · Strategy · Trading · Explain         │
│   lazy-loaded pages · TanStack Query cache · Zustand WS live store       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTPS / WebSocket
┌───────────────────────────────▼─────────────────────────────────────────┐
│                       FastAPI Backend (Render)                           │
│   REST + WebSocket + /metrics   │   rate-limit, CORS, request context    │
└───────────────────────────────┬─────────────────────────────────────────┘
           ┌────────────────────┼─────────────────────┬────────────────┐
           ▼                    ▼                     ▼                ▼
  ┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌─────────┐
  │ Services       │  │ ModelFactory     │  │ Strategy Registry│  │ Agents  │
  │ · Training     │  │ 6 model types    │  │ 4 strategies     │  │ (LLM)   │
  │ · Predict      │  │ · logistic       │  │ · MA cross       │  └─────────┘
  │ · Backtest     │  │ · random forest  │  │ · RSI            │
  │ · Explain      │  │ · xgboost        │  │ · Bollinger      │
  │ · Optimization │  │ · lightgbm       │  │ · Sentiment      │
  │ · News         │  │ · catboost       │  └──────────────────┘
  │ · Search       │  │ · ensemble       │
  └────────┬───────┘  └──────────────────┘
           │
   ┌───────┼──────────┬──────────────┬──────────────┐
   ▼       ▼          ▼              ▼              ▼
┌──────┐┌───────┐┌──────────┐┌──────────────┐┌──────────────┐
│Redis ││ Kafka ││Postgres  ││ ArtifactStore││ Functions    │
│cache ││events ││(Supabase)││(local/S3)    ││ (serverless) │
└──────┘└───────┘└──────────┘└──────────────┘└──────────────┘

Distributed layer (k8s/ for local Minikube demo):
  · quant-ai-api (2 replicas + HPA 2-5)
  · quant-ai-consumer (subscribes prediction_events, exposes /stats/{ticker})
  · prometheus + grafana (pre-provisioned 6-panel dashboard)
```

---

## Quick start

### Docker Compose (full stack locally)

```bash
git clone https://github.com/jigangz/quant-ai.git
cd quant-ai
cp .env.example .env
# Edit .env with DATABASE_URL (Supabase or local Postgres)
docker-compose up --build
# API → http://localhost:8000
# Consumer → http://localhost:8001
# Prometheus → http://localhost:9090
# Grafana → http://localhost:3000 (admin/admin)
```

### Backend only

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Configure .env at minimum: DATABASE_URL
uvicorn app.main:app --reload
```

### Frontend only

```bash
cd quant-ai-ui
npm install
VITE_API_BASE=http://localhost:8000 npm run dev
# Opens at http://localhost:5173
```

### Kubernetes (Minikube, full distributed stack)

```bash
minikube start --cpus=4 --memory=6g
minikube addons enable metrics-server
eval $(minikube docker-env)
docker build -t quant-ai:latest --target production .
docker build -t quant-ai-consumer:latest -f Dockerfile.consumer .
cp k8s/secret.example.yaml k8s/secret.yaml  # fill in values
kubectl create configmap grafana-dashboards -n quant-ai \
    --from-file=observability/dashboards/quant-ai.json \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/
minikube service -n quant-ai api      # open API
minikube service -n quant-ai grafana  # open dashboard
```

Full runbook: [`k8s/README.md`](k8s/README.md)

---

## Time-series data integrity

Time-series needs special handling — random shuffling leaks future data into training.

```python
# WRONG — future leaks into train
X_train, X_test = train_test_split(X, shuffle=True)

# RIGHT — chronological split
# 2020-01 → 2023-06 : train     (70%)
# 2023-07 → 2023-09 : validation (15%)
# 2023-10 → 2024-01 : test       (15%)
```

The `DatasetBuilder` enforces this via `SplitConfig` with `train_ratio`/`val_ratio`. Stacking ensembles use `KFold(shuffle=False)` to preserve ordering in OOF predictions.

---

## Data pipeline

Live frontend is backed by real market data, not mocks:

| Component | Detail |
|-----------|--------|
| Source | yfinance (free, no API key needed) |
| Loader | `scripts/backfill_prices.py` — idempotent (`ON CONFLICT DO NOTHING`) |
| Target | Supabase `prices` table (5010 rows = 10 tickers × 501 trading days × 2 years) |
| Tickers | AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM, V, WMT |
| RLS | `service_role` full access + public read via anon key |
| Schema | `scripts/create_prices_table.sql` (declarative, programmatic creation via SQLAlchemy) |

Re-run any time without duplicates:

```bash
python -m scripts.backfill_prices
```

---

## Performance optimizations

Frontend first-screen load went from ~4 seconds (cold Render + 710KB JS + full 500KB payload) down to <1.5s (warm Render + 335KB JS + 8KB payload):

| Fix | Technique | Impact |
|-----|-----------|--------|
| Render cold-start | `.github/workflows/keepalive.yml` cron `*/10 * * * *` pings `/health` | no more 30s cold-start on free tier |
| First-screen JS | `React.lazy()` + `<Suspense>` per page, 6 route-level chunks | 710KB → 335KB (-53%) |
| Screener payload | `useMarket(ticker, lookback=5)` — only need last 2 closes for %change | 500KB → 8KB (-98%) per screener load |

See `quant-ai-ui/src/api/queries.js` (`normalizeMarket` + `useScreenerTickers`) and `quant-ai-ui/src/app/router.jsx` (lazy routes).

---

## Testing

```bash
# 274 unit + integration tests
pytest tests/ -v --ignore=tests/contract

# 45 contract tests
pytest tests/contract/ -v

# Ruff lint
ruff check app/ --ignore F401,F841,E501,F541,E402

# Frontend build check
cd quant-ai-ui && npm run build
```

CI runs all of the above plus Docker build + post-deploy health check on every push to `main`.

---

## Distributed systems features

Everything below actually runs — not just aspirational — with manifests, metrics, and a Grafana dashboard:

| Component | Role | Lives in |
|-----------|------|----------|
| K8s Deployments | API (2+ replicas) and consumer as separate pods | `k8s/deployment-*.yaml` |
| Horizontal Pod Autoscaler | Auto-scale api 2-5 pods on CPU > 70% | `k8s/hpa-api.yaml` |
| Liveness / readiness probes | Health-based pod restarts + traffic gating | `k8s/deployment-api.yaml` |
| Kafka prediction event stream | `/predict` fires events → consumer aggregates per-ticker stats | `app/services/prediction_event_publisher.py` + `app/workers/events_consumer.py` |
| Prometheus metrics | Auto HTTP metrics + 3 custom ML metrics (`PREDICT_TOTAL`, `PREDICT_CONFIDENCE`, `MODEL_INFERENCE_SECONDS`) | `app/core/metrics.py` |
| Grafana dashboard | 6 panels (RPS, p95 latency, predictions/min, confidence heatmap, inference time, pod count) | `observability/dashboards/quant-ai.json` |

CAP analysis and production scale-up plan in [`docs/architecture/distributed.md`](docs/architecture/distributed.md).

---

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `dev` | Environment (dev/test/prod) |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `REDIS_URL` | — | Redis URL (empty → in-memory) |
| `CACHE_BACKEND` | `memory` | `memory` or `redis` |
| `BROKER_BACKEND` | `redis` | `kafka`, `redis`, or `memory` |
| `QUEUE_BACKEND` | `redis` | `sqs`, `redis`, or `memory` |
| `STORAGE_BACKEND` | `local` | `local`, `supabase`, or `s3` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `DEFAULT_MODEL_TYPE` | `logistic` | Default ML model |

---

## Project structure

```
quant-ai/
├── app/                      # FastAPI backend
│   ├── api/                  # REST route handlers
│   ├── services/             # Business logic
│   ├── ml/                   # Models, features, dataset
│   │   └── hyperparam/       # Optuna multi-objective + strategy optimizers
│   ├── strategies/           # Trading strategies (MA, RSI, Bollinger, Sentiment)
│   ├── workers/              # Kafka events consumer
│   ├── infra/                # Broker, queue, cache abstractions
│   ├── db/                   # Repos (prices, news, models, optimization runs)
│   └── core/                 # Settings, logging, metrics
├── quant-ai-ui/              # React frontend
├── k8s/                      # Kubernetes manifests (Minikube-ready)
├── observability/            # Prometheus + Grafana configs + dashboard
├── docs/                     # Architecture + specs + implementation plans
├── scripts/                  # Helper scripts
├── tests/                    # 274 unit + 45 contract tests
├── docker-compose.yml        # Full local stack
├── Dockerfile                # Multi-stage API image
├── Dockerfile.consumer       # Events consumer image
└── requirements.txt
```

---

## License

MIT
