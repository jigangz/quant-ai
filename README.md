# Quant AI

ML-powered stock direction prediction platform with backtesting, explainability, and a full React frontend.

**TL;DR:**
- Predicts stock price direction using ML models (Logistic, XGBoost, LightGBM, CatBoost)
- Handles time-series data properly (no look-ahead bias)
- Full backtesting with transaction costs and position sizing
- Model versioning and experiment tracking
- React 19 + Tailwind CSS dark-theme frontend with 6 pages

---

## Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Screener | `/` | Hot-ticker table with sort by change% or volume; click to open Dashboard |
| Dashboard | `/dashboard?ticker=AAPL` | Market data, live prediction, SHAP explain |
| Training | `/training` | Submit training jobs, poll status, promote models |
| Strategy | `/strategy` | Select strategy, set params, generate signals, run backtest |
| Trading | `/trading` | Paper-trade: place orders, view portfolio P&L, live WebSocket prices |
| Explain | `/explain` | SHAP feature importance for any ticker |

---

## API Endpoints

### Core

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status |
| `/data/market?ticker=X` | GET | Latest market prices for ticker |
| `/features/groups` | GET | Available feature groups |
| `/predict` | POST | Run prediction with promoted model |
| `/explain?ticker=X` | GET | SHAP feature importances |
| `/search?q=X` | GET | Search news / documents |

### Training & Models

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train` | POST | Start async training job |
| `/runs/{id}` | GET | Training run status |
| `/runs` | GET | List recent runs |
| `/models` | GET | List models |
| `/models/{id}/promote` | POST | Promote model to production |
| `/models/promoted` | GET | Get promoted model |
| `/backtest` | POST | Run ML backtest |

### Strategies

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/strategies` | GET | List available strategies |
| `/api/strategies/{name}` | GET | Strategy schema + params |
| `/api/strategies/{name}/signals` | POST | Generate trading signals |
| `/api/strategies/{name}/backtest` | POST | Run strategy backtest |

### Paper Trading

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trading/orders` | POST | Place order |
| `/api/trading/orders` | GET | List orders |
| `/api/trading/orders/{id}` | DELETE | Cancel order |
| `/api/trading/portfolio` | GET | Portfolio positions + P&L |
| `/api/trading/portfolio/history` | GET | Portfolio value history |
| `/api/trading/portfolio/reset` | POST | Reset portfolio |
| `/api/trading/trades` | GET | Recent trades |
| `/api/trading/ws/prices` | WS | Live price feed |

### Intelligence

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agents/technical` | POST | Technical analysis agent |
| `/rag/answer` | POST | RAG question answering |
| `/news?ticker=X` | GET | News for ticker |

---

## Quick Start

### Docker

```bash
git clone https://github.com/jigangz/quant-ai.git
cd quant-ai
cp .env.example .env
# Edit .env with your DATABASE_URL (Supabase or local Postgres)
docker-compose up
```

### Local (Backend)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Set ENV=dev, DATABASE_URL, etc. in .env
uvicorn app.main:app --reload
```

### Local (Frontend)

```bash
cd quant-ai-ui
npm install
npm run dev
# Opens at http://localhost:5173 — proxies API to http://localhost:8000
```

### Verify

```bash
curl http://localhost:8000/health
# {"status": "ok", ...}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI (Python 3.9+) |
| ML Models | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Database | PostgreSQL (Supabase) / SQLite (test) |
| Model Storage | Local filesystem / S3 |
| Cache / Queue | Redis (or in-memory for dev) |
| Explainability | SHAP |
| RAG | FAISS + sentence-transformers |
| Frontend | React 19, Vite |
| UI Framework | Tailwind CSS v3 (dark theme) |
| Routing | React Router v7 |
| CI | GitHub Actions |
| Deploy | Render (backend) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       React Frontend (Vite)                          │
│  Screener | Dashboard | Training | Strategy | Trading | Explain      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP / WebSocket
┌───────────────────────────▼─────────────────────────────────────────┐
│                           FastAPI Backend                            │
│  /health  /data  /train  /predict  /backtest  /explain  /api/*      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                         Service Layer                                │
│  TrainingService | PredictionService | BacktestEngine | ShapExplainer│
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                           Data Layer                                 │
│  MarketProvider | ModelRegistry | PricesRepo | ArtifactStore         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Leak Prevention

Time-series data requires special handling. Random shuffling leaks future data into training.

```python
# Wrong: random split leaks future data
X_train, X_test = train_test_split(X, shuffle=True)

# Correct: split by date (70% train / 15% val / 15% test)
# 2020-01 to 2023-06 → train
# 2023-07 to 2023-09 → validation
# 2023-10 to 2024-01 → test
```

---

## Model Versioning

```
artifacts/
├── xgboost_AAPL_20240131/
│   ├── model.joblib
│   ├── metadata.json
│   └── metrics.json
```

---

## Testing

```bash
# Unit + integration tests (213 tests)
pytest tests/ -v --ignore=tests/contract

# Contract tests (39 tests)
pytest tests/contract/ -v

# Frontend build check
cd quant-ai-ui && npm run build
```

---

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `dev` | Environment (dev/test/prod) |
| `DATABASE_URL` | — | PostgreSQL or SQLite URL |
| `REDIS_URL` | — | Redis (leave empty for in-memory) |
| `DEFAULT_MODEL_TYPE` | `logistic` | Default ML model |
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `CACHE_BACKEND` | `memory` | `memory` or `redis` |

---

## License

MIT
