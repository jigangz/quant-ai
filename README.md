# Quant AI

ML-powered stock direction prediction platform with backtesting and explainability.

**TL;DR:**
- Predicts stock price direction using ML models
- Handles time-series data properly (no look-ahead bias)
- Full backtesting with transaction costs and position sizing
- Model versioning and experiment tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           API Layer                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ /health  │ │  /train  │ │ /predict │ │/backtest │ │ /explain │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
└───────┼────────────┼────────────┼────────────┼────────────┼─────────┘
        │            │            │            │            │
┌───────┼────────────┼────────────┼────────────┼────────────┼─────────┐
│       │     Service Layer       │            │            │         │
│       │    ┌────────────────────┴───┐  ┌─────┴─────┐  ┌───┴────┐   │
│       │    │   TrainingService      │  │ Backtest  │  │  SHAP  │   │
│       │    │ - DatasetBuilder       │  │  Engine   │  │Explainer│  │
│       │    │ - ModelFactory         │  └───────────┘  └────────┘   │
│       │    └────────────────────────┘                              │
└───────┼─────────────────────────────────────────────────────────────┘
        │
┌───────┼─────────────────────────────────────────────────────────────┐
│       │    ML Layer                                                 │
│  ┌────┴────┐  ┌───────────┐  ┌────────────┐  ┌─────────────────┐   │
│  │ Feature │  │   Model   │  │   Label    │  │   Time-Series   │   │
│  │Registry │  │  Factory  │  │  Generator │  │     Splitter    │   │
│  └─────────┘  └───────────┘  └────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
        │
┌───────┼─────────────────────────────────────────────────────────────┐
│       │         Data Layer                                          │
│  ┌────┴────────┐  ┌────────────────┐  ┌─────────────────────────┐  │
│  │   Market    │  │     Model      │  │      Artifacts          │  │
│  │  Provider   │  │   Registry     │  │    (local/S3)           │  │
│  └─────────────┘  └────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Leak Prevention

Time-series data requires special handling. Random shuffling leaks future data into training.

```python
# Wrong: random split leaks future data
X_train, X_test = train_test_split(X, shuffle=True)

# Correct: split by date
# 2020-01 to 2023-06 → train
# 2023-07 to 2023-09 → validation
# 2023-10 to 2024-01 → test
```

Implementation in `DatasetBuilder._time_series_split()`:
```python
def _time_series_split(self, df):
    unique_dates = df["date"].unique()
    train_end = unique_dates[int(len(unique_dates) * 0.7)]
    train_df = df[df["date"] <= train_end]
    # val and test follow sequentially
```

---

## Model Versioning

Each trained model is tracked with full metadata.

```
artifacts/
├── xgboost_AAPL_20240131/
│   ├── model.joblib
│   ├── metadata.json
│   └── metrics.json
```

Registry schema:
```python
class ModelRecord:
    id: str
    name: str
    version: int
    model_type: str        # logistic, xgboost, lightgbm, catboost
    tickers: list[str]
    feature_groups: list[str]
    metrics: dict          # {val_auc: 0.62, val_f1: 0.58}
    artifact_path: str
    created_at: datetime
```

---

## Backtest Metrics

**Classification:**
| Metric | Target |
|--------|--------|
| AUC | > 0.55 |
| F1 | > 0.50 |

**Strategy:**
| Metric | Target |
|--------|--------|
| Sharpe | > 1.0 |
| Max Drawdown | < 20% |
| Win Rate | > 50% |

---

## Version History

| Version | Features | Status |
|---------|----------|--------|
| V1 | Data pipeline, baseline model, SHAP | ✅ |
| V2 | Multi-ticker, model registry, backtesting | ✅ |
| V3 | Async training, 5 models, RAG, agents | ✅ |
| V4 | Real-time, alerts, multi-user | Planned |

---

## Quick Start

### Docker

```bash
git clone https://github.com/jigangz/quant-ai.git
cd quant-ai
cp .env.example .env
docker-compose up
```

### Local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Verify

```bash
curl http://localhost:8000/health
```

---

## Demo

```bash
# Quick check
python scripts/demo_30s.py

# Full demo
python scripts/demo_2min.py

# V3 showcase (recommended)
python scripts/demo_v3.py
python scripts/demo_v3.py --quick  # skip training
```

The V3 demo shows:
- Training 3 models (Logistic, XGBoost, LightGBM)
- Backtest comparison (Sharpe, returns, drawdown)
- Model promotion to production
- Predictions with promoted model
- Technical analysis with SHAP
- RAG-based explanations

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status |
| `/train` | POST | Train model (async by default) |
| `/runs/{id}` | GET | Training run status |
| `/models` | GET | List models |
| `/models/{id}/promote` | POST | Promote to production |
| `/predict` | POST | Get prediction |
| `/backtest` | POST | Run backtest |
| `/agents/technical` | POST | Technical analysis |
| `/rag/answer` | POST | Question answering |

---

## Project Structure

```
app/
├── api/           # FastAPI routes
├── backtest/      # Backtest engine + metrics
├── db/            # Model registry
├── explain/       # SHAP explainer
├── jobs/          # Async job queue (Redis/RQ)
├── ml/
│   ├── dataset/   # DatasetBuilder
│   ├── features/  # Feature registry
│   ├── hyperparam/# Optuna search
│   └── models/    # Model factory (5 types)
├── providers/     # Data providers (Yahoo)
├── rag/           # FAISS + RAG
└── services/      # Business logic
```

---

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default |
|----------|---------|
| `ENV` | dev |
| `REDIS_URL` | redis://localhost:6379 |
| `DEFAULT_MODEL_TYPE` | logistic |
| `STORAGE_BACKEND` | local |

---

## License

MIT
