# Quant AI Backend

A data-driven quantitative research and prediction platform.

## Version Overview

| Version | Focus | Status |
|---------|-------|--------|
| **V1** | Data collection, feature engineering, baseline model, SHAP explainability | ✅ Complete |
| **V2** | Multi-ticker, multi-model, training API, model registry, backtesting | 🚧 In Progress |
| **V3** | Async training, experiment tracking, UI training panel, RAG | 📋 Planned |

---

## V2 Development Progress

| Batch | Task | Status |
|-------|------|--------|
| 1 | Settings + Docker | ✅ Complete |
| 2 | Provider Abstraction | ⬜ Pending |
| 3 | DatasetBuilder + Multi-ticker | ⬜ Pending |
| 4 | Feature Groups System | ⬜ Pending |
| 5 | ModelFactory + TrainingService | ⬜ Pending |
| 6 | /train API + Model Registry | ⬜ Pending |
| 7 | Backtesting Engine | ⬜ Pending |
| 8 | Contract Tests + CI | ⬜ Pending |
| 9 | Cloud Deploy + Observability | ⬜ Pending |
| 10 | Demo + Documentation | ⬜ Pending |

---

## Quick Start

### Docker (Recommended)

```bash
# Clone and start
git clone https://github.com/jigangz/quant-ai.git
cd quant-ai
cp .env.example .env
docker-compose up

# Verify
curl http://localhost:8000/health
```

### Local Development

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Run
uvicorn app.main:app --reload

# Verify
curl http://localhost:8000/health
```

---

## Project Structure

```
app/
├── api/              # FastAPI routes
│   ├── health.py     # /health (with settings info)
│   ├── market.py     # /data/market
│   ├── predict.py    # /predict
│   └── explain.py    # /explain (SHAP)
├── core/
│   ├── config.py     # Legacy config (V1)
│   └── settings.py   # Pydantic Settings (V2)
├── db/               # Database layer
├── ml/
│   ├── features/     # Feature engineering
│   ├── labels/       # Label generation
│   └── split/        # Train/val/test split
├── providers/        # Data providers (Yahoo, etc.)
├── explain/          # SHAP explainer
└── main.py           # FastAPI app

scripts/
├── train.py          # Baseline training
├── optuna_train.py   # Hyperparameter tuning
└── init.sql          # Database schema

docs/
└── env-setup.md      # Environment setup guide
```

---

## API Endpoints

### Health Check

```
GET /health
```

Returns service status and public settings (no secrets):

```json
{
  "status": "ok",
  "settings": {
    "env": "dev",
    "providers_enabled": ["market"],
    "default_feature_groups": ["ta_basic", "volatility"],
    "default_model_type": "logistic",
    "supabase_configured": false
  }
}
```

### Market Data

```
GET /data/market?ticker=AAPL&period=1y&limit=500
```

### Predict

```
POST /predict
{
  "ticker": "AAPL",
  "horizons": [5],
  "features": {}
}
```

### Explain (SHAP)

```
GET /explain?ticker=AAPL
```

---

## Configuration

All settings are loaded from environment variables. See `.env.example` for full list.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (dev/prod/test) | `dev` |
| `DATABASE_URL` | Database connection | `sqlite:///:memory:` |
| `PROVIDERS_ENABLED` | Data providers | `market` |
| `DEFAULT_FEATURE_GROUPS` | Feature groups | `ta_basic,volatility` |
| `DEFAULT_MODEL_TYPE` | Model type | `logistic` |
| `STORAGE_BACKEND` | Artifact storage | `local` |

### Supabase (V2)

For model registry in production:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

---

## Training

### V1: Script-based

```bash
python scripts/train.py
```

Trains LogisticRegression on AAPL, saves to `artifacts/`.

### V2: API-based (Coming Soon)

```
POST /train
{
  "tickers": ["AAPL", "MSFT"],
  "model_type": "xgboost",
  "feature_groups": ["ta_basic", "volatility"],
  "date_range": ["2020-01-01", "2024-01-01"]
}
```

---

## V1 Features (Complete)

- ✅ Yahoo Finance data provider
- ✅ Technical indicators (SMA, EMA, RSI, MACD, Bollinger)
- ✅ Future return labels
- ✅ Time-series train/val/test split
- ✅ Logistic Regression baseline
- ✅ Optuna hyperparameter tuning
- ✅ SHAP explainability
- ✅ Docker support
- ✅ CI/CD (GitHub Actions)

## V2 Features (In Progress)

- 🚧 Multi-ticker support
- 🚧 Multiple model types (RF, XGBoost, LightGBM)
- 🚧 Feature groups system
- 🚧 Training API (`/train`)
- 🚧 Model registry (Supabase)
- 🚧 Backtesting engine
- ⬜ Walk-forward validation

## V3 Features (Planned)

- ⬜ Async training (Job Queue)
- ⬜ Experiment tracking
- ⬜ UI training panel
- ⬜ Lightweight RAG
- ⬜ Technical analysis agent

---

## License

MIT

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes
4. Run tests: `pytest`
5. Run linting: `ruff check .`
6. Submit PR
