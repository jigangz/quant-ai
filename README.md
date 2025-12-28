# Quant AI Backend

Quant AI is a two-phase project to build a data-driven research and trading platform.

- **Version 1** focuses on a clean backend service for collecting market data, generating features, training baseline models, and exposing simple REST APIs.
- **Version 2** extends the backend with GenAI/RAG support, trading agents, and a richer UI.

This repository contains the backend codebase.

---

## Quick start

The backend is written in **Python** and uses **FastAPI**, **SQLAlchemy**, and **scikit-learn**.

### Run locally

1. **Clone and install dependencies**
   - Install **Python 3.10**
   - Clone this repo
   - Run:
     - `pip install -r requirements.txt`

   The `requirements.txt` file lists FastAPI, SQLAlchemy, yfinance, scikit-learn, Optuna, and SHAP.

2. **Configure the database**
   - By default the code uses an in-memory SQLite database:
     - `DATABASE_URL=sqlite:///:memory:`
   - For persistent storage, set `DATABASE_URL` in a `.env` file to point at your Postgres instance, e.g.:
     - `postgresql+psycopg://user:password@host:port/db`

   The `prices` table must exist with columns:
   - `(ticker, date, open, high, low, close, volume)`
   - `(ticker, date)` as a **unique key**

3. **Run the API**
   - Start the server:
     - `uvicorn app.main:app --reload`
   - Verify health endpoint:
     - `GET /health` → `{"status": "ok"}`

4. **Run tests and linting**
   - The repo includes a GitHub Actions workflow that installs dependencies, checks code style with **ruff**, and runs **pytest** on every push.
   - Locally:
     - `pytest`

5. **Train a model**
   - Use `scripts/train.py` to train a baseline model on historical prices.
   - It loads prices from the database, adds technical indicators, creates labels, splits data chronologically, trains a logistic regression model, and saves metrics + model artifacts in `artifacts/`.

---

## Repository structure

```
app/
  api/            # FastAPI route handlers
    health.py     # /health endpoint
    market.py     # /data/market endpoint for OHLCV data
    explain.py    # /explain endpoint for SHAP explainability
  core/           # configuration (Pydantic settings)
  db/             # database engine and repository functions
  explain/        # SHAP explainer
  ml/             # feature engineering, labels, splits
  providers/      # external data providers (Yahoo Finance implemented)
  rag/            # retrieval-augmented generation (placeholder)
  main.py         # FastAPI application definition
configs/          # reserved for environment-specific YAML configs (empty for now)
scripts/          # training scripts (baseline and Optuna tuning)
tests/            # pytest tests (currently only /health)
```

---

## API modules

### Health check (GET /health)
Returns:
```json
{"status": "ok"}
```
Signals that the service is running.

### Market data (GET /data/market)
Fetches OHLCV data for a ticker.

Flow:
1. Check cached data in the database.
2. If not cached, call the Yahoo Finance provider.
3. Upsert records and return the latest N rows.

Query parameters:
- `ticker` (string)
- `period` (e.g. `1mo`, `3mo`, `1y`)
- `limit` (number of rows)

Data is cached per ticker so repeated requests avoid hitting the external API.

### Explainability (GET /explain)
Loads a trained model and returns the top-k SHAP features for a given ticker.

`ShapExplainer`:
- Retrieves price data
- Builds features
- Generates labels
- Computes SHAP values using the model’s classifier + imputer steps

Returns a JSON payload containing:
- ticker
- sample count
- features ranked by mean absolute SHAP value

---

## Data provider

Market data is retrieved via the Yahoo Finance provider in `app/providers/yahoo.py`.

- `fetch_ohlcv` wraps `yfinance.download`
- Resets indices
- Renames columns
- Adds a `ticker` column

By default it downloads the maximum history to support long rolling windows.

Additional providers (e.g. sentiment, news) will be added in future versions.

---

## Feature engineering and labels

### Technical indicators
Implemented in `app/ml/features/technical.py`.

Given an OHLCV DataFrame, it computes:
- Moving averages (5-day, 20-day)
- Exponential moving averages (12-day, 26-day)
- MACD
- 20-day rolling volatility
- 14-day RSI

The function sorts by date and uses rolling windows so there is **no leakage of future information**.

### Labels
The label generator (`app/ml/labels/returns.py`) creates a binary label for the **next 5-day return direction** (configurable via `HORIZON_DAYS`).

Steps:
- Shift close price by `-HORIZON_DAYS`
- Compute future return
- Label = 1 if future return > 0

Rows with missing future data are dropped to prevent leakage.

---

## Machine-learning pipeline

Training scripts live in `scripts/`.

### Baseline training (`scripts/train.py`)
Steps:
1. Load price data for a hard-coded ticker (currently `AAPL`) from the database
2. Add technical features using `add_technical_features`
3. Generate labels using `add_future_return_label`
4. Build X/y via `build_xy` (selects only predefined feature groups)
5. Chronological split into train/val/test via `time_series_split` (default 70%/15%/15%)
6. Fit logistic regression pipeline:
   - median imputer
   - balanced class weights
7. Evaluate on validation set:
   - accuracy
   - AUC
8. Save model artifact:
   - `artifacts/model.joblib`

### Hyper-parameter tuning (`scripts/optuna_train.py`)
- Defines an Optuna objective sampling logistic regression parameters:
  - `C`, class weight, `l1_ratio`
- Maximizes validation AUC
- Runs 20 trials by default
- Saves best parameters and trial history as artifacts

---

## Explainability with SHAP

`ShapExplainer`:
- Loads a trained model
- Recomputes feature matrix for a ticker
- Uses SHAP `LinearExplainer` to compute per-feature contributions
- Returns top-k features ranked by mean absolute SHAP value

This endpoint forms the basis for future research explainability and will feed into the RAG/agent components in Version 2.

---

## Current progress (V1.1–V1.6)

The following milestones from the v1 roadmap have been completed:

| Milestone | Implementation details |
|---|---|
| **V1.1 Project skeleton and CI** | Repo includes a FastAPI app (`app/main.py`) with a `/health` endpoint. GitHub Actions runs ruff and pytest on every push. |
| **V1.2 Yahoo Finance provider and market API** | Implemented in `app/providers/yahoo.py` using yfinance. `/data/market` fetches from DB or Yahoo and caches results. |
| **V1.3 Technical indicators and data splits** | `add_technical_features` computes MA, EMA, MACD, volatility, RSI without future leakage. `time_series_split` performs chronological splitting. |
| **V1.4 Baseline model** | Logistic regression classifier with a median imputer trained via `scripts/train.py`, saving metrics and a model artifact. |
| **V1.5 Hyper-parameter tuning** | `scripts/optuna_train.py` runs multiple Optuna trials and records best parameters. |
| **V1.6 SHAP explainability** | `ShapExplainer` computes SHAP values for the trained model and exposes them via `/explain`. |

Work for the remaining v1 scope (vector database, services layer, UI, Docker/cloud deployment) is still pending.

---

## Roadmap

### Version 1 (backend & ML)

- **Project skeleton and CI ✓**: FastAPI app, `/health`, placeholder classes for agents and RAG, basic configuration, GitHub Actions pipeline.
- **Market data provider ✓**: Yahoo Finance provider, Postgres/SQLite persistence, `/data/market` endpoint with caching.
- **Feature engineering ✓**: Technical indicators (MA, EMA, MACD, volatility, RSI), chronological train/val/test split to avoid leakage.
- **Baseline model ✓**: 5-day future return direction task, logistic regression baseline, metrics + artifacts.
- **Hyper-parameter tuning ✓**: Optuna optimization for logistic regression hyper-parameters.
- **Explainability ✓**: SHAP feature importance via API.
- **Vector database & retrieval (pending)**: integrate a vector store (FAISS in v1) to store SHAP summaries, research notes, and error cases; implement `/search` endpoint for similarity search.
- **Service layer integration (pending)**: build services to orchestrate data collection, feature generation, model prediction, and explanation; add placeholder endpoints `/agents/run` and `/rag/answer`.
- **Minimal UI (pending)**: dashboard with price charts, predictions, indicators; explainability and retrieval pages.
- **Docker & cloud deployment (pending)**: Docker / docker-compose (API + database) and deployment instructions (Fly.io/Render/Cloud Run).

### Version 2 (GenAI, RAG & agents)

- **Environment preparation**: GPU-enabled environment (Ubuntu + CUDA + PyTorch) if deep models are used; document GPU usage.
- **Multi-provider abstraction**: Provider base class with `fetch`; add sentiment and news providers as placeholders or simple implementations.
- **Cloud vector database**: abstract vector store behind an interface; integrate managed solution (Pinecone or Qdrant); configure via environment variables.
- **Retrieval-Augmented Generation (RAG)**: implement `/rag/answer` by retrieving relevant notes from the vector store and assembling prompts for an LLM; return evidence + summarized results.
- **Trading agents**: agents (technical, sentiment, news) producing structured research conclusions; bull/bear debate + risk manager for risk-adjusted recommendations.
- **UI research room**: analyst cards, bull/bear debate, risk panels; exportable research report.
- **Validation & safety**: schema validation on LLM outputs, restrict responses to retrieved evidence, fallback templates for invalid outputs.
- **Production deployment & observability**: deploy backend + UI; structured logging + basic monitoring; secrets via GitHub Actions + env vars.
- **Final demo**: architecture diagrams, demo scripts, end-to-end run instructions.

---

## Planned UI and user-configurable training

A key extension requested for the next iteration is a front-end that allows users to:
- select tickers
- choose training windows
- pick models
- add custom features

### Dynamic training parameters
Expose a `/train` API endpoint accepting JSON:
- ticker list
- date range
- feature groups
- model type

Move training logic from `scripts/train.py` into a service layer so it can be invoked via API.

### UI controls
Build a simple web UI (React or Vue) with:
- ticker inputs
- date range selectors
- feature group checkboxes
- model dropdown (logistic regression, XGBoost, etc.)
- hyper-parameter tuning options

### Progress feedback
Display:
- training status
- performance metrics
- feature importance

After training, update the explainability page to reflect the user-selected model.

### Persistence
Allow users to:
- save trained models
- load models later for predictions/explanations
- version models by ticker and training date

These changes require both backend additions (service layer + training API) and front-end work, and align naturally with the remaining Version 1 tasks.