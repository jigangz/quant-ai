# Optuna Multi-Objective Optimization — Design Spec

> **Scope:** Phase 3, Sub-project 1 of 3 (Optuna → Ensemble → RL)
> **Goal:** Enhance Quant AI with multi-objective Bayesian hyperparameter optimization for ML models and automated strategy parameter tuning, with results persisted and accessible from the existing Training/Strategy pages.

---

## 1. Architecture

### New Files

```
app/ml/hyperparam/
├── search.py               # MODIFY — add "optuna_multi" mode
├── spaces.py               # NO CHANGE
├── multi_objective.py      # NEW — multi-objective optimization (NSGA-II)
└── strategy_optimizer.py   # NEW — strategy parameter optimizer

app/services/
└── optimization_service.py # NEW — orchestration layer

app/api/
└── optimize.py             # NEW — REST API endpoints

app/db/
└── optimization_repo.py    # NEW — persistence layer

quant-ai-ui/src/api/
└── client.js               # MODIFY — add optimize API functions

quant-ai-ui/src/pages/
├── Training.jsx            # MODIFY — add Auto-Optimize button + results display
└── Strategy.jsx            # MODIFY — add Optimize Parameters button + results display
```

### Data Flow

```
User clicks "Auto-Optimize" (Training page)
    ↓
POST /api/optimize/model
    ↓
OptimizationService.optimize_model(request)
    ↓
MultiObjectiveSearch.run()
    ├── Objective 1: val_auc (prediction accuracy)
    ├── Objective 2: backtest_sharpe (backtest profitability)
    └── Optuna NSGAIISampler → Pareto front
    ↓
Persist optimization run to DB
    ↓
Return best_params + Pareto front + all trial data

---

User clicks "Optimize Parameters" (Strategy page)
    ↓
POST /api/optimize/strategy
    ↓
OptimizationService.optimize_strategy(request)
    ↓
StrategyOptimizer.run()
    ├── For each trial: instantiate strategy → generate_signals() → BacktestEngine.run()
    ├── Objective: sharpe_ratio (or user-selected metric)
    └── Optuna TPESampler → find optimal params
    ↓
Persist + return results
```

---

## 2. Multi-Objective Model Optimization

### New Class: `MultiObjectiveSearch` (`app/ml/hyperparam/multi_objective.py`)

```python
class MultiObjectiveSearch:
    def __init__(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        backtest_data: dict,        # {ticker: prices_df} for backtest — built by DatasetBuilder during training
        base_params: dict | None = None,  # Starting params (optional, from previous training)
    )

    def run(
        self,
        n_trials: int = 50,
        timeout: int | None = 300,
    ) -> MultiObjectiveResult

    def _objective(self, trial: optuna.Trial) -> tuple[float, float]:
        # 1. Sample params from search space (reuse spaces.py)
        # 2. Train model, compute val_auc
        # 3. Run backtest with trained model, compute sharpe
        # return (val_auc, backtest_sharpe)

    def _select_recommended(
        self, pareto_front: list[ParetoPoint]
    ) -> ParetoPoint:
        # Min-max normalize, pick closest to ideal point (1.0, 1.0)
```

### Data Models

```python
class ParetoPoint(BaseModel):
    params: dict[str, Any]
    val_auc: float
    backtest_sharpe: float

class MultiObjectiveResult(BaseModel):
    pareto_front: list[ParetoPoint]
    recommended_params: dict[str, Any]
    recommended_val_auc: float
    recommended_backtest_sharpe: float
    all_trials: list[TrialResult]  # Reuses TrialResult from hyperparam/search.py
    n_trials: int
    total_time_seconds: float
```

### Integration with Existing Search

Add `"optuna_multi"` as a new `search_mode` in `SearchConfig`. The existing `HyperparamSearch.run()` dispatches to `MultiObjectiveSearch` when mode is `"optuna_multi"`. This keeps the single entry point.

---

## 3. Strategy Parameter Optimization

### New Class: `StrategyOptimizer` (`app/ml/hyperparam/strategy_optimizer.py`)

```python
class StrategyOptimizer:
    def __init__(
        self,
        strategy_name: str,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
    )

    def run(
        self,
        n_trials: int = 100,
        timeout: int | None = 300,
        metric: str = "sharpe_ratio",
        param_overrides: dict | None = None,
    ) -> StrategyOptResult

    def _infer_search_space(
        self, schema: dict, overrides: dict | None
    ) -> dict:
        # Auto-derive from Pydantic Parameters schema

    def _objective(self, trial: optuna.Trial) -> float:
        # 1. Sample params from inferred space
        # 2. Create strategy instance with params
        # 3. Generate signals
        # 4. Run backtest
        # 5. Return target metric value
```

### Search Space Auto-Derivation

| Pydantic Field Type | Optuna Mapping | Range |
|---|---|---|
| `int` with `ge/le` | `suggest_int(ge, le)` | From constraints |
| `float` with `ge/le` | `suggest_float(ge, le)` | From constraints |
| `Literal["a","b","c"]` | `suggest_categorical` | From choices |
| Unconstrained `int` | `suggest_int(default*0.5, default*2)` | Around default |
| Unconstrained `float` | `suggest_float(default*0.5, default*2)` | Around default |

Users can override any param range via `param_overrides`:
```json
{
  "fast_period": {"low": 5, "high": 50},
  "slow_period": {"low": 20, "high": 200}
}
```

### Data Model

```python
class StrategyOptResult(BaseModel):
    best_params: dict[str, Any]
    best_metric: float
    metric_name: str
    all_trials: list[TrialResult]
    n_trials: int
    total_time_seconds: float
    strategy_name: str
    ticker: str
```

---

## 4. API Endpoints

### New Router: `app/api/optimize.py`

| Endpoint | Method | Description |
|---|---|---|
| `/api/optimize/model` | POST | Trigger multi-objective model hyperparameter optimization |
| `/api/optimize/strategy` | POST | Trigger strategy parameter optimization |
| `/api/optimize/runs` | GET | List optimization history |
| `/api/optimize/runs/{id}` | GET | Get single optimization run details (all trials) |

### Request/Response Models

```python
# POST /api/optimize/model
class OptimizeModelRequest(BaseModel):
    tickers: list[str]
    model_type: str  # logistic, xgboost, lightgbm, catboost, random_forest
    n_trials: int = Field(default=50, ge=5, le=200)
    timeout: int | None = Field(default=300, ge=10, le=3600)
    feature_groups: list[str] = ["ta_basic", "momentum"]

# POST /api/optimize/strategy
class OptimizeStrategyRequest(BaseModel):
    strategy_name: str
    ticker: str
    n_trials: int = Field(default=100, ge=5, le=500)
    timeout: int | None = Field(default=300, ge=10, le=3600)
    metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, win_rate, max_drawdown
    param_overrides: dict[str, dict] | None = None
    start_date: date | None = None
    end_date: date | None = None
```

### Error Handling

- 400: Invalid model_type, unknown strategy_name, invalid metric
- 422: Pydantic validation (n_trials out of range, etc.)
- 500: Optuna internal failure (wrapped with error message)

---

## 5. Persistence

### New Table: `optimization_runs`

| Column | Type | Description |
|---|---|---|
| id | TEXT (uuid) | Primary key |
| type | TEXT | "model" or "strategy" |
| config | JSON | Request parameters |
| best_params | JSON | Optimal parameters found |
| best_metrics | JSON | Best metric values |
| pareto_front | JSON | Pareto front points (model optimization only, null for strategy) |
| all_trials | JSON | All trial data |
| n_trials | INTEGER | Completed trials |
| duration_seconds | REAL | Total time |
| created_at | TIMESTAMP | Creation time |

### Repository: `app/db/optimization_repo.py`

```python
class OptimizationRepo:
    async def save_run(self, run: OptimizationRun) -> str  # returns id
    async def get_run(self, run_id: str) -> OptimizationRun | None
    async def list_runs(
        self, type: str | None = None, limit: int = 20
    ) -> list[OptimizationRun]
```

Uses the existing database connection from `app/db/`.

---

## 6. Frontend Integration

### Training Page (`Training.jsx`)

- Add "Auto-Optimize" button below model type selector
- On click: call `POST /api/optimize/model` with selected model_type and tickers
- Show loading state during optimization
- Display results:
  - Recommended params (auto-fill into training form)
  - Pareto scatter plot (val_auc vs sharpe) using Tailwind-styled CSS dots
  - Summary: "Found optimal params in {n_trials} trials ({time}s). Recommended: val_auc={x}, sharpe={y}"

### Strategy Page (`Strategy.jsx`)

- Add "Optimize Parameters" button next to parameter form
- On click: call `POST /api/optimize/strategy` with current strategy + ticker
- Display results:
  - Auto-fill optimal params into form
  - Summary: "Best {metric}={value} found in {n_trials} trials ({time}s)"
  - Top 5 trials table (params + metric)

### API Client (`client.js`)

Add functions:
- `optimizeModel(request)` → POST `/api/optimize/model`
- `optimizeStrategy(request)` → POST `/api/optimize/strategy`
- `listOptimizationRuns(type?)` → GET `/api/optimize/runs`
- `getOptimizationRun(id)` → GET `/api/optimize/runs/{id}`

---

## 7. Testing

| Test File | Coverage | Tests |
|---|---|---|
| `tests/test_multi_objective.py` | MultiObjectiveSearch, Pareto selection, edge cases | 5+ |
| `tests/test_strategy_optimizer.py` | StrategyOptimizer, search space inference, param overrides | 5+ |
| `tests/test_optimization_service.py` | OptimizationService orchestration, persistence | 4+ |
| `tests/test_api_optimize.py` | API endpoints, request validation, response format | 4+ |

### Test Approach

- Mock `ModelFactory.create` and `BacktestEngine.run` — no real model training or data fetching
- Optuna tests use `n_trials=3` + fixed seed for speed and reproducibility
- Strategy search space inference tested with real Pydantic schemas
- API tests use FastAPI TestClient, mock service layer
- Follow existing pattern: `patch` at router boundary (SIGN-022)

### Verify Command (unchanged)

```bash
pytest tests/ -v --ignore=tests/contract -p no:cacheprovider
```

---

## 8. Constraints

- **Python 3.9 compat**: Every new file must have `from __future__ import annotations` at top. Use `Optional[str]` not `str | None` in runtime contexts (SIGN-024).
- **Tailwind dark theme**: All UI uses custom colors `bg-surface`, `bg-surface-card`, `text-up`, `text-down`, `bg-accent`. No inline styles (SIGN-023).
- **No new dependencies**: Optuna already in `requirements-full.txt`. NSGA-II sampler is built into Optuna.
- **Frontend from quant-ai-ui/**: All npm commands run from `quant-ai-ui/` directory (SIGN-021).

---

## 9. Out of Scope

- Real-time optimization progress via WebSocket (future enhancement)
- Automatic re-training triggered by optimization results
- Multi-ticker strategy optimization (one ticker at a time for now)
- Ensemble models (separate sub-project spec)
- RL agent (separate sub-project spec)
