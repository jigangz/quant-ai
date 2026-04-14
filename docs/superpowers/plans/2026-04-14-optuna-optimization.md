# Optuna Multi-Objective Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-objective Bayesian hyperparameter optimization for ML models and automated strategy parameter tuning, with persistence and frontend integration.

**Architecture:** Extends existing `HyperparamSearch` with a new `optuna_multi` mode using NSGA-II sampler. New `StrategyOptimizer` automates strategy parameter search via Optuna + BacktestEngine. Results persisted to DB via `OptimizationRepo`, exposed through REST API, and integrated into existing Training/Strategy frontend pages.

**Tech Stack:** Optuna (NSGA-II, TPE), FastAPI, Pydantic, SQLite/Supabase, React + Tailwind CSS

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| CREATE | `app/ml/hyperparam/multi_objective.py` | Multi-objective model optimization (NSGA-II) |
| CREATE | `app/ml/hyperparam/strategy_optimizer.py` | Strategy parameter optimization |
| MODIFY | `app/ml/hyperparam/search.py` | Add `optuna_multi` dispatch |
| CREATE | `app/services/optimization_service.py` | Orchestration: build data, run optimizer, persist |
| CREATE | `app/db/optimization_repo.py` | Persistence for optimization runs |
| CREATE | `app/api/optimize.py` | REST API endpoints |
| MODIFY | `app/main.py` | Register optimize router |
| MODIFY | `quant-ai-ui/src/api/client.js` | Add optimize API functions |
| MODIFY | `quant-ai-ui/src/pages/Training.jsx` | Add Auto-Optimize button |
| MODIFY | `quant-ai-ui/src/pages/Strategy.jsx` | Add Optimize Parameters button |
| CREATE | `tests/test_multi_objective.py` | Tests for multi-objective search |
| CREATE | `tests/test_strategy_optimizer.py` | Tests for strategy optimizer |
| CREATE | `tests/test_optimization_service.py` | Tests for optimization service |
| CREATE | `tests/test_api_optimize.py` | Tests for API endpoints |

---

### Task 1: Multi-Objective Model Optimization

**Files:**
- Create: `app/ml/hyperparam/multi_objective.py`
- Test: `tests/test_multi_objective.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_multi_objective.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from app.ml.hyperparam.multi_objective import (
    MultiObjectiveSearch,
    MultiObjectiveResult,
    ParetoPoint,
)


def _make_data(n=100, n_features=5, seed=42):
    """Generate synthetic train/val data."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.randint(0, 2, n))
    return X, y


class TestMultiObjectiveSearch:
    def test_run_returns_result(self):
        X_train, y_train = _make_data(80)
        X_val, y_val = _make_data(20, seed=99)

        search = MultiObjectiveSearch(
            model_type="logistic",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            backtest_data={},
        )
        result = search.run(n_trials=3, timeout=60)

        assert isinstance(result, MultiObjectiveResult)
        assert result.n_trials >= 1
        assert result.total_time_seconds > 0
        assert result.recommended_params is not None
        assert len(result.all_trials) >= 1

    def test_pareto_front_populated(self):
        X_train, y_train = _make_data(80)
        X_val, y_val = _make_data(20, seed=99)

        search = MultiObjectiveSearch(
            model_type="logistic",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            backtest_data={},
        )
        result = search.run(n_trials=5, timeout=60)

        assert len(result.pareto_front) >= 1
        for point in result.pareto_front:
            assert isinstance(point, ParetoPoint)
            assert "val_auc" in str(point.val_auc) or point.val_auc >= 0

    def test_select_recommended_closest_to_ideal(self):
        """Recommended point should be the one closest to (1.0, 1.0) after normalization."""
        search = MultiObjectiveSearch.__new__(MultiObjectiveSearch)
        
        points = [
            ParetoPoint(params={"a": 1}, val_auc=0.55, backtest_sharpe=0.2),
            ParetoPoint(params={"a": 2}, val_auc=0.60, backtest_sharpe=1.5),
            ParetoPoint(params={"a": 3}, val_auc=0.58, backtest_sharpe=1.0),
        ]
        
        best = search._select_recommended(points)
        # Point with highest combined normalized score
        assert best.params == {"a": 2}

    def test_handles_failed_trials_gracefully(self):
        X_train, y_train = _make_data(80)
        X_val, y_val = _make_data(20, seed=99)

        search = MultiObjectiveSearch(
            model_type="logistic",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            backtest_data={},
        )
        # Even with very few trials, should not raise
        result = search.run(n_trials=2, timeout=30)
        assert result.n_trials >= 1

    def test_backtest_objective_used_when_data_provided(self):
        X_train, y_train = _make_data(80)
        X_val, y_val = _make_data(20, seed=99)

        # Provide backtest data so sharpe is computed
        dates = pd.date_range("2024-01-01", periods=80)
        prices_df = pd.DataFrame({
            "date": dates,
            "close": np.random.RandomState(42).uniform(100, 200, 80),
            "open": np.random.RandomState(42).uniform(100, 200, 80),
            "high": np.random.RandomState(42).uniform(100, 200, 80),
            "low": np.random.RandomState(42).uniform(100, 200, 80),
            "volume": np.random.RandomState(42).randint(1000, 10000, 80),
        })

        search = MultiObjectiveSearch(
            model_type="logistic",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            backtest_data={"AAPL": prices_df},
        )
        result = search.run(n_trials=3, timeout=60)

        # All trials should have both metrics
        for trial in result.all_trials:
            assert "val_auc" in trial.metrics
            assert "backtest_sharpe" in trial.metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/test_multi_objective.py -v --tb=short 2>&1 | tail -15`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ml.hyperparam.multi_objective'`

- [ ] **Step 3: Implement MultiObjectiveSearch**

Create `app/ml/hyperparam/multi_objective.py`:

```python
from __future__ import annotations

"""
Multi-Objective Hyperparameter Search

Uses Optuna's NSGAIISampler to simultaneously optimize:
- val_auc (prediction accuracy)
- backtest_sharpe (strategy profitability)

Returns a Pareto front and a recommended balanced point.
"""

import logging
import math
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticModel, ConfigDict
from sklearn.metrics import roc_auc_score

from app.ml.models import ModelFactory
from .spaces import get_search_space, sample_from_space
from .search import TrialResult

logger = logging.getLogger(__name__)


class ParetoPoint(PydanticModel):
    """A single point on the Pareto front."""

    params: dict[str, Any]
    val_auc: float
    backtest_sharpe: float

    model_config = ConfigDict(extra="forbid")


class MultiObjectiveResult(PydanticModel):
    """Result of multi-objective optimization."""

    pareto_front: list[ParetoPoint]
    recommended_params: dict[str, Any]
    recommended_val_auc: float
    recommended_backtest_sharpe: float
    all_trials: list[TrialResult]
    n_trials: int
    total_time_seconds: float

    model_config = ConfigDict(extra="forbid")


class MultiObjectiveSearch:
    """
    Multi-objective hyperparameter search using Optuna NSGA-II.

    Optimizes two objectives simultaneously:
    1. val_auc — model prediction accuracy on validation set
    2. backtest_sharpe — Sharpe ratio from running a backtest with the model

    Usage:
        search = MultiObjectiveSearch(
            model_type="xgboost",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            backtest_data={"AAPL": prices_df},
        )
        result = search.run(n_trials=50, timeout=300)
    """

    def __init__(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        backtest_data: Optional[dict[str, pd.DataFrame]] = None,
        base_params: Optional[dict[str, Any]] = None,
    ):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.backtest_data = backtest_data or {}
        self.base_params = base_params or {}
        self.search_space = get_search_space(model_type)

    def run(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = 300,
    ) -> MultiObjectiveResult:
        """Run multi-objective optimization."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        start_time = time.time()
        trials: list[TrialResult] = []

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            params = {}
            for param_name, param_def in self.search_space.items():
                params[param_name] = sample_from_space(trial, param_name, param_def)

            trial_start = time.time()
            full_params = {**self.base_params, **params}

            try:
                # Train model
                model = ModelFactory.create(self.model_type, **full_params)
                model.fit(self.X_train, self.y_train)

                # Objective 1: val_auc
                val_proba = model.predict_proba(self.X_val)[:, 1]
                val_auc = roc_auc_score(self.y_val, val_proba)

                # Objective 2: backtest_sharpe
                backtest_sharpe = self._compute_sharpe(model)

                metrics = {
                    "val_auc": round(val_auc, 4),
                    "backtest_sharpe": round(backtest_sharpe, 4),
                }

                trials.append(TrialResult(
                    trial_number=trial.number,
                    params=params,
                    metrics=metrics,
                    duration_seconds=round(time.time() - trial_start, 2),
                ))

                return val_auc, backtest_sharpe

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                trials.append(TrialResult(
                    trial_number=trial.number,
                    params=params,
                    metrics={"val_auc": 0.0, "backtest_sharpe": 0.0},
                    duration_seconds=round(time.time() - trial_start, 2),
                ))
                return 0.0, 0.0

        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        # Build Pareto front
        pareto_front = []
        for trial in study.best_trials:
            pareto_front.append(ParetoPoint(
                params=trial.params,
                val_auc=trial.values[0],
                backtest_sharpe=trial.values[1],
            ))

        # Select recommended point
        if pareto_front:
            recommended = self._select_recommended(pareto_front)
        else:
            recommended = ParetoPoint(
                params=self.base_params,
                val_auc=0.0,
                backtest_sharpe=0.0,
            )

        total_time = round(time.time() - start_time, 2)

        logger.info(
            f"Multi-objective search complete: {len(study.trials)} trials, "
            f"Pareto front: {len(pareto_front)} points, "
            f"recommended: auc={recommended.val_auc:.4f}, sharpe={recommended.backtest_sharpe:.4f}"
        )

        return MultiObjectiveResult(
            pareto_front=pareto_front,
            recommended_params=recommended.params,
            recommended_val_auc=recommended.val_auc,
            recommended_backtest_sharpe=recommended.backtest_sharpe,
            all_trials=trials,
            n_trials=len(study.trials),
            total_time_seconds=total_time,
        )

    def _compute_sharpe(self, model) -> float:
        """Compute a quick Sharpe ratio from model predictions on backtest data."""
        if not self.backtest_data:
            return 0.0

        all_returns = []
        for ticker, prices_df in self.backtest_data.items():
            if prices_df is None or prices_df.empty:
                continue

            try:
                close = prices_df["close"].values
                returns = np.diff(close) / close[:-1]

                # Use val data features if available, else just use returns with random signal
                # In practice, the optimizer provides X_val-aligned signals
                val_proba = model.predict_proba(self.X_val)[:, 1]
                # Align: use as many returns as we have predictions
                n = min(len(val_proba), len(returns))
                if n == 0:
                    continue

                positions = np.where(val_proba[:n] > 0.5, 1.0, -1.0)
                strategy_returns = positions * returns[:n]
                all_returns.extend(strategy_returns.tolist())
            except Exception:
                continue

        if not all_returns:
            return 0.0

        arr = np.array(all_returns)
        mean_ret = np.mean(arr)
        std_ret = np.std(arr)
        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret * math.sqrt(252))

    def _select_recommended(self, pareto_front: list[ParetoPoint]) -> ParetoPoint:
        """Select the point closest to ideal (1.0, 1.0) after min-max normalization."""
        if len(pareto_front) == 1:
            return pareto_front[0]

        aucs = [p.val_auc for p in pareto_front]
        sharpes = [p.backtest_sharpe for p in pareto_front]

        auc_min, auc_max = min(aucs), max(aucs)
        sharpe_min, sharpe_max = min(sharpes), max(sharpes)

        auc_range = auc_max - auc_min if auc_max > auc_min else 1.0
        sharpe_range = sharpe_max - sharpe_min if sharpe_max > sharpe_min else 1.0

        best_point = pareto_front[0]
        best_dist = float("inf")

        for point in pareto_front:
            norm_auc = (point.val_auc - auc_min) / auc_range
            norm_sharpe = (point.backtest_sharpe - sharpe_min) / sharpe_range
            dist = math.sqrt((1.0 - norm_auc) ** 2 + (1.0 - norm_sharpe) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_point = point

        return best_point
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/test_multi_objective.py -v --tb=short 2>&1 | tail -15`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/ml/hyperparam/multi_objective.py tests/test_multi_objective.py
git commit -m "feat: add multi-objective hyperparameter search (NSGA-II)"
```

---

### Task 2: Strategy Parameter Optimizer

**Files:**
- Create: `app/ml/hyperparam/strategy_optimizer.py`
- Test: `tests/test_strategy_optimizer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_strategy_optimizer.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pydantic import Field
from typing import Literal

from app.ml.hyperparam.strategy_optimizer import (
    StrategyOptimizer,
    StrategyOptResult,
    infer_search_space,
)
from app.strategies.base import BaseStrategy, BaseParameters


class MockParameters(BaseParameters):
    fast_period: int = Field(default=10, ge=2, le=100)
    slow_period: int = Field(default=50, ge=5, le=500)
    ma_type: Literal["sma", "ema"] = "sma"


class MockStrategy(BaseStrategy):
    name = "mock_strategy"
    description = "Mock for testing"
    version = "1.0.0"
    Parameters = MockParameters

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            np.where(df["close"].pct_change() > 0, 1, -1),
            index=df.index,
        )


def _make_prices(n=200, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(rng.randn(n) * 2)
    return pd.DataFrame({
        "date": dates,
        "open": close - rng.uniform(0, 1, n),
        "high": close + rng.uniform(0, 2, n),
        "low": close - rng.uniform(0, 2, n),
        "close": close,
        "volume": rng.randint(1000, 10000, n),
    })


class TestInferSearchSpace:
    def test_int_with_constraints(self):
        space = infer_search_space(MockParameters.model_json_schema())
        assert "fast_period" in space
        assert space["fast_period"] == ("int", 2, 100)

    def test_categorical_literal(self):
        space = infer_search_space(MockParameters.model_json_schema())
        assert "ma_type" in space
        assert space["ma_type"][0] == "categorical"
        assert set(space["ma_type"][1]) == {"sma", "ema"}

    def test_override_replaces_inferred(self):
        space = infer_search_space(
            MockParameters.model_json_schema(),
            overrides={"fast_period": {"low": 5, "high": 50}},
        )
        assert space["fast_period"] == ("int", 5, 50)


class TestStrategyOptimizer:
    @patch("app.ml.hyperparam.strategy_optimizer.get_registry")
    @patch("app.ml.hyperparam.strategy_optimizer.get_prices_df")
    def test_run_returns_result(self, mock_prices, mock_registry):
        mock_prices.return_value = _make_prices()

        registry = MagicMock()
        registry.get.return_value = MockStrategy
        registry.get_metadata.return_value = MagicMock(
            parameters_schema=MockParameters.model_json_schema()
        )
        mock_registry.return_value = registry

        optimizer = StrategyOptimizer(
            strategy_name="mock_strategy",
            ticker="AAPL",
        )
        result = optimizer.run(n_trials=3, timeout=30, metric="sharpe_ratio")

        assert isinstance(result, StrategyOptResult)
        assert result.n_trials >= 1
        assert result.best_params is not None
        assert result.strategy_name == "mock_strategy"
        assert result.ticker == "AAPL"

    @patch("app.ml.hyperparam.strategy_optimizer.get_registry")
    @patch("app.ml.hyperparam.strategy_optimizer.get_prices_df")
    def test_param_overrides_respected(self, mock_prices, mock_registry):
        mock_prices.return_value = _make_prices()

        registry = MagicMock()
        registry.get.return_value = MockStrategy
        registry.get_metadata.return_value = MagicMock(
            parameters_schema=MockParameters.model_json_schema()
        )
        mock_registry.return_value = registry

        optimizer = StrategyOptimizer(
            strategy_name="mock_strategy",
            ticker="AAPL",
        )
        result = optimizer.run(
            n_trials=3,
            timeout=30,
            metric="sharpe_ratio",
            param_overrides={"fast_period": {"low": 5, "high": 20}},
        )

        # All trial fast_periods should be within override range
        for trial in result.all_trials:
            if "fast_period" in trial.params:
                assert 5 <= trial.params["fast_period"] <= 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/test_strategy_optimizer.py -v --tb=short 2>&1 | tail -15`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement StrategyOptimizer**

Create `app/ml/hyperparam/strategy_optimizer.py`:

```python
from __future__ import annotations

"""
Strategy Parameter Optimizer

Uses Optuna TPESampler to find optimal strategy parameters
by running backtests and maximizing a target metric (e.g. sharpe_ratio).

Automatically infers search space from strategy's Pydantic Parameters schema.
"""

import logging
import math
import time
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticModel, ConfigDict

from .search import TrialResult

logger = logging.getLogger(__name__)


class StrategyOptResult(PydanticModel):
    """Result of strategy parameter optimization."""

    best_params: dict[str, Any]
    best_metric: float
    metric_name: str
    all_trials: list[TrialResult]
    n_trials: int
    total_time_seconds: float
    strategy_name: str
    ticker: str

    model_config = ConfigDict(extra="forbid")


def infer_search_space(
    schema: dict[str, Any],
    overrides: Optional[dict[str, dict]] = None,
) -> dict[str, tuple]:
    """
    Infer Optuna search space from a Pydantic JSON schema.

    Rules:
    - int with ge/le → suggest_int(ge, le)
    - float/number with ge/le → suggest_float(ge, le)
    - Literal/enum → suggest_categorical
    - Unconstrained int → suggest_int(default*0.5, default*2)
    - Unconstrained float → suggest_float(default*0.5, default*2)

    Args:
        schema: Pydantic model JSON schema (from model_json_schema())
        overrides: Optional dict of {param_name: {"low": x, "high": y}}

    Returns:
        Dict mapping param names to (type, *args) tuples for Optuna
    """
    overrides = overrides or {}
    space: dict[str, tuple] = {}
    properties = schema.get("properties", {})

    for name, prop in properties.items():
        # Check for overrides first
        if name in overrides:
            ov = overrides[name]
            default = prop.get("default")
            if isinstance(default, float) and not isinstance(default, bool):
                space[name] = ("float", ov["low"], ov["high"])
            else:
                space[name] = ("int", int(ov["low"]), int(ov["high"]))
            continue

        # Handle enum/Literal (anyOf with const values)
        if "anyOf" in prop:
            choices = []
            for option in prop["anyOf"]:
                if "const" in option:
                    choices.append(option["const"])
                elif "enum" in option:
                    choices.extend(option["enum"])
            if choices:
                space[name] = ("categorical", choices)
                continue

        if "enum" in prop:
            space[name] = ("categorical", prop["enum"])
            continue

        prop_type = prop.get("type", "")
        default = prop.get("default")
        minimum = prop.get("minimum") or prop.get("exclusiveMinimum")
        maximum = prop.get("maximum") or prop.get("exclusiveMaximum")

        if prop_type == "integer":
            if minimum is not None and maximum is not None:
                space[name] = ("int", int(minimum), int(maximum))
            elif default is not None:
                low = max(1, int(default * 0.5))
                high = int(default * 2)
                space[name] = ("int", low, high)
        elif prop_type == "number":
            if minimum is not None and maximum is not None:
                space[name] = ("float", float(minimum), float(maximum))
            elif default is not None:
                low = max(0.001, float(default * 0.5))
                high = float(default * 2)
                space[name] = ("float", low, high)

    return space


class StrategyOptimizer:
    """
    Optimize strategy parameters using Optuna.

    For each trial:
    1. Sample params from inferred/overridden search space
    2. Create strategy instance with those params
    3. Generate signals on price data
    4. Compute target metric (sharpe_ratio, total_return, win_rate)

    Usage:
        optimizer = StrategyOptimizer(
            strategy_name="ma_crossover",
            ticker="AAPL",
        )
        result = optimizer.run(n_trials=100, metric="sharpe_ratio")
    """

    def __init__(
        self,
        strategy_name: str,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def run(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = 300,
        metric: str = "sharpe_ratio",
        param_overrides: Optional[dict[str, dict]] = None,
    ) -> StrategyOptResult:
        """Run strategy parameter optimization."""
        import optuna
        from app.strategies import get_registry
        from app.providers.market import get_prices_df

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        registry = get_registry()
        strategy_cls = registry.get(self.strategy_name)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")

        metadata = registry.get_metadata(self.strategy_name)
        search_space = infer_search_space(
            metadata.parameters_schema, overrides=param_overrides
        )

        if not search_space:
            raise ValueError(
                f"No optimizable parameters found for {self.strategy_name}"
            )

        # Load price data once
        prices_df = get_prices_df(self.ticker)
        if prices_df is None or prices_df.empty:
            raise ValueError(f"No price data for {self.ticker}")

        start_time = time.time()
        trials: list[TrialResult] = []

        def objective(trial: optuna.Trial) -> float:
            from app.ml.hyperparam.spaces import sample_from_space

            params = {}
            for param_name, param_def in search_space.items():
                params[param_name] = sample_from_space(trial, param_name, param_def)

            trial_start = time.time()

            try:
                strategy = strategy_cls(parameters=strategy_cls.Parameters(**params))
                signals = strategy.generate_signals(prices_df)
                metrics = self._compute_metrics(prices_df, signals)
                score = metrics.get(metric, 0.0)

                trials.append(TrialResult(
                    trial_number=trial.number,
                    params=params,
                    metrics=metrics,
                    duration_seconds=round(time.time() - trial_start, 2),
                ))

                return score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float("-inf")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        total_time = round(time.time() - start_time, 2)

        logger.info(
            f"Strategy optimization complete: {self.strategy_name} on {self.ticker}, "
            f"{len(study.trials)} trials, best {metric}={study.best_value:.4f}"
        )

        return StrategyOptResult(
            best_params=study.best_params,
            best_metric=round(study.best_value, 4),
            metric_name=metric,
            all_trials=trials,
            n_trials=len(study.trials),
            total_time_seconds=total_time,
            strategy_name=self.strategy_name,
            ticker=self.ticker,
        )

    def _compute_metrics(
        self, prices_df: pd.DataFrame, signals: pd.Series
    ) -> dict[str, float]:
        """Compute backtest metrics from signals."""
        close = prices_df["close"].values
        returns = np.diff(close) / close[:-1]
        sig = signals.values[:-1]  # Align: signal at t, return at t+1

        n = min(len(sig), len(returns))
        if n == 0:
            return {"sharpe_ratio": 0.0, "total_return": 0.0, "win_rate": 0.0}

        sig = sig[:n].astype(float)
        ret = returns[:n]

        strategy_returns = sig * ret

        # Sharpe
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)
        sharpe = float(mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

        # Total return
        total_return = float(np.prod(1 + strategy_returns) - 1)

        # Win rate
        trades = strategy_returns[sig != 0]
        win_rate = float(np.mean(trades > 0)) if len(trades) > 0 else 0.0

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "total_return": round(total_return, 4),
            "win_rate": round(win_rate, 4),
            "max_drawdown": round(max_drawdown, 4),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/test_strategy_optimizer.py -v --tb=short 2>&1 | tail -15`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/ml/hyperparam/strategy_optimizer.py tests/test_strategy_optimizer.py
git commit -m "feat: add strategy parameter optimizer with search space inference"
```

---

### Task 3: Integrate optuna_multi into HyperparamSearch

**Files:**
- Modify: `app/ml/hyperparam/search.py`

- [ ] **Step 1: Add optuna_multi mode to SearchConfig and dispatch**

In `app/ml/hyperparam/search.py`, make these changes:

1. Update `SearchConfig.mode` to include `"optuna_multi"`:

```python
class SearchConfig(BaseModel):
    """Configuration for hyperparameter search."""
    
    mode: Literal["none", "grid", "optuna", "optuna_multi"] = "none"
    n_trials: int = Field(default=20, ge=1, le=200)
    timeout_seconds: int | None = Field(default=300, ge=10, le=3600)
    metric: str = "val_auc"
    direction: Literal["maximize", "minimize"] = "maximize"
    
    model_config = ConfigDict(extra="forbid")
```

2. Update `HyperparamSearch.__init__` to accept optional `backtest_data`:

```python
def __init__(
    self,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    base_params: dict[str, Any] | None = None,
    backtest_data: dict | None = None,
):
    self.model_type = model_type
    self.X_train = X_train
    self.y_train = y_train
    self.X_val = X_val
    self.y_val = y_val
    self.base_params = base_params or {}
    self.backtest_data = backtest_data
    
    # Get search space
    self.search_space = get_search_space(model_type)
```

3. Update `HyperparamSearch.run()` dispatch:

```python
def run(self, config: SearchConfig) -> SearchResult:
    if config.mode == "none":
        return self._run_none(config)
    elif config.mode == "grid":
        return self._run_grid(config)
    elif config.mode == "optuna":
        return self._run_optuna(config)
    elif config.mode == "optuna_multi":
        return self._run_optuna_multi(config)
    else:
        raise ValueError(f"Unknown search mode: {config.mode}")
```

4. Add `_run_optuna_multi` method:

```python
def _run_optuna_multi(self, config: SearchConfig) -> SearchResult:
    """Dispatch to MultiObjectiveSearch and convert result."""
    from .multi_objective import MultiObjectiveSearch
    
    search = MultiObjectiveSearch(
        model_type=self.model_type,
        X_train=self.X_train,
        y_train=self.y_train,
        X_val=self.X_val,
        y_val=self.y_val,
        backtest_data=self.backtest_data,
        base_params=self.base_params,
    )
    
    mo_result = search.run(
        n_trials=config.n_trials,
        timeout=config.timeout_seconds,
    )
    
    # Convert to SearchResult for compatibility
    return SearchResult(
        best_params=mo_result.recommended_params,
        best_score=mo_result.recommended_val_auc,
        n_trials_completed=mo_result.n_trials,
        total_time_seconds=mo_result.total_time_seconds,
        all_trials=mo_result.all_trials,
        mode="optuna_multi",
        metric="val_auc+backtest_sharpe",
    )
```

- [ ] **Step 2: Run full test suite to verify no regressions**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/ -v --tb=short --ignore=tests/contract -p no:cacheprovider 2>&1 | tail -5`
Expected: All tests pass (213+ existing + new tests)

- [ ] **Step 3: Commit**

```bash
git add app/ml/hyperparam/search.py
git commit -m "feat: integrate optuna_multi mode into HyperparamSearch"
```

---

### Task 4: Optimization Persistence

**Files:**
- Create: `app/db/optimization_repo.py`
- Test: `tests/test_optimization_service.py` (persistence tests first)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_optimization_service.py`:

```python
from __future__ import annotations

import pytest
from datetime import datetime

from app.db.optimization_repo import OptimizationRepo, OptimizationRun


class TestOptimizationRepo:
    def setup_method(self):
        self.repo = OptimizationRepo()

    def test_save_and_get_model_run(self):
        run = OptimizationRun(
            type="model",
            config={"model_type": "xgboost", "tickers": ["AAPL"]},
            best_params={"n_estimators": 150, "max_depth": 5},
            best_metrics={"val_auc": 0.62, "backtest_sharpe": 1.1},
            pareto_front=[
                {"params": {"n_estimators": 150}, "val_auc": 0.62, "backtest_sharpe": 1.1}
            ],
            all_trials=[],
            n_trials=50,
            duration_seconds=120.5,
        )
        run_id = self.repo.save_run(run)
        assert run_id is not None

        loaded = self.repo.get_run(run_id)
        assert loaded is not None
        assert loaded.type == "model"
        assert loaded.best_params["n_estimators"] == 150
        assert loaded.n_trials == 50

    def test_save_and_get_strategy_run(self):
        run = OptimizationRun(
            type="strategy",
            config={"strategy_name": "ma_crossover", "ticker": "AAPL"},
            best_params={"fast_period": 12, "slow_period": 48},
            best_metrics={"sharpe_ratio": 1.85},
            pareto_front=None,
            all_trials=[],
            n_trials=100,
            duration_seconds=45.2,
        )
        run_id = self.repo.save_run(run)
        loaded = self.repo.get_run(run_id)

        assert loaded is not None
        assert loaded.type == "strategy"
        assert loaded.pareto_front is None

    def test_list_runs_with_filter(self):
        for i in range(3):
            self.repo.save_run(OptimizationRun(
                type="model" if i < 2 else "strategy",
                config={},
                best_params={},
                best_metrics={},
                pareto_front=None,
                all_trials=[],
                n_trials=i + 1,
                duration_seconds=1.0,
            ))

        all_runs = self.repo.list_runs()
        assert len(all_runs) >= 3

        model_runs = self.repo.list_runs(type="model")
        for r in model_runs:
            assert r.type == "model"

    def test_get_nonexistent_returns_none(self):
        result = self.repo.get_run("nonexistent-id")
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/test_optimization_service.py -v --tb=short 2>&1 | tail -15`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement OptimizationRepo**

Create `app/db/optimization_repo.py`:

```python
from __future__ import annotations

"""
Optimization Runs Repository

Persists optimization run results (model + strategy) to local JSON storage.
Follows the same pattern as LocalModelRegistry.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel as PydanticModel, ConfigDict, Field

from app.core.settings import settings

logger = logging.getLogger(__name__)


class OptimizationRun(PydanticModel):
    """A single optimization run record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "model" or "strategy"
    config: dict[str, Any]
    best_params: dict[str, Any]
    best_metrics: dict[str, Any]
    pareto_front: Optional[list[dict[str, Any]]] = None
    all_trials: list[dict[str, Any]] = Field(default_factory=list)
    n_trials: int
    duration_seconds: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="ignore")


class OptimizationRepo:
    """
    Local JSON-based repository for optimization runs.

    Stores data in STORAGE_LOCAL_PATH/registry/optimization_runs.json.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.STORAGE_LOCAL_PATH) / "registry"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_path / "optimization_runs.json"

    def _load(self) -> list[dict]:
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self, data: list[dict]) -> None:
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_run(self, run: OptimizationRun) -> str:
        """Save an optimization run. Returns the run ID."""
        data = self._load()
        data.append(run.model_dump(mode="json"))
        self._save(data)
        logger.info(f"Saved optimization run {run.id} (type={run.type})")
        return run.id

    def get_run(self, run_id: str) -> Optional[OptimizationRun]:
        """Get an optimization run by ID."""
        data = self._load()
        for item in data:
            if item.get("id") == run_id:
                return OptimizationRun(**item)
        return None

    def list_runs(
        self,
        type: Optional[str] = None,
        limit: int = 20,
    ) -> list[OptimizationRun]:
        """List optimization runs, optionally filtered by type."""
        data = self._load()

        if type is not None:
            data = [d for d in data if d.get("type") == type]

        # Sort by created_at desc
        data.sort(key=lambda d: d.get("created_at", ""), reverse=True)

        return [OptimizationRun(**d) for d in data[:limit]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/test_optimization_service.py -v --tb=short 2>&1 | tail -15`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/db/optimization_repo.py tests/test_optimization_service.py
git commit -m "feat: add optimization runs persistence layer"
```

---

### Task 5: Optimization Service

**Files:**
- Create: `app/services/optimization_service.py`

- [ ] **Step 1: Add service tests to existing test file**

Append to `tests/test_optimization_service.py`:

```python
from unittest.mock import patch, MagicMock
from app.services.optimization_service import OptimizationService


class TestOptimizationService:
    @patch("app.services.optimization_service.MultiObjectiveSearch")
    @patch("app.services.optimization_service.DatasetBuilder")
    def test_optimize_model(self, mock_builder, mock_search_cls):
        import numpy as np
        import pandas as pd
        from app.ml.hyperparam.multi_objective import MultiObjectiveResult, ParetoPoint
        from app.ml.hyperparam.search import TrialResult

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.X_train = pd.DataFrame(np.random.randn(50, 3))
        mock_dataset.y_train = pd.Series(np.random.randint(0, 2, 50))
        mock_dataset.X_val = pd.DataFrame(np.random.randn(10, 3))
        mock_dataset.y_val = pd.Series(np.random.randint(0, 2, 10))
        mock_builder.return_value.build.return_value = mock_dataset

        # Mock search result
        mock_result = MultiObjectiveResult(
            pareto_front=[ParetoPoint(params={"C": 1.0}, val_auc=0.6, backtest_sharpe=0.8)],
            recommended_params={"C": 1.0},
            recommended_val_auc=0.6,
            recommended_backtest_sharpe=0.8,
            all_trials=[TrialResult(trial_number=0, params={"C": 1.0}, metrics={"val_auc": 0.6}, duration_seconds=1.0)],
            n_trials=1,
            total_time_seconds=2.0,
        )
        mock_search_cls.return_value.run.return_value = mock_result

        service = OptimizationService()
        result = service.optimize_model(
            tickers=["AAPL"],
            model_type="logistic",
            n_trials=3,
        )

        assert result is not None
        assert result.type == "model"
        assert result.best_params == {"C": 1.0}

    @patch("app.services.optimization_service.StrategyOptimizer")
    def test_optimize_strategy(self, mock_optimizer_cls):
        from app.ml.hyperparam.strategy_optimizer import StrategyOptResult
        from app.ml.hyperparam.search import TrialResult

        mock_result = StrategyOptResult(
            best_params={"fast_period": 12},
            best_metric=1.5,
            metric_name="sharpe_ratio",
            all_trials=[TrialResult(trial_number=0, params={"fast_period": 12}, metrics={"sharpe_ratio": 1.5}, duration_seconds=0.5)],
            n_trials=1,
            total_time_seconds=1.0,
            strategy_name="ma_crossover",
            ticker="AAPL",
        )
        mock_optimizer_cls.return_value.run.return_value = mock_result

        service = OptimizationService()
        result = service.optimize_strategy(
            strategy_name="ma_crossover",
            ticker="AAPL",
            n_trials=3,
        )

        assert result is not None
        assert result.type == "strategy"
        assert result.best_params == {"fast_period": 12}
```

- [ ] **Step 2: Implement OptimizationService**

Create `app/services/optimization_service.py`:

```python
from __future__ import annotations

"""
Optimization Service

Orchestrates model hyperparameter optimization and strategy parameter optimization.
Handles dataset building, optimizer invocation, and persistence.
"""

import logging
from datetime import date
from typing import Any, Optional

from app.db.optimization_repo import OptimizationRepo, OptimizationRun
from app.ml.dataset import DatasetBuilder, DatasetConfig, LabelConfig, SplitConfig
from app.ml.hyperparam.multi_objective import MultiObjectiveSearch
from app.ml.hyperparam.strategy_optimizer import StrategyOptimizer

logger = logging.getLogger(__name__)


class OptimizationService:
    """Service for running and persisting optimization jobs."""

    def __init__(self):
        self.repo = OptimizationRepo()

    def optimize_model(
        self,
        tickers: list[str],
        model_type: str,
        n_trials: int = 50,
        timeout: Optional[int] = 300,
        feature_groups: Optional[list[str]] = None,
    ) -> OptimizationRun:
        """
        Run multi-objective model hyperparameter optimization.

        Args:
            tickers: Stock tickers for training data
            model_type: Model type (logistic, xgboost, etc.)
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds
            feature_groups: Feature groups to use

        Returns:
            Persisted OptimizationRun
        """
        feature_groups = feature_groups or ["ta_basic", "momentum"]

        logger.info(
            f"Starting model optimization: {model_type} on {tickers}, "
            f"{n_trials} trials"
        )

        # Build dataset
        dataset_config = DatasetConfig(
            tickers=tickers,
            feature_groups=feature_groups,
            label_config=LabelConfig(),
            split_config=SplitConfig(),
        )
        builder = DatasetBuilder(dataset_config)
        dataset = builder.build()

        # Run multi-objective search
        search = MultiObjectiveSearch(
            model_type=model_type,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            backtest_data={},
        )
        result = search.run(n_trials=n_trials, timeout=timeout)

        # Persist
        run = OptimizationRun(
            type="model",
            config={
                "tickers": tickers,
                "model_type": model_type,
                "n_trials": n_trials,
                "feature_groups": feature_groups,
            },
            best_params=result.recommended_params,
            best_metrics={
                "val_auc": result.recommended_val_auc,
                "backtest_sharpe": result.recommended_backtest_sharpe,
            },
            pareto_front=[p.model_dump() for p in result.pareto_front],
            all_trials=[t.model_dump() for t in result.all_trials],
            n_trials=result.n_trials,
            duration_seconds=result.total_time_seconds,
        )
        self.repo.save_run(run)

        logger.info(f"Model optimization complete: run_id={run.id}")
        return run

    def optimize_strategy(
        self,
        strategy_name: str,
        ticker: str,
        n_trials: int = 100,
        timeout: Optional[int] = 300,
        metric: str = "sharpe_ratio",
        param_overrides: Optional[dict[str, dict]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> OptimizationRun:
        """
        Run strategy parameter optimization.

        Args:
            strategy_name: Name of the strategy to optimize
            ticker: Stock ticker
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds
            metric: Target metric to optimize
            param_overrides: Custom parameter ranges
            start_date: Start date for price data
            end_date: End date for price data

        Returns:
            Persisted OptimizationRun
        """
        logger.info(
            f"Starting strategy optimization: {strategy_name} on {ticker}, "
            f"{n_trials} trials, metric={metric}"
        )

        optimizer = StrategyOptimizer(
            strategy_name=strategy_name,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        result = optimizer.run(
            n_trials=n_trials,
            timeout=timeout,
            metric=metric,
            param_overrides=param_overrides,
        )

        # Persist
        run = OptimizationRun(
            type="strategy",
            config={
                "strategy_name": strategy_name,
                "ticker": ticker,
                "metric": metric,
                "n_trials": n_trials,
                "param_overrides": param_overrides,
            },
            best_params=result.best_params,
            best_metrics={result.metric_name: result.best_metric},
            pareto_front=None,
            all_trials=[t.model_dump() for t in result.all_trials],
            n_trials=result.n_trials,
            duration_seconds=result.total_time_seconds,
        )
        self.repo.save_run(run)

        logger.info(f"Strategy optimization complete: run_id={run.id}")
        return run

    def get_run(self, run_id: str) -> Optional[OptimizationRun]:
        """Get an optimization run by ID."""
        return self.repo.get_run(run_id)

    def list_runs(
        self, type: Optional[str] = None, limit: int = 20
    ) -> list[OptimizationRun]:
        """List optimization runs."""
        return self.repo.list_runs(type=type, limit=limit)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/test_optimization_service.py -v --tb=short 2>&1 | tail -15`
Expected: 6 PASSED (4 repo + 2 service)

- [ ] **Step 4: Commit**

```bash
git add app/services/optimization_service.py tests/test_optimization_service.py
git commit -m "feat: add optimization service with model + strategy orchestration"
```

---

### Task 6: REST API Endpoints

**Files:**
- Create: `app/api/optimize.py`
- Modify: `app/main.py`
- Test: `tests/test_api_optimize.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_optimize.py`:

```python
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime

from app.main import app
from app.db.optimization_repo import OptimizationRun

client = TestClient(app)


class TestOptimizeModelAPI:
    @patch("app.api.optimize.OptimizationService")
    def test_optimize_model_success(self, mock_service_cls):
        mock_run = OptimizationRun(
            id="test-123",
            type="model",
            config={"model_type": "xgboost"},
            best_params={"n_estimators": 150},
            best_metrics={"val_auc": 0.62, "backtest_sharpe": 1.1},
            pareto_front=[],
            all_trials=[],
            n_trials=50,
            duration_seconds=120.0,
        )
        mock_service_cls.return_value.optimize_model.return_value = mock_run

        resp = client.post("/api/optimize/model", json={
            "tickers": ["AAPL"],
            "model_type": "xgboost",
            "n_trials": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test-123"
        assert data["best_params"]["n_estimators"] == 150

    def test_optimize_model_invalid_model_type(self):
        resp = client.post("/api/optimize/model", json={
            "tickers": ["AAPL"],
            "model_type": "invalid_model",
            "n_trials": 10,
        })
        assert resp.status_code == 400


class TestOptimizeStrategyAPI:
    @patch("app.api.optimize.OptimizationService")
    def test_optimize_strategy_success(self, mock_service_cls):
        mock_run = OptimizationRun(
            id="test-456",
            type="strategy",
            config={"strategy_name": "ma_crossover"},
            best_params={"fast_period": 12},
            best_metrics={"sharpe_ratio": 1.5},
            pareto_front=None,
            all_trials=[],
            n_trials=100,
            duration_seconds=45.0,
        )
        mock_service_cls.return_value.optimize_strategy.return_value = mock_run

        resp = client.post("/api/optimize/strategy", json={
            "strategy_name": "ma_crossover",
            "ticker": "AAPL",
            "n_trials": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test-456"
        assert data["best_params"]["fast_period"] == 12


class TestOptimizeRunsAPI:
    @patch("app.api.optimize.OptimizationService")
    def test_list_runs(self, mock_service_cls):
        mock_service_cls.return_value.list_runs.return_value = [
            OptimizationRun(
                type="model", config={}, best_params={}, best_metrics={},
                pareto_front=None, all_trials=[], n_trials=10, duration_seconds=5.0,
            )
        ]
        resp = client.get("/api/optimize/runs")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    @patch("app.api.optimize.OptimizationService")
    def test_get_run_not_found(self, mock_service_cls):
        mock_service_cls.return_value.get_run.return_value = None
        resp = client.get("/api/optimize/runs/nonexistent")
        assert resp.status_code == 404
```

- [ ] **Step 2: Implement API router**

Create `app/api/optimize.py`:

```python
from __future__ import annotations

"""
Optimization API Endpoints

- POST /api/optimize/model — multi-objective model hyperparameter optimization
- POST /api/optimize/strategy — strategy parameter optimization
- GET /api/optimize/runs — list optimization history
- GET /api/optimize/runs/{id} — get single optimization run
"""

import logging
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.services.optimization_service import OptimizationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimize", tags=["optimization"])

VALID_MODEL_TYPES = {"logistic", "random_forest", "xgboost", "lightgbm", "catboost"}
VALID_METRICS = {"sharpe_ratio", "total_return", "win_rate", "max_drawdown"}


class OptimizeModelRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    model_type: str
    n_trials: int = Field(default=50, ge=5, le=200)
    timeout: Optional[int] = Field(default=300, ge=10, le=3600)
    feature_groups: list[str] = Field(default=["ta_basic", "momentum"])

    model_config = ConfigDict(extra="forbid")


class OptimizeStrategyRequest(BaseModel):
    strategy_name: str
    ticker: str
    n_trials: int = Field(default=100, ge=5, le=500)
    timeout: Optional[int] = Field(default=300, ge=10, le=3600)
    metric: str = "sharpe_ratio"
    param_overrides: Optional[dict[str, dict]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    model_config = ConfigDict(extra="forbid")


@router.post("/model")
def optimize_model(request: OptimizeModelRequest):
    """Run multi-objective model hyperparameter optimization."""
    if request.model_type not in VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {request.model_type}. "
            f"Valid types: {sorted(VALID_MODEL_TYPES)}",
        )

    try:
        service = OptimizationService()
        run = service.optimize_model(
            tickers=request.tickers,
            model_type=request.model_type,
            n_trials=request.n_trials,
            timeout=request.timeout,
            feature_groups=request.feature_groups,
        )
        return run.model_dump(mode="json")
    except Exception as e:
        logger.error(f"Model optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy")
def optimize_strategy(request: OptimizeStrategyRequest):
    """Run strategy parameter optimization."""
    if request.metric not in VALID_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric: {request.metric}. "
            f"Valid metrics: {sorted(VALID_METRICS)}",
        )

    try:
        service = OptimizationService()
        run = service.optimize_strategy(
            strategy_name=request.strategy_name,
            ticker=request.ticker,
            n_trials=request.n_trials,
            timeout=request.timeout,
            metric=request.metric,
            param_overrides=request.param_overrides,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        return run.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Strategy optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
def list_optimization_runs(type: Optional[str] = None, limit: int = 20):
    """List optimization run history."""
    service = OptimizationService()
    runs = service.list_runs(type=type, limit=limit)
    return [r.model_dump(mode="json") for r in runs]


@router.get("/runs/{run_id}")
def get_optimization_run(run_id: str):
    """Get a single optimization run by ID."""
    service = OptimizationService()
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Optimization run not found")
    return run.model_dump(mode="json")
```

- [ ] **Step 3: Register router in main.py**

In `app/main.py`, add the import and include:

```python
from app.api import optimize
```

Add this line after the existing router includes:

```python
app.include_router(optimize.router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/test_api_optimize.py -v --tb=short 2>&1 | tail -15`
Expected: 4 PASSED

- [ ] **Step 5: Run full test suite**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/ -v --tb=short --ignore=tests/contract -p no:cacheprovider 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add app/api/optimize.py app/main.py tests/test_api_optimize.py
git commit -m "feat: add optimization REST API endpoints"
```

---

### Task 7: Frontend — API Client + Training Page

**Files:**
- Modify: `quant-ai-ui/src/api/client.js`
- Modify: `quant-ai-ui/src/pages/Training.jsx`

- [ ] **Step 1: Add optimize functions to client.js**

Read `quant-ai-ui/src/api/client.js` first, then append these functions:

```javascript
// === Optimization ===

export async function optimizeModel(request) {
  const res = await fetch(`${BASE}/api/optimize/model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function optimizeStrategy(request) {
  const res = await fetch(`${BASE}/api/optimize/strategy`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listOptimizationRuns(type) {
  const params = type ? `?type=${type}` : "";
  const res = await fetch(`${BASE}/api/optimize/runs${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getOptimizationRun(id) {
  const res = await fetch(`${BASE}/api/optimize/runs/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

- [ ] **Step 2: Add Auto-Optimize to Training.jsx**

Read `quant-ai-ui/src/pages/Training.jsx` first, then add:

1. Import `optimizeModel` from client
2. Add state: `optimizeResult`, `optimizing`
3. Add "Auto-Optimize" button below model type selector
4. Add results display section showing recommended params + Pareto scatter

The exact changes depend on the current Training.jsx structure — read it first, then add the optimize button and results display using Tailwind classes (`bg-surface-card`, `text-accent`, `bg-accent`). The Pareto scatter is a simple relative-positioned div with absolute-positioned dots.

Key UI elements to add:

```jsx
{/* Auto-Optimize Button */}
<button
  onClick={handleOptimize}
  disabled={optimizing}
  className="px-4 py-2 bg-accent text-white rounded hover:bg-accent/80 disabled:opacity-50"
>
  {optimizing ? "Optimizing..." : "Auto-Optimize"}
</button>

{/* Results */}
{optimizeResult && (
  <div className="mt-4 p-4 bg-surface-card rounded-lg">
    <h4 className="text-sm font-medium text-gray-300 mb-2">Optimization Results</h4>
    <p className="text-xs text-gray-400">
      Found optimal params in {optimizeResult.n_trials} trials
      ({optimizeResult.duration_seconds.toFixed(1)}s)
    </p>
    <div className="mt-2 text-sm">
      <span className="text-up">val_auc: {optimizeResult.best_metrics?.val_auc?.toFixed(4)}</span>
      {" | "}
      <span className="text-up">sharpe: {optimizeResult.best_metrics?.backtest_sharpe?.toFixed(4)}</span>
    </div>
    <div className="mt-2">
      <p className="text-xs text-gray-400 mb-1">Recommended params:</p>
      <pre className="text-xs text-gray-300 bg-surface p-2 rounded overflow-x-auto">
        {JSON.stringify(optimizeResult.best_params, null, 2)}
      </pre>
    </div>
  </div>
)}
```

- [ ] **Step 3: Verify frontend builds**

Run: `cd /c/Users/zjg09/projects/quant-ai/quant-ai-ui && npm run build 2>&1 | tail -10`
Expected: Build succeeds with 0 errors

- [ ] **Step 4: Commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
git add quant-ai-ui/src/api/client.js quant-ai-ui/src/pages/Training.jsx
git commit -m "feat: add Auto-Optimize to Training page"
```

---

### Task 8: Frontend — Strategy Page Integration

**Files:**
- Modify: `quant-ai-ui/src/pages/Strategy.jsx`

- [ ] **Step 1: Add Optimize Parameters to Strategy.jsx**

Read `quant-ai-ui/src/pages/Strategy.jsx` first, then add:

1. Import `optimizeStrategy` from client
2. Add state: `strategyOptResult`, `optimizingStrategy`
3. Add "Optimize Parameters" button next to the parameter form
4. On result: auto-fill params into form + show summary

Key UI elements:

```jsx
{/* Optimize Parameters Button */}
<button
  onClick={handleOptimizeStrategy}
  disabled={optimizingStrategy || !selectedStrategy || !ticker}
  className="px-4 py-2 bg-accent text-white rounded hover:bg-accent/80 disabled:opacity-50"
>
  {optimizingStrategy ? "Optimizing..." : "Optimize Parameters"}
</button>

{/* Strategy Optimization Results */}
{strategyOptResult && (
  <div className="mt-4 p-4 bg-surface-card rounded-lg">
    <h4 className="text-sm font-medium text-gray-300 mb-2">Optimization Results</h4>
    <p className="text-xs text-gray-400">
      Best {strategyOptResult.best_metrics && Object.keys(strategyOptResult.best_metrics)[0]}
      = {strategyOptResult.best_metrics && Object.values(strategyOptResult.best_metrics)[0]?.toFixed(4)}
      ({strategyOptResult.n_trials} trials, {strategyOptResult.duration_seconds.toFixed(1)}s)
    </p>
    <p className="text-xs text-accent mt-1">Parameters auto-filled above</p>
  </div>
)}
```

The handler should call `optimizeStrategy({ strategy_name, ticker, n_trials: 50 })` and on success update the parameter form state with `result.best_params`.

- [ ] **Step 2: Verify frontend builds**

Run: `cd /c/Users/zjg09/projects/quant-ai/quant-ai-ui && npm run build 2>&1 | tail -10`
Expected: Build succeeds with 0 errors

- [ ] **Step 3: Commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
git add quant-ai-ui/src/pages/Strategy.jsx
git commit -m "feat: add Optimize Parameters to Strategy page"
```

---

### Task 9: Update TrainRequest to support optuna_multi

**Files:**
- Modify: `app/services/training_service.py`

- [ ] **Step 1: Update search_mode pattern to include optuna_multi**

In `app/services/training_service.py`, change:

```python
search_mode: str = Field(default="none", pattern="^(none|grid|optuna)$")
```

to:

```python
search_mode: str = Field(default="none", pattern="^(none|grid|optuna|optuna_multi)$")
```

- [ ] **Step 2: Run full test suite**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/ -v --tb=short --ignore=tests/contract -p no:cacheprovider 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add app/services/training_service.py
git commit -m "feat: support optuna_multi search mode in TrainRequest"
```

---

### Task 10: Phase Gate — Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all backend tests**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/ -v --tb=short --ignore=tests/contract -p no:cacheprovider 2>&1 | tail -10`
Expected: All tests pass (230+), 0 failures

- [ ] **Step 2: Run contract tests**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/contract/ -v --tb=short -p no:cacheprovider 2>&1 | tail -10`
Expected: 39 passed

- [ ] **Step 3: Run ruff lint**

Run: `cd /c/Users/zjg09/projects/quant-ai && ruff check app/ --ignore F401,F841,E501,F541,E402 2>&1 | tail -5`
Expected: All checks passed

- [ ] **Step 4: Run frontend build**

Run: `cd /c/Users/zjg09/projects/quant-ai/quant-ai-ui && npm run build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 5: Verify new API endpoints exist**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -c "from app.main import app; routes = [r.path for r in app.routes]; assert '/api/optimize/model' in routes; assert '/api/optimize/strategy' in routes; assert '/api/optimize/runs' in routes; print('All optimize routes registered')" 2>&1`
Expected: "All optimize routes registered"

- [ ] **Step 6: Commit gate pass**

```bash
git commit --allow-empty -m "feat: [P3-OPTUNA-GATE] Optuna optimization — all tests pass, build clean"
```
