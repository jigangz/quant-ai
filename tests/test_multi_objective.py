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
