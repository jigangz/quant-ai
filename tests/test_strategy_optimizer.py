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
