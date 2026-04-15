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
