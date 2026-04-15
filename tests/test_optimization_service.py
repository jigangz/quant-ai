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
