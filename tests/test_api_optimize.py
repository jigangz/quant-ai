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
