"""
Tests for GET /models filter expansion (V4 Pivot Phase 2 · G3).

Adds `ticker` and `label_type` query params to /models list endpoint.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.db.model_registry import LocalModelRegistry, ModelRecord
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def registry_with_mixed_models(tmp_path):
    """Fresh LocalModelRegistry seeded with 4 models across label_types + tickers.

    Constructs registry then overrides its paths to tmp_path so this fixture
    is fully isolated from other tests' persisted state (the LocalModelRegistry
    reads `settings.STORAGE_LOCAL_PATH` once at __init__).
    """
    registry = LocalModelRegistry()
    storage_path = tmp_path / "registry"
    storage_path.mkdir(parents=True, exist_ok=True)
    registry.storage_path = storage_path
    registry.models_file = storage_path / "models.json"
    registry.runs_file = storage_path / "training_runs.json"

    seeds = [
        dict(
            name="dir_aapl",
            model_type="logistic",
            tickers=["AAPL"],
            feature_groups=["ta_basic"],
            label_type="direction",
        ),
        dict(
            name="vol_aapl_xgb",
            model_type="xgboost",
            tickers=["AAPL"],
            feature_groups=["ta_basic"],
            label_type="volatility",
        ),
        dict(
            name="vol_msft",
            model_type="random_forest",
            tickers=["MSFT"],
            feature_groups=["ta_basic"],
            label_type="volatility",
        ),
        dict(
            name="dir_multi",
            model_type="logistic",
            tickers=["AAPL", "MSFT", "GOOGL"],
            feature_groups=["ta_basic"],
            label_type="direction",
        ),
    ]
    for s in seeds:
        registry.insert_model(ModelRecord(**s))
    return registry


class TestLocalRegistryFilters:
    def test_no_filter_returns_all(self, registry_with_mixed_models):
        rows = registry_with_mixed_models.list_models()
        assert len(rows) == 4

    def test_label_type_volatility_filter(self, registry_with_mixed_models):
        rows = registry_with_mixed_models.list_models(label_type="volatility")
        assert len(rows) == 2
        assert all(r.label_type == "volatility" for r in rows)

    def test_label_type_direction_filter(self, registry_with_mixed_models):
        rows = registry_with_mixed_models.list_models(label_type="direction")
        assert len(rows) == 2
        assert all(r.label_type == "direction" for r in rows)

    def test_ticker_filter_single(self, registry_with_mixed_models):
        """ticker=AAPL matches 3 records (2 single-ticker + 1 multi)."""
        rows = registry_with_mixed_models.list_models(ticker="AAPL")
        assert len(rows) == 3

    def test_ticker_filter_msft(self, registry_with_mixed_models):
        """ticker=MSFT matches 2 records."""
        rows = registry_with_mixed_models.list_models(ticker="MSFT")
        assert len(rows) == 2

    def test_combined_ticker_and_label_type(self, registry_with_mixed_models):
        """ticker=AAPL + label_type=volatility -> just 1 model (vol_aapl_xgb)."""
        rows = registry_with_mixed_models.list_models(
            ticker="AAPL", label_type="volatility"
        )
        assert len(rows) == 1
        assert rows[0].name == "vol_aapl_xgb"

    def test_unknown_ticker_returns_empty(self, registry_with_mixed_models):
        rows = registry_with_mixed_models.list_models(ticker="ZZZZ")
        assert rows == []

    def test_unknown_label_type_returns_empty(self, registry_with_mixed_models):
        rows = registry_with_mixed_models.list_models(label_type="regime")
        assert rows == []


class TestModelsEndpointFilters:
    """Contract tests for GET /models?ticker=&label_type= via FastAPI TestClient."""

    def test_endpoint_accepts_label_type_query(self, client):
        r = client.get("/models?label_type=volatility")
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        assert "total" in body

    def test_endpoint_accepts_ticker_query(self, client):
        r = client.get("/models?ticker=AAPL")
        assert r.status_code == 200

    def test_endpoint_accepts_combined_filter(self, client):
        r = client.get("/models?ticker=AAPL&label_type=volatility")
        assert r.status_code == 200

    def test_endpoint_invalid_limit_rejected(self, client):
        r = client.get("/models?limit=0")
        assert r.status_code == 422
