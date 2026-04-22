"""
Tests for /agents/technical response enriched with model metadata
(V4 Pivot Phase 2 · G6).

Lets the frontend Dashboard show the "model source tag" and Sub 4 Dialog
refresh the right indicator after切模型.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.agents import TechnicalAnalysisResponse, _build_model_metadata
from app.db.model_registry import LocalModelRegistry, ModelRecord, get_model_registry
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ==========================================================================
# _build_model_metadata helper
# ==========================================================================


class TestBuildModelMetadata:
    def test_returns_empty_for_unknown_model(self):
        result = _build_model_metadata("nonexistent_xyz_12345")
        assert result == {}

    def test_returns_metadata_when_model_exists(self, monkeypatch, tmp_path):
        """Register a model locally, then verify helper pulls its metadata."""
        # Override the global registry to a fresh instance with isolated storage
        registry = LocalModelRegistry()
        storage = tmp_path / "registry"
        storage.mkdir(parents=True, exist_ok=True)
        registry.storage_path = storage
        registry.models_file = storage / "models.json"
        registry.runs_file = storage / "training_runs.json"

        record = ModelRecord(
            name="test_vol_aapl",
            model_type="xgboost",
            tickers=["AAPL"],
            feature_groups=["ta_basic"],
            label_type="volatility",
            horizon_days=10,
            version=3,
            metrics={"val_rmse": 0.1, "val_mae": 0.08},
        )
        registry.insert_model(record)

        # Monkeypatch factory to return our isolated registry
        import app.db.model_registry as registry_module
        monkeypatch.setattr(registry_module, "_registry_instance", registry)

        result = _build_model_metadata(record.id)
        assert result["model_name"] == "test_vol_aapl"
        assert result["model_type"] == "xgboost"
        assert result["model_label_type"] == "volatility"
        assert result["model_horizon_days"] == 10
        assert result["model_version"] == 3
        assert result["model_tickers"] == ["AAPL"]
        assert result["model_metrics"] == {"val_rmse": 0.1, "val_mae": 0.08}
        assert result["model_created_at"] is not None


# ==========================================================================
# TechnicalAnalysisResponse schema exposes new fields
# ==========================================================================


class TestSchema:
    def test_response_schema_has_model_metadata_fields(self):
        fields = TechnicalAnalysisResponse.model_fields
        assert "model_name" in fields
        assert "model_label_type" in fields
        assert "model_horizon_days" in fields
        assert "model_version" in fields
        assert "model_tickers" in fields
        assert "model_metrics" in fields
        assert "model_created_at" in fields

    def test_all_model_fields_default_to_none_or_empty(self):
        r = TechnicalAnalysisResponse(
            success=False, ticker="AAPL", timestamp="2026-04-22T00:00:00",
        )
        assert r.model_name is None
        assert r.model_label_type is None
        assert r.model_horizon_days is None
        assert r.model_version is None
        assert r.model_tickers == []
        assert r.model_metrics == {}
        assert r.model_created_at is None


# ==========================================================================
# Contract: endpoint includes fields (values may be null when no model)
# ==========================================================================


class TestAgentsEndpointContract:
    def test_endpoint_returns_new_fields_in_response_shape(self, client):
        """Even on error paths, the response must include the new V4 P2 fields (possibly null)."""
        r = client.post("/agents/technical", json={"ticker": "AAPL"})
        assert r.status_code == 200
        body = r.json()
        # The 6 new metadata keys must be present
        for key in (
            "model_name",
            "model_type",
            "model_label_type",
            "model_horizon_days",
            "model_version",
            "model_tickers",
            "model_metrics",
            "model_created_at",
        ):
            assert key in body, f"missing key in response: {key}"
