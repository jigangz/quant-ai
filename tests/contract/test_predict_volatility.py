"""
Contract tests for POST /predict/volatility endpoint (V4 Pivot Phase 2).

These tests validate request schema + response shape, not model accuracy.
Actual regression training + accuracy validation comes in D10-D11 integration.
"""

from __future__ import annotations


class TestPredictVolatilityContract:
    """Contract tests for POST /predict/volatility."""

    def test_missing_ticker_returns_422(self, client):
        """Missing required 'ticker' field → 422."""
        response = client.post("/predict/volatility", json={})
        assert response.status_code == 422

    def test_invalid_horizon_returns_422(self, client):
        """horizon_days out of [1, 60] → 422."""
        response = client.post(
            "/predict/volatility",
            json={"ticker": "AAPL", "horizon_days": 100},
        )
        assert response.status_code == 422

    def test_invalid_horizon_zero_returns_422(self, client):
        response = client.post(
            "/predict/volatility",
            json={"ticker": "AAPL", "horizon_days": 0},
        )
        assert response.status_code == 422

    def test_minimal_request_succeeds_schema(self, client):
        """Minimal valid request — only ticker — returns 200 with success/error structure."""
        response = client.post("/predict/volatility", json={"ticker": "AAPL"})
        assert response.status_code == 200
        data = response.json()
        # Either success=True (if a vol model exists) or success=False with error
        assert "success" in data
        assert "ticker" in data
        assert data["ticker"] == "AAPL"
        if not data["success"]:
            assert "error" in data

    def test_no_model_available_returns_graceful_error(self, client):
        """With no trained/promoted vol model, endpoint returns graceful error (not 500)."""
        response = client.post(
            "/predict/volatility",
            json={"ticker": "XXYZ_NO_DATA", "model_id": "does_not_exist"},
        )
        # Must not 500; either 200 with success=False or 404
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is False
            assert "error" in data

    def test_horizon_pass_through_in_response(self, client):
        """When success, response includes the horizon_days passed in."""
        response = client.post(
            "/predict/volatility",
            json={"ticker": "AAPL", "horizon_days": 10},
        )
        assert response.status_code == 200
        data = response.json()
        if data.get("success") is True:
            assert data.get("horizon_days") == 10
            assert "predicted_volatility" in data
            assert "annualized" in data

    def test_extra_fields_rejected(self, client):
        """Pydantic strict schema rejects unknown fields is not required;
        ensure it at least doesn't crash on extra."""
        # Pydantic default is ignore-extras; sanity check 200 OK path
        response = client.post(
            "/predict/volatility",
            json={"ticker": "AAPL", "unknown_field": "whatever"},
        )
        assert response.status_code in (200, 422)
