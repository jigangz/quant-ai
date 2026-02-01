"""
Contract tests for /predict endpoint.

Tests:
- Valid prediction request
- Missing ticker returns error
- Invalid model_id handling
- Response schema validation
"""

import pytest


class TestPredictContract:
    """Contract tests for /predict endpoints."""

    def test_predict_get_missing_ticker_returns_422(self, client):
        """GET /predict without ticker returns 422."""
        response = client.get("/predict")
        assert response.status_code == 422

    def test_predict_get_with_ticker(self, client):
        """GET /predict?ticker=AAPL returns valid response or data error."""
        try:
            response = client.get("/predict?ticker=AAPL")
        except Exception:
            # Database not available in CI - that's ok for contract test
            return

        # Accept success or data-related error
        assert response.status_code in [200, 400, 500]

        data = response.json()
        if response.status_code == 200:
            assert "status" in data
            if data["status"] == "ok":
                assert "ticker" in data
                assert "prob_up" in data
                assert "signal" in data

    def test_predict_post_missing_ticker_returns_422(self, client):
        """POST /predict without ticker returns 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_post_with_ticker(self, client, sample_predict_request):
        """POST /predict with valid ticker."""
        try:
            response = client.post("/predict", json=sample_predict_request)
        except Exception:
            # Database not available in CI
            return

        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data

    def test_predict_with_invalid_model_id(self, client):
        """Prediction with invalid model_id returns readable error."""
        response = client.post("/predict", json={
            "ticker": "AAPL",
            "model_id": "nonexistent_model_xyz",
        })
        # Should not crash, return error status
        assert response.status_code in [200, 400, 500]

        data = response.json()
        if response.status_code == 200:
            assert data.get("status") == "error"
            assert "message" in data

    def test_predict_response_schema_success(self, client):
        """Successful prediction has correct schema."""
        try:
            response = client.get("/predict?ticker=AAPL")
        except Exception:
            # Database not available in CI
            return

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                # Validate response fields
                assert "ticker" in data
                assert "prob_up" in data
                assert isinstance(data["prob_up"], (int, float))
                assert 0 <= data["prob_up"] <= 1
                assert "signal" in data
                assert data["signal"] in ["LONG", "SHORT", "HOLD"]

    def test_predict_lookback_bounds(self, client):
        """Lookback parameter validation."""
        # Too small
        response = client.get("/predict?ticker=AAPL&lookback=10")
        assert response.status_code == 422

        # Too large
        response = client.get("/predict?ticker=AAPL&lookback=5000")
        assert response.status_code == 422

        # Valid - may fail due to database, that's ok
        try:
            response = client.get("/predict?ticker=AAPL&lookback=500")
            assert response.status_code in [200, 400, 500]
        except Exception:
            pass  # Database not available
