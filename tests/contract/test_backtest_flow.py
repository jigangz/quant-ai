"""
Contract tests for /backtest endpoint.

Tests:
- Missing model_id returns error
- Invalid model_id returns readable error
- Invalid parameters return 422
- Response schema validation
"""

import pytest


class TestBacktestContract:
    """Contract tests for /backtest endpoints."""

    def test_backtest_missing_model_id_returns_422(self, client):
        """POST /backtest without model_id returns 422."""
        response = client.post("/backtest", json={
            "tickers": ["AAPL"],
        })
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_backtest_invalid_model_id_returns_error(self, client):
        """POST /backtest with invalid model_id returns readable error."""
        response = client.post("/backtest", json={
            "model_id": "nonexistent_model_xyz_12345",
        })
        # Should be 400 with readable error, not 500 crash
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_backtest_invalid_threshold_returns_422(self, client):
        """Invalid signal_threshold returns 422."""
        response = client.post("/backtest", json={
            "model_id": "test",
            "signal_threshold": 1.5,  # Invalid: > 0.9
        })
        assert response.status_code == 422

    def test_backtest_invalid_transaction_cost_returns_422(self, client):
        """Invalid transaction_cost_bps returns 422."""
        response = client.post("/backtest", json={
            "model_id": "test",
            "transaction_cost_bps": 500,  # Invalid: > 100
        })
        assert response.status_code == 422

    def test_backtest_invalid_position_size_returns_422(self, client):
        """Invalid position_size returns 422."""
        response = client.post("/backtest", json={
            "model_id": "test",
            "position_size": 2.0,  # Invalid: > 1.0
        })
        assert response.status_code == 422

    def test_backtest_extra_fields_rejected(self, client):
        """Extra unknown fields are rejected."""
        response = client.post("/backtest", json={
            "model_id": "test",
            "unknown_field": "should_fail",
        })
        assert response.status_code == 422

    def test_backtest_report_endpoint_exists(self, client):
        """POST /backtest/report endpoint exists."""
        response = client.post("/backtest/report", json={
            "model_id": "nonexistent",
        })
        # Should return error, not 404 (endpoint exists)
        assert response.status_code in [400, 422, 500]


class TestBacktestResponseSchema:
    """Tests for backtest response schema validation."""

    def test_error_response_has_detail(self, client):
        """Error responses have readable detail."""
        response = client.post("/backtest", json={
            "model_id": "nonexistent_model",
        })
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"] or "message" in data["detail"]

    def test_validation_error_lists_field(self, client):
        """Validation errors mention the problematic field."""
        response = client.post("/backtest", json={
            "model_id": "test",
            "signal_threshold": "not_a_number",
        })
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Should mention signal_threshold
        detail_str = str(data["detail"]).lower()
        assert "signal_threshold" in detail_str or "type" in detail_str
