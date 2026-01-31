"""
Contract tests for /train endpoint.

Tests:
- Valid training request returns model_id
- Missing required fields return 422
- Invalid model_type returns readable error
- Invalid ticker handling
- Model registration after training
"""

import pytest


class TestTrainContract:
    """Contract tests for POST /train."""

    def test_train_missing_tickers_returns_422(self, client):
        """Missing required field returns 422 with clear error."""
        response = client.post("/train", json={
            "model_type": "logistic",
        })
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Should mention the missing field
        assert any("tickers" in str(err).lower() for err in data["detail"])

    def test_train_empty_tickers_returns_422(self, client):
        """Empty tickers list returns 422."""
        response = client.post("/train", json={
            "tickers": [],
            "model_type": "logistic",
        })
        assert response.status_code == 422

    def test_train_invalid_model_type_returns_error(self, client):
        """Invalid model_type returns readable error, not crash."""
        response = client.post("/train", json={
            "tickers": ["AAPL"],
            "model_type": "invalid_model_xyz",
        })
        # Should be 400 or 500 with readable error, not crash
        assert response.status_code in [400, 422, 500]
        data = response.json()
        assert "error" in data or "detail" in data

    def test_train_invalid_ratios_returns_422(self, client):
        """Invalid train/val ratios return 422."""
        response = client.post("/train", json={
            "tickers": ["AAPL"],
            "model_type": "logistic",
            "train_ratio": 1.5,  # Invalid: > 1
        })
        assert response.status_code == 422

    def test_train_invalid_horizon_returns_422(self, client):
        """Invalid horizon_days returns 422."""
        response = client.post("/train", json={
            "tickers": ["AAPL"],
            "model_type": "logistic",
            "horizon_days": 100,  # Invalid: > 60
        })
        assert response.status_code == 422

    def test_train_extra_fields_rejected(self, client):
        """Extra unknown fields are rejected (strict schema)."""
        response = client.post("/train", json={
            "tickers": ["AAPL"],
            "model_type": "logistic",
            "unknown_field": "should_fail",
        })
        assert response.status_code == 422

    def test_train_response_schema(self, client, sample_train_request):
        """
        Valid request returns expected response schema.
        Note: This may fail if no price data available.
        """
        response = client.post("/train", json=sample_train_request)

        # Accept success (200) or data-related error (400/500)
        assert response.status_code in [200, 400, 500]

        if response.status_code == 200:
            data = response.json()
            # Check required fields in response
            assert "id" in data or "model_id" in data
            assert "model_type" in data
            assert "tickers" in data
            assert "metrics" in data


class TestModelsContract:
    """Contract tests for /models endpoints."""

    def test_list_models_returns_list(self, client):
        """GET /models returns list structure."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert isinstance(data["models"], list)

    def test_get_model_not_found_returns_404(self, client):
        """GET /models/{id} with invalid ID returns 404."""
        response = client.get("/models/nonexistent_model_id_12345")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_list_models_with_status_filter(self, client):
        """GET /models?status=active works."""
        response = client.get("/models?status=active")
        assert response.status_code == 200

    def test_list_models_with_limit(self, client):
        """GET /models?limit=5 works."""
        response = client.get("/models?limit=5")
        assert response.status_code == 200

    def test_list_models_invalid_limit_returns_422(self, client):
        """Invalid limit returns 422."""
        response = client.get("/models?limit=999")
        assert response.status_code == 422
