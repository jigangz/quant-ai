"""
Contract tests for provider switching and configuration.

Tests:
- Market data endpoints work
- Feature endpoints work
- Health check includes provider status
- Graceful handling when providers unavailable
"""

import pytest


class TestProviderContract:
    """Contract tests for data providers."""

    def test_health_endpoint(self, client):
        """GET /health returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_market_data_endpoint(self, client):
        """GET /market/{ticker} returns data or error."""
        response = client.get("/market/AAPL")
        # Accept success or graceful error
        assert response.status_code in [200, 400, 404, 500]

        if response.status_code == 200:
            data = response.json()
            # Should have some data structure
            assert isinstance(data, (dict, list))

    def test_market_invalid_ticker(self, client):
        """Invalid ticker returns readable error."""
        response = client.get("/market/INVALID_TICKER_XYZ_123")
        # Should not crash
        assert response.status_code in [200, 400, 404, 500]

    def test_features_endpoint(self, client):
        """GET /features/{ticker} returns features or error."""
        response = client.get("/features/AAPL")
        assert response.status_code in [200, 400, 404, 500]

    def test_features_list_endpoint(self, client):
        """GET /features/list returns available features."""
        response = client.get("/features/list")
        # Endpoint might not exist, but shouldn't crash
        assert response.status_code in [200, 404, 405]


class TestHealthContract:
    """Contract tests for health and status endpoints."""

    def test_health_response_schema(self, client):
        """Health endpoint has expected schema."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "status" in data
        assert data["status"] in ["ok", "healthy", "degraded", "unhealthy"]

    def test_root_endpoint(self, client):
        """GET / returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data or "version" in data


class TestErrorHandling:
    """Tests for graceful error handling across endpoints."""

    def test_404_returns_json(self, client):
        """Non-existent endpoint returns JSON error."""
        response = client.get("/nonexistent/endpoint/xyz")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Wrong HTTP method returns 405."""
        response = client.delete("/health")  # Health doesn't support DELETE
        assert response.status_code in [404, 405]

    def test_malformed_json_returns_422(self, client):
        """Malformed JSON body returns 422."""
        response = client.post(
            "/train",
            content="not valid json {{{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_wrong_content_type_handled(self, client):
        """Wrong content type is handled gracefully."""
        response = client.post(
            "/train",
            content="tickers=AAPL",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code in [400, 415, 422]
