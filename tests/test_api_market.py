"""Tests for Market Data API."""
from __future__ import annotations
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestGetMarketData:
    def test_happy_path(self):
        mock_rows = [{"ticker": "AAPL", "date": "2024-01-15", "open": 150.0,
                       "high": 155.0, "low": 149.0, "close": 153.0, "volume": 1000000}]
        with patch("app.api.market.get_prices", return_value=mock_rows):
            resp = client.get("/data/market?ticker=AAPL")
            assert resp.status_code == 200
            assert resp.json()[0]["ticker"] == "AAPL"

    def test_missing_ticker_returns_422(self):
        resp = client.get("/data/market")
        assert resp.status_code == 422


class TestListMarketProviders:
    def test_returns_providers(self):
        resp = client.get("/data/market/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "current" in data
