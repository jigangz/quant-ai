"""Tests for Agents API."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestTechnicalAgent:
    def test_no_model_returns_success_false(self):
        mock_cache = MagicMock()
        mock_cache.get_promoted_id.return_value = None
        with patch("app.services.model_cache.get_model_cache", return_value=mock_cache):
            resp = client.post("/agents/technical", json={"ticker": "AAPL"})
            assert resp.status_code == 200
            assert resp.json()["success"] is False

    def test_missing_ticker_returns_422(self):
        resp = client.post("/agents/technical", json={})
        assert resp.status_code == 422

class TestSummaryAgent:
    def test_no_model_returns_success_false(self):
        mock_cache = MagicMock()
        mock_cache.get_promoted_id.return_value = None
        with patch("app.services.model_cache.get_model_cache", return_value=mock_cache):
            resp = client.post("/agents/summary", json={"tickers": ["AAPL"]})
            assert resp.status_code == 200
            assert resp.json()["success"] is False
