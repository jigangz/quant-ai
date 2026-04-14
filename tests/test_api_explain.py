"""Tests for Explain API."""
from __future__ import annotations
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestExplainEndpoint:
    def test_happy_path(self):
        with patch("app.api.explain.explain", return_value={
            "status": "ok", "data": {"ticker": "AAPL", "top_features": []}
        }):
            resp = client.get("/explain?ticker=AAPL")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_missing_ticker_returns_422(self):
        resp = client.get("/explain")
        assert resp.status_code == 422
