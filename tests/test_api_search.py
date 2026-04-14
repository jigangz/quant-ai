"""Tests for Search API."""
from __future__ import annotations
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestSearchEndpoint:
    def test_happy_path(self):
        with patch("app.api.search.search", return_value={
            "status": "ok", "query": "rsi", "results": []
        }):
            resp = client.get("/search?q=rsi")
            assert resp.status_code == 200
            assert resp.json()["query"] == "rsi"

    def test_missing_query_returns_422(self):
        resp = client.get("/search")
        assert resp.status_code == 422
