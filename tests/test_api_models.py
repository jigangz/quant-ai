"""Tests for Models API."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestListModels:
    def test_empty_list(self):
        mock_reg = MagicMock()
        mock_reg.list_models.return_value = []
        with patch("app.api.models.get_model_registry", return_value=mock_reg):
            resp = client.get("/models")
            assert resp.status_code == 200
            assert resp.json()["total"] == 0


class TestListModelTypes:
    def test_happy_path(self):
        resp = client.get("/models/types")
        assert resp.status_code == 200
        types = [t["type"] for t in resp.json()["types"]]
        assert "logistic" in types
        assert "random_forest" in types


class TestGetPromotedModel:
    def test_no_promoted(self):
        mock_cache = MagicMock()
        mock_cache.get_promoted.return_value = (None, None)
        with patch("app.services.model_cache.get_model_cache", return_value=mock_cache):
            resp = client.get("/models/promoted")
            assert resp.status_code == 200
            assert resp.json()["promoted_id"] is None
