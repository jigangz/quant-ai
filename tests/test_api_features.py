"""Tests for Features API."""
from __future__ import annotations
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestListFeatureGroups:
    def test_happy_path(self):
        resp = client.get("/features/groups")
        assert resp.status_code == 200
        data = resp.json()
        assert "groups" in data
        assert data["total_groups"] >= 1


class TestListAllFeatures:
    def test_happy_path(self):
        resp = client.get("/features/all")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1


class TestGetFeatureGroup:
    def test_existing_group(self):
        resp = client.get("/features/groups/ta_basic")
        assert resp.status_code == 200
        assert resp.json()["name"] == "ta_basic"

    def test_nonexistent_returns_404(self):
        resp = client.get("/features/groups/nonexistent_xyz")
        assert resp.status_code == 404
