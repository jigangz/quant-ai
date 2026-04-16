from __future__ import annotations

"""
Contract tests for ensemble training via the /train endpoint.

The endpoint accepts ensemble_config dict for model_type='ensemble'.
Mocks DatasetBuilder.build to avoid hitting real data services.
Uses ?async=true (sync mode) to execute training synchronously in tests.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Patch DatasetBuilder.build to return synthetic data
    from app.ml.dataset import DatasetBuilder

    class FakeDataset:
        def __init__(self):
            rng = np.random.default_rng(7)
            self.X_train = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
            self.y_train = pd.Series((self.X_train["a"] > 0).astype(int))
            self.X_val = pd.DataFrame(rng.normal(size=(15, 3)), columns=["a", "b", "c"])
            self.y_val = pd.Series((self.X_val["a"] > 0).astype(int))
            self.X_test = pd.DataFrame(rng.normal(size=(15, 3)), columns=["a", "b", "c"])
            self.y_test = pd.Series((self.X_test["a"] > 0).astype(int))

            class Meta:
                feature_names = ["a", "b", "c"]
                n_features = 3
                total_samples = 90
                train_samples = 60
                val_samples = 15
                test_samples = 15
                train_date_range = ("2020-01-01", "2020-05-30")
                val_date_range = ("2020-06-01", "2020-06-30")
                test_date_range = ("2020-07-01", "2020-07-30")

            self.metadata = Meta()

    monkeypatch.setattr(DatasetBuilder, "build", lambda self: FakeDataset())

    from app.main import app
    return TestClient(app)


def test_train_ensemble_voting_soft(client):
    # Use ?async=true to run synchronously (DatasetBuilder mock is used)
    resp = client.post("/train?async=true", json={
        "tickers": ["AAPL"],
        "model_type": "ensemble",
        "ensemble_config": {
            "mode": "voting_soft",
            "base_models": ["logistic", "random_forest"],
        },
        "save_model": False,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_type"] == "ensemble"


def test_train_ensemble_stacking_logistic(client):
    # Use ?async=true to run synchronously (DatasetBuilder mock is used)
    resp = client.post("/train?async=true", json={
        "tickers": ["AAPL"],
        "model_type": "ensemble",
        "ensemble_config": {
            "mode": "stacking_logistic",
            "base_models": ["logistic", "random_forest"],
            "cv_folds": 3,
        },
        "save_model": False,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_type"] == "ensemble"


def test_train_ensemble_rejects_missing_config(client):
    resp = client.post("/train", json={
        "tickers": ["AAPL"],
        "model_type": "ensemble",
        "save_model": False,
    })
    assert resp.status_code == 422  # validation error


def test_train_non_ensemble_rejects_config(client):
    resp = client.post("/train", json={
        "tickers": ["AAPL"],
        "model_type": "logistic",
        "ensemble_config": {
            "mode": "voting_soft",
            "base_models": ["logistic", "random_forest"],
        },
        "save_model": False,
    })
    assert resp.status_code == 422
