"""
Shared fixtures for contract tests.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_train_request():
    """Valid training request."""
    return {
        "tickers": ["AAPL"],
        "model_type": "logistic",
        "feature_groups": ["ta_basic"],
        "horizon_days": 5,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "save_model": True,
    }


@pytest.fixture
def sample_predict_request():
    """Valid prediction request."""
    return {
        "ticker": "AAPL",
    }


@pytest.fixture
def sample_backtest_request():
    """Valid backtest request (needs real model_id)."""
    return {
        "model_id": "test_model",
        "tickers": ["AAPL"],
        "signal_threshold": 0.55,
        "transaction_cost_bps": 10,
    }
