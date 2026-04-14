"""Shared fixtures for contract tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.main import app
from app.db.engine import engine


@pytest.fixture(autouse=True)
def _ensure_tables():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume INTEGER,
                UNIQUE(ticker, date)
            )
        """))
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_predict_request():
    return {"ticker": "AAPL"}


@pytest.fixture
def sample_backtest_request():
    return {
        "model_id": "test_model",
        "tickers": ["AAPL"],
        "signal_threshold": 0.55,
        "transaction_cost_bps": 10,
    }


@pytest.fixture
def sample_train_request():
    return {
        "tickers": ["AAPL"],
        "model_type": "logistic",
        "feature_groups": ["ta_basic"],
        "horizon_days": 5,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "save_model": True,
    }
