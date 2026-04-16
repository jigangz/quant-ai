from __future__ import annotations

"""Tests for TrainRequest ensemble_config validation and TrainingService integration."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

from app.services.training_service import TrainRequest, TrainingService


# ===========================================================================
# Validation tests
# ===========================================================================

def test_train_request_ensemble_config_required_when_model_type_ensemble():
    """ensemble_config must be provided when model_type='ensemble'."""
    with pytest.raises(ValidationError, match="ensemble_config is required"):
        TrainRequest(
            tickers=["AAPL"],
            model_type="ensemble",
            # ensemble_config intentionally omitted
        )


def test_train_request_ensemble_config_forbidden_for_non_ensemble():
    """ensemble_config must NOT be provided for other model types."""
    with pytest.raises(ValidationError, match="ensemble_config is forbidden"):
        TrainRequest(
            tickers=["AAPL"],
            model_type="logistic",
            ensemble_config={"mode": "voting_soft", "base_models": ["logistic", "random_forest"]},
        )


def test_train_request_ensemble_config_accepted_when_model_type_ensemble():
    """Valid ensemble TrainRequest is accepted."""
    req = TrainRequest(
        tickers=["AAPL"],
        model_type="ensemble",
        ensemble_config={
            "mode": "voting_soft",
            "base_models": ["logistic", "random_forest"],
        },
    )
    assert req.model_type == "ensemble"
    assert req.ensemble_config["mode"] == "voting_soft"


def test_train_request_normal_model_without_ensemble_config_accepted():
    """Normal model requests without ensemble_config work as before."""
    req = TrainRequest(
        tickers=["AAPL"],
        model_type="logistic",
    )
    assert req.model_type == "logistic"
    assert req.ensemble_config is None


# ===========================================================================
# End-to-end training test (monkeypatched DatasetBuilder)
# ===========================================================================

def _make_mock_dataset():
    """Create a minimal mock dataset for TrainingService tests."""
    rng = np.random.default_rng(0)
    n = 100
    n_features = 5
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series((X["f0"] > 0).astype(int))

    n_train, n_val = 70, 15
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]

    mock_meta = MagicMock()
    mock_meta.total_samples = n
    mock_meta.n_features = n_features
    mock_meta.feature_names = list(X.columns)
    mock_meta.train_samples = n_train
    mock_meta.val_samples = n_val
    mock_meta.test_samples = n - n_train - n_val
    mock_meta.train_date_range = ("2024-01-01", "2024-07-10")
    mock_meta.val_date_range = ("2024-07-11", "2024-08-25")
    mock_meta.test_date_range = ("2024-08-26", "2024-09-15")

    mock_dataset = MagicMock()
    mock_dataset.X_train = X_train
    mock_dataset.y_train = y_train
    mock_dataset.X_val = X_val
    mock_dataset.y_val = y_val
    mock_dataset.X_test = X_test
    mock_dataset.y_test = y_test
    mock_dataset.metadata = mock_meta
    return mock_dataset


def test_training_service_ensemble_end_to_end():
    """TrainingService.train() with model_type='ensemble' returns success."""
    mock_dataset = _make_mock_dataset()

    with patch("app.services.training_service.DatasetBuilder") as MockBuilder:
        MockBuilder.return_value.build.return_value = mock_dataset

        service = TrainingService(artifacts_path="/tmp/test_ens_training")
        result = service.train(TrainRequest(
            tickers=["AAPL"],
            model_type="ensemble",
            ensemble_config={
                "mode": "voting_soft",
                "base_models": ["logistic", "random_forest"],
            },
            save_model=False,
        ))

    assert result.success is True
    assert result.model_type == "ensemble"
    assert "val_auc" in result.metrics or "val_accuracy" in result.metrics
