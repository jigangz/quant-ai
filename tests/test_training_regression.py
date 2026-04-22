"""
End-to-end regression training tests (V4 Pivot P1 Day 10-11).

Verifies:
- Task dispatch: label_type='volatility' or 'return' → regression model
- ModelFactory passes task through to base models
- TrainingService uses regression metrics (MAE/RMSE/R2) instead of accuracy/auc
- ensemble with non-direction label_type raises a clear error
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from app.ml.models.factory import ModelFactory
from app.ml.models.sklearn_models import LogisticModel, RandomForestModel
from app.backtest.metrics import calculate_regression_metrics
from app.services.training_service import (
    LABEL_TYPE_TO_TASK,
    TrainRequest,
    TrainingService,
    _task_for_label_type,
)


# ==========================================================================
# label_type → task mapping
# ==========================================================================


class TestTaskDispatch:
    def test_direction_maps_to_classification(self):
        assert _task_for_label_type("direction") == "classification"

    def test_return_maps_to_regression(self):
        assert _task_for_label_type("return") == "regression"

    def test_volatility_maps_to_regression(self):
        assert _task_for_label_type("volatility") == "regression"

    def test_unknown_defaults_to_classification(self):
        """Safety: unknown label types default to classification (backward compat)."""
        assert _task_for_label_type("unknown_type") == "classification"

    def test_mapping_table_complete(self):
        """All 4 V4 label types are in the map."""
        for t in ("direction", "return", "volatility", "meta_label"):
            assert t in LABEL_TYPE_TO_TASK


# ==========================================================================
# Model Factory task dispatch
# ==========================================================================


class TestModelTaskDispatch:
    def test_logistic_classification_default(self):
        m = ModelFactory.create("logistic")
        assert m.task == "classification"

    def test_logistic_regression_task(self):
        m = ModelFactory.create("logistic", task="regression")
        assert m.task == "regression"
        # Underlying estimator should be Ridge (regression)
        clf_step = m.model.named_steps["clf"]
        from sklearn.linear_model import Ridge
        assert isinstance(clf_step, Ridge)

    def test_random_forest_regression_task(self):
        m = ModelFactory.create("random_forest", task="regression")
        assert m.task == "regression"
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(m.model.named_steps["clf"], RandomForestRegressor)

    def test_predict_proba_raises_for_regression(self):
        """Calling predict_proba on a regression model raises NotImplementedError."""
        import pytest
        m = ModelFactory.create("logistic", task="regression")
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 3)), columns=["f0", "f1", "f2"])
        y = pd.Series(rng.normal(size=20))
        m.fit(X, y)
        with pytest.raises(NotImplementedError, match="predict_proba"):
            m.predict_proba(X)


# ==========================================================================
# Regression metrics
# ==========================================================================


class TestRegressionMetrics:
    def test_perfect_prediction_gives_zero_error(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        m = calculate_regression_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["mape"] == 0.0

    def test_r2_bounded(self):
        """R^2 for any valid prediction is ≤ 1."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.15, 0.18, 0.32, 0.39, 0.55])
        m = calculate_regression_metrics(y_true, y_pred)
        assert m["r2"] is not None
        assert m["r2"] <= 1.0

    def test_qlike_positive_values_only(self):
        """QLIKE undefined when y_true or y_pred <= 0."""
        y_true = np.array([0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.18, 0.32, 0.38, 0.52])
        m = calculate_regression_metrics(y_true, y_pred)
        assert m["qlike"] is not None
        assert m["qlike"] >= 0  # QLIKE >= 0 by construction; min at perfect prediction

    def test_qlike_zero_when_perfect(self):
        y = np.array([0.2, 0.3, 0.4])
        m = calculate_regression_metrics(y, y)
        # QLIKE = (1) - log(1) - 1 = 0
        assert abs(m["qlike"]) < 1e-6

    def test_handles_nan_values(self):
        y_true = np.array([0.1, np.nan, 0.3, 0.4])
        y_pred = np.array([0.12, 0.2, np.nan, 0.38])
        m = calculate_regression_metrics(y_true, y_pred)
        # Only 2 valid pairs remain (indices 0 and 3); MAE computed on those
        assert m["mae"] is not None


# ==========================================================================
# TrainingService end-to-end with regression target
# ==========================================================================


def _make_mock_regression_dataset(seed: int = 0, n: int = 100, n_features: int = 5):
    """Synthetic dataset: y = linear combo of features + small noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    true_weights = rng.normal(size=n_features) * 0.1
    y = pd.Series(X.values @ true_weights + rng.normal(scale=0.01, size=n))

    n_train, n_val = 70, 15
    mock_dataset = MagicMock()
    mock_dataset.X_train = X.iloc[:n_train]
    mock_dataset.y_train = y.iloc[:n_train]
    mock_dataset.X_val = X.iloc[n_train : n_train + n_val]
    mock_dataset.y_val = y.iloc[n_train : n_train + n_val]
    mock_dataset.X_test = X.iloc[n_train + n_val :]
    mock_dataset.y_test = y.iloc[n_train + n_val :]

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
    mock_dataset.metadata = mock_meta
    return mock_dataset


class TestTrainingServiceRegression:
    def test_logistic_regression_training_e2e(self, tmp_path):
        """End-to-end: label_type=volatility + logistic → Ridge regressor, returns regression metrics."""
        mock_dataset = _make_mock_regression_dataset()

        with patch("app.services.training_service.DatasetBuilder") as MockBuilder:
            MockBuilder.return_value.build.return_value = mock_dataset

            service = TrainingService(artifacts_path=str(tmp_path / "artifacts"))
            result = service.train(
                TrainRequest(
                    tickers=["AAPL"],
                    model_type="logistic",
                    label_type="volatility",
                    save_model=False,
                )
            )

        assert result.success is True, f"Training failed: {result.error}"
        assert result.model_type == "logistic"
        # Regression metrics present
        assert any(k.endswith("_mae") for k in result.metrics)
        assert any(k.endswith("_rmse") for k in result.metrics)
        # Classification metrics absent
        assert not any(k.endswith("_accuracy") for k in result.metrics)
        assert not any(k.endswith("_auc") for k in result.metrics)

    def test_random_forest_regression_training_e2e(self, tmp_path):
        mock_dataset = _make_mock_regression_dataset()
        with patch("app.services.training_service.DatasetBuilder") as MockBuilder:
            MockBuilder.return_value.build.return_value = mock_dataset

            service = TrainingService(artifacts_path=str(tmp_path / "artifacts"))
            result = service.train(
                TrainRequest(
                    tickers=["AAPL"],
                    model_type="random_forest",
                    label_type="return",
                    save_model=False,
                )
            )

        assert result.success is True
        assert any(k.endswith("_mae") for k in result.metrics)

    def test_ensemble_regression_raises_clear_error(self, tmp_path):
        """Ensemble + regression → raise ValueError (V4 P1 limitation)."""
        mock_dataset = _make_mock_regression_dataset()
        with patch("app.services.training_service.DatasetBuilder") as MockBuilder:
            MockBuilder.return_value.build.return_value = mock_dataset

            service = TrainingService(artifacts_path=str(tmp_path / "artifacts"))
            result = service.train(
                TrainRequest(
                    tickers=["AAPL"],
                    model_type="ensemble",
                    label_type="volatility",
                    ensemble_config={
                        "mode": "voting_soft",
                        "base_models": ["logistic", "random_forest"],
                    },
                    save_model=False,
                )
            )

        # TrainingService catches and returns success=False
        assert result.success is False
        assert "ensemble" in (result.error or "").lower()

    def test_classification_still_works_backward_compat(self, tmp_path):
        """label_type=direction still produces classification metrics (no regression)."""
        rng = np.random.default_rng(1)
        n, nf = 100, 5
        X = pd.DataFrame(rng.normal(size=(n, nf)), columns=[f"f{i}" for i in range(nf)])
        y = pd.Series((X["f0"] > 0).astype(int))

        mock_dataset = MagicMock()
        mock_dataset.X_train, mock_dataset.y_train = X.iloc[:70], y.iloc[:70]
        mock_dataset.X_val, mock_dataset.y_val = X.iloc[70:85], y.iloc[70:85]
        mock_dataset.X_test, mock_dataset.y_test = X.iloc[85:], y.iloc[85:]
        meta = MagicMock()
        meta.total_samples = n
        meta.n_features = nf
        meta.feature_names = list(X.columns)
        meta.train_samples, meta.val_samples, meta.test_samples = 70, 15, 15
        meta.train_date_range = ("2024-01-01", "2024-07-10")
        meta.val_date_range = ("2024-07-11", "2024-08-25")
        meta.test_date_range = ("2024-08-26", "2024-09-15")
        mock_dataset.metadata = meta

        with patch("app.services.training_service.DatasetBuilder") as MockBuilder:
            MockBuilder.return_value.build.return_value = mock_dataset

            service = TrainingService(artifacts_path=str(tmp_path / "artifacts"))
            result = service.train(
                TrainRequest(
                    tickers=["AAPL"],
                    model_type="logistic",
                    label_type="direction",
                    save_model=False,
                )
            )

        assert result.success is True
        # Classification metrics present
        assert any(k.endswith("_accuracy") for k in result.metrics)
        # Regression metrics NOT mixed in
        assert not any(k.endswith("_mae") for k in result.metrics)
