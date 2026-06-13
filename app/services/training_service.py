from __future__ import annotations

"""
Training Service

Encapsulates the training workflow:
1. Build dataset (DatasetBuilder)
2. Create model (ModelFactory)
3. Train model
4. Evaluate model
5. Save artifacts

Replaces scripts/train.py with a service-oriented approach.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from app.core.settings import settings
from app.ml.dataset import DatasetBuilder, DatasetConfig, LabelConfig, SplitConfig
from app.ml.models import get_model
from app.backtest.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)


# V4 Pivot: multi-task ML — map label_type to sklearn task kind.
LABEL_TYPE_TO_TASK: dict[str, str] = {
    "direction": "classification",
    "return": "regression",
    "volatility": "regression",
    "meta_label": "classification",  # P3 will decide; default binary for now.
    # V5 Phase C: own task kind so _evaluate routes to cross-sectional metrics
    # (Rank IC / precision@top_pct), not global AUC. The underlying model is a
    # plain binary classifier (created with task='classification').
    "xs_strong": "xs_strong",
}


def _task_for_label_type(label_type: str) -> str:
    """Map LabelConfig.label_type → sklearn task kind."""
    return LABEL_TYPE_TO_TASK.get(label_type, "classification")

logger = logging.getLogger(__name__)


class TrainRequest(BaseModel):
    """Request for training a model."""

    # Data
    tickers: list[str] = Field(min_length=1)
    start_date: date | None = None
    end_date: date | None = None

    # Data source (V5 Phase C): "db" reads the prices table; None → yahoo.
    market_provider: str | None = None

    # Features
    feature_groups: list[str] = Field(default=["ta_basic", "momentum"])

    # Labels
    horizon_days: int = Field(default=5, ge=1, le=60)
    label_type: str = "direction"
    # V5 Phase C · xs_strong per-date strong-group fraction (ignored otherwise).
    top_pct: float = Field(default=0.30, gt=0, lt=1)

    # Features
    feature_names: list[str] | None = None  # explicit override (V5 Phase C)

    # Model
    model_type: str = "logistic"
    model_params: dict[str, Any] = Field(default_factory=dict)

    # Hyperparameter search
    search_mode: str = Field(default="none", pattern="^(none|grid|optuna|optuna_multi)$")
    search_trials: int = Field(default=20, ge=1, le=200)
    search_timeout: int | None = Field(default=300, ge=10, le=3600)

    # Split
    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.3)

    # Ensemble (only used when model_type='ensemble')
    ensemble_config: Optional[dict] = None

    # Options
    save_model: bool = True
    model_name: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_ensemble_config(self) -> "TrainRequest":
        if self.model_type == "ensemble":
            if not self.ensemble_config:
                raise ValueError(
                    "ensemble_config is required when model_type='ensemble'"
                )
        else:
            if self.ensemble_config is not None:
                raise ValueError(
                    "ensemble_config is forbidden when model_type != 'ensemble'"
                )
        return self


class TrainResult(BaseModel):
    """Result of training."""

    # Status
    success: bool
    error: str | None = None

    # Model info
    model_id: str | None = None
    model_type: str
    model_path: str | None = None

    # Data info
    tickers: list[str]
    feature_groups: list[str]
    
    # Hyperparameter search info
    search_mode: str = "none"
    search_trials_completed: int = 0
    search_best_params: dict[str, Any] = {}
    search_time_seconds: float = 0.0
    feature_names: list[str] = []
    n_features: int = 0

    # Split info
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_date_range: tuple[str, str] | None = None
    val_date_range: tuple[str, str] | None = None
    test_date_range: tuple[str, str] | None = None

    # Metrics
    metrics: dict[str, float] = {}

    # Timing
    training_time_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainingService:
    """
    Service for training ML models.

    Usage:
        service = TrainingService()
        result = service.train(TrainRequest(
            tickers=["AAPL", "MSFT"],
            model_type="xgboost",
            feature_groups=["ta_basic", "momentum"],
        ))
    """

    def __init__(self, artifacts_path: str | None = None):
        self.artifacts_path = Path(artifacts_path or settings.STORAGE_LOCAL_PATH)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

    def train(self, request: TrainRequest) -> TrainResult:
        """
        Train a model based on the request.

        Args:
            request: Training configuration

        Returns:
            TrainResult with metrics and model info
        """
        import time

        start_time = time.time()
        search_result = None

        try:
            logger.info(f"Starting training: {request.model_type} on {request.tickers}")

            # 1. Build dataset
            dataset_config = DatasetConfig(
                tickers=request.tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                feature_groups=request.feature_groups,
                feature_names=request.feature_names,
                market_provider=request.market_provider,
                label_config=LabelConfig(
                    horizon_days=request.horizon_days,
                    label_type=request.label_type,
                    top_pct=request.top_pct,
                ),
                split_config=SplitConfig(
                    train_ratio=request.train_ratio,
                    val_ratio=request.val_ratio,
                ),
            )

            builder = DatasetBuilder(dataset_config)
            dataset = builder.build()

            logger.info(
                f"Dataset built: {dataset.metadata.total_samples} samples, "
                f"{dataset.metadata.n_features} features"
            )

            # 2. Hyperparameter search (if enabled)
            model_params = request.model_params.copy()
            
            if request.search_mode != "none":
                from app.ml.hyperparam import HyperparamSearch, SearchConfig
                
                logger.info(f"Running {request.search_mode} search ({request.search_trials} trials)")
                
                search = HyperparamSearch(
                    model_type=request.model_type,
                    X_train=dataset.X_train,
                    y_train=dataset.y_train,
                    X_val=dataset.X_val,
                    y_val=dataset.y_val,
                    base_params=request.model_params,
                )
                
                search_config = SearchConfig(
                    mode=request.search_mode,
                    n_trials=request.search_trials,
                    timeout_seconds=request.search_timeout,
                )
                
                search_result = search.run(search_config)
                model_params = {**request.model_params, **search_result.best_params}
                
                logger.info(
                    f"Search complete: best {search_config.metric}={search_result.best_score:.4f}, "
                    f"params={search_result.best_params}"
                )

            # 3. Create model with best params
            task = _task_for_label_type(request.label_type)
            # xs_strong is a binary top-pct classifier under the hood; only the
            # EVAL task differs (cross-sectional metrics). Create as classification.
            model_task = "classification" if task == "xs_strong" else task
            if request.model_type == "ensemble":
                if model_task != "classification":
                    raise ValueError(
                        f"Ensemble model not yet supported for task='{task}' "
                        f"(label_type='{request.label_type}'). V4 Pivot limitation; "
                        f"ensemble regression planned post-P1."
                    )
                model = get_model("ensemble", ensemble_config=request.ensemble_config)
            else:
                model = get_model(request.model_type, task=model_task, **model_params)

            # 4. Train
            logger.info("Training model...")
            model.fit(dataset.X_train, dataset.y_train)

            # 4. Evaluate — dispatch on task (classification / regression / xs_strong)
            groups = None
            if task == "xs_strong":
                groups = {
                    "train": dataset.groups_train,
                    "val": dataset.groups_val,
                    "test": dataset.groups_test,
                }
            metrics = self._evaluate(
                model,
                dataset.X_train,
                dataset.y_train,
                dataset.X_val,
                dataset.y_val,
                dataset.X_test,
                dataset.y_test,
                task=task,
                groups=groups,
                top_pct=request.top_pct,
            )

            logger.info(f"Metrics: {metrics}")

            # 5. Set metadata
            model.set_metadata(
                feature_names=dataset.metadata.feature_names,
                feature_groups=request.feature_groups,
                tickers=request.tickers,
                train_samples=dataset.metadata.train_samples,
                val_samples=dataset.metadata.val_samples,
                train_start_date=dataset.metadata.train_date_range[0],
                train_end_date=dataset.metadata.train_date_range[1],
                metrics=metrics,
            )

            # 6. Save model
            model_path = None
            model_id = None

            if request.save_model:
                model_id = self._generate_model_id(request)
                model_path = self.artifacts_path / model_id
                model.save(model_path)
                logger.info(f"Model saved to: {model_path}")

            training_time = time.time() - start_time

            return TrainResult(
                success=True,
                model_id=model_id,
                model_type=request.model_type,
                model_path=str(model_path) if model_path else None,
                tickers=request.tickers,
                feature_groups=request.feature_groups,
                feature_names=dataset.metadata.feature_names,
                n_features=dataset.metadata.n_features,
                train_samples=dataset.metadata.train_samples,
                val_samples=dataset.metadata.val_samples,
                test_samples=dataset.metadata.test_samples,
                train_date_range=dataset.metadata.train_date_range,
                val_date_range=dataset.metadata.val_date_range,
                test_date_range=dataset.metadata.test_date_range,
                metrics=metrics,
                training_time_seconds=training_time,
                # Search info
                search_mode=request.search_mode,
                search_trials_completed=search_result.n_trials_completed if search_result else 0,
                search_best_params=search_result.best_params if search_result else {},
                search_time_seconds=search_result.total_time_seconds if search_result else 0.0,
            )

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return TrainResult(
                success=False,
                error=str(e),
                model_type=request.model_type,
                tickers=request.tickers,
                feature_groups=request.feature_groups,
                training_time_seconds=time.time() - start_time,
            )

    def _evaluate(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        task: str = "classification",
        groups: dict | None = None,
        top_pct: float = 0.30,
    ) -> dict[str, float]:
        """Evaluate model on all splits — dispatches on task
        (classification / regression / xs_strong)."""
        splits = [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]

        # V5 Phase C: cross-sectional ranking — score each split, group by date,
        # compute Rank IC + precision@top_pct (NOT AUC). Needs the carried
        # [date, future_return] groups, aligned row-wise to X.
        if task == "xs_strong":
            from app.backtest.metrics import calculate_xs_metrics

            groups = groups or {}
            metrics = {}
            for split_name, X, _y in splits:
                if len(X) == 0:
                    continue
                g = groups.get(split_name)
                if g is None or len(g) == 0:
                    continue
                scores = model.predict_proba(X)[:, 1]
                xs = calculate_xs_metrics(
                    scores,
                    g["future_return"].to_numpy(),
                    g["date"].to_numpy(),
                    top_pct=top_pct,
                )
                for k, v in xs.items():
                    if v is not None:
                        metrics[f"{split_name}_{k}"] = v
            return metrics

        metrics = {}

        for split_name, X, y in splits:
            if len(X) == 0:
                continue

            # Drop NaN labels (regression tail-of-series rows for vol/return targets)
            import pandas as pd  # local to avoid top-level import churn
            mask = pd.Series(y).notna().values
            X_clean = X.loc[mask] if hasattr(X, "loc") else X[mask]
            y_clean = pd.Series(y).loc[mask].values if hasattr(y, "loc") else y[mask]
            if len(y_clean) == 0:
                continue

            y_pred = model.predict(X_clean)

            if task == "classification":
                y_prob = model.predict_proba(X_clean)[:, 1]
                cls_metrics = calculate_classification_metrics(y_clean, y_pred, y_prob)
                for k, v in cls_metrics.items():
                    if v is not None:
                        metrics[f"{split_name}_{k}"] = v
            elif task == "regression":
                reg_metrics = calculate_regression_metrics(y_clean, y_pred)
                for k, v in reg_metrics.items():
                    if v is not None:
                        metrics[f"{split_name}_{k}"] = v
            else:
                raise ValueError(f"Unknown task: {task}")

        return metrics

    def _generate_model_id(self, request: TrainRequest) -> str:
        """Generate a unique model ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        tickers_str = "_".join(request.tickers[:3])
        if len(request.tickers) > 3:
            tickers_str += f"_plus{len(request.tickers) - 3}"

        name = request.model_name or request.model_type
        return f"{name}_{tickers_str}_{timestamp}"
