from __future__ import annotations

"""
XGBoost Model Implementation

Provides XGBoost wrapper with optional GPU support.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .base import BaseModel

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. XGBoostModel will not be available.")


class XGBoostModel(BaseModel):
    """XGBoost wrapper — Classifier or Regressor based on task."""

    model_type = "xgboost"

    def __init__(
        self,
        task: str = "classification",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_gpu: bool = False,
        **kwargs,
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        super().__init__(
            task=task,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_gpu=use_gpu,
            **kwargs,
        )
        self.task = task

        tree_method = "hist"
        device = "cuda" if use_gpu else "cpu"

        common_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            device=device,
            random_state=42,
        )
        if task == "classification":
            clf = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", **common_kwargs,
            )
        elif task == "regression":
            clf = xgb.XGBRegressor(eval_metric="rmse", **common_kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError(
                f"predict_proba not available for task='{self.task}'."
            )
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        clf = self.model.named_steps["clf"]
        importance = clf.feature_importances_

        return dict(
            sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        )
