from __future__ import annotations

"""
LightGBM Model Implementation

Provides LightGBM wrapper with imputation.
Fast gradient boosting framework.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .base import BaseModel

logger = logging.getLogger(__name__)

# Try to import LightGBM
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. LightGBMModel will not be available.")


class LightGBMModel(BaseModel):
    """
    LightGBM model with imputation.
    
    Features:
    - Fast training with histogram-based algorithm
    - Native categorical feature support
    - Lower memory usage than XGBoost
    """

    model_type = "lightgbm"

    def __init__(
        self,
        task: str = "classification",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        class_weight: str | None = "balanced",
        **kwargs,
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

        super().__init__(
            task=task,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            class_weight=class_weight,
            **kwargs,
        )
        self.task = task

        common_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        if task == "classification":
            clf = lgb.LGBMClassifier(class_weight=class_weight, **common_kwargs)
        elif task == "regression":
            clf = lgb.LGBMRegressor(**common_kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """Fit the model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
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
