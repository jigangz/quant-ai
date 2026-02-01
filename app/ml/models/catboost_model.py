"""
CatBoost Model Implementation

Provides CatBoost wrapper with imputation.
Gradient boosting with native categorical feature support.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .base import BaseModel

logger = logging.getLogger(__name__)

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not installed. CatBoostModel will not be available.")


class CatBoostModel(BaseModel):
    """
    CatBoost model with imputation.
    
    Features:
    - Native categorical feature support (no encoding needed)
    - Ordered boosting to reduce overfitting
    - Fast GPU training
    - Robust to outliers
    """

    model_type = "catboost"

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        border_count: int = 254,
        auto_class_weights: str = "Balanced",
        use_gpu: bool = False,
        **kwargs,
    ):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Run: pip install catboost")

        super().__init__(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
            auto_class_weights=auto_class_weights,
            use_gpu=use_gpu,
            **kwargs,
        )

        # Configure device
        task_type = "GPU" if use_gpu else "CPU"

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=learning_rate,
                        l2_leaf_reg=l2_leaf_reg,
                        border_count=border_count,
                        auto_class_weights=auto_class_weights,
                        task_type=task_type,
                        random_state=42,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostModel":
        """Fit the model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        clf = self.model.named_steps["clf"]
        importance = clf.get_feature_importance()

        return dict(
            sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        )
