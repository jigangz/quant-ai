"""
Model Factory

Creates model instances based on model type.
"""

import logging
from typing import Literal, Type

from .base import BaseModel
from .sklearn_models import LogisticModel, RandomForestModel

logger = logging.getLogger(__name__)

# Model type alias
ModelType = Literal["logistic", "random_forest", "xgboost", "lightgbm"]


class ModelFactory:
    """
    Factory for creating ML model instances.
    
    Usage:
        model = ModelFactory.create("logistic", C=0.5)
        model = ModelFactory.create("xgboost", n_estimators=200)
    """
    
    _registry: dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """Register a model class."""
        cls._registry[model_type] = model_class
        logger.debug(f"Registered model: {model_type}")
    
    @classmethod
    def create(cls, model_type: str, **params) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            **params: Model-specific parameters
        
        Returns:
            Model instance
        
        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {available}"
            )
        
        model_class = cls._registry[model_type]
        logger.info(f"Creating model: {model_type}")
        return model_class(**params)
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model types."""
        return list(cls._registry.keys())
    
    @classmethod
    def get_model_info(cls) -> list[dict]:
        """Get info about all registered models."""
        return [
            {
                "type": model_type,
                "class": model_class.__name__,
                "module": model_class.__module__,
            }
            for model_type, model_class in cls._registry.items()
        ]


# Register default models
ModelFactory.register("logistic", LogisticModel)
ModelFactory.register("random_forest", RandomForestModel)

# Try to register XGBoost
try:
    from .xgboost_model import XGBoostModel, XGBOOST_AVAILABLE
    if XGBOOST_AVAILABLE:
        ModelFactory.register("xgboost", XGBoostModel)
except ImportError:
    pass

# Try to register LightGBM
try:
    import lightgbm as lgb
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import numpy as np
    import pandas as pd
    
    class LightGBMModel(BaseModel):
        """LightGBM model."""
        
        model_type = "lightgbm"
        
        def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 6,
            learning_rate: float = 0.1,
            num_leaves: int = 31,
            **kwargs
        ):
            super().__init__(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                **kwargs
            )
            
            self.model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    random_state=42,
                    verbose=-1,
                )),
            ])
        
        def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
            self.model.fit(X, y)
            self.is_fitted = True
            return self
        
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return self.model.predict(X)
        
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return self.model.predict_proba(X)
    
    ModelFactory.register("lightgbm", LightGBMModel)
except ImportError:
    logger.debug("LightGBM not installed")


# Convenience function
def get_model(model_type: str, **params) -> BaseModel:
    """Create a model instance (convenience function)."""
    return ModelFactory.create(model_type, **params)
