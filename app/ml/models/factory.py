"""
Model Factory

Creates model instances based on model type.
Supports: logistic, random_forest, xgboost, lightgbm, catboost
"""

import logging
from typing import Literal, Type

from .base import BaseModel
from .sklearn_models import LogisticModel, RandomForestModel

logger = logging.getLogger(__name__)

# Model type alias (all supported models)
ModelType = Literal["logistic", "random_forest", "xgboost", "lightgbm", "catboost"]


class ModelFactory:
    """
    Factory for creating ML model instances.

    Supports 5 model types:
    - logistic: Logistic Regression (sklearn)
    - random_forest: Random Forest (sklearn)
    - xgboost: XGBoost gradient boosting
    - lightgbm: LightGBM gradient boosting
    - catboost: CatBoost gradient boosting

    Usage:
        model = ModelFactory.create("logistic", C=0.5)
        model = ModelFactory.create("xgboost", n_estimators=200)
        model = ModelFactory.create("catboost", depth=8, use_gpu=True)
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
                f"Unknown model type: {model_type}. Available: {available}"
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
                "available": True,
            }
            for model_type, model_class in cls._registry.items()
        ]

    @classmethod
    def is_available(cls, model_type: str) -> bool:
        """Check if a model type is available."""
        return model_type in cls._registry


# ===================================
# Register Models
# ===================================

# Always available: sklearn models
ModelFactory.register("logistic", LogisticModel)
ModelFactory.register("random_forest", RandomForestModel)

# XGBoost (optional)
try:
    from .xgboost_model import XGBoostModel, XGBOOST_AVAILABLE

    if XGBOOST_AVAILABLE:
        ModelFactory.register("xgboost", XGBoostModel)
        logger.info("XGBoost registered")
except ImportError:
    logger.debug("XGBoost not installed")

# LightGBM (optional)
try:
    from .lightgbm_model import LightGBMModel, LIGHTGBM_AVAILABLE

    if LIGHTGBM_AVAILABLE:
        ModelFactory.register("lightgbm", LightGBMModel)
        logger.info("LightGBM registered")
except ImportError:
    logger.debug("LightGBM not installed")

# CatBoost (optional)
try:
    from .catboost_model import CatBoostModel, CATBOOST_AVAILABLE

    if CATBOOST_AVAILABLE:
        ModelFactory.register("catboost", CatBoostModel)
        logger.info("CatBoost registered")
except ImportError:
    logger.debug("CatBoost not installed")


# ===================================
# Convenience Functions
# ===================================

def get_model(model_type: str, **params) -> BaseModel:
    """Create a model instance (convenience function)."""
    return ModelFactory.create(model_type, **params)


def list_available_models() -> list[str]:
    """List all available model types."""
    return ModelFactory.list_models()
