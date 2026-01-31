"""Feature engineering module."""

from .technical import add_technical_features, FEATURE_GROUPS, ALL_FEATURES
from .registry import (
    FeatureRegistry,
    FeatureGroup,
    feature_registry,
    register_default_groups,
)

__all__ = [
    # Legacy
    "add_technical_features",
    "FEATURE_GROUPS",
    "ALL_FEATURES",
    # Registry
    "FeatureRegistry",
    "FeatureGroup",
    "feature_registry",
    "register_default_groups",
]
