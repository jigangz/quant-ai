"""
Hyperparameter Search Module

Supports:
- none: No search, use provided params
- grid: Grid search over param combinations
- optuna: Bayesian optimization with Optuna
"""

from .search import HyperparamSearch, SearchConfig, SearchResult
from .spaces import get_search_space, SEARCH_SPACES

__all__ = [
    "HyperparamSearch",
    "SearchConfig",
    "SearchResult",
    "get_search_space",
    "SEARCH_SPACES",
]
