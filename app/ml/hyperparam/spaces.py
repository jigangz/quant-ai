"""
Search Spaces for Hyperparameter Optimization

Defines parameter ranges for each model type.
"""

from typing import Any

# Search space definitions
# Format: {param_name: (type, *args)}
# Types: int, float, categorical, log_float, log_int

SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "logistic": {
        "C": ("log_float", 0.001, 10.0),
        "max_iter": ("int", 500, 2000),
        "class_weight": ("categorical", ["balanced", None]),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 3, 15),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "class_weight": ("categorical", ["balanced", "balanced_subsample", None]),
    },
    "xgboost": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("log_float", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 0.01, 0.3),
        "num_leaves": ("int", 15, 127),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "catboost": {
        "iterations": ("int", 50, 300),
        "depth": ("int", 4, 10),
        "learning_rate": ("log_float", 0.01, 0.3),
        "l2_leaf_reg": ("log_float", 1.0, 10.0),
    },
}


def get_search_space(model_type: str) -> dict[str, tuple]:
    """
    Get the search space for a model type.
    
    Args:
        model_type: One of logistic, random_forest, xgboost, lightgbm, catboost
        
    Returns:
        Dict mapping param names to (type, *args) tuples
    """
    if model_type not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for: {model_type}")
    return SEARCH_SPACES[model_type]


def sample_from_space(trial, param_name: str, param_def: tuple) -> Any:
    """
    Sample a parameter value using an Optuna trial.
    
    Args:
        trial: Optuna trial object
        param_name: Name of the parameter
        param_def: (type, *args) tuple
        
    Returns:
        Sampled value
    """
    param_type = param_def[0]
    args = param_def[1:]
    
    if param_type == "int":
        return trial.suggest_int(param_name, args[0], args[1])
    elif param_type == "float":
        return trial.suggest_float(param_name, args[0], args[1])
    elif param_type == "log_float":
        return trial.suggest_float(param_name, args[0], args[1], log=True)
    elif param_type == "log_int":
        return trial.suggest_int(param_name, args[0], args[1], log=True)
    elif param_type == "categorical":
        return trial.suggest_categorical(param_name, args[0])
    else:
        raise ValueError(f"Unknown param type: {param_type}")


def generate_grid(space: dict[str, tuple], max_combinations: int = 100) -> list[dict]:
    """
    Generate grid of parameter combinations.
    
    For continuous params, samples a few discrete values.
    Limits total combinations to max_combinations.
    """
    import itertools
    
    param_values = {}
    
    for param_name, param_def in space.items():
        param_type = param_def[0]
        args = param_def[1:]
        
        if param_type == "int":
            # Sample 3-5 values
            low, high = args
            step = max(1, (high - low) // 4)
            param_values[param_name] = list(range(low, high + 1, step))[:5]
        elif param_type in ("float", "log_float"):
            # Sample 3 values
            low, high = args
            if param_type == "log_float":
                import numpy as np
                param_values[param_name] = list(np.geomspace(low, high, 3))
            else:
                param_values[param_name] = [low, (low + high) / 2, high]
        elif param_type == "categorical":
            param_values[param_name] = list(args[0])
    
    # Generate combinations
    keys = list(param_values.keys())
    values = [param_values[k] for k in keys]
    
    combinations = list(itertools.product(*values))[:max_combinations]
    
    return [dict(zip(keys, combo)) for combo in combinations]
