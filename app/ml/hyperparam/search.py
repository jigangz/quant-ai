"""
Hyperparameter Search Engine

Supports:
- none: No search
- grid: Grid search
- optuna: Bayesian optimization
"""

import logging
import time
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score

from app.ml.models import ModelFactory
from .spaces import get_search_space, sample_from_space, generate_grid

logger = logging.getLogger(__name__)


class SearchConfig(BaseModel):
    """Configuration for hyperparameter search."""
    
    mode: Literal["none", "grid", "optuna"] = "none"
    n_trials: int = Field(default=20, ge=1, le=200)
    timeout_seconds: int | None = Field(default=300, ge=10, le=3600)
    metric: str = "val_auc"
    direction: Literal["maximize", "minimize"] = "maximize"
    
    class Config:
        extra = "forbid"


class TrialResult(BaseModel):
    """Result of a single trial."""
    
    trial_number: int
    params: dict[str, Any]
    metrics: dict[str, float]
    duration_seconds: float


class SearchResult(BaseModel):
    """Result of hyperparameter search."""
    
    best_params: dict[str, Any]
    best_score: float
    n_trials_completed: int
    total_time_seconds: float
    all_trials: list[TrialResult] = []
    
    # Search config used
    mode: str
    metric: str


class HyperparamSearch:
    """
    Hyperparameter search engine.
    
    Usage:
        search = HyperparamSearch(
            model_type="xgboost",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
        )
        result = search.run(SearchConfig(mode="optuna", n_trials=50))
        best_params = result.best_params
    """
    
    def __init__(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_params: dict[str, Any] | None = None,
    ):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.base_params = base_params or {}
        
        # Get search space
        self.search_space = get_search_space(model_type)
    
    def run(self, config: SearchConfig) -> SearchResult:
        """
        Run hyperparameter search.
        
        Args:
            config: Search configuration
            
        Returns:
            SearchResult with best params and all trial results
        """
        if config.mode == "none":
            return self._run_none(config)
        elif config.mode == "grid":
            return self._run_grid(config)
        elif config.mode == "optuna":
            return self._run_optuna(config)
        else:
            raise ValueError(f"Unknown search mode: {config.mode}")
    
    def _evaluate(self, params: dict[str, Any]) -> dict[str, float]:
        """Evaluate a parameter configuration."""
        # Merge with base params
        full_params = {**self.base_params, **params}
        
        # Create and train model
        model = ModelFactory.create(self.model_type, **full_params)
        model.fit(self.X_train, self.y_train)
        
        # Get predictions
        train_proba = model.predict_proba(self.X_train)[:, 1]
        val_proba = model.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            "train_auc": roc_auc_score(self.y_train, train_proba),
            "val_auc": roc_auc_score(self.y_val, val_proba),
        }
        
        return metrics
    
    def _run_none(self, config: SearchConfig) -> SearchResult:
        """No search - just evaluate base params."""
        start_time = time.time()
        
        metrics = self._evaluate(self.base_params)
        
        return SearchResult(
            best_params=self.base_params,
            best_score=metrics.get(config.metric, 0.0),
            n_trials_completed=1,
            total_time_seconds=time.time() - start_time,
            all_trials=[
                TrialResult(
                    trial_number=0,
                    params=self.base_params,
                    metrics=metrics,
                    duration_seconds=time.time() - start_time,
                )
            ],
            mode="none",
            metric=config.metric,
        )
    
    def _run_grid(self, config: SearchConfig) -> SearchResult:
        """Grid search over parameter combinations."""
        start_time = time.time()
        
        # Generate grid
        param_grid = generate_grid(self.search_space, max_combinations=config.n_trials)
        logger.info(f"Grid search: {len(param_grid)} combinations")
        
        trials = []
        best_score = float("-inf") if config.direction == "maximize" else float("inf")
        best_params = {}
        
        for i, params in enumerate(param_grid):
            # Check timeout
            if config.timeout_seconds and (time.time() - start_time) > config.timeout_seconds:
                logger.warning(f"Grid search timeout after {i} trials")
                break
            
            trial_start = time.time()
            
            try:
                metrics = self._evaluate(params)
                score = metrics.get(config.metric, 0.0)
                
                # Update best
                is_better = (
                    score > best_score if config.direction == "maximize"
                    else score < best_score
                )
                if is_better:
                    best_score = score
                    best_params = params
                
                trials.append(TrialResult(
                    trial_number=i,
                    params=params,
                    metrics=metrics,
                    duration_seconds=time.time() - trial_start,
                ))
                
                logger.debug(f"Trial {i}: {config.metric}={score:.4f}")
                
            except Exception as e:
                logger.warning(f"Trial {i} failed: {e}")
        
        return SearchResult(
            best_params=best_params,
            best_score=best_score,
            n_trials_completed=len(trials),
            total_time_seconds=time.time() - start_time,
            all_trials=trials,
            mode="grid",
            metric=config.metric,
        )
    
    def _run_optuna(self, config: SearchConfig) -> SearchResult:
        """Optuna Bayesian optimization."""
        import optuna
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        start_time = time.time()
        trials = []
        
        def objective(trial: optuna.Trial) -> float:
            # Sample params from search space
            params = {}
            for param_name, param_def in self.search_space.items():
                params[param_name] = sample_from_space(trial, param_name, param_def)
            
            trial_start = time.time()
            
            try:
                metrics = self._evaluate(params)
                score = metrics.get(config.metric, 0.0)
                
                trials.append(TrialResult(
                    trial_number=trial.number,
                    params=params,
                    metrics=metrics,
                    duration_seconds=time.time() - trial_start,
                ))
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float("-inf") if config.direction == "maximize" else float("inf")
        
        # Create study
        study = optuna.create_study(
            direction=config.direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout_seconds,
            show_progress_bar=False,
        )
        
        logger.info(
            f"Optuna search complete: {len(study.trials)} trials, "
            f"best {config.metric}={study.best_value:.4f}"
        )
        
        return SearchResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials_completed=len(study.trials),
            total_time_seconds=time.time() - start_time,
            all_trials=trials,
            mode="optuna",
            metric=config.metric,
        )
