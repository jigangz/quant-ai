from __future__ import annotations

"""
Optimization Service

Orchestrates model hyperparameter optimization and strategy parameter optimization.
Handles dataset building, optimizer invocation, and persistence.
"""

import logging
from datetime import date
from typing import Any, Optional

from app.db.optimization_repo import OptimizationRepo, OptimizationRun
from app.ml.dataset import DatasetBuilder, DatasetConfig, LabelConfig, SplitConfig
from app.ml.hyperparam.multi_objective import MultiObjectiveSearch
from app.ml.hyperparam.strategy_optimizer import StrategyOptimizer

logger = logging.getLogger(__name__)


class OptimizationService:
    """Service for running and persisting optimization jobs."""

    def __init__(self):
        self.repo = OptimizationRepo()

    def optimize_model(
        self,
        tickers: list[str],
        model_type: str,
        n_trials: int = 50,
        timeout: Optional[int] = 300,
        feature_groups: Optional[list[str]] = None,
    ) -> OptimizationRun:
        """
        Run multi-objective model hyperparameter optimization.

        Args:
            tickers: Stock tickers for training data
            model_type: Model type (logistic, xgboost, etc.)
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds
            feature_groups: Feature groups to use

        Returns:
            Persisted OptimizationRun
        """
        feature_groups = feature_groups or ["ta_basic", "momentum"]

        logger.info(
            f"Starting model optimization: {model_type} on {tickers}, "
            f"{n_trials} trials"
        )

        # Build dataset
        dataset_config = DatasetConfig(
            tickers=tickers,
            feature_groups=feature_groups,
            label_config=LabelConfig(),
            split_config=SplitConfig(),
        )
        builder = DatasetBuilder(dataset_config)
        dataset = builder.build()

        # Run multi-objective search
        search = MultiObjectiveSearch(
            model_type=model_type,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            backtest_data={},
        )
        result = search.run(n_trials=n_trials, timeout=timeout)

        # Persist
        run = OptimizationRun(
            type="model",
            config={
                "tickers": tickers,
                "model_type": model_type,
                "n_trials": n_trials,
                "feature_groups": feature_groups,
            },
            best_params=result.recommended_params,
            best_metrics={
                "val_auc": result.recommended_val_auc,
                "backtest_sharpe": result.recommended_backtest_sharpe,
            },
            pareto_front=[p.model_dump() for p in result.pareto_front],
            all_trials=[t.model_dump() for t in result.all_trials],
            n_trials=result.n_trials,
            duration_seconds=result.total_time_seconds,
        )
        self.repo.save_run(run)

        logger.info(f"Model optimization complete: run_id={run.id}")
        return run

    def optimize_strategy(
        self,
        strategy_name: str,
        ticker: str,
        n_trials: int = 100,
        timeout: Optional[int] = 300,
        metric: str = "sharpe_ratio",
        param_overrides: Optional[dict[str, dict]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> OptimizationRun:
        """
        Run strategy parameter optimization.

        Args:
            strategy_name: Name of the strategy to optimize
            ticker: Stock ticker
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds
            metric: Target metric to optimize
            param_overrides: Custom parameter ranges
            start_date: Start date for price data
            end_date: End date for price data

        Returns:
            Persisted OptimizationRun
        """
        logger.info(
            f"Starting strategy optimization: {strategy_name} on {ticker}, "
            f"{n_trials} trials, metric={metric}"
        )

        optimizer = StrategyOptimizer(
            strategy_name=strategy_name,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        result = optimizer.run(
            n_trials=n_trials,
            timeout=timeout,
            metric=metric,
            param_overrides=param_overrides,
        )

        # Persist
        run = OptimizationRun(
            type="strategy",
            config={
                "strategy_name": strategy_name,
                "ticker": ticker,
                "metric": metric,
                "n_trials": n_trials,
                "param_overrides": param_overrides,
            },
            best_params=result.best_params,
            best_metrics={result.metric_name: result.best_metric},
            pareto_front=None,
            all_trials=[t.model_dump() for t in result.all_trials],
            n_trials=result.n_trials,
            duration_seconds=result.total_time_seconds,
        )
        self.repo.save_run(run)

        logger.info(f"Strategy optimization complete: run_id={run.id}")
        return run

    def get_run(self, run_id: str) -> Optional[OptimizationRun]:
        """Get an optimization run by ID."""
        return self.repo.get_run(run_id)

    def list_runs(
        self, type: Optional[str] = None, limit: int = 20
    ) -> list[OptimizationRun]:
        """List optimization runs."""
        return self.repo.list_runs(type=type, limit=limit)
