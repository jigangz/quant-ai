from __future__ import annotations

"""
Multi-Objective Hyperparameter Search

Uses Optuna's NSGAIISampler to simultaneously optimize:
- val_auc (prediction accuracy)
- backtest_sharpe (strategy profitability)

Returns a Pareto front and a recommended balanced point.
"""

import logging
import math
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticModel, ConfigDict
from sklearn.metrics import roc_auc_score

from app.ml.models import ModelFactory
from .spaces import get_search_space, sample_from_space
from .search import TrialResult

logger = logging.getLogger(__name__)


class ParetoPoint(PydanticModel):
    """A single point on the Pareto front."""

    params: dict[str, Any]
    val_auc: float
    backtest_sharpe: float

    model_config = ConfigDict(extra="forbid")


class MultiObjectiveResult(PydanticModel):
    """Result of multi-objective optimization."""

    pareto_front: list[ParetoPoint]
    recommended_params: dict[str, Any]
    recommended_val_auc: float
    recommended_backtest_sharpe: float
    all_trials: list[TrialResult]
    n_trials: int
    total_time_seconds: float

    model_config = ConfigDict(extra="forbid")


class MultiObjectiveSearch:
    """
    Multi-objective hyperparameter search using Optuna NSGA-II.

    Optimizes two objectives simultaneously:
    1. val_auc — model prediction accuracy on validation set
    2. backtest_sharpe — Sharpe ratio from running a backtest with the model

    Usage:
        search = MultiObjectiveSearch(
            model_type="xgboost",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            backtest_data={"AAPL": prices_df},
        )
        result = search.run(n_trials=50, timeout=300)
    """

    def __init__(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        backtest_data: Optional[dict[str, pd.DataFrame]] = None,
        base_params: Optional[dict[str, Any]] = None,
    ):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.backtest_data = backtest_data or {}
        self.base_params = base_params or {}
        self.search_space = get_search_space(model_type)

    def run(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = 300,
    ) -> MultiObjectiveResult:
        """Run multi-objective optimization."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        start_time = time.time()
        trials: list[TrialResult] = []

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            params = {}
            for param_name, param_def in self.search_space.items():
                params[param_name] = sample_from_space(trial, param_name, param_def)

            trial_start = time.time()
            full_params = {**self.base_params, **params}

            try:
                # Train model
                model = ModelFactory.create(self.model_type, **full_params)
                model.fit(self.X_train, self.y_train)

                # Objective 1: val_auc
                val_proba = model.predict_proba(self.X_val)[:, 1]
                val_auc = roc_auc_score(self.y_val, val_proba)

                # Objective 2: backtest_sharpe
                backtest_sharpe = self._compute_sharpe(model)

                metrics = {
                    "val_auc": round(val_auc, 4),
                    "backtest_sharpe": round(backtest_sharpe, 4),
                }

                trials.append(TrialResult(
                    trial_number=trial.number,
                    params=params,
                    metrics=metrics,
                    duration_seconds=round(time.time() - trial_start, 2),
                ))

                return val_auc, backtest_sharpe

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                trials.append(TrialResult(
                    trial_number=trial.number,
                    params=params,
                    metrics={"val_auc": 0.0, "backtest_sharpe": 0.0},
                    duration_seconds=round(time.time() - trial_start, 2),
                ))
                return 0.0, 0.0

        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        # Build Pareto front
        pareto_front = []
        for trial in study.best_trials:
            pareto_front.append(ParetoPoint(
                params=trial.params,
                val_auc=trial.values[0],
                backtest_sharpe=trial.values[1],
            ))

        # Select recommended point
        if pareto_front:
            recommended = self._select_recommended(pareto_front)
        else:
            recommended = ParetoPoint(
                params=self.base_params,
                val_auc=0.0,
                backtest_sharpe=0.0,
            )

        total_time = round(time.time() - start_time, 2)

        logger.info(
            f"Multi-objective search complete: {len(study.trials)} trials, "
            f"Pareto front: {len(pareto_front)} points, "
            f"recommended: auc={recommended.val_auc:.4f}, sharpe={recommended.backtest_sharpe:.4f}"
        )

        return MultiObjectiveResult(
            pareto_front=pareto_front,
            recommended_params=recommended.params,
            recommended_val_auc=recommended.val_auc,
            recommended_backtest_sharpe=recommended.backtest_sharpe,
            all_trials=trials,
            n_trials=len(study.trials),
            total_time_seconds=total_time,
        )

    def _compute_sharpe(self, model) -> float:
        """Compute a quick Sharpe ratio from model predictions on backtest data."""
        if not self.backtest_data:
            return 0.0

        all_returns = []
        for ticker, prices_df in self.backtest_data.items():
            if prices_df is None or prices_df.empty:
                continue

            try:
                close = prices_df["close"].values
                returns = np.diff(close) / close[:-1]

                # Use val data features if available, else just use returns with random signal
                # In practice, the optimizer provides X_val-aligned signals
                val_proba = model.predict_proba(self.X_val)[:, 1]
                # Align: use as many returns as we have predictions
                n = min(len(val_proba), len(returns))
                if n == 0:
                    continue

                positions = np.where(val_proba[:n] > 0.5, 1.0, -1.0)
                strategy_returns = positions * returns[:n]
                all_returns.extend(strategy_returns.tolist())
            except Exception:
                continue

        if not all_returns:
            return 0.0

        arr = np.array(all_returns)
        mean_ret = np.mean(arr)
        std_ret = np.std(arr)
        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret * math.sqrt(252))

    def _select_recommended(self, pareto_front: list[ParetoPoint]) -> ParetoPoint:
        """Select the point closest to ideal (1.0, 1.0) after min-max normalization."""
        if len(pareto_front) == 1:
            return pareto_front[0]

        aucs = [p.val_auc for p in pareto_front]
        sharpes = [p.backtest_sharpe for p in pareto_front]

        auc_min, auc_max = min(aucs), max(aucs)
        sharpe_min, sharpe_max = min(sharpes), max(sharpes)

        auc_range = auc_max - auc_min if auc_max > auc_min else 1.0
        sharpe_range = sharpe_max - sharpe_min if sharpe_max > sharpe_min else 1.0

        best_point = pareto_front[0]
        best_dist = float("inf")

        for point in pareto_front:
            norm_auc = (point.val_auc - auc_min) / auc_range
            norm_sharpe = (point.backtest_sharpe - sharpe_min) / sharpe_range
            dist = math.sqrt((1.0 - norm_auc) ** 2 + (1.0 - norm_sharpe) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_point = point

        return best_point
