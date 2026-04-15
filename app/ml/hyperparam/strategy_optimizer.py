from __future__ import annotations

"""
Strategy Parameter Optimizer

Uses Optuna TPESampler to find optimal strategy parameters
by running backtests and maximizing a target metric (e.g. sharpe_ratio).

Automatically infers search space from strategy's Pydantic Parameters schema.
"""

import logging
import math
import time
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticModel, ConfigDict

from .search import TrialResult
from app.strategies import get_registry
from app.db.prices_repo import get_prices_df

logger = logging.getLogger(__name__)


class StrategyOptResult(PydanticModel):
    """Result of strategy parameter optimization."""

    best_params: dict[str, Any]
    best_metric: float
    metric_name: str
    all_trials: list[TrialResult]
    n_trials: int
    total_time_seconds: float
    strategy_name: str
    ticker: str

    model_config = ConfigDict(extra="forbid")


def infer_search_space(
    schema: dict[str, Any],
    overrides: Optional[dict[str, dict]] = None,
) -> dict[str, tuple]:
    """
    Infer Optuna search space from a Pydantic JSON schema.

    Rules:
    - int with ge/le → suggest_int(ge, le)
    - float/number with ge/le → suggest_float(ge, le)
    - Literal/enum → suggest_categorical
    - Unconstrained int → suggest_int(default*0.5, default*2)
    - Unconstrained float → suggest_float(default*0.5, default*2)

    Args:
        schema: Pydantic model JSON schema (from model_json_schema())
        overrides: Optional dict of {param_name: {"low": x, "high": y}}

    Returns:
        Dict mapping param names to (type, *args) tuples for Optuna
    """
    overrides = overrides or {}
    space: dict[str, tuple] = {}
    properties = schema.get("properties", {})

    for name, prop in properties.items():
        # Check for overrides first
        if name in overrides:
            ov = overrides[name]
            default = prop.get("default")
            if isinstance(default, float) and not isinstance(default, bool):
                space[name] = ("float", ov["low"], ov["high"])
            else:
                space[name] = ("int", int(ov["low"]), int(ov["high"]))
            continue

        # Handle enum/Literal (anyOf with const values)
        if "anyOf" in prop:
            choices = []
            for option in prop["anyOf"]:
                if "const" in option:
                    choices.append(option["const"])
                elif "enum" in option:
                    choices.extend(option["enum"])
            if choices:
                space[name] = ("categorical", choices)
                continue

        if "enum" in prop:
            space[name] = ("categorical", prop["enum"])
            continue

        prop_type = prop.get("type", "")
        default = prop.get("default")
        minimum = prop.get("minimum") or prop.get("exclusiveMinimum")
        maximum = prop.get("maximum") or prop.get("exclusiveMaximum")

        if prop_type == "integer":
            if minimum is not None and maximum is not None:
                space[name] = ("int", int(minimum), int(maximum))
            elif default is not None:
                low = max(1, int(default * 0.5))
                high = int(default * 2)
                space[name] = ("int", low, high)
        elif prop_type == "number":
            if minimum is not None and maximum is not None:
                space[name] = ("float", float(minimum), float(maximum))
            elif default is not None:
                low = max(0.001, float(default * 0.5))
                high = float(default * 2)
                space[name] = ("float", low, high)

    return space


class StrategyOptimizer:
    """
    Optimize strategy parameters using Optuna.

    For each trial:
    1. Sample params from inferred/overridden search space
    2. Create strategy instance with those params
    3. Generate signals on price data
    4. Compute target metric (sharpe_ratio, total_return, win_rate)

    Usage:
        optimizer = StrategyOptimizer(
            strategy_name="ma_crossover",
            ticker="AAPL",
        )
        result = optimizer.run(n_trials=100, metric="sharpe_ratio")
    """

    def __init__(
        self,
        strategy_name: str,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def run(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = 300,
        metric: str = "sharpe_ratio",
        param_overrides: Optional[dict[str, dict]] = None,
    ) -> StrategyOptResult:
        """Run strategy parameter optimization."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        registry = get_registry()
        strategy_cls = registry.get(self.strategy_name)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")

        metadata = registry.get_metadata(self.strategy_name)
        search_space = infer_search_space(
            metadata.parameters_schema, overrides=param_overrides
        )

        if not search_space:
            raise ValueError(
                f"No optimizable parameters found for {self.strategy_name}"
            )

        # Load price data once
        prices_df = get_prices_df(self.ticker)
        if prices_df is None or prices_df.empty:
            raise ValueError(f"No price data for {self.ticker}")

        start_time = time.time()
        trials: list[TrialResult] = []

        def objective(trial: optuna.Trial) -> float:
            from app.ml.hyperparam.spaces import sample_from_space

            params = {}
            for param_name, param_def in search_space.items():
                params[param_name] = sample_from_space(trial, param_name, param_def)

            trial_start = time.time()

            try:
                strategy = strategy_cls(parameters=strategy_cls.Parameters(**params))
                signals = strategy.generate_signals(prices_df)
                metrics = self._compute_metrics(prices_df, signals)
                score = metrics.get(metric, 0.0)

                trials.append(TrialResult(
                    trial_number=trial.number,
                    params=params,
                    metrics=metrics,
                    duration_seconds=round(time.time() - trial_start, 2),
                ))

                return score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float("-inf")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        total_time = round(time.time() - start_time, 2)

        logger.info(
            f"Strategy optimization complete: {self.strategy_name} on {self.ticker}, "
            f"{len(study.trials)} trials, best {metric}={study.best_value:.4f}"
        )

        return StrategyOptResult(
            best_params=study.best_params,
            best_metric=round(study.best_value, 4),
            metric_name=metric,
            all_trials=trials,
            n_trials=len(study.trials),
            total_time_seconds=total_time,
            strategy_name=self.strategy_name,
            ticker=self.ticker,
        )

    def _compute_metrics(
        self, prices_df: pd.DataFrame, signals: pd.Series
    ) -> dict[str, float]:
        """Compute backtest metrics from signals."""
        close = prices_df["close"].values
        returns = np.diff(close) / close[:-1]
        sig = signals.values[:-1]  # Align: signal at t, return at t+1

        n = min(len(sig), len(returns))
        if n == 0:
            return {"sharpe_ratio": 0.0, "total_return": 0.0, "win_rate": 0.0}

        sig = sig[:n].astype(float)
        ret = returns[:n]

        strategy_returns = sig * ret

        # Sharpe
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)
        sharpe = float(mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

        # Total return
        total_return = float(np.prod(1 + strategy_returns) - 1)

        # Win rate
        trades = strategy_returns[sig != 0]
        win_rate = float(np.mean(trades > 0)) if len(trades) > 0 else 0.0

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "total_return": round(total_return, 4),
            "win_rate": round(win_rate, 4),
            "max_drawdown": round(max_drawdown, 4),
        }
