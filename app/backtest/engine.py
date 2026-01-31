"""
Backtest Engine

Runs backtests on trained models:
1. Load model from registry
2. Generate predictions on historical data
3. Simulate trading strategy
4. Calculate performance metrics
5. Compare to Buy & Hold benchmark
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.backtest.metrics import (
    calculate_classification_metrics,
    calculate_strategy_metrics,
    generate_report_markdown,
)
from app.core.settings import settings
from app.db.model_registry import get_model_registry
from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.ml.labels.returns import add_future_return_label
from app.ml.features.build import build_xy

logger = logging.getLogger(__name__)


# ===================================
# Request / Response Models
# ===================================
class BacktestRequest(BaseModel):
    """Request for running a backtest."""

    model_id: str
    tickers: list[str] | None = None  # Override model tickers
    start_date: date | None = None
    end_date: date | None = None

    # Strategy params
    signal_threshold: float = Field(default=0.55, ge=0.5, le=0.9)
    transaction_cost_bps: int = Field(default=10, ge=0, le=100)  # basis points

    # Position sizing
    position_size: float = Field(default=1.0, ge=0.1, le=1.0)  # fraction of capital

    class Config:
        extra = "forbid"


class BacktestResult(BaseModel):
    """Result of a backtest."""

    # Status
    success: bool
    error: str | None = None

    # Config echo
    model_id: str
    tickers: list[str]
    start_date: str | None = None
    end_date: str | None = None
    transaction_cost_bps: int = 10

    # Metrics
    strategy_metrics: dict[str, Any] = {}
    classification_metrics: dict[str, float] = {}

    # Report
    report_markdown: str = ""

    # Details
    n_samples: int = 0
    n_trades: int = 0
    equity_curve: list[float] = []
    benchmark_curve: list[float] = []

    class Config:
        extra = "ignore"


# ===================================
# Backtest Engine
# ===================================
class BacktestEngine:
    """
    Engine for running backtests on trained models.

    Usage:
        engine = BacktestEngine()
        result = engine.run(BacktestRequest(model_id="..."))
    """

    def __init__(self):
        self.registry = get_model_registry()
        self.model_cache: dict[str, Any] = {}

    def load_model(self, model_id: str):
        """Load model from registry."""
        if model_id in self.model_cache:
            return self.model_cache[model_id]

        record = self.registry.get_model(model_id)
        if not record:
            raise ValueError(f"Model not found: {model_id}")

        if not record.artifact_path:
            raise ValueError(f"Model has no artifact: {model_id}")

        artifact_path = Path(record.artifact_path)
        if not artifact_path.exists():
            artifact_path = Path(f"{record.artifact_path}.joblib")

        if not artifact_path.exists():
            raise ValueError(f"Model artifact not found: {artifact_path}")

        model = joblib.load(artifact_path)
        self.model_cache[model_id] = model
        return model

    def run(self, request: BacktestRequest) -> BacktestResult:
        """
        Run backtest.

        Args:
            request: Backtest configuration

        Returns:
            BacktestResult with metrics and report
        """
        try:
            logger.info(f"Starting backtest for model: {request.model_id}")

            # 1. Load model
            model = self.load_model(request.model_id)
            model_record = self.registry.get_model(request.model_id)

            # Use model's tickers if not overridden
            tickers = request.tickers or model_record.tickers
            if not tickers:
                raise ValueError("No tickers specified")

            # 2. Load and prepare data for each ticker
            all_predictions = []
            all_actuals = []
            all_probs = []
            all_returns = []
            all_dates = []

            for ticker in tickers:
                data = self._prepare_data(ticker, request.start_date, request.end_date)
                if data is None or len(data) == 0:
                    logger.warning(f"No data for {ticker}")
                    continue

                X, y, returns, dates = data

                # Generate predictions
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1]

                all_predictions.extend(y_pred)
                all_actuals.extend(y)
                all_probs.extend(y_prob)
                all_returns.extend(returns)
                all_dates.extend(dates)

            if len(all_predictions) == 0:
                return BacktestResult(
                    success=False,
                    error="No data available for backtest",
                    model_id=request.model_id,
                    tickers=tickers,
                )

            # Convert to arrays
            y_pred = np.array(all_predictions)
            y_true = np.array(all_actuals)
            y_prob = np.array(all_probs)
            returns = np.array(all_returns)

            # 3. Calculate classification metrics
            classification_metrics = calculate_classification_metrics(
                y_true, y_pred, y_prob
            )

            # 4. Simulate trading strategy
            strategy_returns, benchmark_returns = self._simulate_strategy(
                y_prob=y_prob,
                returns=returns,
                signal_threshold=request.signal_threshold,
                transaction_cost_bps=request.transaction_cost_bps,
                position_size=request.position_size,
            )

            # 5. Calculate strategy metrics
            strategy_returns_series = pd.Series(strategy_returns)
            benchmark_returns_series = pd.Series(benchmark_returns)

            strategy_metrics = calculate_strategy_metrics(
                strategy_returns_series,
                benchmark_returns_series,
            )

            # 6. Generate equity curves
            equity_curve = list((1 + strategy_returns_series).cumprod())
            benchmark_curve = list((1 + benchmark_returns_series).cumprod())

            # 7. Generate report
            config = {
                "model_id": request.model_id,
                "tickers": tickers,
                "start_date": str(request.start_date) if request.start_date else min(all_dates).strftime("%Y-%m-%d") if all_dates else "N/A",
                "end_date": str(request.end_date) if request.end_date else max(all_dates).strftime("%Y-%m-%d") if all_dates else "N/A",
                "transaction_cost_bps": request.transaction_cost_bps,
            }
            report_md = generate_report_markdown(
                strategy_metrics, classification_metrics, config
            )

            logger.info(f"Backtest complete: Sharpe={strategy_metrics.get('sharpe')}")

            return BacktestResult(
                success=True,
                model_id=request.model_id,
                tickers=tickers,
                start_date=config["start_date"],
                end_date=config["end_date"],
                transaction_cost_bps=request.transaction_cost_bps,
                strategy_metrics=strategy_metrics,
                classification_metrics=classification_metrics,
                report_markdown=report_md,
                n_samples=len(y_pred),
                n_trades=strategy_metrics.get("n_trades", 0),
                equity_curve=equity_curve,
                benchmark_curve=benchmark_curve,
            )

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return BacktestResult(
                success=False,
                error=str(e),
                model_id=request.model_id,
                tickers=request.tickers or [],
            )

    def _prepare_data(
        self,
        ticker: str,
        start_date: date | None,
        end_date: date | None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list] | None:
        """
        Prepare data for backtesting.

        Returns:
            Tuple of (X, y, returns, dates) or None
        """
        # Load price data
        rows = get_prices(ticker, limit=2000)
        if not rows:
            return None

        df = pd.DataFrame(rows)

        # Filter by date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if start_date:
                df = df[df["date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["date"] <= pd.to_datetime(end_date)]

        if len(df) < 50:  # Need minimum data
            return None

        # Feature engineering
        df_feat = add_technical_features(df)
        df_labeled = add_future_return_label(df_feat)

        # Build X, y
        X, y = build_xy(df_labeled)

        if X.empty or len(y) == 0:
            return None

        # Get actual returns (for PnL calculation)
        # Use next-day return as the actual return
        if "close" in df.columns:
            df_aligned = df.iloc[-len(X):]
            returns = df_aligned["close"].pct_change().fillna(0).values
        else:
            returns = np.zeros(len(X))

        # Get dates
        if "date" in df.columns:
            dates = df["date"].iloc[-len(X):].tolist()
        else:
            dates = list(range(len(X)))

        return X, y.values, returns, dates

    def _simulate_strategy(
        self,
        y_prob: np.ndarray,
        returns: np.ndarray,
        signal_threshold: float,
        transaction_cost_bps: int,
        position_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate trading strategy.

        Strategy:
        - Long when prob > threshold
        - Flat when prob <= threshold
        - Apply transaction costs on position changes

        Returns:
            Tuple of (strategy_returns, benchmark_returns)
        """
        n = len(y_prob)
        strategy_returns = np.zeros(n)
        benchmark_returns = returns.copy()

        transaction_cost = transaction_cost_bps / 10000  # Convert bps to decimal

        # Generate signals: 1 = long, 0 = flat
        signals = (y_prob > signal_threshold).astype(float) * position_size

        # Previous position
        prev_position = 0.0

        for i in range(n):
            current_position = signals[i]

            # Calculate return from position
            if i > 0:
                strategy_returns[i] = prev_position * returns[i]

            # Apply transaction cost on position change
            position_change = abs(current_position - prev_position)
            if position_change > 0:
                strategy_returns[i] -= position_change * transaction_cost

            prev_position = current_position

        return strategy_returns, benchmark_returns
