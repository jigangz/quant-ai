"""
Backtest Engine

Runs backtests on trained models:
1. Load model from registry
2. Generate predictions on historical data
3. Simulate trading strategy with position sizing
4. Apply transaction costs and slippage
5. Calculate performance metrics
6. Compare to Buy & Hold benchmark

V3 Enhancements:
- Position sizing: fixed, volatility_scaled, kelly
- Slippage simulation
- Multi-ticker portfolio backtesting
- Portfolio weighting: equal, market_cap, inverse_vol
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

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
    
    # Position sizing
    position_sizing: Literal["fixed", "volatility_scaled", "kelly"] = "fixed"
    position_size: float = Field(default=1.0, ge=0.1, le=2.0)  # base size or max
    volatility_lookback: int = Field(default=20, ge=5, le=60)  # for vol scaling
    kelly_fraction: float = Field(default=0.25, ge=0.1, le=0.5)  # kelly fraction
    
    # Transaction costs
    enable_costs: bool = True
    transaction_cost_bps: int = Field(default=10, ge=0, le=100)  # basis points
    
    # Slippage
    enable_slippage: bool = True
    slippage_bps: int = Field(default=5, ge=0, le=50)  # basis points
    
    # Portfolio (multi-ticker)
    portfolio_weighting: Literal["equal", "market_cap", "inverse_vol"] = "equal"
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "daily"

    class Config:
        extra = "forbid"


class TickerResult(BaseModel):
    """Per-ticker backtest results."""
    
    ticker: str
    n_samples: int
    n_trades: int
    weight: float
    sharpe: float | None = None
    total_return: float | None = None
    win_rate: float | None = None


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
    
    # Config details
    position_sizing: str = "fixed"
    transaction_cost_bps: int = 10
    slippage_bps: int = 5
    portfolio_weighting: str = "equal"

    # Metrics
    strategy_metrics: dict[str, Any] = {}
    classification_metrics: dict[str, float] = {}
    
    # Per-ticker results
    ticker_results: list[TickerResult] = []

    # Report
    report_markdown: str = ""

    # Details
    n_samples: int = 0
    n_trades: int = 0
    equity_curve: list[float] = []
    benchmark_curve: list[float] = []
    
    # Drawdown
    max_drawdown: float = 0.0
    drawdown_curve: list[float] = []

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
        """Run backtest."""
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
            ticker_data = {}
            for ticker in tickers:
                data = self._prepare_data(ticker, request.start_date, request.end_date)
                if data is not None and len(data[0]) > 0:
                    ticker_data[ticker] = data
                else:
                    logger.warning(f"No data for {ticker}")

            if not ticker_data:
                return BacktestResult(
                    success=False,
                    error="No data available for backtest",
                    model_id=request.model_id,
                    tickers=tickers,
                )

            # 3. Calculate portfolio weights
            weights = self._calculate_weights(
                ticker_data, 
                request.portfolio_weighting
            )

            # 4. Run backtest for each ticker
            ticker_results = []
            all_strategy_returns = []
            all_benchmark_returns = []
            all_predictions = []
            all_actuals = []
            all_probs = []
            total_trades = 0

            for ticker, (X, y, returns, dates, volatility) in ticker_data.items():
                weight = weights.get(ticker, 1.0 / len(ticker_data))
                
                # Generate predictions
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1]

                # Calculate position sizes
                positions = self._calculate_positions(
                    y_prob=y_prob,
                    volatility=volatility,
                    request=request,
                )

                # Simulate strategy
                strat_ret, bench_ret, n_trades = self._simulate_strategy(
                    positions=positions,
                    returns=returns,
                    weight=weight,
                    request=request,
                )

                # Store results
                ticker_results.append(TickerResult(
                    ticker=ticker,
                    n_samples=len(y),
                    n_trades=n_trades,
                    weight=weight,
                    sharpe=self._quick_sharpe(strat_ret),
                    total_return=float(np.sum(strat_ret)),
                    win_rate=float(np.mean(strat_ret > 0)) if len(strat_ret) > 0 else None,
                ))

                all_strategy_returns.append(strat_ret)
                all_benchmark_returns.append(bench_ret)
                all_predictions.extend(y_pred)
                all_actuals.extend(y)
                all_probs.extend(y_prob)
                total_trades += n_trades

            # 5. Aggregate portfolio returns
            portfolio_returns = self._aggregate_returns(
                all_strategy_returns, 
                list(weights.values())
            )
            benchmark_returns = self._aggregate_returns(
                all_benchmark_returns,
                list(weights.values())
            )

            # 6. Calculate metrics
            y_pred = np.array(all_predictions)
            y_true = np.array(all_actuals)
            y_prob = np.array(all_probs)

            classification_metrics = calculate_classification_metrics(
                y_true, y_pred, y_prob
            )

            strategy_metrics = calculate_strategy_metrics(
                pd.Series(portfolio_returns),
                pd.Series(benchmark_returns),
            )
            strategy_metrics["n_trades"] = total_trades

            # 7. Generate curves
            equity_curve = list((1 + pd.Series(portfolio_returns)).cumprod())
            benchmark_curve = list((1 + pd.Series(benchmark_returns)).cumprod())
            
            # Drawdown curve
            cumulative = pd.Series(equity_curve)
            running_max = cumulative.cummax()
            drawdown_curve = list((cumulative - running_max) / running_max)
            max_drawdown = float(min(drawdown_curve)) if drawdown_curve else 0.0

            # 8. Generate report
            start_str = str(request.start_date) if request.start_date else "N/A"
            end_str = str(request.end_date) if request.end_date else "N/A"

            config = {
                "model_id": request.model_id,
                "tickers": tickers,
                "start_date": start_str,
                "end_date": end_str,
                "transaction_cost_bps": request.transaction_cost_bps if request.enable_costs else 0,
                "slippage_bps": request.slippage_bps if request.enable_slippage else 0,
                "position_sizing": request.position_sizing,
                "portfolio_weighting": request.portfolio_weighting,
            }
            report_md = generate_report_markdown(
                strategy_metrics, classification_metrics, config
            )

            logger.info(f"Backtest complete: Sharpe={strategy_metrics.get('sharpe')}")

            return BacktestResult(
                success=True,
                model_id=request.model_id,
                tickers=tickers,
                start_date=start_str,
                end_date=end_str,
                position_sizing=request.position_sizing,
                transaction_cost_bps=request.transaction_cost_bps if request.enable_costs else 0,
                slippage_bps=request.slippage_bps if request.enable_slippage else 0,
                portfolio_weighting=request.portfolio_weighting,
                strategy_metrics=strategy_metrics,
                classification_metrics=classification_metrics,
                ticker_results=ticker_results,
                report_markdown=report_md,
                n_samples=len(y_pred),
                n_trades=total_trades,
                equity_curve=equity_curve,
                benchmark_curve=benchmark_curve,
                max_drawdown=max_drawdown,
                drawdown_curve=drawdown_curve,
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
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list, np.ndarray] | None:
        """
        Prepare data for backtesting.

        Returns:
            Tuple of (X, y, returns, dates, volatility) or None
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

        if len(df) < 50:
            return None

        # Feature engineering
        df_feat = add_technical_features(df)
        df_labeled = add_future_return_label(df_feat)

        # Build X, y
        X, y = build_xy(df_labeled)

        if X.empty or len(y) == 0:
            return None

        # Get actual returns
        df_aligned = df.iloc[-len(X):]
        if "close" in df.columns:
            returns = df_aligned["close"].pct_change().fillna(0).values
        else:
            returns = np.zeros(len(X))

        # Calculate rolling volatility (for position sizing)
        if "close" in df.columns:
            volatility = df_aligned["close"].pct_change().rolling(20).std().fillna(0.02).values
        else:
            volatility = np.full(len(X), 0.02)

        # Get dates
        if "date" in df.columns:
            dates = df["date"].iloc[-len(X):].tolist()
        else:
            dates = list(range(len(X)))

        return X, y.values, returns, dates, volatility

    def _calculate_weights(
        self,
        ticker_data: dict,
        weighting: str,
    ) -> dict[str, float]:
        """Calculate portfolio weights for each ticker."""
        tickers = list(ticker_data.keys())
        n = len(tickers)
        
        if weighting == "equal":
            return {t: 1.0 / n for t in tickers}
        
        elif weighting == "inverse_vol":
            # Inverse volatility weighting
            vols = {}
            for ticker, (X, y, returns, dates, volatility) in ticker_data.items():
                avg_vol = np.mean(volatility) if len(volatility) > 0 else 0.02
                vols[ticker] = 1.0 / max(avg_vol, 0.001)
            
            total = sum(vols.values())
            return {t: v / total for t, v in vols.items()}
        
        elif weighting == "market_cap":
            # For now, fall back to equal (would need market cap data)
            logger.warning("Market cap weighting not implemented, using equal")
            return {t: 1.0 / n for t in tickers}
        
        return {t: 1.0 / n for t in tickers}

    def _calculate_positions(
        self,
        y_prob: np.ndarray,
        volatility: np.ndarray,
        request: BacktestRequest,
    ) -> np.ndarray:
        """
        Calculate position sizes based on sizing method.
        
        Returns:
            Array of position sizes (0 to max_size)
        """
        n = len(y_prob)
        positions = np.zeros(n)
        
        # Base signal
        signals = (y_prob > request.signal_threshold).astype(float)
        
        if request.position_sizing == "fixed":
            positions = signals * request.position_size
        
        elif request.position_sizing == "volatility_scaled":
            # Scale inversely with volatility
            target_vol = 0.02  # 2% target daily vol
            vol_scale = target_vol / np.maximum(volatility, 0.001)
            vol_scale = np.clip(vol_scale, 0.1, request.position_size)
            positions = signals * vol_scale
        
        elif request.position_sizing == "kelly":
            # Kelly criterion: f = p - (1-p)/b
            # Simplified: use probability as edge estimate
            edge = y_prob - 0.5  # Edge over random
            kelly_size = np.clip(edge * 2, 0, request.kelly_fraction)
            positions = signals * kelly_size * request.position_size
        
        return positions

    def _simulate_strategy(
        self,
        positions: np.ndarray,
        returns: np.ndarray,
        weight: float,
        request: BacktestRequest,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Simulate trading strategy with costs and slippage.

        Returns:
            Tuple of (strategy_returns, benchmark_returns, n_trades)
        """
        n = len(positions)
        strategy_returns = np.zeros(n)
        benchmark_returns = returns.copy() * weight

        # Costs
        transaction_cost = request.transaction_cost_bps / 10000 if request.enable_costs else 0
        slippage = request.slippage_bps / 10000 if request.enable_slippage else 0
        total_cost = transaction_cost + slippage

        prev_position = 0.0
        n_trades = 0

        for i in range(n):
            current_position = positions[i] * weight

            # Calculate return from position
            if i > 0:
                strategy_returns[i] = prev_position * returns[i]

            # Apply costs on position change
            position_change = abs(current_position - prev_position)
            if position_change > 0.01:  # Threshold to count as trade
                strategy_returns[i] -= position_change * total_cost
                n_trades += 1

            prev_position = current_position

        return strategy_returns, benchmark_returns, n_trades

    def _aggregate_returns(
        self,
        returns_list: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        """Aggregate returns from multiple tickers."""
        if not returns_list:
            return np.array([])
        
        # Align lengths (use shortest)
        min_len = min(len(r) for r in returns_list)
        
        portfolio_returns = np.zeros(min_len)
        for returns, weight in zip(returns_list, weights):
            portfolio_returns += returns[:min_len] * weight
        
        return portfolio_returns

    def _quick_sharpe(self, returns: np.ndarray) -> float | None:
        """Quick Sharpe ratio calculation."""
        if len(returns) < 2:
            return None
        std = np.std(returns)
        if std == 0:
            return None
        return float(np.mean(returns) / std * np.sqrt(252))
