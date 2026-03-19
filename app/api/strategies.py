"""
Strategies API

Provides endpoints for listing, inspecting, and using trading strategies.
"""

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.strategies import get_registry, StrategyMetadata
from app.db.prices_repo import get_prices_df


router = APIRouter(prefix="/api/strategies", tags=["Strategies"])


# ===================================
# Response Models
# ===================================


class StrategyListItem(BaseModel):
    """Summary of a strategy for listing."""
    name: str
    description: str
    version: str


class StrategyDetail(BaseModel):
    """Full strategy details including parameter schema."""
    name: str
    description: str
    version: str
    author: str
    required_columns: List[str]
    parameters_schema: Dict[str, Any]


class SignalsRequest(BaseModel):
    """Request body for generating signals."""
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol")
    start_date: Optional[date] = Field(None, description="Start date for signals")
    end_date: Optional[date] = Field(None, description="End date for signals")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Strategy-specific parameters"
    )


class SignalRecord(BaseModel):
    """Single signal record."""
    date: str
    signal: int  # -1, 0, 1
    signal_label: str  # "short", "hold", "long"


class SignalsResponse(BaseModel):
    """Response containing generated signals."""
    strategy: str
    ticker: str
    start_date: str
    end_date: str
    total_signals: int
    long_signals: int
    short_signals: int
    hold_signals: int
    signals: List[SignalRecord]


class BacktestRequest(BaseModel):
    """Request body for strategy backtest."""
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol")
    start_date: Optional[date] = Field(None, description="Backtest start date")
    end_date: Optional[date] = Field(None, description="Backtest end date")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Strategy-specific parameters"
    )
    initial_capital: float = Field(
        default=100000.0,
        gt=0,
        description="Initial capital for backtest"
    )
    transaction_cost_bps: float = Field(
        default=10.0,
        ge=0,
        description="Transaction cost in basis points"
    )


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: Optional[float]
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    profitable_trades: int
    losing_trades: int


class BacktestResponse(BaseModel):
    """Response containing backtest results."""
    strategy: str
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    metrics: BacktestMetrics


# ===================================
# Helper Functions
# ===================================


def _signal_to_label(signal: int) -> str:
    """Convert signal integer to label."""
    if signal == 1:
        return "long"
    elif signal == -1:
        return "short"
    return "hold"


def _get_price_data(ticker: str, limit: int = 500):
    """
    Get price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        limit: Maximum rows to return
        
    Returns:
        DataFrame with price data
        
    Raises:
        HTTPException: If no data found
    """
    df = get_prices_df(ticker.upper(), limit=limit)
    
    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price data found for {ticker}. Fetch market data first."
        )
    
    return df


# ===================================
# GET /api/strategies
# ===================================


@router.get("", response_model=List[StrategyListItem])
def list_strategies():
    """
    List all available trading strategies.
    
    Returns:
        List of strategy summaries with name, description, and version.
    """
    registry = get_registry()
    strategies = []
    
    for name in registry.list_strategies():
        metadata = registry.get_metadata(name)
        if metadata:
            strategies.append(StrategyListItem(
                name=metadata.name,
                description=metadata.description,
                version=metadata.version,
            ))
    
    return strategies


# ===================================
# GET /api/strategies/{name}
# ===================================


@router.get("/{name}", response_model=StrategyDetail)
def get_strategy(name: str):
    """
    Get detailed information about a specific strategy.
    
    Args:
        name: Strategy name (e.g., "ma_crossover", "rsi_strategy")
        
    Returns:
        Strategy details including parameter schema for configuration.
    """
    registry = get_registry()
    metadata = registry.get_metadata(name)
    
    if metadata is None:
        available = ", ".join(registry.list_strategies())
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{name}' not found. Available: {available}"
        )
    
    return StrategyDetail(
        name=metadata.name,
        description=metadata.description,
        version=metadata.version,
        author=metadata.author,
        required_columns=metadata.required_columns,
        parameters_schema=metadata.parameters_schema,
    )


# ===================================
# POST /api/strategies/{name}/signals
# ===================================


@router.post("/{name}/signals", response_model=SignalsResponse)
def generate_signals(name: str, request: SignalsRequest):
    """
    Generate trading signals for a ticker using a strategy.
    
    Args:
        name: Strategy name
        request: SignalsRequest with ticker, date range, and optional parameters
        
    Returns:
        SignalsResponse with signals for each date in the range.
    """
    registry = get_registry()
    
    # Get strategy class
    strategy_class = registry.get(name)
    if strategy_class is None:
        available = ", ".join(registry.list_strategies())
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{name}' not found. Available: {available}"
        )
    
    # Create strategy instance with parameters
    try:
        strategy = registry.create_instance(name, request.parameters)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameters: {str(e)}"
        )
    
    # Get price data
    df = _get_price_data(request.ticker)
    
    # Ensure date column
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = df.index.astype(str)
    
    # Filter by date range
    if request.start_date:
        df = df[df.index >= str(request.start_date)]
    if request.end_date:
        df = df[df.index <= str(request.end_date)]
    
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No data in the specified date range"
        )
    
    # Generate signals
    try:
        signals_series = strategy.generate_signals(df)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error generating signals: {str(e)}"
        )
    
    # Build response
    signals = []
    for idx, sig in signals_series.items():
        signals.append(SignalRecord(
            date=str(idx),
            signal=int(sig),
            signal_label=_signal_to_label(int(sig)),
        ))
    
    long_count = sum(1 for s in signals if s.signal == 1)
    short_count = sum(1 for s in signals if s.signal == -1)
    hold_count = sum(1 for s in signals if s.signal == 0)
    
    return SignalsResponse(
        strategy=name,
        ticker=request.ticker.upper(),
        start_date=signals[0].date if signals else "",
        end_date=signals[-1].date if signals else "",
        total_signals=len(signals),
        long_signals=long_count,
        short_signals=short_count,
        hold_signals=hold_count,
        signals=signals,
    )


# ===================================
# POST /api/strategies/{name}/backtest
# ===================================


@router.post("/{name}/backtest", response_model=BacktestResponse)
def run_strategy_backtest(name: str, request: BacktestRequest):
    """
    Run a backtest using a strategy on historical data.
    
    Simulates trading based on strategy signals and computes performance metrics.
    
    Args:
        name: Strategy name
        request: BacktestRequest with ticker, dates, parameters, and capital
        
    Returns:
        BacktestResponse with performance metrics.
    """
    registry = get_registry()
    
    # Get strategy class
    strategy_class = registry.get(name)
    if strategy_class is None:
        available = ", ".join(registry.list_strategies())
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{name}' not found. Available: {available}"
        )
    
    # Create strategy instance
    try:
        strategy = registry.create_instance(name, request.parameters)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameters: {str(e)}"
        )
    
    # Get price data
    df = _get_price_data(request.ticker)
    
    # Ensure we have required columns
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = df.index.astype(str)
    
    # Filter by date range
    if request.start_date:
        df = df[df.index >= str(request.start_date)]
    if request.end_date:
        df = df[df.index <= str(request.end_date)]
    
    if len(df) < 10:
        raise HTTPException(
            status_code=400,
            detail="Insufficient data for backtest (need at least 10 bars)"
        )
    
    # Generate signals
    try:
        signals = strategy.generate_signals(df)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error generating signals: {str(e)}"
        )
    
    # Simple backtest simulation
    capital = request.initial_capital
    position = 0  # -1, 0, or 1
    entry_price = 0.0
    transaction_cost = request.transaction_cost_bps / 10000
    
    trades = []
    portfolio_values = [capital]
    
    prices = df["close"].values
    signal_values = signals.values
    
    for i in range(1, len(prices)):
        current_price = prices[i]
        current_signal = signal_values[i]
        
        # Check if position needs to change
        if current_signal != position and current_signal != 0:
            # Close existing position
            if position != 0:
                pnl = (current_price - entry_price) * position
                trade_cost = abs(entry_price) * transaction_cost + abs(current_price) * transaction_cost
                capital += pnl - trade_cost
                trades.append({
                    "type": "close",
                    "pnl": pnl - trade_cost,
                    "profitable": pnl > 0
                })
            
            # Open new position
            position = int(current_signal)
            entry_price = current_price
            capital -= abs(current_price) * transaction_cost
        
        # Mark-to-market
        if position != 0:
            unrealized_pnl = (current_price - entry_price) * position
            portfolio_values.append(capital + unrealized_pnl)
        else:
            portfolio_values.append(capital)
    
    # Close any remaining position
    if position != 0:
        final_price = prices[-1]
        pnl = (final_price - entry_price) * position
        trade_cost = abs(entry_price) * transaction_cost + abs(final_price) * transaction_cost
        capital += pnl - trade_cost
        trades.append({
            "type": "close",
            "pnl": pnl - trade_cost,
            "profitable": pnl > 0
        })
    
    final_capital = capital
    
    # Calculate metrics
    total_return = (final_capital - request.initial_capital) / request.initial_capital * 100
    
    # Annualized return (assume ~252 trading days)
    trading_days = len(prices)
    years = trading_days / 252
    annualized_return = ((final_capital / request.initial_capital) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    
    # Max drawdown
    peak = portfolio_values[0]
    max_dd = 0
    for val in portfolio_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Win rate
    profitable_trades = sum(1 for t in trades if t["profitable"])
    total_trades = len(trades)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Sharpe ratio (simplified - daily returns)
    import numpy as np
    returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else [0]
    sharpe = None
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    
    return BacktestResponse(
        strategy=name,
        ticker=request.ticker.upper(),
        start_date=str(df.index[0]),
        end_date=str(df.index[-1]),
        initial_capital=request.initial_capital,
        final_capital=round(final_capital, 2),
        metrics=BacktestMetrics(
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(annualized_return, 2),
            sharpe_ratio=round(sharpe, 2) if sharpe else None,
            max_drawdown_pct=round(max_dd, 2),
            win_rate_pct=round(win_rate, 2),
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            losing_trades=total_trades - profitable_trades,
        ),
    )
