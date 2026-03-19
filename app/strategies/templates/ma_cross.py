"""
Moving Average Crossover Strategy

A classic trend-following strategy that generates signals based on the crossover
of fast and slow moving averages.

Signals:
- Long (1): Fast MA crosses above Slow MA
- Short (-1): Fast MA crosses below Slow MA
- Hold (0): No crossover
"""

from typing import List, Literal, Optional

import pandas as pd
from pydantic import Field

from app.strategies.base import BaseStrategy, BaseParameters


class MovingAverageCrossoverParameters(BaseParameters):
    """Parameters for Moving Average Crossover strategy."""
    
    fast_period: int = Field(
        default=10,
        ge=2,
        le=100,
        description="Period for the fast moving average"
    )
    slow_period: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Period for the slow moving average"
    )
    ma_type: Literal["sma", "ema"] = Field(
        default="sma",
        description="Type of moving average: 'sma' (Simple) or 'ema' (Exponential)"
    )
    signal_on_cross_only: bool = Field(
        default=False,
        description="If True, only signal on crossover bars. If False, maintain position while trend persists."
    )


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    Uses the crossover of two moving averages to generate trend-following signals.
    When the fast MA crosses above the slow MA, it signals a bullish trend (long).
    When the fast MA crosses below the slow MA, it signals a bearish trend (short).
    """
    
    name = "ma_crossover"
    description = "Moving Average Crossover - trend following strategy using fast/slow MA crossover"
    version = "1.0.0"
    required_columns: List[str] = ["close"]
    Parameters = MovingAverageCrossoverParameters
    
    def __init__(
        self,
        parameters: Optional[MovingAverageCrossoverParameters] = None,
        **kwargs
    ):
        super().__init__(parameters, **kwargs)
        
        # Validate fast < slow
        if self.params.fast_period >= self.params.slow_period:
            raise ValueError(
                f"fast_period ({self.params.fast_period}) must be less than "
                f"slow_period ({self.params.slow_period})"
            )
    
    def _calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate moving average based on ma_type parameter."""
        if self.params.ma_type == "ema":
            return series.ewm(span=period, adjust=False).mean()
        else:
            return series.rolling(window=period).mean()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with signals: 1 (long), -1 (short), 0 (neutral/no data)
        """
        missing = self.validate_data(df)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        close = df["close"]
        
        # Calculate MAs
        fast_ma = self._calculate_ma(close, self.params.fast_period)
        slow_ma = self._calculate_ma(close, self.params.slow_period)
        
        # Initialize signals
        signals = pd.Series(0, index=df.index, dtype=int)
        
        # Need enough data for slow MA
        valid_idx = ~(fast_ma.isna() | slow_ma.isna())
        
        if self.params.signal_on_cross_only:
            # Signal only on crossover bars
            fast_above = fast_ma > slow_ma
            fast_above_prev = fast_above.shift(1)
            
            # Long signal: fast crosses above slow
            long_cross = fast_above & ~fast_above_prev
            # Short signal: fast crosses below slow
            short_cross = ~fast_above & fast_above_prev
            
            signals[long_cross & valid_idx] = 1
            signals[short_cross & valid_idx] = -1
        else:
            # Maintain position while trend persists
            signals[valid_idx & (fast_ma > slow_ma)] = 1
            signals[valid_idx & (fast_ma < slow_ma)] = -1
        
        return signals
