from __future__ import annotations

"""
Bollinger Bands Breakout Strategy

A volatility-based strategy that generates signals when price breaks out
of Bollinger Bands.

Signals:
- Long (1): Price breaks above upper band (momentum breakout)
- Short (-1): Price breaks below lower band (momentum breakdown)
- Hold (0): Price within bands

Can also be configured for mean-reversion (reverse signals).
"""

from typing import List, Literal, Optional

import pandas as pd
from pydantic import Field

from app.strategies.base import BaseStrategy, BaseParameters


class BollingerBreakoutParameters(BaseParameters):
    """Parameters for Bollinger Bands Breakout strategy."""
    
    bb_period: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Period for Bollinger Bands calculation (moving average)"
    )
    bb_std: float = Field(
        default=2.0,
        ge=0.5,
        le=4.0,
        description="Number of standard deviations for band width"
    )
    strategy_mode: Literal["breakout", "mean_reversion"] = Field(
        default="breakout",
        description="'breakout': trade in direction of breakout. 'mean_reversion': fade the breakout"
    )
    use_close_cross: bool = Field(
        default=True,
        description="If True, signal on close crossing band. If False, signal on high/low touching band"
    )


class BollingerBreakout(BaseStrategy):
    """
    Bollinger Bands Breakout Strategy.
    
    Uses Bollinger Bands to identify volatility breakouts or mean-reversion opportunities.
    
    In breakout mode:
    - Long when price breaks above upper band (momentum continuation)
    - Short when price breaks below lower band (momentum continuation)
    
    In mean-reversion mode:
    - Short when price breaks above upper band (expect pullback)
    - Long when price breaks below lower band (expect bounce)
    """
    
    name = "bollinger_breakout"
    description = "Bollinger Bands Breakout - volatility-based breakout or mean-reversion strategy"
    version = "1.0.0"
    required_columns: List[str] = ["close"]
    Parameters = BollingerBreakoutParameters
    
    def __init__(
        self,
        parameters: Optional[BollingerBreakoutParameters] = None,
        **kwargs
    ):
        super().__init__(parameters, **kwargs)
        
        # If using high/low cross, we need those columns
        if not self.params.use_close_cross:
            self.required_columns = ["close", "high", "low"]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            df: DataFrame with 'close' column (and 'high'/'low' if use_close_cross=False)
            
        Returns:
            Series with signals: 1 (long), -1 (short), 0 (neutral)
        """
        missing = self.validate_data(df)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        close = df["close"]
        
        # Calculate Bollinger Bands manually
        middle_band = close.rolling(window=self.params.bb_period).mean()
        std = close.rolling(window=self.params.bb_period).std()
        upper_band = middle_band + (self.params.bb_std * std)
        lower_band = middle_band - (self.params.bb_std * std)
        
        # Initialize signals
        signals = pd.Series(0, index=df.index, dtype=int)
        
        # Valid data points
        valid_idx = ~(upper_band.isna() | lower_band.isna())
        
        if self.params.use_close_cross:
            # Use close price for signals
            price = close
        else:
            # Use high for upper band, low for lower band
            price_upper = df["high"]
            price_lower = df["low"]
        
        if self.params.use_close_cross:
            # Close-based signals
            above_upper = close > upper_band
            below_lower = close < lower_band
            
            if self.params.strategy_mode == "breakout":
                # Breakout: go with the momentum
                signals[valid_idx & above_upper] = 1
                signals[valid_idx & below_lower] = -1
            else:
                # Mean reversion: fade the extremes
                signals[valid_idx & above_upper] = -1
                signals[valid_idx & below_lower] = 1
        else:
            # High/Low based signals
            above_upper = df["high"] > upper_band
            below_lower = df["low"] < lower_band
            
            if self.params.strategy_mode == "breakout":
                signals[valid_idx & above_upper] = 1
                signals[valid_idx & below_lower] = -1
            else:
                signals[valid_idx & above_upper] = -1
                signals[valid_idx & below_lower] = 1
        
        return signals
