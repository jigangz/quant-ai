from __future__ import annotations

"""
RSI (Relative Strength Index) Strategy

A mean-reversion strategy that generates signals based on RSI overbought
and oversold conditions.

Signals:
- Long (1): RSI drops below oversold level (market oversold, expect bounce)
- Short (-1): RSI rises above overbought level (market overbought, expect pullback)
- Hold (0): RSI in neutral zone
"""

from typing import List, Optional

import pandas as pd
from pydantic import Field, ValidationInfo, field_validator

from app.strategies.base import BaseStrategy, BaseParameters


class RSIParameters(BaseParameters):
    """Parameters for RSI strategy."""
    
    rsi_period: int = Field(
        default=14,
        ge=2,
        le=100,
        description="Period for RSI calculation"
    )
    overbought: float = Field(
        default=70.0,
        ge=50.0,
        le=95.0,
        description="RSI level above which market is considered overbought"
    )
    oversold: float = Field(
        default=30.0,
        ge=5.0,
        le=50.0,
        description="RSI level below which market is considered oversold"
    )
    exit_on_neutral: bool = Field(
        default=False,
        description="If True, exit position when RSI returns to neutral zone (between overbought and oversold)"
    )
    
    @field_validator("oversold")
    def oversold_less_than_overbought(cls, v: float, info: ValidationInfo) -> float:
        if "overbought" in info.data and v >= info.data["overbought"]:
            raise ValueError("oversold must be less than overbought")
        return v


class RSIStrategy(BaseStrategy):
    """
    RSI Overbought/Oversold Strategy.
    
    Uses the Relative Strength Index to identify overbought and oversold conditions.
    When RSI drops below the oversold level, it signals a potential buying opportunity.
    When RSI rises above the overbought level, it signals a potential selling opportunity.
    
    This is a mean-reversion strategy that assumes extreme RSI values will revert to the mean.
    """
    
    name = "rsi_strategy"
    description = "RSI Overbought/Oversold - mean reversion strategy using RSI extremes"
    version = "1.0.0"
    required_columns: List[str] = ["close"]
    Parameters = RSIParameters
    
    def __init__(
        self,
        parameters: Optional[RSIParameters] = None,
        **kwargs
    ):
        super().__init__(parameters, **kwargs)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI levels.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series with signals: 1 (long when oversold), -1 (short when overbought), 0 (neutral)
        """
        missing = self.validate_data(df)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        close = df["close"]
        
        # Calculate RSI manually
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.params.rsi_period).mean()
        avg_loss = loss.rolling(window=self.params.rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Initialize signals
        signals = pd.Series(0, index=df.index, dtype=int)
        
        # Valid data points (RSI needs warmup period)
        valid_idx = ~rsi.isna()
        
        if self.params.exit_on_neutral:
            # Simple level-based signals
            signals[valid_idx & (rsi < self.params.oversold)] = 1
            signals[valid_idx & (rsi > self.params.overbought)] = -1
            # Neutral zone stays 0
        else:
            # Maintain position until opposite signal
            # Track when we enter overbought/oversold
            current_signal = 0
            for i, (idx, r) in enumerate(rsi.items()):
                if pd.isna(r):
                    signals.iloc[i] = 0
                elif r < self.params.oversold:
                    current_signal = 1
                    signals.iloc[i] = current_signal
                elif r > self.params.overbought:
                    current_signal = -1
                    signals.iloc[i] = current_signal
                else:
                    signals.iloc[i] = current_signal
        
        return signals
