"""
News Sentiment Driven Strategy

A sentiment-based strategy that generates signals based on news sentiment scores.
This strategy uses a sentiment_score column (typically from NLP analysis of news)
to make trading decisions.

Signals:
- Long (1): Sentiment score above positive threshold (bullish news)
- Short (-1): Sentiment score below negative threshold (bearish news)
- Hold (0): Sentiment in neutral zone
"""

from typing import List, Optional

import pandas as pd
from pydantic import Field, validator

from app.strategies.base import BaseStrategy, BaseParameters


class SentimentDrivenParameters(BaseParameters):
    """Parameters for Sentiment Driven strategy."""
    
    positive_threshold: float = Field(
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="Sentiment score above this level triggers long signal"
    )
    negative_threshold: float = Field(
        default=-0.3,
        ge=-1.0,
        le=1.0,
        description="Sentiment score below this level triggers short signal"
    )
    sentiment_column: str = Field(
        default="sentiment_score",
        description="Name of the column containing sentiment scores"
    )
    smoothing_period: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Rolling window to smooth sentiment scores (1 = no smoothing)"
    )
    require_confirmation: bool = Field(
        default=False,
        description="If True, require consecutive periods of extreme sentiment to signal"
    )
    confirmation_periods: int = Field(
        default=2,
        ge=2,
        le=10,
        description="Number of consecutive periods required for confirmation (if enabled)"
    )
    
    @validator("negative_threshold")
    def thresholds_valid(cls, v, values):
        if "positive_threshold" in values and v >= values["positive_threshold"]:
            raise ValueError("negative_threshold must be less than positive_threshold")
        return v


class SentimentDriven(BaseStrategy):
    """
    News Sentiment Driven Strategy.
    
    Uses sentiment scores (typically from news or social media analysis) to
    generate trading signals. The sentiment score is expected to be in the
    range [-1, 1] where:
    - -1 = extremely negative
    - 0 = neutral
    - +1 = extremely positive
    
    The strategy can optionally smooth sentiment scores and require confirmation
    (consecutive extreme readings) before generating signals.
    """
    
    name = "sentiment_driven"
    description = "News Sentiment Driven - uses sentiment scores from news/social analysis"
    version = "1.0.0"
    required_columns: List[str] = ["sentiment_score"]  # Will be updated based on params
    Parameters = SentimentDrivenParameters
    
    def __init__(
        self,
        parameters: Optional[SentimentDrivenParameters] = None,
        **kwargs
    ):
        super().__init__(parameters, **kwargs)
        
        # Update required columns based on sentiment_column parameter
        self.required_columns = [self.params.sentiment_column]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on sentiment scores.
        
        Args:
            df: DataFrame with sentiment_score column (or configured column name)
            
        Returns:
            Series with signals: 1 (long/bullish), -1 (short/bearish), 0 (neutral)
        """
        missing = self.validate_data(df)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        sentiment = df[self.params.sentiment_column].copy()
        
        # Apply smoothing if configured
        if self.params.smoothing_period > 1:
            sentiment = sentiment.rolling(
                window=self.params.smoothing_period,
                min_periods=1
            ).mean()
        
        # Initialize signals
        signals = pd.Series(0, index=df.index, dtype=int)
        
        # Valid data points
        valid_idx = ~sentiment.isna()
        
        # Basic signals
        bullish = sentiment > self.params.positive_threshold
        bearish = sentiment < self.params.negative_threshold
        
        if self.params.require_confirmation:
            # Require consecutive periods of extreme sentiment
            n = self.params.confirmation_periods
            
            # Count consecutive bullish/bearish periods
            bullish_confirmed = bullish.rolling(window=n).sum() >= n
            bearish_confirmed = bearish.rolling(window=n).sum() >= n
            
            signals[valid_idx & bullish_confirmed] = 1
            signals[valid_idx & bearish_confirmed] = -1
        else:
            # Simple threshold signals
            signals[valid_idx & bullish] = 1
            signals[valid_idx & bearish] = -1
        
        return signals
