"""
Built-in Strategy Templates

This module provides ready-to-use trading strategy implementations:
- MovingAverageCrossover: Classic MA crossover strategy
- RSIStrategy: RSI overbought/oversold strategy
- BollingerBreakout: Bollinger Bands breakout strategy
- SentimentDriven: News sentiment-based strategy
"""

from app.strategies.templates.ma_cross import MovingAverageCrossover
from app.strategies.templates.rsi_strategy import RSIStrategy
from app.strategies.templates.bollinger_breakout import BollingerBreakout
from app.strategies.templates.sentiment_driven import SentimentDriven

__all__ = [
    "MovingAverageCrossover",
    "RSIStrategy",
    "BollingerBreakout",
    "SentimentDriven",
]
