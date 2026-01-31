"""
Data Providers Package

Provides unified interfaces for fetching market, sentiment, and news data
from various sources.

Usage:
    from app.providers import get_market_provider, get_sentiment_provider
    
    market = get_market_provider()
    df = market.fetch("AAPL", start_date=date(2024, 1, 1))
"""

from app.providers.base import (
    BaseProvider,
    MarketProvider,
    SentimentProvider,
    NewsProvider,
    OHLCVRecord,
    SentimentRecord,
    NewsRecord,
)
from app.providers.factory import (
    ProviderFactory,
    get_market_provider,
    get_sentiment_provider,
    get_news_provider,
)
from app.providers.market.yahoo import YahooMarketProvider
from app.providers.sentiment.mock import MockSentimentProvider
from app.providers.news.mock import MockNewsProvider

__all__ = [
    # Base classes
    "BaseProvider",
    "MarketProvider", 
    "SentimentProvider",
    "NewsProvider",
    # Schemas
    "OHLCVRecord",
    "SentimentRecord",
    "NewsRecord",
    # Factory
    "ProviderFactory",
    "get_market_provider",
    "get_sentiment_provider",
    "get_news_provider",
    # Implementations
    "YahooMarketProvider",
    "MockSentimentProvider",
    "MockNewsProvider",
]
