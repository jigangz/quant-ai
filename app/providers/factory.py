"""
Provider Factory

Creates and manages data provider instances based on configuration.
Implements dependency injection pattern for easy testing and switching.
"""

import logging
from typing import Literal

from app.core.settings import settings
from app.providers.base import (
    BaseProvider,
    MarketProvider,
    SentimentProvider,
    NewsProvider,
)
from app.providers.market.yahoo import YahooMarketProvider
from app.providers.sentiment.mock import MockSentimentProvider
from app.providers.news.mock import MockNewsProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating data provider instances.

    Providers are cached and reused for efficiency.
    """

    _instances: dict[str, BaseProvider] = {}

    @classmethod
    def get_market_provider(cls, provider_name: str | None = None) -> MarketProvider:
        """
        Get a market data provider.

        Args:
            provider_name: Provider name (default: from settings)

        Returns:
            MarketProvider instance
        """
        name = provider_name or settings.MARKET_PROVIDER
        cache_key = f"market:{name}"

        if cache_key not in cls._instances:
            if name == "yahoo":
                cls._instances[cache_key] = YahooMarketProvider()
            # Future: add polygon, alpha_vantage, etc.
            else:
                raise ValueError(f"Unknown market provider: {name}")

            logger.info(f"Created market provider: {name}")

        return cls._instances[cache_key]

    @classmethod
    def get_sentiment_provider(cls, provider_name: str = "mock") -> SentimentProvider:
        """
        Get a sentiment data provider.

        Args:
            provider_name: Provider name (default: mock)

        Returns:
            SentimentProvider instance
        """
        cache_key = f"sentiment:{provider_name}"

        if cache_key not in cls._instances:
            if provider_name == "mock":
                cls._instances[cache_key] = MockSentimentProvider()
            # Future: add reddit, stocktwits, etc.
            else:
                raise ValueError(f"Unknown sentiment provider: {provider_name}")

            logger.info(f"Created sentiment provider: {provider_name}")

        return cls._instances[cache_key]

    @classmethod
    def get_news_provider(cls, provider_name: str = "mock") -> NewsProvider:
        """
        Get a news data provider.

        Args:
            provider_name: Provider name (default: mock)

        Returns:
            NewsProvider instance
        """
        cache_key = f"news:{provider_name}"

        if cache_key not in cls._instances:
            if provider_name == "mock":
                cls._instances[cache_key] = MockNewsProvider()
            # Future: add newsapi, benzinga, etc.
            else:
                raise ValueError(f"Unknown news provider: {provider_name}")

            logger.info(f"Created news provider: {provider_name}")

        return cls._instances[cache_key]

    @classmethod
    def get_provider(
        cls,
        provider_type: Literal["market", "sentiment", "news"],
        provider_name: str | None = None,
    ) -> BaseProvider:
        """
        Get a provider by type.

        Args:
            provider_type: Type of provider
            provider_name: Specific provider name (optional)

        Returns:
            Provider instance
        """
        if provider_type == "market":
            return cls.get_market_provider(provider_name)
        elif provider_type == "sentiment":
            return cls.get_sentiment_provider(provider_name or "mock")
        elif provider_type == "news":
            return cls.get_news_provider(provider_name or "mock")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @classmethod
    def get_enabled_providers(cls) -> dict[str, BaseProvider]:
        """
        Get all enabled providers based on settings.

        Returns:
            Dict of {provider_type: provider_instance}
        """
        providers = {}

        for provider_type in settings.providers_list:
            try:
                providers[provider_type] = cls.get_provider(provider_type)
            except ValueError as e:
                logger.warning(f"Could not create provider: {e}")

        return providers

    @classmethod
    def clear_cache(cls):
        """Clear all cached provider instances."""
        cls._instances.clear()
        logger.info("Provider cache cleared")


# Convenience functions
def get_market_provider(provider_name: str | None = None) -> MarketProvider:
    """Get market provider (convenience function)."""
    return ProviderFactory.get_market_provider(provider_name)


def get_sentiment_provider(provider_name: str = "mock") -> SentimentProvider:
    """Get sentiment provider (convenience function)."""
    return ProviderFactory.get_sentiment_provider(provider_name)


def get_news_provider(provider_name: str = "mock") -> NewsProvider:
    """Get news provider (convenience function)."""
    return ProviderFactory.get_news_provider(provider_name)
