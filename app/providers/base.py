"""
Base Provider Interface

All data providers (market, sentiment, news) implement this interface
to ensure consistent data schema across different data sources.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


# ===================================
# Standard Schemas
# ===================================


class OHLCVRecord(BaseModel):
    """Standard OHLCV record schema."""

    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    class Config:
        json_encoders = {date: lambda v: v.isoformat()}


class SentimentRecord(BaseModel):
    """Standard sentiment record schema."""

    ticker: str
    date: date
    source: str  # e.g., "reddit", "twitter", "stocktwits"
    sentiment_score: float = Field(ge=-1.0, le=1.0)  # -1 (bearish) to 1 (bullish)
    volume: int = 0  # Number of posts/mentions
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    class Config:
        json_encoders = {date: lambda v: v.isoformat()}


class NewsRecord(BaseModel):
    """Standard news record schema."""

    ticker: str
    date: date
    source: str  # e.g., "reuters", "bloomberg", "yahoo"
    headline: str
    summary: str | None = None
    url: str | None = None
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)

    class Config:
        json_encoders = {date: lambda v: v.isoformat()}


# ===================================
# Base Provider Interface
# ===================================


class BaseProvider(ABC):
    """Abstract base class for all data providers."""

    @property
    @abstractmethod
    def provider_type(self) -> Literal["market", "sentiment", "news"]:
        """Return the provider type."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'yahoo', 'reddit')."""
        pass

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch data for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            **kwargs: Provider-specific parameters

        Returns:
            DataFrame with standardized schema
        """
        pass

    def validate_ticker(self, ticker: str) -> str:
        """Validate and normalize ticker symbol."""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        return ticker.upper().strip()

    def validate_date_range(
        self, start_date: date | None, end_date: date | None
    ) -> tuple[date | None, date | None]:
        """Validate date range."""
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must be before end_date")
        return start_date, end_date


class MarketProvider(BaseProvider):
    """Base class for market data providers."""

    @property
    def provider_type(self) -> Literal["market", "sentiment", "news"]:
        return "market"

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch OHLCV data."""
        pass


class SentimentProvider(BaseProvider):
    """Base class for sentiment data providers."""

    @property
    def provider_type(self) -> Literal["market", "sentiment", "news"]:
        return "sentiment"

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch sentiment data."""
        pass


class NewsProvider(BaseProvider):
    """Base class for news data providers."""

    @property
    def provider_type(self) -> Literal["market", "sentiment", "news"]:
        return "news"

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch news data."""
        pass
