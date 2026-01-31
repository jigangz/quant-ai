"""
Mock Sentiment Provider

Generates synthetic sentiment data based on price movements.
Used for development/testing until real sentiment API is integrated.
"""

import logging
from datetime import date, timedelta
import random

import numpy as np
import pandas as pd

from app.providers.base import SentimentProvider

logger = logging.getLogger(__name__)


class MockSentimentProvider(SentimentProvider):
    """
    Mock sentiment provider that generates synthetic data.

    Sentiment is loosely correlated with price movements to simulate
    realistic sentiment-price relationships.
    """

    def __init__(self, seed: int | None = None):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @property
    def provider_name(self) -> str:
        return "mock"

    def fetch(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        source: str = "mock_social",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate mock sentiment data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            source: Data source name

        Returns:
            DataFrame with sentiment data
        """
        ticker = self.validate_ticker(ticker)
        start_date, end_date = self.validate_date_range(start_date, end_date)

        # Default to last 30 days
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        logger.info(
            f"Generating mock sentiment for {ticker} from {start_date} to {end_date}"
        )

        # Generate date range (business days only)
        dates = pd.bdate_range(start=start_date, end=end_date).date

        records = []
        for d in dates:
            # Generate random sentiment with some autocorrelation
            base_sentiment = np.random.normal(0, 0.3)
            sentiment_score = np.clip(base_sentiment, -1, 1)

            # Generate volume based on absolute sentiment (more extreme = more activity)
            base_volume = int(abs(sentiment_score) * 500 + np.random.poisson(100))

            # Split into bullish/bearish/neutral
            total = base_volume
            if sentiment_score > 0.1:
                bullish = int(total * (0.5 + sentiment_score * 0.3))
                bearish = int(total * (0.2 - sentiment_score * 0.1))
            elif sentiment_score < -0.1:
                bullish = int(total * (0.2 + sentiment_score * 0.1))
                bearish = int(total * (0.5 - sentiment_score * 0.3))
            else:
                bullish = int(total * 0.3)
                bearish = int(total * 0.3)
            neutral = total - bullish - bearish

            records.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "source": source,
                    "sentiment_score": round(sentiment_score, 4),
                    "volume": total,
                    "bullish_count": max(0, bullish),
                    "bearish_count": max(0, bearish),
                    "neutral_count": max(0, neutral),
                }
            )

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} mock sentiment records for {ticker}")
        return df
