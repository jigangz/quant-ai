"""
Mock News Provider

Generates synthetic news headlines for development/testing.
Used until real news API is integrated.
"""

import logging
from datetime import date, timedelta
import random

import numpy as np
import pandas as pd

from app.providers.base import NewsProvider

logger = logging.getLogger(__name__)


# Template headlines for different sentiment levels
BULLISH_HEADLINES = [
    "{ticker} stock surges on strong earnings report",
    "{ticker} announces new product launch, shares jump",
    "{ticker} beats analyst expectations, stock rallies",
    "Analysts upgrade {ticker} to 'Buy' rating",
    "{ticker} secures major partnership deal",
    "{ticker} reports record revenue growth",
    "Institutional investors increase stake in {ticker}",
    "{ticker} expansion plans drive stock higher",
]

BEARISH_HEADLINES = [
    "{ticker} stock falls on disappointing guidance",
    "{ticker} misses earnings estimates, shares drop",
    "Analysts downgrade {ticker} citing concerns",
    "{ticker} faces regulatory scrutiny",
    "{ticker} reports declining sales figures",
    "Competition pressures {ticker} market share",
    "{ticker} announces layoffs amid restructuring",
    "Supply chain issues impact {ticker} outlook",
]

NEUTRAL_HEADLINES = [
    "{ticker} reports in-line quarterly results",
    "{ticker} maintains market position",
    "Analysts remain neutral on {ticker}",
    "{ticker} CEO speaks at industry conference",
    "{ticker} announces board changes",
    "{ticker} files routine regulatory documents",
    "Market watches {ticker} ahead of earnings",
    "{ticker} trading volume remains steady",
]


class MockNewsProvider(NewsProvider):
    """
    Mock news provider that generates synthetic headlines.

    Headlines are generated with random sentiment to simulate
    real news flow.
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
        limit: int = 100,
        source: str = "mock_news",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate mock news data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            limit: Maximum number of articles
            source: Data source name

        Returns:
            DataFrame with news data
        """
        ticker = self.validate_ticker(ticker)
        start_date, end_date = self.validate_date_range(start_date, end_date)

        # Default to last 30 days
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        logger.info(
            f"Generating mock news for {ticker} from {start_date} to {end_date}"
        )

        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date).date

        records = []
        for d in dates:
            # Random number of articles per day (0-3)
            num_articles = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]

            for _ in range(num_articles):
                if len(records) >= limit:
                    break

                # Random sentiment
                sentiment = random.choices(
                    ["bullish", "bearish", "neutral"], weights=[0.35, 0.25, 0.40]
                )[0]

                if sentiment == "bullish":
                    headline = random.choice(BULLISH_HEADLINES).format(ticker=ticker)
                    sentiment_score = round(random.uniform(0.3, 0.9), 2)
                elif sentiment == "bearish":
                    headline = random.choice(BEARISH_HEADLINES).format(ticker=ticker)
                    sentiment_score = round(random.uniform(-0.9, -0.3), 2)
                else:
                    headline = random.choice(NEUTRAL_HEADLINES).format(ticker=ticker)
                    sentiment_score = round(random.uniform(-0.2, 0.2), 2)

                records.append(
                    {
                        "ticker": ticker,
                        "date": d,
                        "source": source,
                        "headline": headline,
                        "summary": f"Mock summary for: {headline}",
                        "url": f"https://mock-news.example.com/{ticker}/{d}",
                        "sentiment_score": sentiment_score,
                        "relevance_score": round(random.uniform(0.7, 1.0), 2),
                    }
                )

            if len(records) >= limit:
                break

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} mock news records for {ticker}")
        return df
