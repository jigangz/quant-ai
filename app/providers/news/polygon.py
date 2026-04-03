from __future__ import annotations

"""
Polygon.io News Provider

Fetches real news data from Polygon.io API and uses Claude Haiku
for sentiment analysis and categorization.
"""

import json
import logging
import time
from datetime import date, timedelta

import httpx
import pandas as pd

from app.core.settings import settings
from app.providers.base import NewsProvider

logger = logging.getLogger(__name__)

# News categories for classification
CATEGORIES = [
    "earnings",
    "product",
    "policy",
    "market",
    "competition",
    "management",
    "other",
]

SENTIMENT_PROMPT = """\
You are a financial news analyst. For each news article below, provide:
1. sentiment_score: float from -1.0 (very bearish) to 1.0 (very bullish)
2. category: one of {categories}
3. bullish_reason: brief reason for bullish sentiment (or null)
4. bearish_reason: brief reason for bearish sentiment (or null)
5. relevance_score: float from 0.0 to 1.0 indicating relevance to the ticker

Respond ONLY with a JSON array. Each element must have keys:
  index, sentiment_score, category, bullish_reason, bearish_reason, relevance_score

Articles:
{articles}
"""


def _analyze_sentiment_batch(
    articles: list[dict],
    ticker: str,
) -> list[dict]:
    """
    Analyze sentiment for a batch of articles using Claude Haiku.

    Args:
        articles: List of dicts with 'index', 'title', 'description'.
        ticker: Stock ticker for context.

    Returns:
        List of sentiment dicts keyed by index.
    """
    if not settings.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set, skipping sentiment analysis")
        return [
            {
                "index": a["index"],
                "sentiment_score": None,
                "category": "other",
                "bullish_reason": None,
                "bearish_reason": None,
                "relevance_score": None,
            }
            for a in articles
        ]

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed, skipping sentiment analysis")
        return [
            {
                "index": a["index"],
                "sentiment_score": None,
                "category": "other",
                "bullish_reason": None,
                "bearish_reason": None,
                "relevance_score": None,
            }
            for a in articles
        ]

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    formatted_articles = "\n".join(
        f"[{a['index']}] {a['title']} — {a.get('description', '') or ''}"
        for a in articles
    )

    prompt = SENTIMENT_PROMPT.format(
        categories=", ".join(CATEGORIES),
        articles=formatted_articles,
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze these news articles about {ticker}:\n\n{prompt}",
                }
            ],
        )
        raw = response.content[0].text.strip()

        # Extract JSON array from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            logger.error("No JSON array found in Claude response")
            raise ValueError("Invalid response format")

        results = json.loads(raw[start:end])
        return results

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return [
            {
                "index": a["index"],
                "sentiment_score": None,
                "category": "other",
                "bullish_reason": None,
                "bearish_reason": None,
                "relevance_score": None,
            }
            for a in articles
        ]


class PolygonNewsProvider(NewsProvider):
    """
    Polygon.io news provider with Claude Haiku sentiment analysis.

    Requires POLYGON_API_KEY in settings.
    Free tier rate limiting is applied automatically (0.2s between requests).
    """

    BASE_URL = "https://api.polygon.io/v2/reference/news"

    @property
    def provider_name(self) -> str:
        return "polygon"

    def fetch(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch news from Polygon.io and analyze sentiment with Claude.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date (unused by Polygon news endpoint).
            end_date: End date (unused by Polygon news endpoint).
            limit: Maximum number of articles to fetch.

        Returns:
            DataFrame with columns: ticker, date, headline, summary, url,
            source, sentiment_score, category, bullish_reason, bearish_reason,
            relevance_score.

        Raises:
            ValueError: If POLYGON_API_KEY is not configured.
        """
        if not settings.POLYGON_API_KEY:
            raise ValueError(
                "POLYGON_API_KEY is required for Polygon news provider. "
                "Set it in .env or environment variables."
            )

        ticker = self.validate_ticker(ticker)

        # Fetch from Polygon API
        params = {
            "ticker": ticker,
            "limit": min(limit, 100),
            "apiKey": settings.POLYGON_API_KEY,
        }

        logger.info(f"Fetching news from Polygon.io for {ticker}, limit={limit}")

        try:
            # Rate limit for free tier
            time.sleep(0.2)
            resp = httpx.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error(f"Polygon API request failed: {e}")
            return pd.DataFrame()

        results = data.get("results", [])
        if not results:
            logger.info(f"No news found for {ticker}")
            return pd.DataFrame()

        logger.info(f"Fetched {len(results)} articles for {ticker}")

        # Parse articles
        articles = []
        for i, item in enumerate(results):
            published = item.get("published_utc", "")
            article_date = published[:10] if published else str(date.today())
            publisher = item.get("publisher", {})

            articles.append(
                {
                    "index": i,
                    "ticker": ticker,
                    "date": article_date,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "url": item.get("article_url", ""),
                    "source": publisher.get("name", "unknown") if isinstance(publisher, dict) else str(publisher),
                }
            )

        # Sentiment analysis in batches
        batch_size = settings.NEWS_BATCH_SIZE
        all_sentiments: dict[int, dict] = {}

        for batch_start in range(0, len(articles), batch_size):
            batch = articles[batch_start : batch_start + batch_size]
            batch_input = [
                {
                    "index": a["index"],
                    "title": a["title"],
                    "description": a["description"],
                }
                for a in batch
            ]

            sentiments = _analyze_sentiment_batch(batch_input, ticker)
            for s in sentiments:
                all_sentiments[s["index"]] = s

        # Build final records
        records = []
        for a in articles:
            s = all_sentiments.get(a["index"], {})
            records.append(
                {
                    "ticker": a["ticker"],
                    "date": a["date"],
                    "headline": a["title"],
                    "summary": a["description"],
                    "url": a["url"],
                    "source": a["source"],
                    "sentiment_score": s.get("sentiment_score"),
                    "category": s.get("category", "other"),
                    "bullish_reason": s.get("bullish_reason"),
                    "bearish_reason": s.get("bearish_reason"),
                    "relevance_score": s.get("relevance_score"),
                }
            )

        df = pd.DataFrame(records)
        logger.info(
            f"Polygon news provider returned {len(df)} articles for {ticker}"
        )
        return df
