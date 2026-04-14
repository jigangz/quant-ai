"""Tests for News Data API."""
from __future__ import annotations
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestGetNews:
    def test_happy_path(self):
        mock_rows = [{"ticker": "AAPL", "date": "2024-06-01", "headline": "Test",
                       "summary": "S", "url": "https://x.com", "source": "mock",
                       "category": "tech", "sentiment_score": 0.5,
                       "bullish_reason": None, "bearish_reason": None,
                       "relevance_score": 0.8, "created_at": "2024-06-01T00:00:00"}]
        with patch("app.api.news.get_news_for_ticker", return_value=mock_rows):
            resp = client.get("/data/news/AAPL")
            assert resp.status_code == 200
            assert len(resp.json()) == 1

    def test_empty_result(self):
        with patch("app.api.news.get_news_for_ticker", return_value=[]):
            resp = client.get("/data/news/ZZZZ")
            assert resp.status_code == 200
            assert resp.json() == []

class TestSentimentSummary:
    def test_happy_path(self):
        with patch("app.api.news.get_news_sentiment_summary", return_value={
            "ticker": "AAPL", "date": "2024-06-01", "news_count": 5,
            "avg_sentiment": 0.3, "positive_count": 3, "negative_count": 1,
            "neutral_count": 1, "category_distribution": {"tech": 5}
        }):
            resp = client.get("/data/news/AAPL/sentiment-summary?date=2024-06-01")
            assert resp.status_code == 200
            assert resp.json()["news_count"] == 5
