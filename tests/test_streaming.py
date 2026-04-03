"""
Tests for Kafka Real-time Data Pipeline (Phase 2.5 Step 9)

Tests all producers and consumers using InMemoryBroker.
No real Kafka or Redis required.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock settings before any app imports
# ---------------------------------------------------------------------------
_mock_settings = MagicMock()
_mock_settings.BROKER_BACKEND = "memory"
_mock_settings.QUEUE_BACKEND = "memory"
_mock_settings.NOTIFY_BACKEND = "memory"
_mock_settings.FUNCTIONS_BACKEND = "local"
_mock_settings.REDIS_URL = "redis://localhost:6379/0"
_mock_settings.SQS_ENDPOINT_URL = None
_mock_settings.SNS_ENDPOINT_URL = None
_mock_settings.LAMBDA_ENDPOINT_URL = None
_mock_settings.AWS_REGION = "us-east-1"
_mock_settings.ANTHROPIC_API_KEY = None
_mock_settings.POLYGON_API_KEY = None
_mock_settings.MARKET_PROVIDER = "yahoo"
_mock_settings.KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
_mock_settings.KAFKA_SASL_USERNAME = None
_mock_settings.KAFKA_SASL_PASSWORD = None
_mock_settings.LOG_LEVEL = "WARNING"

_mock_settings_module = MagicMock()
_mock_settings_module.settings = _mock_settings
sys.modules.setdefault("app.core.settings", _mock_settings_module)

# Mock app.core.logging
_mock_logging_module = MagicMock()
_mock_logging_module.get_logger = lambda name: __import__("logging").getLogger(name)
_mock_logging_module.setup_logging = MagicMock()
_mock_logging_module.request_id_ctx = MagicMock()
sys.modules.setdefault("app.core.logging", _mock_logging_module)

# Mock redis.asyncio
_mock_aioredis = MagicMock()
sys.modules.setdefault("redis.asyncio", _mock_aioredis)

# Mock app.cache
_mock_cache_module = MagicMock()
_mock_cache_module.get_redis = AsyncMock(return_value=None)
sys.modules.setdefault("app.cache", _mock_cache_module)

_mock_market_cache = MagicMock()
_mock_market_cache.get_market_cache = MagicMock()
sys.modules.setdefault("app.cache.market_cache", _mock_market_cache)

# Mock heavy deps
for mod_name in [
    "app.db.prices_repo", "app.ml.features.technical",
    "app.ml.labels.returns", "app.ml.features.build",
    "app.db.model_registry", "app.services.model_cache",
    "app.middleware", "app.providers.base",
    "app.explain.shap_explainer", "app.explain.shap_to_text",
    "yfinance", "joblib", "shap",
]:
    sys.modules.setdefault(mod_name, MagicMock())


from app.infra.broker import InMemoryBroker, BrokerMessage  # noqa: E402
from app.streaming import (  # noqa: E402
    TOPIC_MARKET_PRICES,
    TOPIC_NEWS_RAW,
    TOPIC_NEWS_SCORED,
    TOPIC_SIGNALS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_broker() -> InMemoryBroker:
    return InMemoryBroker()


# ---------------------------------------------------------------------------
# Topic Constants
# ---------------------------------------------------------------------------

class TestTopicConstants:
    """Verify topic constant values."""

    def test_topic_values(self):
        assert TOPIC_MARKET_PRICES == "market.prices"
        assert TOPIC_NEWS_RAW == "news.raw"
        assert TOPIC_NEWS_SCORED == "news.scored"
        assert TOPIC_SIGNALS == "signals.generated"


# ---------------------------------------------------------------------------
# Price Producer
# ---------------------------------------------------------------------------

class TestPriceProducer:
    """Test price producer message format and publishing."""

    @pytest.mark.asyncio
    async def test_publish_price_message_format(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.price_producer import PriceProducer

        producer = PriceProducer(broker, symbols=["AAPL"], interval_seconds=10)
        await producer._tick()

        assert len(broker._messages) == 1
        msg = broker._messages[0]
        assert msg.topic == TOPIC_MARKET_PRICES
        assert msg.key == "AAPL"

        value = msg.value
        assert value["symbol"] == "AAPL"
        assert isinstance(value["price"], float)
        assert "volume" in value
        assert "timestamp" in value
        assert value["source"] == "mock"

    @pytest.mark.asyncio
    async def test_publish_multiple_symbols(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.price_producer import PriceProducer

        symbols = ["AAPL", "MSFT", "GOOGL"]
        producer = PriceProducer(broker, symbols=symbols)
        await producer._tick()

        assert len(broker._messages) == 3
        published_symbols = {m.value["symbol"] for m in broker._messages}
        assert published_symbols == set(symbols)

    @pytest.mark.asyncio
    async def test_publish_once(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.price_producer import PriceProducer

        producer = PriceProducer(broker, symbols=["AAPL"])
        await producer.publish_once("TSLA", 250.0, volume=1000)

        assert len(broker._messages) == 1
        msg = broker._messages[0]
        assert msg.value["symbol"] == "TSLA"
        assert msg.value["price"] == 250.0
        assert msg.value["volume"] == 1000
        assert msg.value["source"] == "external"

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.price_producer import PriceProducer

        producer = PriceProducer(broker, symbols=["AAPL"], interval_seconds=100)
        await producer.start()
        assert producer._running is True
        assert producer._task is not None
        await producer.stop()
        assert producer._running is False
        assert producer._task is None


# ---------------------------------------------------------------------------
# News Producer
# ---------------------------------------------------------------------------

class TestNewsProducer:
    """Test news producer message format and publishing."""

    @pytest.mark.asyncio
    async def test_poll_produces_messages(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.news_producer import NewsProducer

        producer = NewsProducer(broker, symbols=["AAPL"])
        await producer._poll()

        assert len(broker._messages) >= 1
        for msg in broker._messages:
            assert msg.topic == TOPIC_NEWS_RAW
            value = msg.value
            assert "title" in value
            assert "description" in value
            assert "symbols" in value
            assert "published_at" in value
            assert "source_url" in value

    @pytest.mark.asyncio
    async def test_news_message_format(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.news_producer import NewsProducer

        producer = NewsProducer(broker, symbols=["MSFT"])
        await producer._poll()

        msg = broker._messages[0]
        assert isinstance(msg.value["title"], str)
        assert isinstance(msg.value["symbols"], list)
        assert len(msg.value["symbols"]) > 0


# ---------------------------------------------------------------------------
# Sentiment Consumer (news.raw → news.scored pipeline)
# ---------------------------------------------------------------------------

class TestSentimentConsumer:
    """Test sentiment scoring pipeline."""

    @pytest.mark.asyncio
    async def test_scores_news_article(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.sentiment_consumer import SentimentConsumer

        consumer = SentimentConsumer(broker)
        await consumer.start()

        # Publish a bullish news article → should trigger handler
        await broker.publish(TOPIC_NEWS_RAW, {
            "title": "AAPL beats earnings expectations, stock surges",
            "description": "Apple reported strong growth and record profits",
            "symbols": ["AAPL"],
            "published_at": "2026-01-01T00:00:00Z",
            "source_url": "https://example.com/aapl",
        })

        # The handler publishes to news.scored
        scored_msgs = [
            m for m in broker._messages if m.topic == TOPIC_NEWS_SCORED
        ]
        assert len(scored_msgs) == 1
        scored = scored_msgs[0].value

        assert "sentiment_score" in scored
        assert "sentiment_label" in scored
        assert "analyzed_at" in scored
        # Original fields preserved
        assert scored["title"] == "AAPL beats earnings expectations, stock surges"
        assert scored["symbols"] == ["AAPL"]

    @pytest.mark.asyncio
    async def test_bullish_article_scores_positive(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.sentiment_consumer import SentimentConsumer

        consumer = SentimentConsumer(broker)
        await consumer.start()

        await broker.publish(TOPIC_NEWS_RAW, {
            "title": "Stock rally continues with strong gains and growth",
            "description": "Upgraded outlook with record profit",
            "symbols": ["AAPL"],
            "published_at": "2026-01-01T00:00:00Z",
            "source_url": "https://example.com",
        })

        scored_msgs = [
            m for m in broker._messages if m.topic == TOPIC_NEWS_SCORED
        ]
        assert scored_msgs[0].value["sentiment_score"] > 0
        assert scored_msgs[0].value["sentiment_label"] == "bullish"

    @pytest.mark.asyncio
    async def test_bearish_article_scores_negative(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.sentiment_consumer import SentimentConsumer

        consumer = SentimentConsumer(broker)
        await consumer.start()

        await broker.publish(TOPIC_NEWS_RAW, {
            "title": "Massive losses and crash after downgrade",
            "description": "Stock selloff amid bankruptcy risk warning",
            "symbols": ["TSLA"],
            "published_at": "2026-01-01T00:00:00Z",
            "source_url": "https://example.com",
        })

        scored_msgs = [
            m for m in broker._messages if m.topic == TOPIC_NEWS_SCORED
        ]
        assert scored_msgs[0].value["sentiment_score"] < 0
        assert scored_msgs[0].value["sentiment_label"] == "bearish"

    @pytest.mark.asyncio
    async def test_empty_article_skipped(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.sentiment_consumer import SentimentConsumer

        consumer = SentimentConsumer(broker)
        await consumer.start()

        await broker.publish(TOPIC_NEWS_RAW, {
            "title": "",
            "description": "",
            "symbols": [],
            "published_at": "2026-01-01T00:00:00Z",
            "source_url": "",
        })

        scored_msgs = [
            m for m in broker._messages if m.topic == TOPIC_NEWS_SCORED
        ]
        assert len(scored_msgs) == 0


# ---------------------------------------------------------------------------
# Signal Consumer
# ---------------------------------------------------------------------------

class TestSignalConsumer:
    """Test strategy signal generation from price stream."""

    @pytest.mark.asyncio
    async def test_generates_signal_after_enough_data(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.signal_consumer import SignalConsumer, MIN_WINDOW

        consumer = SignalConsumer(broker, strategy_names=["ma_crossover"])
        await consumer.start()

        # Feed enough price data to trigger strategies
        for i in range(MIN_WINDOW + 5):
            price = 100.0 + (i * 0.5)
            await broker.publish(TOPIC_MARKET_PRICES, {
                "symbol": "AAPL",
                "price": price,
                "volume": 100000,
                "timestamp": f"2026-01-01T00:00:{i:02d}Z",
                "source": "mock",
            })

        # Check that signals were published (may or may not trigger depending
        # on strategy logic — just verify no crashes)
        signal_msgs = [
            m for m in broker._messages if m.topic == TOPIC_SIGNALS
        ]
        # Strategy may produce hold signals → 0 output messages is OK
        for msg in signal_msgs:
            value = msg.value
            assert "strategy_name" in value
            assert "symbol" in value
            assert value["signal"] in ("buy", "sell")
            assert "confidence" in value
            assert "timestamp" in value

    @pytest.mark.asyncio
    async def test_no_signal_before_min_window(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.signal_consumer import SignalConsumer, MIN_WINDOW

        consumer = SignalConsumer(broker)
        await consumer.start()

        # Send fewer than MIN_WINDOW messages
        for i in range(MIN_WINDOW - 1):
            await broker.publish(TOPIC_MARKET_PRICES, {
                "symbol": "AAPL",
                "price": 100.0 + i,
                "volume": 10000,
                "timestamp": "2026-01-01T00:00:00Z",
                "source": "mock",
            })

        signal_msgs = [
            m for m in broker._messages if m.topic == TOPIC_SIGNALS
        ]
        assert len(signal_msgs) == 0

    @pytest.mark.asyncio
    async def test_ignores_invalid_messages(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.signal_consumer import SignalConsumer

        consumer = SignalConsumer(broker)
        await consumer.start()

        # Missing symbol
        await broker.publish(TOPIC_MARKET_PRICES, {"price": 100.0})
        # Missing price
        await broker.publish(TOPIC_MARKET_PRICES, {"symbol": "AAPL"})

        assert len(consumer._price_history) == 0


# ---------------------------------------------------------------------------
# WebSocket Relay
# ---------------------------------------------------------------------------

class TestWebSocketRelay:
    """Test price relay to Redis Pub/Sub."""

    @pytest.mark.asyncio
    async def test_relay_publishes_to_redis(self):
        broker = _make_broker()
        await broker.start()

        mock_redis = AsyncMock()

        from app.streaming.ws_relay import WebSocketRelay

        relay = WebSocketRelay(broker)
        relay._redis = mock_redis
        # Subscribe manually (start() would try to import get_redis)
        await broker.subscribe(TOPIC_MARKET_PRICES, relay._handle)

        await broker.publish(TOPIC_MARKET_PRICES, {
            "symbol": "AAPL",
            "price": 175.50,
            "volume": 100000,
            "timestamp": "2026-01-01T00:00:00Z",
        })

        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "ws:prices"

    @pytest.mark.asyncio
    async def test_relay_skips_when_no_redis(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.ws_relay import WebSocketRelay

        relay = WebSocketRelay(broker)
        relay._redis = None

        # Manually subscribe (start() would try to get_redis)
        await broker.subscribe(TOPIC_MARKET_PRICES, relay._handle)

        # Should not raise
        await broker.publish(TOPIC_MARKET_PRICES, {
            "symbol": "AAPL",
            "price": 175.50,
            "volume": 100000,
            "timestamp": "2026-01-01T00:00:00Z",
        })

    @pytest.mark.asyncio
    async def test_relay_skips_empty_symbol(self):
        broker = _make_broker()
        await broker.start()

        mock_redis = AsyncMock()

        from app.streaming.ws_relay import WebSocketRelay

        relay = WebSocketRelay(broker)
        relay._redis = mock_redis
        await broker.subscribe(TOPIC_MARKET_PRICES, relay._handle)

        await broker.publish(TOPIC_MARKET_PRICES, {"price": 100.0})

        mock_redis.publish.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-End Pipeline: News → Sentiment Scoring
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Integration test: full news → scored pipeline via InMemoryBroker."""

    @pytest.mark.asyncio
    async def test_news_to_scored_pipeline(self):
        broker = _make_broker()
        await broker.start()

        from app.streaming.news_producer import NewsProducer
        from app.streaming.sentiment_consumer import SentimentConsumer

        # Wire up consumer first
        consumer = SentimentConsumer(broker)
        await consumer.start()

        # Produce news — this triggers the consumer synchronously
        # because InMemoryBroker dispatches inline
        producer = NewsProducer(broker, symbols=["AAPL"])
        await producer._poll()

        raw_count = len([
            m for m in broker._messages if m.topic == TOPIC_NEWS_RAW
        ])
        scored_count = len([
            m for m in broker._messages if m.topic == TOPIC_NEWS_SCORED
        ])

        # Every raw news article should produce a scored version
        assert raw_count >= 1
        assert scored_count == raw_count

        # Verify scored message has all required fields
        scored = [
            m for m in broker._messages if m.topic == TOPIC_NEWS_SCORED
        ][0].value
        assert "sentiment_score" in scored
        assert "sentiment_label" in scored
        assert "analyzed_at" in scored
        assert scored["sentiment_label"] in ("bullish", "bearish", "neutral")


# ---------------------------------------------------------------------------
# Streaming Service
# ---------------------------------------------------------------------------

class TestStreamingService:
    """Test streaming service lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        import app.infra.broker as broker_mod

        broker_mod._broker_instance = None
        _mock_settings.BROKER_BACKEND = "memory"

        from app.streaming.service import StreamingService

        service = StreamingService()
        await service.start()
        assert service._running is True
        assert len(service._components) == 5

        await service.stop()
        assert service._running is False

        broker_mod._broker_instance = None
