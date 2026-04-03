"""
Tests for Redis Cache Layer (Phase 2.5 Step 6)

Tests all 5 cache modules:
- 6.1 Market Price Cache
- 6.2 Trading Session Cache
- 6.3 Distributed Rate Limiting
- 6.4 WebSocket Pub/Sub
- 6.5 Model Artifact Cache

Uses mocked Redis to avoid requiring a running Redis instance.
Avoids importing modules that chain to settings.py for Python 3.9 compat.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import sys
import time
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers: mock out settings import chain for Python 3.9 compat
# ---------------------------------------------------------------------------

def _ensure_settings_mock():
    """
    Ensure app.core.settings is importable on Python <3.10.
    If it's already imported, skip. Otherwise inject a mock.
    """
    if "app.core.settings" in sys.modules:
        return
    try:
        from app.core.settings import settings  # noqa
    except TypeError:
        # Python 3.9: str | None syntax fails.  Inject a minimal mock.
        mock_settings = MagicMock()
        mock_settings.REDIS_URL = "redis://localhost:6379/0"
        mock_settings.ENV = "test"
        mock_settings.DEBUG = False
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.LOG_FORMAT = "text"
        mock_settings.RATE_LIMIT_PER_MINUTE = 60
        mock_mod = MagicMock()
        mock_mod.settings = mock_settings
        sys.modules.setdefault("app.core.settings", mock_mod)

        # Also mock logging module to avoid secondary import error
        try:
            from app.core.logging import get_logger  # noqa
        except Exception:
            import logging

            mock_log_mod = MagicMock()
            mock_log_mod.get_logger = logging.getLogger
            sys.modules.setdefault("app.core.logging", mock_log_mod)


_ensure_settings_mock()


# ---------------------------------------------------------------------------
# 6.1  Market Price Cache
# ---------------------------------------------------------------------------

class TestMarketPriceCache:
    """Tests for MarketPriceCache."""

    @pytest.fixture
    def cache(self):
        from app.cache.market_cache import MarketPriceCache
        return MarketPriceCache(ttl=5)

    @pytest.mark.asyncio
    async def test_set_and_get_price_memory_fallback(self, cache):
        """When Redis is unavailable, should use in-memory cache."""
        with patch("app.cache.market_cache.get_redis", new_callable=AsyncMock, return_value=None):
            await cache.set_price("AAPL", 175.50)
            price = await cache.get_price("AAPL")
            assert price == 175.50

    @pytest.mark.asyncio
    async def test_get_price_returns_none_when_not_cached(self, cache):
        with patch("app.cache.market_cache.get_redis", new_callable=AsyncMock, return_value=None):
            price = await cache.get_price("AAPL")
            assert price is None

    @pytest.mark.asyncio
    async def test_expired_price_returns_none(self, cache):
        """Expired cache entries should return None."""
        with patch("app.cache.market_cache.get_redis", new_callable=AsyncMock, return_value=None):
            await cache.set_price("AAPL", 175.50)
            # Force expiry
            cache._mem_cache["AAPL"] = (175.50, time.time() - 10)
            price = await cache.get_price("AAPL")
            assert price is None

    @pytest.mark.asyncio
    async def test_set_prices_bulk(self, cache):
        with patch("app.cache.market_cache.get_redis", new_callable=AsyncMock, return_value=None):
            await cache.set_prices({"AAPL": 175.50, "GOOGL": 140.25})
            assert await cache.get_price("AAPL") == 175.50
            assert await cache.get_price("GOOGL") == 140.25

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        with patch("app.cache.market_cache.get_redis", new_callable=AsyncMock, return_value=None):
            await cache.set_price("AAPL", 175.50)
            await cache.invalidate("AAPL")
            assert await cache.get_price("AAPL") is None

    @pytest.mark.asyncio
    async def test_symbol_case_insensitive(self, cache):
        with patch("app.cache.market_cache.get_redis", new_callable=AsyncMock, return_value=None):
            await cache.set_price("aapl", 175.50)
            assert await cache.get_price("AAPL") == 175.50

    @pytest.mark.asyncio
    async def test_redis_backend(self, cache):
        """Test with a mocked Redis client."""
        mock_redis = AsyncMock()
        mock_pipe = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_pipe.execute = AsyncMock(return_value=[True, True, True, True])

        now_str = str(time.time()).encode()
        mock_redis.hget = AsyncMock(side_effect=lambda key, field: {
            ("cache:market:prices", "AAPL"): b"175.50",
            ("cache:market:prices:ts", "AAPL"): now_str,
        }.get((key, field)))

        with patch("app.cache.market_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
            await cache.set_price("AAPL", 175.50)
            price = await cache.get_price("AAPL")
            assert price == 175.50


# ---------------------------------------------------------------------------
# 6.2  Trading Session Cache
# ---------------------------------------------------------------------------

class TestTradingSessionCache:
    """Tests for TradingSessionCache."""

    @pytest.fixture
    def cache(self):
        from app.cache.trading_cache import TradingSessionCache
        return TradingSessionCache()

    @pytest.mark.asyncio
    async def test_save_portfolio_state_with_redis(self, cache):
        mock_redis = AsyncMock()
        mock_pipe = MagicMock()
        mock_pipe.set = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[True, True])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)

        with patch("app.cache.trading_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
            positions = {"AAPL": {"quantity": 10, "avg_cost": 150.0, "realized_pnl": 0.0}}
            result = await cache.save_portfolio_state(
                cash=85000.0,
                starting_cash=100000.0,
                total_realized_pnl=500.0,
                positions=positions,
            )
            assert result is True
            assert mock_pipe.set.call_count == 2

    @pytest.mark.asyncio
    async def test_save_returns_false_when_redis_unavailable(self, cache):
        with patch("app.cache.trading_cache.get_redis", new_callable=AsyncMock, return_value=None):
            result = await cache.save_portfolio_state(
                cash=85000.0, starting_cash=100000.0,
                total_realized_pnl=0.0, positions={},
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_load_returns_none_when_redis_unavailable(self, cache):
        with patch("app.cache.trading_cache.get_redis", new_callable=AsyncMock, return_value=None):
            state = await cache.load_portfolio_state()
            assert state is None

    @pytest.mark.asyncio
    async def test_save_and_load_trades(self, cache):
        mock_redis = AsyncMock()
        stored = {}

        async def mock_set(key, value):
            stored[key] = value
            return True

        async def mock_get(key):
            return stored.get(key)

        mock_redis.set = AsyncMock(side_effect=mock_set)
        mock_redis.get = AsyncMock(side_effect=mock_get)

        with patch("app.cache.trading_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
            trades = [{"symbol": "AAPL", "quantity": 10, "price": 150.0}]
            assert await cache.save_trades(trades) is True
            loaded = await cache.load_trades()
            assert loaded == trades

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=4)

        with patch("app.cache.trading_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
            result = await cache.clear()
            assert result is True
            mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_and_load_history(self, cache):
        mock_redis = AsyncMock()
        stored = {}

        async def mock_set(key, value):
            stored[key] = value
            return True

        async def mock_get(key):
            return stored.get(key)

        mock_redis.set = AsyncMock(side_effect=mock_set)
        mock_redis.get = AsyncMock(side_effect=mock_get)

        with patch("app.cache.trading_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
            history = [{"timestamp": "2024-01-01", "cash": 100000.0}]
            assert await cache.save_history(history) is True
            loaded = await cache.load_history()
            assert loaded == history


# ---------------------------------------------------------------------------
# 6.3  Distributed Rate Limiting
# ---------------------------------------------------------------------------

class TestDistributedRateLimiter:
    """Tests for Redis-backed rate limiter."""

    def _make_middleware(self, **kwargs):
        from app.middleware.rate_limit import RateLimitMiddleware
        app_mock = MagicMock()
        return RateLimitMiddleware(app_mock, **kwargs)

    def test_memory_fallback_rate_limiting(self):
        middleware = self._make_middleware(requests_per_minute=5)

        for _ in range(5):
            is_limited, remaining, _ = middleware._is_rate_limited_memory("1.2.3.4")
            if not is_limited:
                middleware._requests["1.2.3.4"].append(time.time())

        is_limited, remaining, _ = middleware._is_rate_limited_memory("1.2.3.4")
        assert is_limited is True
        assert remaining == 0

    def test_different_ips_independent(self):
        middleware = self._make_middleware(requests_per_minute=2)

        middleware._requests["1.1.1.1"].append(time.time())
        middleware._requests["1.1.1.1"].append(time.time())

        is_limited, _, _ = middleware._is_rate_limited_memory("1.1.1.1")
        assert is_limited is True

        is_limited, _, _ = middleware._is_rate_limited_memory("2.2.2.2")
        assert is_limited is False

    def test_client_ip_extraction(self):
        middleware = self._make_middleware()

        # X-Forwarded-For
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "10.0.0.1, 10.0.0.2"}
        assert middleware._get_client_ip(request) == "10.0.0.1"

        # X-Real-IP
        request.headers = {"X-Real-IP": "10.0.0.3"}
        assert middleware._get_client_ip(request) == "10.0.0.3"

        # Direct connection
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        assert middleware._get_client_ip(request) == "127.0.0.1"

    @pytest.mark.asyncio
    async def test_redis_rate_limiting(self):
        middleware = self._make_middleware(requests_per_minute=3)

        mock_redis = AsyncMock()
        mock_pipe = MagicMock()
        mock_pipe.zremrangebyscore = MagicMock()
        mock_pipe.zcard = MagicMock()
        mock_pipe.zadd = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 2, True, True])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        mock_redis.zrange = AsyncMock(return_value=[(b"ts", time.time())])

        is_limited, remaining, _ = await middleware._is_rate_limited_redis("1.2.3.4", mock_redis)
        assert is_limited is False
        assert remaining == 1

    @pytest.mark.asyncio
    async def test_redis_rate_limited_when_exceeded(self):
        middleware = self._make_middleware(requests_per_minute=3)

        mock_redis = AsyncMock()
        mock_pipe = MagicMock()
        mock_pipe.zremrangebyscore = MagicMock()
        mock_pipe.zcard = MagicMock()
        mock_pipe.zadd = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 5, True, True])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        mock_redis.zrange = AsyncMock(return_value=[(b"ts", time.time() - 30)])

        is_limited, remaining, _ = await middleware._is_rate_limited_redis("1.2.3.4", mock_redis)
        assert is_limited is True
        assert remaining == 0

    def test_window_expiry_clears_old_requests(self):
        middleware = self._make_middleware(requests_per_minute=2)

        # Add old requests (61 seconds ago)
        old_time = time.time() - 61
        middleware._requests["1.1.1.1"] = [old_time, old_time]

        # Should not be limited since old requests are outside window
        is_limited, remaining, _ = middleware._is_rate_limited_memory("1.1.1.1")
        assert is_limited is False
        assert remaining == 2


# ---------------------------------------------------------------------------
# 6.4  WebSocket Pub/Sub
# ---------------------------------------------------------------------------

class TestWebSocketPubSub:
    """Tests for Redis Pub/Sub integration in WebSocket streaming."""

    def test_price_generator_tick(self):
        from app.trading.websocket import PriceGenerator

        gen = PriceGenerator()
        updates = gen.tick(["AAPL", "GOOGL"])
        assert "AAPL" in updates
        assert "GOOGL" in updates
        assert updates["AAPL"].price > 0
        assert updates["AAPL"].symbol == "AAPL"

    def test_price_generator_unknown_symbol(self):
        from app.trading.websocket import PriceGenerator

        gen = PriceGenerator()
        price = gen.get_price("XYZ123")
        assert price > 0

    @pytest.mark.asyncio
    async def test_connection_manager_subscribe(self):
        from app.trading.websocket import ConnectionManager

        manager = ConnectionManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()

        await manager.connect(ws)
        manager.subscribe(ws, ["AAPL", "GOOGL"])

        symbols = manager.get_all_symbols()
        assert "AAPL" in symbols
        assert "GOOGL" in symbols

    @pytest.mark.asyncio
    async def test_connection_manager_broadcast(self):
        from app.trading.websocket import ConnectionManager, PriceGenerator

        manager = ConnectionManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()

        await manager.connect(ws)
        manager.subscribe(ws, ["AAPL"])

        gen = PriceGenerator()
        updates = gen.tick(["AAPL", "GOOGL"])
        await manager.broadcast(updates)

        ws.send_json.assert_called_once()
        sent = ws.send_json.call_args[0][0]
        assert "AAPL" in sent["data"]
        assert "GOOGL" not in sent["data"]

    @pytest.mark.asyncio
    async def test_broadcast_raw(self):
        from app.trading.websocket import ConnectionManager

        manager = ConnectionManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()

        await manager.connect(ws)
        manager.subscribe(ws, ["AAPL"])

        message = {
            "data": {
                "AAPL": {"price": 175.50, "symbol": "AAPL"},
                "GOOGL": {"price": 140.25, "symbol": "GOOGL"},
            }
        }
        await manager.broadcast_raw(message)

        ws.send_json.assert_called_once()
        sent = ws.send_json.call_args[0][0]
        assert "AAPL" in sent["data"]
        assert "GOOGL" not in sent["data"]

    @pytest.mark.asyncio
    async def test_publish_to_redis(self):
        from app.trading.websocket import PriceStreamer, PriceGenerator, ConnectionManager, PRICE_CHANNEL

        manager = ConnectionManager()
        gen = PriceGenerator()
        streamer = PriceStreamer(manager, gen)

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)

        with patch.object(streamer, "_get_publish_redis", new_callable=AsyncMock, return_value=mock_redis):
            updates = gen.tick(["AAPL"])
            result = await streamer._publish_to_redis(updates)
            assert result is True
            mock_redis.publish.assert_called_once()
            call_args = mock_redis.publish.call_args
            assert call_args[0][0] == PRICE_CHANNEL

    @pytest.mark.asyncio
    async def test_publish_fallback_when_redis_unavailable(self):
        from app.trading.websocket import PriceStreamer, PriceGenerator, ConnectionManager

        manager = ConnectionManager()
        gen = PriceGenerator()
        streamer = PriceStreamer(manager, gen)

        with patch.object(streamer, "_get_publish_redis", new_callable=AsyncMock, return_value=None):
            updates = gen.tick(["AAPL"])
            result = await streamer._publish_to_redis(updates)
            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_cleanup(self):
        from app.trading.websocket import ConnectionManager

        manager = ConnectionManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()

        await manager.connect(ws)
        assert manager.connection_count == 1

        manager.disconnect(ws)
        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        from app.trading.websocket import ConnectionManager

        manager = ConnectionManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()

        await manager.connect(ws)
        manager.subscribe(ws, ["AAPL", "GOOGL"])
        manager.unsubscribe(ws, ["GOOGL"])

        symbols = manager.get_all_symbols()
        assert "AAPL" in symbols
        assert "GOOGL" not in symbols


# ---------------------------------------------------------------------------
# 6.5  Model Artifact Cache
# ---------------------------------------------------------------------------

class TestModelArtifactCache:
    """Tests for ModelArtifactCache."""

    @pytest.fixture
    def cache(self):
        from app.cache.model_cache import ModelArtifactCache
        return ModelArtifactCache(ttl=3600, max_mem_entries=5)

    @pytest.mark.asyncio
    async def test_set_and_get_memory_fallback(self, cache):
        with patch("app.cache.model_cache.get_redis", new_callable=AsyncMock, return_value=None):
            model = {"type": "logistic", "weights": [1.0, 2.0, 3.0]}
            await cache.set("/models/v1", model)
            loaded = await cache.get("/models/v1")
            assert loaded == model

    @pytest.mark.asyncio
    async def test_get_returns_none_when_not_cached(self, cache):
        with patch("app.cache.model_cache.get_redis", new_callable=AsyncMock, return_value=None):
            result = await cache.get("/models/nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        with patch("app.cache.model_cache.get_redis", new_callable=AsyncMock, return_value=None):
            await cache.set("/models/v1", {"weights": [1, 2]})
            await cache.invalidate("/models/v1")
            assert await cache.get("/models/v1") is None

    @pytest.mark.asyncio
    async def test_memory_eviction(self, cache):
        """When max entries exceeded, oldest should be evicted."""
        cache._max_mem_entries = 3
        with patch("app.cache.model_cache.get_redis", new_callable=AsyncMock, return_value=None):
            await cache.set("m1", "model1")
            await cache.set("m2", "model2")
            await cache.set("m3", "model3")
            await cache.set("m4", "model4")  # should evict m1

            assert await cache.get("m1") is None
            assert await cache.get("m4") == "model4"

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        with patch("app.cache.model_cache.get_redis", new_callable=AsyncMock, return_value=None):
            await cache.set("m1", "model1")
            await cache.clear()
            assert await cache.get("m1") is None

    @pytest.mark.asyncio
    async def test_redis_backend(self, cache):
        mock_redis = AsyncMock()
        model = {"type": "rf", "weights": [1, 2, 3]}
        pickled = pickle.dumps(model)

        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=pickled)

        with patch("app.cache.model_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
            await cache.set("/models/v1", model)
            mock_redis.set.assert_called_once()

            loaded = await cache.get("/models/v1")
            assert loaded == model

    @pytest.mark.asyncio
    async def test_cache_key_deterministic(self, cache):
        key1 = cache._cache_key("/models/v1")
        key2 = cache._cache_key("/models/v1")
        key3 = cache._cache_key("/models/v2")
        assert key1 == key2
        assert key1 != key3


# ---------------------------------------------------------------------------
# Shared Redis connection helper
# ---------------------------------------------------------------------------

class TestRedisConnection:
    """Tests for the shared async Redis connection helper."""

    @pytest.mark.asyncio
    async def test_get_redis_returns_none_when_unavailable(self):
        import app.cache as cache_module
        cache_module._redis_client = None

        with patch("app.cache.aioredis.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(side_effect=ConnectionError("refused"))
            mock_from_url.return_value = mock_client

            result = await cache_module.get_redis()
            assert result is None

    @pytest.mark.asyncio
    async def test_close_redis(self):
        import app.cache as cache_module

        mock_client = AsyncMock()
        cache_module._redis_client = mock_client

        await cache_module.close_redis()
        mock_client.aclose.assert_called_once()
        assert cache_module._redis_client is None

    @pytest.mark.asyncio
    async def test_get_redis_resets_on_lost_connection(self):
        import app.cache as cache_module

        mock_client = AsyncMock()
        # First ping succeeds (cached), second fails
        mock_client.ping = AsyncMock(side_effect=ConnectionError("lost"))
        cache_module._redis_client = mock_client

        result = await cache_module.get_redis()
        assert result is None
        assert cache_module._redis_client is None
