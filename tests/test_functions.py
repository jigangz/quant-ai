"""
Tests for Serverless Functions (Phase 2.5 Step 8)

Tests function handlers via LocalRunner, keyword sentiment fallback,
alert trigger evaluation, data fetcher caching, SHAP generator wrapper,
and REST API endpoints.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock settings before any app imports
# ---------------------------------------------------------------------------
_mock_settings = MagicMock()
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

_mock_settings_module = MagicMock()
_mock_settings_module.settings = _mock_settings
sys.modules.setdefault("app.core.settings", _mock_settings_module)

# Mock app.core.logging to avoid 3.10+ syntax issues on older Python
_mock_logging_module = MagicMock()
_mock_logging_module.get_logger = lambda name: __import__("logging").getLogger(name)
_mock_logging_module.setup_logging = MagicMock()
_mock_logging_module.request_id_ctx = MagicMock()
sys.modules.setdefault("app.core.logging", _mock_logging_module)

# Mock redis.asyncio to avoid requiring Redis
_mock_aioredis = MagicMock()
sys.modules.setdefault("redis.asyncio", _mock_aioredis)

# Mock app.cache and submodules to avoid Redis dependency
_mock_cache_module = MagicMock()
_mock_cache_module.get_redis = AsyncMock(return_value=None)
sys.modules.setdefault("app.cache", _mock_cache_module)

_mock_market_cache = MagicMock()
_mock_market_cache.get_market_cache = MagicMock()
sys.modules.setdefault("app.cache.market_cache", _mock_market_cache)

# Mock heavy app dependencies that may not be installed or have 3.10+ syntax
for mod_name in [
    "app.db.prices_repo", "app.ml.features.technical",
    "app.ml.labels.returns", "app.ml.features.build",
    "app.db.model_registry", "app.services.model_cache",
    "app.middleware", "app.providers.base",
    "app.explain.shap_explainer", "app.explain.shap_to_text",
    "yfinance", "joblib", "shap",
]:
    sys.modules.setdefault(mod_name, MagicMock())


# ---------------------------------------------------------------------------
# Sentiment Analysis Tests
# ---------------------------------------------------------------------------


class TestSentimentAnalyzer:
    """Test sentiment analysis function handler."""

    @pytest.mark.asyncio
    async def test_keyword_bullish_sentiment(self):
        from app.functions.sentiment_analyzer import handle_sentiment_analysis

        result = await handle_sentiment_analysis({
            "text": "AAPL beats earnings expectations, stock surged to record highs"
        })

        assert result["success"] is True
        assert result["method"] == "keyword"
        assert result["score"] > 0  # Bullish keywords

    @pytest.mark.asyncio
    async def test_keyword_bearish_sentiment(self):
        from app.functions.sentiment_analyzer import handle_sentiment_analysis

        result = await handle_sentiment_analysis({
            "text": "Company reports massive losses, stock crashed after downgrade warning"
        })

        assert result["success"] is True
        assert result["method"] == "keyword"
        assert result["score"] < 0  # Bearish keywords

    @pytest.mark.asyncio
    async def test_keyword_neutral_sentiment(self):
        from app.functions.sentiment_analyzer import handle_sentiment_analysis

        result = await handle_sentiment_analysis({
            "text": "The company held its annual meeting today"
        })

        assert result["success"] is True
        assert result["method"] == "keyword"
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_missing_text_returns_error(self):
        from app.functions.sentiment_analyzer import handle_sentiment_analysis

        result = await handle_sentiment_analysis({})
        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self):
        from app.functions.sentiment_analyzer import handle_sentiment_analysis

        result = await handle_sentiment_analysis({"text": ""})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Alert Trigger Tests
# ---------------------------------------------------------------------------


class TestAlertTrigger:
    """Test alert trigger function handler."""

    @pytest.fixture(autouse=True)
    def setup_notifier(self):
        import app.infra.notify as notify_mod

        notify_mod._notifier_instance = None
        _mock_settings.NOTIFY_BACKEND = "memory"
        self.notifier = notify_mod.get_notifier()
        yield
        notify_mod._notifier_instance = None

    @pytest.mark.asyncio
    async def test_price_breakout_above(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "AAPL",
            "price": 185.0,
            "rules": [
                {"type": "price_breakout", "threshold": 180.0, "direction": "above"}
            ],
        })

        assert result["success"] is True
        assert len(result["triggered"]) == 1
        assert result["triggered"][0]["type"] == "price_breakout"
        assert result["triggered"][0]["direction"] == "above"

    @pytest.mark.asyncio
    async def test_price_breakout_below(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "AAPL",
            "price": 170.0,
            "rules": [
                {"type": "price_breakout", "threshold": 175.0, "direction": "below"}
            ],
        })

        assert result["success"] is True
        assert len(result["triggered"]) == 1
        assert result["triggered"][0]["direction"] == "below"

    @pytest.mark.asyncio
    async def test_price_breakout_not_triggered(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "AAPL",
            "price": 175.0,
            "rules": [
                {"type": "price_breakout", "threshold": 180.0, "direction": "above"}
            ],
        })

        assert result["success"] is True
        assert len(result["triggered"]) == 0

    @pytest.mark.asyncio
    async def test_rsi_overbought(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "TSLA",
            "price": 200.0,
            "rsi": 75.0,
            "rules": [
                {"type": "rsi_threshold", "overbought": 70, "oversold": 30}
            ],
        })

        assert result["success"] is True
        assert len(result["triggered"]) == 1
        assert result["triggered"][0]["condition"] == "overbought"

    @pytest.mark.asyncio
    async def test_rsi_oversold(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "TSLA",
            "price": 150.0,
            "rsi": 25.0,
            "rules": [
                {"type": "rsi_threshold", "overbought": 70, "oversold": 30}
            ],
        })

        assert result["success"] is True
        assert len(result["triggered"]) == 1
        assert result["triggered"][0]["condition"] == "oversold"

    @pytest.mark.asyncio
    async def test_rsi_normal_not_triggered(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "TSLA",
            "price": 180.0,
            "rsi": 50.0,
            "rules": [
                {"type": "rsi_threshold", "overbought": 70, "oversold": 30}
            ],
        })

        assert result["success"] is True
        assert len(result["triggered"]) == 0

    @pytest.mark.asyncio
    async def test_macd_bullish_crossover(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "MSFT",
            "price": 350.0,
            "macd": 1.5,
            "signal": 1.2,
            "prev_macd": 1.0,
            "prev_signal": 1.3,
            "rules": [{"type": "macd_crossover"}],
        })

        assert result["success"] is True
        assert len(result["triggered"]) == 1
        assert result["triggered"][0]["crossover_type"] == "bullish"

    @pytest.mark.asyncio
    async def test_macd_bearish_crossover(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "MSFT",
            "price": 340.0,
            "macd": 1.0,
            "signal": 1.3,
            "prev_macd": 1.5,
            "prev_signal": 1.2,
            "rules": [{"type": "macd_crossover"}],
        })

        assert result["success"] is True
        assert len(result["triggered"]) == 1
        assert result["triggered"][0]["crossover_type"] == "bearish"

    @pytest.mark.asyncio
    async def test_multiple_rules_evaluated(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "AAPL",
            "price": 185.0,
            "rsi": 75.0,
            "rules": [
                {"type": "price_breakout", "threshold": 180.0, "direction": "above"},
                {"type": "rsi_threshold", "overbought": 70, "oversold": 30},
            ],
        })

        assert result["success"] is True
        assert result["evaluated"] == 2
        assert len(result["triggered"]) == 2

    @pytest.mark.asyncio
    async def test_missing_symbol_returns_error(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({"price": 100.0, "rules": []})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_alerts_published_to_notifier(self):
        from app.functions.alert_trigger import handle_alert_trigger

        result = await handle_alert_trigger({
            "symbol": "AAPL",
            "price": 185.0,
            "rules": [
                {"type": "price_breakout", "threshold": 180.0, "direction": "above"}
            ],
        })

        assert result["success"] is True
        # InMemoryNotifier should have the alert in history
        assert len(self.notifier._history) == 1
        assert self.notifier._history[0].payload["type"] == "price_breakout"


# ---------------------------------------------------------------------------
# Data Fetcher Tests
# ---------------------------------------------------------------------------


class TestDataFetcher:
    """Test data fetcher function handler."""

    @pytest.mark.asyncio
    async def test_fetch_with_mock_provider(self):
        from app.functions.data_fetcher import handle_data_fetch

        mock_cache = MagicMock()
        mock_cache.set_prices = AsyncMock()

        with patch("app.functions.data_fetcher.get_market_cache", return_value=mock_cache), \
             patch("app.functions.data_fetcher._fetch_prices", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"AAPL": 185.0, "MSFT": 350.0}

            result = await handle_data_fetch({"symbols": ["AAPL", "MSFT"]})

        assert result["success"] is True
        assert result["cached"] == 2
        assert result["fetched"]["AAPL"] == 185.0
        mock_cache.set_prices.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_handles_partial_failures(self):
        from app.functions.data_fetcher import handle_data_fetch

        mock_cache = MagicMock()
        mock_cache.set_prices = AsyncMock()

        with patch("app.functions.data_fetcher.get_market_cache", return_value=mock_cache), \
             patch("app.functions.data_fetcher._fetch_prices", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"AAPL": 185.0, "INVALID": None}

            result = await handle_data_fetch({"symbols": ["AAPL", "INVALID"]})

        assert result["success"] is True
        assert result["cached"] == 1  # Only AAPL cached
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_fetch_uses_default_symbols(self):
        from app.functions.data_fetcher import handle_data_fetch, DEFAULT_SYMBOLS

        mock_cache = MagicMock()
        mock_cache.set_prices = AsyncMock()

        with patch("app.functions.data_fetcher.get_market_cache", return_value=mock_cache), \
             patch("app.functions.data_fetcher._fetch_prices", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {s: 100.0 for s in DEFAULT_SYMBOLS}

            result = await handle_data_fetch({})

        assert result["total"] == len(DEFAULT_SYMBOLS)
        mock_fetch.assert_called_once_with(DEFAULT_SYMBOLS)

    @pytest.mark.asyncio
    async def test_fetch_no_valid_prices_skips_cache(self):
        from app.functions.data_fetcher import handle_data_fetch

        mock_cache = MagicMock()
        mock_cache.set_prices = AsyncMock()

        with patch("app.functions.data_fetcher.get_market_cache", return_value=mock_cache), \
             patch("app.functions.data_fetcher._fetch_prices", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"AAPL": None}

            result = await handle_data_fetch({"symbols": ["AAPL"]})

        assert result["cached"] == 0
        mock_cache.set_prices.assert_not_called()


# ---------------------------------------------------------------------------
# SHAP Generator Tests
# ---------------------------------------------------------------------------


class TestShapGenerator:
    """Test SHAP generator function handler."""

    @pytest.mark.asyncio
    async def test_missing_ticker_returns_error(self):
        from app.functions.shap_generator import handle_shap_generation

        result = await handle_shap_generation({})
        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_wraps_shap_explainer(self):
        from app.functions.shap_generator import handle_shap_generation

        mock_result = {
            "success": True,
            "ticker": "AAPL",
            "top_features": [{"feature": "rsi_14", "shap_value": 0.5}],
        }

        with patch("app.functions.shap_generator.ShapExplainer") as MockExplainer:
            instance = MockExplainer.return_value
            instance.explain.return_value = mock_result

            result = await handle_shap_generation({
                "ticker": "AAPL",
                "model_id": "test-model",
                "lookback": 200,
                "top_k": 5,
            })

        assert result["success"] is True
        assert result["ticker"] == "AAPL"
        instance.explain.assert_called_once_with(
            ticker="AAPL",
            model_id="test-model",
            lookback=200,
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_default_parameters(self):
        from app.functions.shap_generator import handle_shap_generation

        with patch("app.functions.shap_generator.ShapExplainer") as MockExplainer:
            instance = MockExplainer.return_value
            instance.explain.return_value = {"success": True, "ticker": "MSFT"}

            await handle_shap_generation({"ticker": "MSFT"})

        instance.explain.assert_called_once_with(
            ticker="MSFT",
            model_id=None,
            lookback=500,
            top_k=10,
        )


# ---------------------------------------------------------------------------
# LocalRunner Integration Tests
# ---------------------------------------------------------------------------


class TestLocalRunnerIntegration:
    """Test functions registered and invoked via LocalRunner."""

    @pytest.fixture(autouse=True)
    def setup_runner(self):
        import app.infra.functions as func_mod
        import app.infra.notify as notify_mod

        func_mod._runner_instance = None
        notify_mod._notifier_instance = None
        _mock_settings.FUNCTIONS_BACKEND = "local"
        _mock_settings.NOTIFY_BACKEND = "memory"

        # Register functions
        from app.functions import register_all_functions
        register_all_functions()

        self.runner = func_mod.get_function_runner()
        yield
        func_mod._runner_instance = None
        notify_mod._notifier_instance = None

    @pytest.mark.asyncio
    async def test_invoke_sentiment_analyzer(self):
        result = await self.runner.invoke("sentiment-analyzer", {
            "text": "Stock rally continues with strong gains"
        })
        assert result["success"] is True
        assert result["score"] > 0

    @pytest.mark.asyncio
    async def test_invoke_alert_trigger(self):
        result = await self.runner.invoke("alert-trigger", {
            "symbol": "AAPL",
            "price": 185.0,
            "rules": [
                {"type": "price_breakout", "threshold": 180.0, "direction": "above"}
            ],
        })
        assert result["success"] is True
        assert len(result["triggered"]) == 1

    @pytest.mark.asyncio
    async def test_invoke_data_fetcher(self):
        with patch("app.functions.data_fetcher.get_market_cache") as mock_cache_fn, \
             patch("app.functions.data_fetcher._fetch_prices", new_callable=AsyncMock) as mock_fetch:
            mock_cache = MagicMock()
            mock_cache.set_prices = AsyncMock()
            mock_cache_fn.return_value = mock_cache
            mock_fetch.return_value = {"AAPL": 185.0}

            result = await self.runner.invoke("data-fetcher", {
                "symbols": ["AAPL"]
            })

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_invoke_shap_generator(self):
        with patch("app.functions.shap_generator.ShapExplainer") as MockExplainer:
            instance = MockExplainer.return_value
            instance.explain.return_value = {"success": True, "ticker": "AAPL"}

            result = await self.runner.invoke("shap-generator", {
                "ticker": "AAPL"
            })

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_invoke_unknown_function_raises(self):
        with pytest.raises(ValueError, match="No handler registered"):
            await self.runner.invoke("nonexistent-function", {})


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------


def _build_functions_test_client():
    """Build a test client with a minimal functions router."""
    from typing import Any, Dict
    from fastapi.testclient import TestClient
    from fastapi import FastAPI, APIRouter, HTTPException
    from pydantic import BaseModel
    from app.functions import FUNCTIONS
    from app.infra.functions import get_function_runner

    router = APIRouter(prefix="/api/v1/functions")

    class _InvokeRequest(BaseModel):
        payload: Dict[str, Any] = {}
        async_invoke: bool = False

    @router.get("")
    async def list_functions():
        return {
            "functions": [
                {"name": n, "handler": h.__module__ + "." + h.__name__}
                for n, h in FUNCTIONS.items()
            ],
        }

    @router.post("/{name}/invoke")
    async def invoke_function(name: str, body: _InvokeRequest):
        if name not in FUNCTIONS:
            raise HTTPException(status_code=404, detail=f"Function not found: {name}")
        runner = get_function_runner()
        result = await runner.invoke(name, body.payload, async_invoke=body.async_invoke)
        return {"function": name, "result": result, "async_invoke": body.async_invoke}

    test_app = FastAPI()
    test_app.include_router(router)
    return TestClient(test_app)


class TestFunctionsAPI:
    """Test functions REST API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import app.infra.functions as func_mod
        import app.infra.notify as notify_mod

        func_mod._runner_instance = None
        notify_mod._notifier_instance = None
        _mock_settings.FUNCTIONS_BACKEND = "local"
        _mock_settings.NOTIFY_BACKEND = "memory"

        from app.functions import register_all_functions
        register_all_functions()
        yield
        func_mod._runner_instance = None
        notify_mod._notifier_instance = None

    @pytest.fixture
    def client(self):
        return _build_functions_test_client()

    def test_list_functions(self, client):
        resp = client.get("/api/v1/functions")
        assert resp.status_code == 200
        data = resp.json()
        names = [f["name"] for f in data["functions"]]
        assert "sentiment-analyzer" in names
        assert "alert-trigger" in names
        assert "data-fetcher" in names
        assert "shap-generator" in names

    def test_invoke_sentiment_function(self, client):
        resp = client.post("/api/v1/functions/sentiment-analyzer/invoke", json={
            "payload": {"text": "Stock surged on strong earnings beat"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["function"] == "sentiment-analyzer"
        assert data["result"]["success"] is True
        assert data["result"]["score"] > 0

    def test_invoke_unknown_function_404(self, client):
        resp = client.post("/api/v1/functions/nonexistent/invoke", json={
            "payload": {},
        })
        assert resp.status_code == 404

    def test_invoke_with_empty_payload(self, client):
        resp = client.post("/api/v1/functions/sentiment-analyzer/invoke", json={
            "payload": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["success"] is False


# ---------------------------------------------------------------------------
# Keyword Sentiment Scorer Unit Tests
# ---------------------------------------------------------------------------


class TestKeywordSentiment:
    """Test the keyword-based sentiment fallback scorer."""

    def test_pure_bullish(self):
        from app.functions.sentiment_analyzer import _keyword_sentiment

        score = _keyword_sentiment("strong growth and record profits")
        assert score > 0

    def test_pure_bearish(self):
        from app.functions.sentiment_analyzer import _keyword_sentiment

        score = _keyword_sentiment("massive losses and bankruptcy risk")
        assert score < 0

    def test_neutral_no_keywords(self):
        from app.functions.sentiment_analyzer import _keyword_sentiment

        score = _keyword_sentiment("the sky is blue today")
        assert score == 0.0

    def test_mixed_sentiment(self):
        from app.functions.sentiment_analyzer import _keyword_sentiment

        score = _keyword_sentiment("strong growth but high risk of losses")
        # Mixed — score should be between -1 and 1
        assert -1 <= score <= 1
