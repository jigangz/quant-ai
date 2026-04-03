"""
Scheduled Data Fetcher Function

Fetches latest market data for configured symbols and stores in cache.
Can be triggered by EventBridge schedule or called locally.
"""

from __future__ import annotations

import logging
from typing import Any

from app.cache.market_cache import get_market_cache
from app.core.settings import settings

logger = logging.getLogger(__name__)

# Default symbols to fetch when none specified
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


async def _fetch_prices(symbols: list[str]) -> dict[str, float | None]:
    """Fetch latest prices for symbols via the configured market provider."""
    prices: dict[str, float | None] = {}

    provider_name = settings.MARKET_PROVIDER
    if provider_name == "yahoo":
        try:
            import yfinance as yf

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        prices[symbol] = float(hist["Close"].iloc[-1])
                    else:
                        prices[symbol] = None
                        logger.warning(f"No data returned for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
                    prices[symbol] = None
        except ImportError:
            logger.error("yfinance not installed")
    elif provider_name == "polygon":
        # Polygon requires API key — skip if not configured
        if not settings.POLYGON_API_KEY:
            logger.warning("POLYGON_API_KEY not set, skipping Polygon fetch")
        else:
            try:
                import httpx

                for symbol in symbols:
                    try:
                        resp = httpx.get(
                            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev",
                            params={"apiKey": settings.POLYGON_API_KEY},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        results = resp.json().get("results", [])
                        if results:
                            prices[symbol] = float(results[0]["c"])
                        else:
                            prices[symbol] = None
                    except Exception as e:
                        logger.warning(f"Polygon fetch failed for {symbol}: {e}")
                        prices[symbol] = None
            except ImportError:
                logger.error("httpx not installed")
    else:
        logger.warning(f"Unknown market provider: {provider_name}")

    return prices


# ---------------------------------------------------------------------------
# Function handler
# ---------------------------------------------------------------------------

async def handle_data_fetch(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Data fetcher function handler.

    Payload:
        symbols (list[str]): Ticker symbols to fetch. Defaults to DEFAULT_SYMBOLS.

    Returns:
        fetched (dict): Symbol → price mapping.
        cached (int): Number of prices stored in cache.
    """
    symbols = payload.get("symbols", DEFAULT_SYMBOLS)
    if not symbols:
        symbols = DEFAULT_SYMBOLS

    logger.info(f"Fetching market data for {len(symbols)} symbols")

    prices = await _fetch_prices(symbols)

    # Store successful fetches in cache
    cache = get_market_cache()
    valid_prices = {s: p for s, p in prices.items() if p is not None}

    if valid_prices:
        await cache.set_prices(valid_prices)
        logger.info(f"Cached {len(valid_prices)} prices")

    return {
        "success": True,
        "fetched": {s: p for s, p in prices.items()},
        "cached": len(valid_prices),
        "total": len(symbols),
    }
