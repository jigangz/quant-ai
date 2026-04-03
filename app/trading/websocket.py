from __future__ import annotations

"""
WebSocket Price Streaming with Redis Pub/Sub

Real-time price updates via WebSocket for paper trading.
Includes mock price generator for demo mode.

Publishes price updates to Redis Pub/Sub so multiple API instances
can share the same price stream for horizontal scaling.
"""

import asyncio
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Set

from fastapi import WebSocket

from app.core.logging import get_logger
from app.trading.models import PriceUpdate

logger = get_logger(__name__)

# Redis Pub/Sub channel for price updates
PRICE_CHANNEL = "ws:prices"


class PriceGenerator:
    """
    Mock price generator for demo mode.

    Generates realistic-looking price movements based on random walk
    with mean reversion tendencies.
    """

    # Base prices for common symbols
    BASE_PRICES = {
        "AAPL": 175.50,
        "GOOGL": 140.25,
        "MSFT": 375.00,
        "AMZN": 155.75,
        "META": 350.00,
        "NVDA": 480.50,
        "TSLA": 245.00,
        "SPY": 475.25,
        "QQQ": 400.00,
        "IWM": 200.00,
    }

    def __init__(self):
        self._prices: Dict[str, float] = dict(self.BASE_PRICES)
        self._prev_prices: Dict[str, float] = dict(self.BASE_PRICES)
        self._volumes: Dict[str, int] = {}

    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        symbol = symbol.upper()
        if symbol not in self._prices:
            # Generate initial price for unknown symbol
            base = 50 + (hash(symbol) % 450)
            self._prices[symbol] = float(base)
            self._prev_prices[symbol] = float(base)
        return self._prices[symbol]

    def tick(self, symbols: Optional[List[str]] = None) -> Dict[str, PriceUpdate]:
        """
        Generate a price tick for specified symbols.

        Args:
            symbols: Symbols to update. If None, updates all known symbols.

        Returns:
            Dict of symbol to PriceUpdate.
        """
        if symbols is None:
            symbols = list(self._prices.keys())

        updates: Dict[str, PriceUpdate] = {}

        for symbol in symbols:
            symbol = symbol.upper()

            # Ensure symbol exists
            if symbol not in self._prices:
                self.get_price(symbol)

            prev_price = self._prices[symbol]

            # Random walk with mean reversion
            # Volatility ~0.1-0.3% per tick
            volatility = 0.001 + random.random() * 0.002
            change_pct = random.gauss(0, volatility)

            # Mean reversion toward base price (if known)
            base = self.BASE_PRICES.get(symbol, prev_price)
            reversion = (base - prev_price) / base * 0.05
            change_pct += reversion

            # Calculate new price
            new_price = prev_price * (1 + change_pct)
            new_price = max(0.01, new_price)  # Floor at $0.01
            new_price = round(new_price, 2)

            # Calculate change
            price_change = new_price - prev_price
            pct_change = (price_change / prev_price * 100) if prev_price > 0 else 0

            # Simulate volume
            base_volume = 100000 + (hash(symbol) % 900000)
            volume = int(base_volume * (0.8 + random.random() * 0.4))

            # Store new price
            self._prev_prices[symbol] = prev_price
            self._prices[symbol] = new_price
            self._volumes[symbol] = self._volumes.get(symbol, 0) + volume

            updates[symbol] = PriceUpdate(
                symbol=symbol,
                price=new_price,
                change=round(price_change, 2),
                change_pct=round(pct_change, 4),
                volume=self._volumes[symbol],
                timestamp=datetime.utcnow(),
            )

        return updates


class ConnectionManager:
    """
    Manages WebSocket connections for price streaming.

    Handles connection lifecycle and message broadcasting.
    """

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self._connections.add(websocket)
        self._subscriptions[websocket] = set()
        logger.info(f"WebSocket connected: {len(self._connections)} total")

    def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection."""
        self._connections.discard(websocket)
        self._subscriptions.pop(websocket, None)
        logger.info(f"WebSocket disconnected: {len(self._connections)} remaining")

    def subscribe(self, websocket: WebSocket, symbols: List[str]) -> None:
        """Subscribe a connection to symbols."""
        if websocket in self._subscriptions:
            self._subscriptions[websocket].update(s.upper() for s in symbols)
            logger.debug(
                f"Subscribed to {symbols}: {self._subscriptions[websocket]}"
            )

    def unsubscribe(self, websocket: WebSocket, symbols: List[str]) -> None:
        """Unsubscribe a connection from symbols."""
        if websocket in self._subscriptions:
            for symbol in symbols:
                self._subscriptions[websocket].discard(symbol.upper())

    def get_all_symbols(self) -> Set[str]:
        """Get all currently subscribed symbols across all connections."""
        symbols: Set[str] = set()
        for subs in self._subscriptions.values():
            symbols.update(subs)
        return symbols

    async def broadcast(self, updates: Dict[str, PriceUpdate]) -> None:
        """
        Broadcast price updates to subscribed connections.

        Each connection only receives updates for its subscribed symbols.
        """
        disconnected: List[WebSocket] = []

        for websocket, symbols in self._subscriptions.items():
            # Filter updates for this connection's subscriptions
            relevant = {
                symbol: update.model_dump(mode="json")
                for symbol, update in updates.items()
                if symbol in symbols
            }

            if relevant:
                try:
                    await websocket.send_json({
                        "type": "price_update",
                        "data": relevant,
                    })
                except Exception as e:
                    logger.warning(f"Failed to send to websocket: {e}")
                    disconnected.append(websocket)

        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_raw(self, message: dict) -> None:
        """
        Broadcast a raw message dict to relevant subscribed connections.

        Used by the Redis Pub/Sub subscriber to relay messages from other instances.
        """
        updates_data = message.get("data", {})
        disconnected: List[WebSocket] = []

        for websocket, symbols in self._subscriptions.items():
            relevant = {
                sym: data for sym, data in updates_data.items()
                if sym in symbols
            }
            if relevant:
                try:
                    await websocket.send_json({
                        "type": "price_update",
                        "data": relevant,
                    })
                except Exception as e:
                    logger.warning(f"Failed to relay to websocket: {e}")
                    disconnected.append(websocket)

        for ws in disconnected:
            self.disconnect(ws)

    async def send_personal(self, websocket: WebSocket, message: dict) -> bool:
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            return False

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self._connections)


class PriceStreamer:
    """
    Coordinates price generation, Redis Pub/Sub publishing, and WebSocket streaming.

    Publishes generated prices to a Redis Pub/Sub channel so that
    multiple API instances can subscribe and relay to their local
    WebSocket connections. Falls back to direct broadcasting when
    Redis is unavailable.
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        price_generator: Optional[PriceGenerator] = None,
        interval_ms: int = 1000,
    ):
        self._manager = connection_manager or ConnectionManager()
        self._generator = price_generator or PriceGenerator()
        self._interval_ms = interval_ms
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._subscriber_task: Optional[asyncio.Task] = None
        self._redis_pubsub = None

    @property
    def manager(self) -> ConnectionManager:
        """Get the connection manager."""
        return self._manager

    @property
    def generator(self) -> PriceGenerator:
        """Get the price generator."""
        return self._generator

    async def _get_publish_redis(self):
        """Get a Redis client for publishing. Returns None if unavailable."""
        try:
            from app.cache import get_redis
            return await get_redis()
        except Exception:
            return None

    async def start(self) -> None:
        """Start the price streaming background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._stream_loop())
        self._subscriber_task = asyncio.create_task(self._subscribe_loop())
        logger.info(f"Price streamer started (interval: {self._interval_ms}ms)")

    async def stop(self) -> None:
        """Stop the price streaming background task."""
        self._running = False
        for task in [self._task, self._subscriber_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._subscriber_task = None

        if self._redis_pubsub is not None:
            try:
                await self._redis_pubsub.unsubscribe(PRICE_CHANNEL)
                await self._redis_pubsub.aclose()
            except Exception:
                pass
            self._redis_pubsub = None

        logger.info("Price streamer stopped")

    async def _publish_to_redis(self, updates: Dict[str, PriceUpdate]) -> bool:
        """Publish price updates to Redis Pub/Sub channel."""
        redis_client = await self._get_publish_redis()
        if redis_client is None:
            return False

        try:
            message = {
                "type": "price_update",
                "data": {
                    symbol: update.model_dump(mode="json")
                    for symbol, update in updates.items()
                },
            }
            await redis_client.publish(PRICE_CHANNEL, json.dumps(message).encode())
            return True
        except Exception as e:
            logger.warning(f"Failed to publish prices to Redis: {e}")
            return False

    async def _subscribe_loop(self) -> None:
        """Subscribe to Redis Pub/Sub and relay messages to local WebSocket clients."""
        while self._running:
            try:
                redis_client = await self._get_publish_redis()
                if redis_client is None:
                    await asyncio.sleep(5)
                    continue

                self._redis_pubsub = redis_client.pubsub()
                await self._redis_pubsub.subscribe(PRICE_CHANNEL)
                logger.info("Subscribed to Redis price channel")

                async for raw_message in self._redis_pubsub.listen():
                    if not self._running:
                        break
                    if raw_message["type"] != "message":
                        continue

                    try:
                        message = json.loads(raw_message["data"])
                        await self._manager.broadcast_raw(message)
                    except Exception as e:
                        logger.warning(f"Error processing pubsub message: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Redis subscriber error, reconnecting: {e}")
                self._redis_pubsub = None
                await asyncio.sleep(2)

    async def _stream_loop(self) -> None:
        """Main streaming loop."""
        while self._running:
            try:
                # Get all subscribed symbols
                symbols = list(self._manager.get_all_symbols())

                if symbols:
                    # Generate price updates
                    updates = self._generator.tick(symbols)

                    # Try to publish to Redis Pub/Sub
                    published = await self._publish_to_redis(updates)

                    if not published:
                        # Direct broadcast as fallback
                        await self._manager.broadcast(updates)

                # Wait for next tick
                await asyncio.sleep(self._interval_ms / 1000)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in price stream loop: {e}")
                await asyncio.sleep(1)  # Back off on error


# Global singleton instances
_connection_manager: Optional[ConnectionManager] = None
_price_streamer: Optional[PriceStreamer] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


def get_price_streamer() -> PriceStreamer:
    """Get the global price streamer."""
    global _price_streamer
    if _price_streamer is None:
        _price_streamer = PriceStreamer(get_connection_manager())
    return _price_streamer
