from __future__ import annotations

"""
Matching Engine

Local order matching engine for paper trading.
Supports market and limit orders with immediate fill simulation.
"""

import logging
from datetime import datetime
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple
from uuid import UUID

from app.trading.models import (
    Order,
    OrderCreate,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Trade,
)
from app.trading.portfolio import PortfolioManager, get_portfolio_manager

logger = logging.getLogger(__name__)


class PriceProvider:
    """
    Abstract price provider interface.
    
    Implementations should provide current prices for order matching.
    """
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        raise NotImplementedError
    
    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols."""
        return {s: p for s in symbols if (p := self.get_price(s)) is not None}


class MockPriceProvider(PriceProvider):
    """
    Mock price provider for testing and demo mode.
    
    Returns pre-configured prices or simulated prices based on symbol.
    """
    
    # Default prices for common symbols
    DEFAULT_PRICES = {
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
    
    def __init__(self, prices: Optional[Dict[str, float]] = None):
        self._prices: Dict[str, float] = dict(self.DEFAULT_PRICES)
        if prices:
            self._prices.update({k.upper(): v for k, v in prices.items()})
        self._lock = Lock()
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get price for symbol, generating one if not found."""
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._prices:
                # Generate a reasonable price for unknown symbols
                # Use hash to get consistent price per symbol
                base = 50 + (hash(symbol) % 450)  # $50-$500 range
                self._prices[symbol] = float(base)
            return self._prices[symbol]
    
    def set_price(self, symbol: str, price: float) -> None:
        """Set price for a symbol."""
        with self._lock:
            self._prices[symbol.upper()] = price
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Bulk update prices."""
        with self._lock:
            self._prices.update({k.upper(): v for k, v in prices.items()})


class MatchingEngine:
    """
    Local order matching engine.
    
    Handles order submission, matching, and lifecycle management.
    All operations are thread-safe.
    """
    
    def __init__(
        self,
        portfolio: Optional[PortfolioManager] = None,
        price_provider: Optional[PriceProvider] = None,
    ):
        self._portfolio = portfolio or get_portfolio_manager()
        self._price_provider = price_provider or MockPriceProvider()
        self._orders: Dict[UUID, Order] = {}
        self._lock = Lock()
        self._order_callbacks: List[Callable[[Order, Optional[Trade]], None]] = []
    
    @property
    def price_provider(self) -> PriceProvider:
        """Get the price provider."""
        return self._price_provider
    
    def register_callback(
        self, callback: Callable[[Order, Optional[Trade]], None]
    ) -> None:
        """Register a callback for order updates."""
        self._order_callbacks.append(callback)
    
    def _notify_callbacks(self, order: Order, trade: Optional[Trade] = None) -> None:
        """Notify all registered callbacks of an order update."""
        for callback in self._order_callbacks:
            try:
                callback(order, trade)
            except Exception as e:
                logger.error(f"Order callback error: {e}")
    
    def submit_order(self, order_request: OrderCreate) -> Tuple[Order, Optional[Trade]]:
        """
        Submit a new order for execution.
        
        Market orders are filled immediately at current price.
        Limit orders are checked against current price:
          - Buy limit: fills if current price <= limit price
          - Sell limit: fills if current price >= limit price
        
        Args:
            order_request: Order creation request.
            
        Returns:
            Tuple of (order, trade) - trade is None if not filled.
        """
        # Validate limit price for limit orders
        if order_request.order_type == OrderType.LIMIT and order_request.limit_price is None:
            raise ValueError("Limit price required for limit orders")
        
        symbol = order_request.symbol.upper()
        
        # Get current price
        current_price = self._price_provider.get_price(symbol)
        if current_price is None:
            raise ValueError(f"Unable to get price for {symbol}")
        
        # Create order
        order = Order(
            symbol=symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            limit_price=order_request.limit_price,
        )
        
        with self._lock:
            # Store order
            self._orders[order.id] = order
            
            # Determine execution price
            if order.order_type == OrderType.MARKET:
                exec_price = current_price
            else:
                # Limit order - check if it can fill
                if order.side == OrderSide.BUY:
                    if current_price <= order.limit_price:
                        exec_price = current_price  # Fill at market, not limit
                    else:
                        # Order remains pending
                        logger.info(
                            f"Limit order pending: {order.side.value} {order.quantity} {symbol} "
                            f"@ ${order.limit_price:.2f} (current: ${current_price:.2f})"
                        )
                        return order, None
                else:  # SELL
                    if current_price >= order.limit_price:
                        exec_price = current_price
                    else:
                        logger.info(
                            f"Limit order pending: {order.side.value} {order.quantity} {symbol} "
                            f"@ ${order.limit_price:.2f} (current: ${current_price:.2f})"
                        )
                        return order, None
            
            # Check if order can execute
            can_execute, reason = self._portfolio.can_execute_order(
                symbol, order.side, order.quantity, exec_price
            )
            
            if not can_execute:
                order.status = OrderStatus.REJECTED
                order.reject_reason = reason
                order.updated_at = datetime.utcnow()
                logger.warning(f"Order rejected: {reason}")
                self._notify_callbacks(order, None)
                return order, None
            
            # Execute the order
            trade = self._execute_order(order, exec_price)
            
            return order, trade
    
    def _execute_order(self, order: Order, exec_price: float) -> Trade:
        """Execute an order at the given price. Must be called with lock held."""
        # Apply fill to portfolio
        trade = self._portfolio.apply_fill(order, exec_price, order.quantity)
        
        # Update order status
        order.filled_quantity = order.quantity
        order.fill_price = exec_price
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()
        order.updated_at = datetime.utcnow()
        
        logger.info(
            f"Order filled: {order.side.value} {order.quantity} {order.symbol} @ ${exec_price:.2f}"
        )
        
        self._notify_callbacks(order, trade)
        return trade
    
    def cancel_order(self, order_id: UUID) -> Optional[Order]:
        """
        Cancel a pending order.
        
        Args:
            order_id: ID of order to cancel.
            
        Returns:
            Cancelled order, or None if not found/not cancellable.
        """
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                logger.warning(f"Order {order_id} not found")
                return None
            
            if order.status != OrderStatus.PENDING:
                logger.warning(
                    f"Cannot cancel order {order_id}: status is {order.status.value}"
                )
                return None
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            logger.info(f"Order cancelled: {order_id}")
            self._notify_callbacks(order, None)
            
            return order
    
    def get_order(self, order_id: UUID) -> Optional[Order]:
        """Get an order by ID."""
        with self._lock:
            return self._orders.get(order_id)
    
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """
        Get orders with optional filtering.
        
        Args:
            status: Filter by order status.
            symbol: Filter by symbol.
            limit: Maximum orders to return.
            
        Returns:
            List of orders, newest first.
        """
        with self._lock:
            orders = list(self._orders.values())
            
            if status is not None:
                orders = [o for o in orders if o.status == status]
            
            if symbol is not None:
                symbol = symbol.upper()
                orders = [o for o in orders if o.symbol == symbol]
            
            # Sort by creation time, newest first
            orders.sort(key=lambda o: o.created_at, reverse=True)
            
            return orders[:limit]
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return self.get_orders(status=OrderStatus.PENDING)
    
    def check_limit_orders(self) -> List[Tuple[Order, Trade]]:
        """
        Check and execute any limit orders that can now fill.
        
        Called when prices update to check if pending limit orders
        can be executed.
        
        Returns:
            List of (order, trade) tuples for filled orders.
        """
        filled: List[Tuple[Order, Trade]] = []
        
        with self._lock:
            pending = [
                o for o in self._orders.values()
                if o.status == OrderStatus.PENDING
            ]
            
            for order in pending:
                current_price = self._price_provider.get_price(order.symbol)
                if current_price is None:
                    continue
                
                should_fill = False
                if order.side == OrderSide.BUY and current_price <= order.limit_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                    should_fill = True
                
                if should_fill:
                    can_execute, reason = self._portfolio.can_execute_order(
                        order.symbol, order.side, order.quantity, current_price
                    )
                    
                    if can_execute:
                        trade = self._execute_order(order, current_price)
                        filled.append((order, trade))
                    else:
                        # Reject the order
                        order.status = OrderStatus.REJECTED
                        order.reject_reason = reason
                        order.updated_at = datetime.utcnow()
                        self._notify_callbacks(order, None)
        
        return filled
    
    def reset(self) -> None:
        """Clear all orders."""
        with self._lock:
            self._orders.clear()
            logger.info("Matching engine reset")


# Global singleton instance
_matching_engine: Optional[MatchingEngine] = None


def get_matching_engine() -> MatchingEngine:
    """Get the global matching engine instance."""
    global _matching_engine
    if _matching_engine is None:
        _matching_engine = MatchingEngine()
    return _matching_engine


def reset_matching_engine() -> MatchingEngine:
    """Reset and return the global matching engine."""
    global _matching_engine
    _matching_engine = MatchingEngine()
    return _matching_engine


# ---------------------------------------------------------------------------
# V4 P3: place_order — high-level order entry with optional meta-label gate
# ---------------------------------------------------------------------------

def _make_order_result(order: Order) -> "OrderResult":
    from app.trading.models import OrderResult
    return OrderResult(
        status=order.status.value,
        reason=order.reject_reason,
        order=order,
    )


def _rejected(reason: str) -> "OrderResult":
    from app.trading.models import OrderResult
    return OrderResult(status="rejected", reason=reason)


def place_order(
    ticker: str,
    side: str,
    qty: float,
    order_type: str = "market",
    limit_price: Optional[float] = None,
    meta_model_id: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> "OrderResult":
    """Submit an order with an optional meta-label quality gate.

    When meta_model_id is None the behavior is byte-identical to calling
    submit_order() directly (backward-compatible legacy path).

    Args:
        ticker: Ticker symbol.
        side: "buy" or "sell".
        qty: Number of shares.
        order_type: "market" or "limit".
        limit_price: Required for limit orders.
        meta_model_id: If given, scores the signal via SignalScoringService
                       before placing. Orders are rejected when score < threshold.
        score_threshold: Probability cutoff for the meta-label gate. Falls back
                         to PaperTradingConfig.default_score_threshold if None.

    Returns:
        OrderResult with status "filled" | "pending" | "rejected".
    """
    # V4 P3: Meta-label gating (opt-in — when meta_model_id is given)
    if meta_model_id is not None:
        from app.services import signal_scoring_service
        from app.trading.models import PaperTradingConfig

        threshold = (
            score_threshold
            if score_threshold is not None
            else PaperTradingConfig().default_score_threshold
        )
        try:
            req = signal_scoring_service.SignalScoreRequest(
                ticker=ticker,
                meta_model_id=meta_model_id,
                signal=1 if side == "buy" else -1,
            )
            resp = signal_scoring_service.score_signal(req)
        except ValueError as e:
            return _rejected(reason=f"meta_model_error:{e}")

        score = resp.get("reliability_score", 0.0)
        if score < threshold:
            return _rejected(
                reason=(
                    f"meta_score_below_threshold:"
                    f"score={score:.3f}:threshold={threshold:.3f}"
                )
            )
        # Apply half-Kelly sizing, clamped to at least 1 share
        hint = resp.get("sizing_hint", {})
        kelly_frac = hint.get("half_kelly_fraction", 0.25)
        cap = hint.get("cap", 0.25)
        qty = max(1, int(qty * kelly_frac / cap))

    # Legacy path: submit via MatchingEngine
    from app.trading.models import OrderCreate, OrderSide, OrderType

    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    type_enum = OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT

    order_create = OrderCreate(
        symbol=ticker,
        side=side_enum,
        order_type=type_enum,
        quantity=float(qty),
        limit_price=limit_price,
    )
    engine = get_matching_engine()
    order, _trade = engine.submit_order(order_create)
    return _make_order_result(order)
