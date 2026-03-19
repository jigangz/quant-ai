"""
Portfolio Manager

Manages portfolio state including cash, positions, and P&L tracking.
All storage is in-memory for MVP.
"""

import logging
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional
from uuid import UUID

from app.trading.models import (
    Order,
    OrderSide,
    Position,
    Portfolio,
    PortfolioSnapshot,
    Trade,
)

logger = logging.getLogger(__name__)

# Default starting cash
DEFAULT_STARTING_CASH = 100_000.0


class PositionState:
    """Internal position tracking state."""
    
    def __init__(self, symbol: str, quantity: float = 0.0, avg_cost: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.realized_pnl = 0.0
    
    def apply_fill(self, side: OrderSide, quantity: float, price: float) -> float:
        """
        Apply a fill to this position.
        
        Returns:
            Realized P&L from this fill (if closing/reducing position).
        """
        realized = 0.0
        
        if side == OrderSide.BUY:
            if self.quantity >= 0:
                # Adding to long position - update avg cost
                total_cost = (self.quantity * self.avg_cost) + (quantity * price)
                self.quantity += quantity
                self.avg_cost = total_cost / self.quantity if self.quantity != 0 else 0.0
            else:
                # Covering short position
                cover_qty = min(quantity, abs(self.quantity))
                realized = cover_qty * (self.avg_cost - price)  # Short profit
                self.realized_pnl += realized
                self.quantity += quantity
                
                if self.quantity > 0:
                    # Flipped to long
                    self.avg_cost = price
                elif self.quantity == 0:
                    self.avg_cost = 0.0
        else:  # SELL
            if self.quantity <= 0:
                # Adding to short position
                total_cost = (abs(self.quantity) * self.avg_cost) + (quantity * price)
                self.quantity -= quantity
                self.avg_cost = total_cost / abs(self.quantity) if self.quantity != 0 else 0.0
            else:
                # Closing long position
                sell_qty = min(quantity, self.quantity)
                realized = sell_qty * (price - self.avg_cost)  # Long profit
                self.realized_pnl += realized
                self.quantity -= quantity
                
                if self.quantity < 0:
                    # Flipped to short
                    self.avg_cost = price
                elif self.quantity == 0:
                    self.avg_cost = 0.0
        
        return realized
    
    def to_position(self, current_price: float) -> Position:
        """Convert to Position model with current price."""
        return Position.calculate(
            symbol=self.symbol,
            quantity=self.quantity,
            avg_cost=self.avg_cost,
            current_price=current_price,
        )


class PortfolioManager:
    """
    Manages portfolio state with thread-safe operations.
    
    All data is stored in-memory. Use reset() to return to initial state.
    """
    
    def __init__(self, starting_cash: float = DEFAULT_STARTING_CASH):
        self._starting_cash = starting_cash
        self._lock = Lock()
        self._reset_internal()
    
    def _reset_internal(self) -> None:
        """Internal reset without lock."""
        self._cash = self._starting_cash
        self._positions: Dict[str, PositionState] = {}
        self._trades: List[Trade] = []
        self._history: List[PortfolioSnapshot] = []
        self._total_realized_pnl = 0.0
        
        # Record initial snapshot
        self._history.append(
            PortfolioSnapshot(
                timestamp=datetime.utcnow(),
                cash=self._cash,
                market_value=0.0,
                total_equity=self._cash,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
            )
        )
    
    def reset(self, starting_cash: Optional[float] = None) -> Portfolio:
        """
        Reset portfolio to initial state.
        
        Args:
            starting_cash: Optional new starting cash (defaults to $100k).
            
        Returns:
            Fresh portfolio state.
        """
        with self._lock:
            if starting_cash is not None:
                self._starting_cash = starting_cash
            self._reset_internal()
            logger.info(f"Portfolio reset to ${self._starting_cash:,.2f}")
            return self._get_portfolio_internal({})
    
    @property
    def cash(self) -> float:
        """Current cash balance."""
        with self._lock:
            return self._cash
    
    @property
    def trades(self) -> List[Trade]:
        """List of all executed trades."""
        with self._lock:
            return list(self._trades)
    
    def check_buying_power(self, amount: float) -> bool:
        """Check if there's enough cash for a purchase."""
        with self._lock:
            return self._cash >= amount
    
    def check_position(self, symbol: str, quantity: float) -> bool:
        """Check if there's enough position to sell."""
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                return False
            return pos.quantity >= quantity
    
    def can_execute_order(
        self, symbol: str, side: OrderSide, quantity: float, price: float
    ) -> tuple:
        """
        Check if an order can be executed.
        
        Returns:
            Tuple of (can_execute: bool, reason: str)
        """
        with self._lock:
            if side == OrderSide.BUY:
                required = quantity * price
                if self._cash < required:
                    return False, f"Insufficient funds: need ${required:,.2f}, have ${self._cash:,.2f}"
            else:  # SELL
                pos = self._positions.get(symbol)
                if pos is None or pos.quantity < quantity:
                    current_qty = pos.quantity if pos else 0
                    return False, f"Insufficient position: need {quantity}, have {current_qty}"
            
            return True, "OK"
    
    def apply_fill(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float,
    ) -> Trade:
        """
        Apply a fill to the portfolio.
        
        Args:
            order: The order being filled.
            fill_price: Execution price.
            fill_quantity: Quantity filled.
            
        Returns:
            Trade record of the fill.
        """
        with self._lock:
            symbol = order.symbol.upper()
            total_value = fill_price * fill_quantity
            
            # Update cash
            if order.side == OrderSide.BUY:
                self._cash -= total_value
            else:
                self._cash += total_value
            
            # Update position
            if symbol not in self._positions:
                self._positions[symbol] = PositionState(symbol)
            
            realized = self._positions[symbol].apply_fill(
                order.side, fill_quantity, fill_price
            )
            self._total_realized_pnl += realized
            
            # Remove position if quantity is zero
            if abs(self._positions[symbol].quantity) < 1e-9:
                del self._positions[symbol]
            
            # Create trade record
            trade = Trade(
                order_id=order.id,
                symbol=symbol,
                side=order.side,
                quantity=fill_quantity,
                price=fill_price,
                total_value=total_value,
            )
            self._trades.append(trade)
            
            logger.info(
                f"Fill applied: {order.side.value} {fill_quantity} {symbol} @ ${fill_price:,.2f} "
                f"(realized P&L: ${realized:,.2f})"
            )
            
            return trade
    
    def get_position(self, symbol: str, current_price: float) -> Optional[Position]:
        """Get current position for a symbol."""
        with self._lock:
            pos = self._positions.get(symbol.upper())
            if pos is None:
                return None
            return pos.to_position(current_price)
    
    def get_portfolio(self, current_prices: Dict[str, float]) -> Portfolio:
        """
        Get full portfolio state.
        
        Args:
            current_prices: Dict mapping symbols to current prices.
            
        Returns:
            Portfolio with all positions and metrics.
        """
        with self._lock:
            return self._get_portfolio_internal(current_prices)
    
    def _get_portfolio_internal(self, current_prices: Dict[str, float]) -> Portfolio:
        """Internal portfolio calculation without lock."""
        positions: List[Position] = []
        total_market_value = 0.0
        total_unrealized_pnl = 0.0
        total_cost_basis = 0.0
        
        for symbol, pos_state in self._positions.items():
            price = current_prices.get(symbol, pos_state.avg_cost)
            position = pos_state.to_position(price)
            positions.append(position)
            total_market_value += position.market_value
            total_unrealized_pnl += position.unrealized_pnl
            total_cost_basis += pos_state.quantity * pos_state.avg_cost
        
        total_equity = self._cash + total_market_value
        pnl_pct = (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis != 0 else 0.0
        
        return Portfolio(
            cash=self._cash,
            positions=positions,
            total_market_value=total_market_value,
            total_equity=total_equity,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_pct=pnl_pct,
            buying_power=self._cash,
        )
    
    def record_snapshot(self, current_prices: Dict[str, float]) -> PortfolioSnapshot:
        """Record a portfolio snapshot for history tracking."""
        with self._lock:
            portfolio = self._get_portfolio_internal(current_prices)
            snapshot = PortfolioSnapshot(
                timestamp=datetime.utcnow(),
                cash=portfolio.cash,
                market_value=portfolio.total_market_value,
                total_equity=portfolio.total_equity,
                realized_pnl=self._total_realized_pnl,
                unrealized_pnl=portfolio.total_unrealized_pnl,
            )
            self._history.append(snapshot)
            return snapshot
    
    def get_history(
        self, limit: Optional[int] = None
    ) -> List[PortfolioSnapshot]:
        """Get portfolio history snapshots."""
        with self._lock:
            if limit:
                return list(self._history[-limit:])
            return list(self._history)
    
    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L."""
        with self._lock:
            return self._total_realized_pnl

    def get_position_symbols(self) -> List[str]:
        """Get list of symbols with open positions."""
        with self._lock:
            return [s for s, p in self._positions.items() if p.quantity != 0]

    def get_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[Trade]:
        """Get trade history with optional filters."""
        with self._lock:
            trades = list(self._trades)
            if symbol:
                trades = [t for t in trades if t.symbol.upper() == symbol.upper()]
            return trades[-limit:]


# Global singleton instance
_portfolio_manager: Optional[PortfolioManager] = None


def get_portfolio_manager() -> PortfolioManager:
    """Get the global portfolio manager instance."""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager()
    return _portfolio_manager


def reset_portfolio_manager(starting_cash: float = DEFAULT_STARTING_CASH) -> PortfolioManager:
    """Reset and return the global portfolio manager."""
    global _portfolio_manager
    _portfolio_manager = PortfolioManager(starting_cash)
    return _portfolio_manager
