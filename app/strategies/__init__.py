"""
Strategy System

Provides a registry for managing trading strategies and their templates.

Usage:
    from app.strategies import StrategyRegistry, get_registry
    
    # Get the global registry
    registry = get_registry()
    
    # List all strategies
    strategies = registry.list_strategies()
    
    # Get a specific strategy class
    StrategyClass = registry.get("ma_crossover")
    
    # Create instance with parameters
    strategy = StrategyClass(fast_period=5, slow_period=20)
    
    # Generate signals
    signals = strategy.generate_signals(df)
"""

from typing import Dict, List, Optional, Type

from app.strategies.base import (
    BaseStrategy,
    BaseParameters,
    StrategyMetadata,
)
from app.strategies.templates import (
    MovingAverageCrossover,
    RSIStrategy,
    BollingerBreakout,
    SentimentDriven,
)


class StrategyRegistry:
    """
    Registry for managing trading strategies.
    
    Provides methods to register, retrieve, and list strategies.
    Comes pre-loaded with built-in strategy templates.
    """
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._register_builtin_strategies()
    
    def _register_builtin_strategies(self):
        """Register all built-in strategy templates."""
        builtin = [
            MovingAverageCrossover,
            RSIStrategy,
            BollingerBreakout,
            SentimentDriven,
        ]
        for strategy_class in builtin:
            self.register(strategy_class)
    
    def register(self, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class.
        
        Args:
            strategy_class: Strategy class (must inherit from BaseStrategy)
            
        Raises:
            ValueError: If strategy_class doesn't inherit from BaseStrategy
            ValueError: If a strategy with the same name is already registered
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(
                f"{strategy_class} must inherit from BaseStrategy"
            )
        
        name = strategy_class.name
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' is already registered")
        
        self._strategies[name] = strategy_class
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            True if strategy was removed, False if not found
        """
        if name in self._strategies:
            del self._strategies[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get a strategy class by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy class or None if not found
        """
        return self._strategies.get(name)
    
    def get_or_raise(self, name: str) -> Type[BaseStrategy]:
        """
        Get a strategy class by name, raising if not found.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy class
            
        Raises:
            KeyError: If strategy not found
        """
        if name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' not found. Available: {available}"
            )
        return self._strategies[name]
    
    def list_strategies(self) -> List[str]:
        """
        List all registered strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def list_metadata(self) -> List[StrategyMetadata]:
        """
        List metadata for all registered strategies.
        
        Returns:
            List of StrategyMetadata objects
        """
        metadata = []
        for strategy_class in self._strategies.values():
            # Create temporary instance to get metadata
            instance = strategy_class()
            metadata.append(instance.get_metadata())
        return metadata
    
    def get_metadata(self, name: str) -> Optional[StrategyMetadata]:
        """
        Get metadata for a specific strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            StrategyMetadata or None if not found
        """
        strategy_class = self.get(name)
        if strategy_class is None:
            return None
        
        instance = strategy_class()
        return instance.get_metadata()
    
    def create_instance(
        self,
        name: str,
        parameters: Optional[Dict] = None
    ) -> BaseStrategy:
        """
        Create a strategy instance with parameters.
        
        Args:
            name: Strategy name
            parameters: Dictionary of parameter values
            
        Returns:
            Strategy instance
            
        Raises:
            KeyError: If strategy not found
            ValueError: If parameters are invalid
        """
        strategy_class = self.get_or_raise(name)
        
        if parameters:
            return strategy_class(**parameters)
        return strategy_class()
    
    def __contains__(self, name: str) -> bool:
        return name in self._strategies
    
    def __len__(self) -> int:
        return len(self._strategies)
    
    def __iter__(self):
        return iter(self._strategies.keys())


# Global registry instance
_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """
    Get the global strategy registry.
    
    Returns:
        StrategyRegistry singleton instance
    """
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry


# Exports
__all__ = [
    # Base classes
    "BaseStrategy",
    "BaseParameters",
    "StrategyMetadata",
    # Templates
    "MovingAverageCrossover",
    "RSIStrategy",
    "BollingerBreakout",
    "SentimentDriven",
    # Registry
    "StrategyRegistry",
    "get_registry",
]
