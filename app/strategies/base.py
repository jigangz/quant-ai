"""
Base Strategy Module

Provides the abstract BaseStrategy class that all trading strategies must inherit from.
Strategies generate signals (-1 for short, 0 for hold, 1 for long) based on market data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel, Field


class BaseParameters(BaseModel):
    """Base class for strategy parameters. All strategies should define their own Parameters class."""
    
    class Config:
        extra = "forbid"


class StrategyMetadata(BaseModel):
    """Metadata about a strategy."""
    
    name: str
    description: str
    version: str
    author: str = "quant-ai"
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    required_columns: List[str] = Field(default_factory=list)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Strategies must implement:
    - name, description, version properties
    - Parameters class (Pydantic model for strategy parameters)
    - generate_signals(df) method that returns a Series of -1/0/1
    
    Example usage:
        class MyStrategy(BaseStrategy):
            name = "my_strategy"
            description = "My custom strategy"
            version = "1.0.0"
            
            class Parameters(BaseParameters):
                threshold: float = 0.5
            
            def generate_signals(self, df: pd.DataFrame) -> pd.Series:
                # Generate signals based on df
                return pd.Series(0, index=df.index)
    """
    
    # Strategy metadata - must be overridden by subclasses
    name: str = "base_strategy"
    description: str = "Base strategy - do not use directly"
    version: str = "1.0.0"
    author: str = "quant-ai"
    
    # Required columns in the input DataFrame
    required_columns: List[str] = ["close"]
    
    # Parameters class - override in subclass
    Parameters: Type[BaseParameters] = BaseParameters
    
    def __init__(self, parameters: Optional[BaseParameters] = None, **kwargs):
        """
        Initialize the strategy with parameters.
        
        Args:
            parameters: Pydantic model instance with strategy parameters
            **kwargs: Alternative way to pass parameters as keyword arguments
        """
        if parameters is not None:
            self.params = parameters
        elif kwargs:
            self.params = self.Parameters(**kwargs)
        else:
            self.params = self.Parameters()
    
    def validate_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate that the input DataFrame has all required columns.
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            List of missing column names (empty if all required columns present)
        """
        if df is None or df.empty:
            return ["DataFrame is empty or None"]
        
        missing = []
        for col in self.required_columns:
            if col not in df.columns:
                missing.append(col)
        
        return missing
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market data.
        
        Args:
            df: DataFrame with market data (must contain required_columns)
            
        Returns:
            pd.Series with signals: -1 (short), 0 (hold/neutral), 1 (long)
            Index should match input DataFrame index.
        """
        pass
    
    def get_metadata(self) -> StrategyMetadata:
        """Get strategy metadata including parameter schema."""
        return StrategyMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            author=self.author,
            parameters_schema=self.Parameters.schema(),
            required_columns=self.required_columns,
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
