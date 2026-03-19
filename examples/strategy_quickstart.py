"""
Strategy System Quick Start

This example demonstrates how to use the Strategy System to:
1. List available strategies
2. Create strategy instances
3. Generate signals
4. Validate and customize parameters
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the strategy registry
from app.strategies import get_registry


def main():
    print("=" * 60)
    print("Strategy System Quick Start")
    print("=" * 60)
    print()
    
    # ===================================
    # 1. Get the registry and list strategies
    # ===================================
    print("1️⃣  Getting strategy registry...")
    registry = get_registry()
    print(f"   Found {len(registry)} strategies\n")
    
    print("Available strategies:")
    for i, name in enumerate(registry.list_strategies(), 1):
        metadata = registry.get_metadata(name)
        print(f"   {i}. {metadata.name}")
        print(f"      {metadata.description}")
        print()
    
    # ===================================
    # 2. Create sample data
    # ===================================
    print("2️⃣  Creating sample price data...")
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price movement
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    
    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low
    }, index=dates)
    
    print(f"   Created {len(df)} days of data")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}\n")
    
    # ===================================
    # 3. Example 1: Moving Average Crossover
    # ===================================
    print("3️⃣  Example 1: Moving Average Crossover Strategy")
    print("   Parameters: fast=5, slow=20, type=SMA\n")
    
    ma_strategy = registry.create_instance('ma_crossover', {
        'fast_period': 5,
        'slow_period': 20,
        'ma_type': 'sma'
    })
    
    # Validate data
    missing = ma_strategy.validate_data(df)
    if missing:
        print(f"   ❌ Missing columns: {missing}")
    else:
        print("   ✅ Data validation passed")
    
    # Generate signals
    signals = ma_strategy.generate_signals(df)
    
    print(f"   Generated {len(signals)} signals:")
    print(f"   • Long signals: {(signals == 1).sum()}")
    print(f"   • Short signals: {(signals == -1).sum()}")
    print(f"   • Neutral: {(signals == 0).sum()}")
    print()
    
    # Show some example signals
    signal_changes = signals[signals.diff() != 0].head(5)
    if len(signal_changes) > 0:
        print("   First 5 signal changes:")
        for date, signal in signal_changes.items():
            signal_name = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}[signal]
            print(f"   • {date.date()}: {signal_name} (signal={signal})")
    print()
    
    # ===================================
    # 4. Example 2: RSI Strategy
    # ===================================
    print("4️⃣  Example 2: RSI Overbought/Oversold Strategy")
    print("   Parameters: period=14, overbought=70, oversold=30\n")
    
    rsi_strategy = registry.create_instance('rsi_strategy', {
        'rsi_period': 14,
        'overbought': 70,
        'oversold': 30,
        'exit_on_neutral': True
    })
    
    signals = rsi_strategy.generate_signals(df)
    
    print(f"   Generated {len(signals)} signals:")
    print(f"   • Long signals: {(signals == 1).sum()}")
    print(f"   • Short signals: {(signals == -1).sum()}")
    print(f"   • Neutral: {(signals == 0).sum()}")
    print()
    
    # ===================================
    # 5. Example 3: Bollinger Bands Breakout
    # ===================================
    print("5️⃣  Example 3: Bollinger Bands Breakout Strategy")
    print("   Parameters: period=20, std=2.0, mode=breakout\n")
    
    bb_strategy = registry.create_instance('bollinger_breakout', {
        'bb_period': 20,
        'bb_std': 2.0,
        'strategy_mode': 'breakout',
        'use_close_cross': True
    })
    
    signals = bb_strategy.generate_signals(df)
    
    print(f"   Generated {len(signals)} signals:")
    print(f"   • Long signals (breakout above): {(signals == 1).sum()}")
    print(f"   • Short signals (breakdown below): {(signals == -1).sum()}")
    print(f"   • Neutral (within bands): {(signals == 0).sum()}")
    print()
    
    # ===================================
    # 6. Accessing parameter schema
    # ===================================
    print("6️⃣  Accessing parameter schema")
    print("   Getting schema for 'ma_crossover'...\n")
    
    metadata = registry.get_metadata('ma_crossover')
    schema = metadata.parameters_schema
    
    print("   Parameters:")
    for param_name, param_info in schema.get('properties', {}).items():
        default = param_info.get('default', 'N/A')
        param_type = param_info.get('type', 'unknown')
        description = param_info.get('description', '')
        
        print(f"   • {param_name} ({param_type})")
        print(f"     Default: {default}")
        if description:
            print(f"     {description}")
    print()
    
    # ===================================
    # Summary
    # ===================================
    print("=" * 60)
    print("✅ Quick Start Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("• Use the REST API endpoints (/api/strategies)")
    print("• Create custom strategies (inherit from BaseStrategy)")
    print("• Run backtests with different parameters")
    print("• Combine strategies for portfolio optimization")
    print()


if __name__ == '__main__':
    main()
