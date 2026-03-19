import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

// Strategy types
export interface StrategyParameter {
  name: string;
  type: 'number' | 'select' | 'boolean';
  label: string;
  default: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  category: 'momentum' | 'mean-reversion' | 'ml-based' | 'hybrid';
  parameters: StrategyParameter[];
}

export interface Signal {
  date: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
}

export interface SignalResult {
  ticker: string;
  strategy_id: string;
  signals: Signal[];
  summary: {
    buy_signals: number;
    sell_signals: number;
    avg_confidence: number;
  };
}

// Mock strategies for fallback
const mockStrategies: Strategy[] = [
  {
    id: 'momentum-rsi',
    name: 'RSI Momentum',
    description: 'Relative Strength Index based momentum strategy',
    category: 'momentum',
    parameters: [
      { name: 'rsi_period', type: 'number', label: 'RSI Period', default: 14, min: 5, max: 50 },
      { name: 'overbought', type: 'number', label: 'Overbought Level', default: 70, min: 60, max: 90 },
      { name: 'oversold', type: 'number', label: 'Oversold Level', default: 30, min: 10, max: 40 },
    ],
  },
  {
    id: 'macd-crossover',
    name: 'MACD Crossover',
    description: 'Moving Average Convergence Divergence crossover signals',
    category: 'momentum',
    parameters: [
      { name: 'fast_period', type: 'number', label: 'Fast EMA', default: 12, min: 5, max: 30 },
      { name: 'slow_period', type: 'number', label: 'Slow EMA', default: 26, min: 15, max: 50 },
      { name: 'signal_period', type: 'number', label: 'Signal Line', default: 9, min: 3, max: 20 },
    ],
  },
  {
    id: 'bollinger-bands',
    name: 'Bollinger Bands',
    description: 'Mean reversion strategy using Bollinger Bands',
    category: 'mean-reversion',
    parameters: [
      { name: 'period', type: 'number', label: 'Period', default: 20, min: 10, max: 50 },
      { name: 'std_dev', type: 'number', label: 'Std Deviations', default: 2, min: 1, max: 3, step: 0.5 },
    ],
  },
  {
    id: 'ml-ensemble',
    name: 'ML Ensemble',
    description: 'Machine learning ensemble combining XGBoost, LSTM, and LightGBM',
    category: 'ml-based',
    parameters: [
      { name: 'model_type', type: 'select', label: 'Primary Model', default: 'ensemble', options: [
        { value: 'xgboost', label: 'XGBoost' },
        { value: 'lstm', label: 'LSTM' },
        { value: 'lightgbm', label: 'LightGBM' },
        { value: 'ensemble', label: 'Ensemble' },
      ]},
      { name: 'confidence_threshold', type: 'number', label: 'Confidence Threshold', default: 0.6, min: 0.5, max: 0.9, step: 0.05 },
      { name: 'use_sentiment', type: 'boolean', label: 'Include Sentiment', default: true },
    ],
  },
  {
    id: 'trend-following',
    name: 'Trend Following',
    description: 'Multi-timeframe trend following with ADX confirmation',
    category: 'hybrid',
    parameters: [
      { name: 'short_ma', type: 'number', label: 'Short MA', default: 10, min: 5, max: 30 },
      { name: 'long_ma', type: 'number', label: 'Long MA', default: 50, min: 20, max: 200 },
      { name: 'adx_threshold', type: 'number', label: 'ADX Threshold', default: 25, min: 15, max: 40 },
    ],
  },
];

// Generate mock signals
function generateMockSignals(ticker: string, strategyId: string): SignalResult {
  const signals: Signal[] = [];
  const baseDate = new Date();
  baseDate.setMonth(baseDate.getMonth() - 3);
  
  let price = 150 + Math.random() * 50;
  
  for (let i = 0; i < 60; i++) {
    const date = new Date(baseDate);
    date.setDate(date.getDate() + i);
    
    price = price * (1 + (Math.random() - 0.48) * 0.03);
    
    const rand = Math.random();
    const signal: 'BUY' | 'SELL' | 'HOLD' = rand < 0.15 ? 'BUY' : rand > 0.85 ? 'SELL' : 'HOLD';
    
    if (signal !== 'HOLD') {
      signals.push({
        date: date.toISOString().slice(0, 10),
        signal,
        confidence: 0.5 + Math.random() * 0.4,
        price: Math.round(price * 100) / 100,
      });
    }
  }
  
  const buySignals = signals.filter(s => s.signal === 'BUY').length;
  const sellSignals = signals.filter(s => s.signal === 'SELL').length;
  
  return {
    ticker,
    strategy_id: strategyId,
    signals,
    summary: {
      buy_signals: buySignals,
      sell_signals: sellSignals,
      avg_confidence: signals.length > 0 
        ? signals.reduce((a, s) => a + s.confidence, 0) / signals.length 
        : 0,
    },
  };
}

export async function fetchStrategies(): Promise<Strategy[]> {
  try {
    const { data } = await api.get('/strategies');
    return data;
  } catch {
    // Fallback to mock data
    return mockStrategies;
  }
}

export async function generateSignals(
  ticker: string,
  strategyId: string,
  params: Record<string, unknown>
): Promise<SignalResult> {
  try {
    const { data } = await api.post('/strategies/signals', {
      ticker,
      strategy_id: strategyId,
      parameters: params,
    });
    return data;
  } catch {
    // Fallback to mock data
    return generateMockSignals(ticker, strategyId);
  }
}
