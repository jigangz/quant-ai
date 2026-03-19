import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

// Trading types
export interface Position {
  ticker: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

export interface Order {
  id: string;
  ticker: string;
  side: 'BUY' | 'SELL';
  type: 'MARKET' | 'LIMIT';
  quantity: number;
  limit_price?: number;
  status: 'PENDING' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  created_at: string;
  filled_at?: string;
  filled_price?: number;
}

export interface Trade {
  id: string;
  ticker: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  total: number;
  executed_at: string;
}

export interface Portfolio {
  cash: number;
  total_value: number;
  positions_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  positions: Position[];
}

export interface EquityPoint {
  date: string;
  value: number;
  cash: number;
  positions_value: number;
}

export interface Quote {
  ticker: string;
  price: number;
  change: number;
  change_pct: number;
  volume: number;
  timestamp: string;
}

// Mock data generators
function generateMockPortfolio(): Portfolio {
  const positions: Position[] = [
    {
      ticker: 'AAPL',
      quantity: 50,
      avg_cost: 178.50,
      current_price: 185.20,
      market_value: 9260,
      unrealized_pnl: 335,
      unrealized_pnl_pct: 3.75,
    },
    {
      ticker: 'GOOGL',
      quantity: 20,
      avg_cost: 142.30,
      current_price: 138.45,
      market_value: 2769,
      unrealized_pnl: -77,
      unrealized_pnl_pct: -2.71,
    },
    {
      ticker: 'MSFT',
      quantity: 30,
      avg_cost: 420.00,
      current_price: 432.15,
      market_value: 12964.50,
      unrealized_pnl: 364.50,
      unrealized_pnl_pct: 2.89,
    },
  ];

  const positionsValue = positions.reduce((sum, p) => sum + p.market_value, 0);
  const totalPnl = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);
  const totalCost = positions.reduce((sum, p) => sum + p.avg_cost * p.quantity, 0);
  const cash = 25000;

  return {
    cash,
    total_value: cash + positionsValue,
    positions_value: positionsValue,
    total_pnl: totalPnl,
    total_pnl_pct: (totalPnl / totalCost) * 100,
    positions,
  };
}

function generateMockOrders(): Order[] {
  return [
    {
      id: 'ord-001',
      ticker: 'NVDA',
      side: 'BUY',
      type: 'LIMIT',
      quantity: 10,
      limit_price: 880.00,
      status: 'PENDING',
      created_at: new Date(Date.now() - 3600000).toISOString(),
    },
    {
      id: 'ord-002',
      ticker: 'AMD',
      side: 'BUY',
      type: 'LIMIT',
      quantity: 25,
      limit_price: 165.50,
      status: 'PENDING',
      created_at: new Date(Date.now() - 7200000).toISOString(),
    },
  ];
}

function generateMockTrades(): Trade[] {
  const trades: Trade[] = [];
  const tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'];
  
  for (let i = 0; i < 15; i++) {
    const ticker = tickers[Math.floor(Math.random() * tickers.length)];
    const side: 'BUY' | 'SELL' = Math.random() > 0.5 ? 'BUY' : 'SELL';
    const quantity = Math.floor(Math.random() * 50) + 5;
    const price = 100 + Math.random() * 400;
    
    trades.push({
      id: `trd-${String(i).padStart(3, '0')}`,
      ticker,
      side,
      quantity,
      price: Math.round(price * 100) / 100,
      total: Math.round(quantity * price * 100) / 100,
      executed_at: new Date(Date.now() - i * 86400000 * Math.random()).toISOString(),
    });
  }
  
  return trades.sort((a, b) => new Date(b.executed_at).getTime() - new Date(a.executed_at).getTime());
}

function generateMockEquityCurve(): EquityPoint[] {
  const points: EquityPoint[] = [];
  const baseDate = new Date();
  baseDate.setMonth(baseDate.getMonth() - 3);
  
  let value = 50000;
  let cash = 30000;
  
  for (let i = 0; i < 90; i++) {
    const date = new Date(baseDate);
    date.setDate(date.getDate() + i);
    
    // Simulate daily returns
    value = value * (1 + (Math.random() - 0.48) * 0.02);
    const positionsValue = value - cash;
    
    // Occasional cash changes (trades)
    if (Math.random() < 0.1) {
      const tradeAmount = (Math.random() - 0.5) * 5000;
      cash = Math.max(5000, cash + tradeAmount);
    }
    
    points.push({
      date: date.toISOString().slice(0, 10),
      value: Math.round(value * 100) / 100,
      cash: Math.round(cash * 100) / 100,
      positions_value: Math.round((value - cash + positionsValue) / 2 * 100) / 100,
    });
  }
  
  return points;
}

function generateMockQuote(ticker: string): Quote {
  const basePrice = {
    'AAPL': 185,
    'GOOGL': 140,
    'MSFT': 430,
    'AMZN': 185,
    'NVDA': 900,
    'META': 500,
    'AMD': 170,
  }[ticker] || 100 + Math.random() * 200;
  
  const change = (Math.random() - 0.5) * 10;
  
  return {
    ticker,
    price: Math.round((basePrice + change) * 100) / 100,
    change: Math.round(change * 100) / 100,
    change_pct: Math.round((change / basePrice) * 10000) / 100,
    volume: Math.floor(Math.random() * 10000000) + 1000000,
    timestamp: new Date().toISOString(),
  };
}

// API functions
export async function fetchPortfolio(): Promise<Portfolio> {
  try {
    const { data } = await api.get('/trading/portfolio');
    return data;
  } catch {
    return generateMockPortfolio();
  }
}

export async function fetchOpenOrders(): Promise<Order[]> {
  try {
    const { data } = await api.get('/trading/orders', { params: { status: 'PENDING' } });
    return data;
  } catch {
    return generateMockOrders();
  }
}

export async function fetchTradeHistory(limit = 20): Promise<Trade[]> {
  try {
    const { data } = await api.get('/trading/trades', { params: { limit } });
    return data;
  } catch {
    return generateMockTrades().slice(0, limit);
  }
}

export async function fetchEquityCurve(): Promise<EquityPoint[]> {
  try {
    const { data } = await api.get('/trading/equity');
    return data;
  } catch {
    return generateMockEquityCurve();
  }
}

export async function fetchQuote(ticker: string): Promise<Quote> {
  try {
    const { data } = await api.get(`/trading/quote/${ticker}`);
    return data;
  } catch {
    return generateMockQuote(ticker);
  }
}

export interface PlaceOrderParams {
  ticker: string;
  side: 'BUY' | 'SELL';
  type: 'MARKET' | 'LIMIT';
  quantity: number;
  limit_price?: number;
}

export async function placeOrder(params: PlaceOrderParams): Promise<Order> {
  try {
    const { data } = await api.post('/trading/orders', params);
    return data;
  } catch {
    // Mock order creation
    return {
      id: `ord-${Date.now()}`,
      ...params,
      status: params.type === 'MARKET' ? 'FILLED' : 'PENDING',
      created_at: new Date().toISOString(),
      filled_at: params.type === 'MARKET' ? new Date().toISOString() : undefined,
      filled_price: params.type === 'MARKET' ? (params.limit_price || 100) : undefined,
    };
  }
}

export async function cancelOrder(orderId: string): Promise<void> {
  try {
    await api.delete(`/trading/orders/${orderId}`);
  } catch {
    // Silent fail for mock
  }
}
