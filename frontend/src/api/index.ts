import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

// Market data
export interface MarketDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export async function fetchMarketData(ticker: string, period = '3mo'): Promise<MarketDataPoint[]> {
  const { data } = await api.get('/data/market', { params: { ticker, period } });
  return data;
}

// News
export interface NewsItem {
  id: string;
  date: string;
  headline: string;   // backend field name
  summary: string;
  source: string;
  url: string;
  category: string;
  sentiment_score: number | null;
  bullish_reason: string | null;
  bearish_reason: string | null;
}

export async function fetchNews(ticker: string, date?: string): Promise<NewsItem[]> {
  const { data } = await api.get(`/data/news/${ticker}`, { params: date ? { date } : {} });
  return data;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export async function fetchNewsCategories(ticker: string): Promise<CategoryCount[]> {
  const { data } = await api.get(`/data/news/${ticker}/categories`);
  return data;
}

export interface SentimentSummary {
  date: string;
  total: number;
  bullish: number;
  bearish: number;
  neutral: number;
  avg_score: number;
}

export async function fetchSentimentSummary(ticker: string, date: string): Promise<SentimentSummary> {
  const { data } = await api.get(`/data/news/${ticker}/sentiment-summary`, { params: { date } });
  return data;
}

export interface SimilarDay {
  date: string;
  similarity: number;
  price_change: number;
  sentiment_score: number;
  news_count: number;
}

export async function fetchSimilarDays(ticker: string, date: string): Promise<SimilarDay[]> {
  const { data } = await api.get(`/data/news/${ticker}/similar-days`, { params: { date } });
  return data;
}

// Prediction
export interface PredictionResult {
  ticker: string;
  date: string;
  horizons: {
    t1: { direction: 'UP' | 'DOWN'; confidence: number };
    t3: { direction: 'UP' | 'DOWN'; confidence: number };
    t5: { direction: 'UP' | 'DOWN'; confidence: number };
  };
}

export async function predict(ticker: string): Promise<PredictionResult> {
  const { data } = await api.post('/predict', { ticker });
  // Adapt backend format: { predictions: [{ horizon, probability, confidence }] }
  // to frontend format: { horizons: { t1, t3, t5 } }
  const horizonMap: Record<number, 't1' | 't3' | 't5'> = { 1: 't1', 3: 't3', 5: 't5' };
  const horizons: PredictionResult['horizons'] = {
    t1: { direction: 'UP', confidence: 0.5 },
    t3: { direction: 'UP', confidence: 0.5 },
    t5: { direction: 'UP', confidence: 0.5 },
  };
  if (data.predictions) {
    for (const p of data.predictions) {
      const key = horizonMap[p.horizon];
      if (key) {
        horizons[key] = {
          direction: (p.probability?.up ?? 0) >= 0.5 ? 'UP' : 'DOWN',
          confidence: p.confidence ?? Math.max(p.probability?.up ?? 0, p.probability?.down ?? 0),
        };
      }
    }
  }
  return { ticker: data.ticker ?? ticker, date: data.date ?? new Date().toISOString().slice(0, 10), horizons };
}

// SHAP / Explain
export interface ShapFeature {
  feature: string;
  value: number;
  shap_value: number;
}

export async function fetchExplanation(ticker: string): Promise<ShapFeature[]> {
  const { data } = await api.post('/explain', { ticker });
  return data;
}

// Backtest
export interface BacktestParams {
  ticker: string;
  start_date: string;
  end_date: string;
  model_type: string;
}

export interface BacktestPoint {
  date: string;
  strategy_return: number;
  benchmark_return: number;
}

export interface BacktestTrade {
  date: string;
  direction: 'LONG' | 'SHORT';
  entry_price: number;
  exit_price: number;
  return_pct: number;
  holding_days: number;
}

export interface BacktestResult {
  equity_curve: BacktestPoint[];
  trades: BacktestTrade[];
  metrics: {
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_return: number;
    num_trades: number;
    avg_return: number;
  };
}

export async function runBacktest(params: BacktestParams): Promise<BacktestResult> {
  const { data } = await api.post('/backtest', params);
  return data;
}

// Models
export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  auc: number;
  f1: number;
  trained_at: string;
  status: 'production' | 'candidate' | 'archived';
  description: string;
  params: Record<string, unknown>;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const { data } = await api.get('/models');
  return data;
}

export async function promoteModel(id: string): Promise<void> {
  await api.post(`/models/${id}/promote`);
}

// Train
export async function trainModel(ticker: string, modelType: string): Promise<{ task_id: string }> {
  const { data } = await api.post('/train', { ticker, model_type: modelType });
  return data;
}
