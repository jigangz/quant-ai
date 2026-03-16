import { useState, useEffect, useCallback } from 'react';
import Card from '../components/UI/Card';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import CandlestickChart from '../components/Charts/CandlestickChart';
import NewsPanel from '../components/News/NewsPanel';
import NewsCategoryTabs from '../components/News/NewsCategoryTabs';
import { fetchMarketData, fetchNews, type MarketDataPoint, type NewsItem } from '../api';

interface Props {
  ticker: string;
  onPriceUpdate: (price: number | null, change: number | null) => void;
}

export default function Dashboard({ ticker, onPriceUpdate }: Props) {
  const [marketData, setMarketData] = useState<MarketDataPoint[]>([]);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [category, setCategory] = useState('All');
  const [loadingMarket, setLoadingMarket] = useState(false);
  const [loadingNews, setLoadingNews] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch market data
  useEffect(() => {
    if (!ticker) return;
    setLoadingMarket(true);
    setError(null);
    fetchMarketData(ticker)
      .then((data) => {
        setMarketData(data);
        if (data.length > 0) {
          const last = data[data.length - 1];
          const prev = data.length > 1 ? data[data.length - 2] : last;
          const change = ((last.close - prev.close) / prev.close) * 100;
          onPriceUpdate(last.close, change);
        }
      })
      .catch(() => setError('Failed to load market data'))
      .finally(() => setLoadingMarket(false));
  }, [ticker]);

  // Fetch news on date click
  useEffect(() => {
    if (!ticker || !selectedDate) return;
    setLoadingNews(true);
    fetchNews(ticker, selectedDate)
      .then(setNews)
      .catch(() => setNews([]))
      .finally(() => setLoadingNews(false));
  }, [ticker, selectedDate]);

  const handleDateClick = useCallback((date: string) => {
    setSelectedDate(date);
  }, []);

  const handleHover = useCallback((_date: string | null, ohlc?: MarketDataPoint) => {
    if (ohlc && marketData.length > 0) {
      const prev = marketData.find((d) => d.date < ohlc.date);
      if (prev) {
        const change = ((ohlc.close - prev.close) / prev.close) * 100;
        onPriceUpdate(ohlc.close, change);
      }
    }
  }, [marketData, onPriceUpdate]);

  // Compute news dots from market data (simplified: show random dots for demo)
  // In production, this should come from a dedicated API
  const newsDots = marketData
    .filter((_, i) => i % 5 === 0) // placeholder: every 5th day
    .map((d) => ({
      date: d.date,
      count: Math.floor(Math.random() * 5) + 1,
      avgSentiment: Math.random() * 2 - 1,
    }));

  // Filter news by category
  const filteredNews =
    category === 'All'
      ? news
      : news.filter((n) => n.category.toLowerCase() === category.toLowerCase());

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <div className="text-4xl mb-3">⚠️</div>
          <div className="text-sm">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-12 gap-4 h-full animate-fade-in">
      {/* K-line chart */}
      <div className="col-span-8">
        <Card className="h-full min-h-[500px]">
          {loadingMarket ? (
            <LoadingSpinner text="Loading market data..." />
          ) : (
            <div className="p-2 h-full">
              <CandlestickChart
                data={marketData}
                newsDots={newsDots}
                onDateClick={handleDateClick}
                onHover={handleHover}
              />
            </div>
          )}
        </Card>
      </div>

      {/* News panel */}
      <div className="col-span-4 flex flex-col gap-4">
        <Card title="News Feed" className="flex-1 overflow-hidden">
          <div className="p-3 flex flex-col h-full">
            <div className="mb-3">
              <NewsCategoryTabs active={category} onChange={setCategory} />
            </div>
            <div className="flex-1 overflow-y-auto pr-1" style={{ maxHeight: 'calc(100vh - 220px)' }}>
              <NewsPanel
                news={filteredNews}
                selectedDate={selectedDate}
                loading={loadingNews}
              />
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
