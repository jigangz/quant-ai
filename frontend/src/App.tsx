import { useState, useCallback } from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Layout/Sidebar';
import Topbar from './components/Layout/Topbar';
import Dashboard from './pages/Dashboard';
import Prediction from './pages/Prediction';
import Backtest from './pages/Backtest';
import Models from './pages/Models';

export default function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [price, setPrice] = useState<number | null>(null);
  const [change, setChange] = useState<number | null>(null);

  const handlePriceUpdate = useCallback((p: number | null, c: number | null) => {
    setPrice(p);
    setChange(c);
  }, []);

  return (
    <div className="min-h-screen bg-dark-bg text-gray-200">
      <Sidebar />
      <div className="ml-[60px] flex flex-col h-screen">
        <Topbar ticker={ticker} price={price} change={change} onTickerChange={setTicker} />
        <main className="flex-1 overflow-auto p-4">
          <Routes>
            <Route
              path="/"
              element={<Dashboard ticker={ticker} onPriceUpdate={handlePriceUpdate} />}
            />
            <Route path="/prediction" element={<Prediction ticker={ticker} />} />
            <Route path="/backtest" element={<Backtest ticker={ticker} />} />
            <Route path="/models" element={<Models />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
