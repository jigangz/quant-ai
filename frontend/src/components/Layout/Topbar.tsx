import { useState } from 'react';
import { Search, TrendingUp, TrendingDown } from 'lucide-react';

interface TopbarProps {
  ticker: string;
  price: number | null;
  change: number | null;
  onTickerChange: (ticker: string) => void;
}

export default function Topbar({ ticker, price, change, onTickerChange }: TopbarProps) {
  const [input, setInput] = useState('');

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const val = input.trim().toUpperCase();
    if (val) {
      onTickerChange(val);
      setInput('');
    }
  }

  const isUp = (change ?? 0) >= 0;

  return (
    <header className="h-14 bg-dark-card border-b border-dark-border flex items-center px-6 gap-6">
      {/* Search */}
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Search ticker..."
            className="w-48 h-8 pl-8 pr-3 text-sm bg-dark-bg border border-dark-border rounded-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-accent transition-colors"
          />
        </div>
      </form>

      {/* Current ticker info */}
      <div className="flex items-center gap-4">
        <span className="text-lg font-bold text-white font-mono">{ticker}</span>
        {price !== null && (
          <span className="text-lg font-mono font-semibold text-gray-200">
            ${price.toFixed(2)}
          </span>
        )}
        {change !== null && (
          <span
            className={`flex items-center gap-1 text-sm font-mono font-semibold ${
              isUp ? 'text-bull' : 'text-bear'
            }`}
          >
            {isUp ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
            {isUp ? '+' : ''}
            {change.toFixed(2)}%
          </span>
        )}
      </div>

      {/* Right spacer / branding */}
      <div className="ml-auto flex items-center gap-2">
        <span className="text-xs text-gray-600">QuantAI v1.0</span>
      </div>
    </header>
  );
}
