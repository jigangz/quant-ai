import { ExternalLink } from 'lucide-react';
import Badge from '../UI/Badge';
import type { NewsItem } from '../../api';

interface Props {
  news: NewsItem[];
  selectedDate: string | null;
  loading: boolean;
}

// Map sentiment score to a gradient background color
function sentimentGradient(score: number): string {
  if (score > 0.3) return 'bg-bull/10 border-l-2 border-l-bull';
  if (score < -0.3) return 'bg-bear/10 border-l-2 border-l-bear';
  return 'bg-dark-hover border-l-2 border-l-gray-600';
}

function sentimentBadgeVariant(label: string): 'success' | 'danger' | 'neutral' {
  if (label === 'bullish') return 'success';
  if (label === 'bearish') return 'danger';
  return 'neutral';
}

const categoryColors: Record<string, 'default' | 'success' | 'danger' | 'warning' | 'info' | 'neutral'> = {
  earnings: 'warning',
  policy: 'info',
  product: 'default',
  market: 'neutral',
  competition: 'danger',
  management: 'success',
};

export default function NewsPanel({ news, selectedDate, loading }: Props) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-sm text-gray-500 animate-pulse-slow">Loading news...</div>
      </div>
    );
  }

  if (!selectedDate) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-600">
        <div className="text-3xl mb-2">📰</div>
        <div className="text-sm">Click a date on the chart to view news</div>
      </div>
    );
  }

  if (news.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-600">
        <div className="text-3xl mb-2">🔍</div>
        <div className="text-sm">No news found for {selectedDate}</div>
      </div>
    );
  }

  // Sort by sentiment score (most bullish first)
  const sorted = [...news].sort((a, b) => b.sentiment_score - a.sentiment_score);

  return (
    <div className="flex flex-col gap-2 animate-fade-in">
      <div className="flex items-center justify-between px-1 mb-1">
        <span className="text-xs text-gray-500 font-mono">{selectedDate}</span>
        <span className="text-xs text-gray-600">{news.length} articles</span>
      </div>
      {sorted.map((item) => (
        <div
          key={item.id}
          className={`p-3 rounded-sm transition-colors ${sentimentGradient(item.sentiment_score)}`}
        >
          <div className="flex items-start gap-2 mb-1.5">
            <Badge variant={sentimentBadgeVariant(item.sentiment_label)}>
              {item.sentiment_label}
            </Badge>
            <Badge variant={categoryColors[item.category] || 'neutral'}>
              {item.category}
            </Badge>
            <span
              className="ml-auto text-xs font-mono px-1.5 py-0.5 rounded-sm"
              style={{
                // Gradient from red to green based on sentiment score
                backgroundColor: `rgba(${item.sentiment_score > 0 ? '16,185,129' : '239,68,68'}, ${Math.abs(item.sentiment_score) * 0.3})`,
                color: item.sentiment_score > 0 ? '#10b981' : item.sentiment_score < 0 ? '#ef4444' : '#888',
              }}
            >
              {item.sentiment_score > 0 ? '+' : ''}
              {item.sentiment_score.toFixed(2)}
            </span>
          </div>

          <a
            href={item.url}
            target="_blank"
            rel="noreferrer"
            className="text-sm text-gray-200 hover:text-white font-medium leading-snug flex items-start gap-1"
          >
            {item.title}
            <ExternalLink className="w-3 h-3 flex-shrink-0 mt-0.5 text-gray-600" />
          </a>

          {item.summary && (
            <p className="text-xs text-gray-500 mt-1.5 leading-relaxed line-clamp-2">
              {item.summary}
            </p>
          )}

          {(item.reason_bull || item.reason_bear) && (
            <div className="mt-2 flex flex-col gap-1">
              {item.reason_bull && (
                <div className="text-xs text-bull/80 flex items-start gap-1">
                  <span className="font-bold">▲</span>
                  <span>{item.reason_bull}</span>
                </div>
              )}
              {item.reason_bear && (
                <div className="text-xs text-bear/80 flex items-start gap-1">
                  <span className="font-bold">▼</span>
                  <span>{item.reason_bear}</span>
                </div>
              )}
            </div>
          )}

          <div className="mt-1.5 text-xs text-gray-600">{item.source}</div>
        </div>
      ))}
    </div>
  );
}
