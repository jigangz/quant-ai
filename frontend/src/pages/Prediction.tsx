import { useState, useEffect, useRef } from 'react';
import { Play, ArrowUpRight, ArrowDownRight, Clock } from 'lucide-react';
import Card from '../components/UI/Card';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import ShapChart from '../components/Charts/ShapChart';
import Badge from '../components/UI/Badge';
import { predict, fetchExplanation, fetchSimilarDays, type PredictionResult, type ShapFeature, type SimilarDay } from '../api';

interface Props {
  ticker: string;
}

// Animated counter component
function AnimatedNumber({ value, suffix = '%' }: { value: number; suffix?: string }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<number>(0);

  useEffect(() => {
    ref.current = 0;
    const target = value;
    const duration = 800;
    const start = performance.now();

    function tick(now: number) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      ref.current = eased * target;
      setDisplay(ref.current);
      if (progress < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  }, [value]);

  return (
    <span className="font-mono font-bold animate-count-up">
      {display.toFixed(1)}{suffix}
    </span>
  );
}

function PredictionCard({
  label,
  direction,
  confidence,
}: {
  label: string;
  direction: 'UP' | 'DOWN';
  confidence: number;
}) {
  const isUp = direction === 'UP';
  return (
    <Card className={`p-4 ${isUp ? 'border-l-2 border-l-bull' : 'border-l-2 border-l-bear'}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-gray-500 font-medium">{label}</span>
        <div
          className={`flex items-center gap-1 px-2 py-1 rounded-sm text-sm font-bold ${
            isUp ? 'bg-bull/20 text-bull' : 'bg-bear/20 text-bear'
          }`}
        >
          {isUp ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
          {direction}
        </div>
      </div>
      <div className="mb-2">
        <div className="text-2xl">
          <AnimatedNumber value={confidence * 100} />
        </div>
        <div className="text-xs text-gray-500 mt-0.5">confidence</div>
      </div>
      {/* Progress bar */}
      <div className="h-1.5 bg-dark-bg rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${isUp ? 'bg-bull' : 'bg-bear'}`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>
    </Card>
  );
}

export default function Prediction({ ticker }: Props) {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [shapData, setShapData] = useState<ShapFeature[]>([]);
  const [similarDays, setSimilarDays] = useState<SimilarDay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict() {
    setLoading(true);
    setError(null);
    try {
      const [pred, shap, similar] = await Promise.all([
        predict(ticker),
        fetchExplanation(ticker),
        fetchSimilarDays(ticker, new Date().toISOString().slice(0, 10)),
      ]);
      setPrediction(pred);
      setShapData(shap);
      setSimilarDays(similar);
    } catch {
      setError('Failed to generate prediction');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-4">
        <span className="text-lg font-semibold text-gray-200">
          Prediction — <span className="font-mono text-accent">{ticker}</span>
        </span>
        <button
          onClick={handlePredict}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent/80 text-white text-sm font-medium rounded-sm transition-colors disabled:opacity-50"
        >
          <Play className="w-4 h-4" />
          {loading ? 'Predicting...' : 'Run Prediction'}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-bear/10 border border-bear/30 rounded-sm text-sm text-bear">
          {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Generating predictions..." />}

      {prediction && !loading && (
        <div className="grid grid-cols-12 gap-4">
          {/* Prediction cards */}
          <div className="col-span-8">
            <div className="grid grid-cols-3 gap-4 mb-4">
              <PredictionCard
                label="T+1 (Next Day)"
                direction={prediction.horizons.t1.direction}
                confidence={prediction.horizons.t1.confidence}
              />
              <PredictionCard
                label="T+3 (3 Days)"
                direction={prediction.horizons.t3.direction}
                confidence={prediction.horizons.t3.confidence}
              />
              <PredictionCard
                label="T+5 (1 Week)"
                direction={prediction.horizons.t5.direction}
                confidence={prediction.horizons.t5.confidence}
              />
            </div>

            {/* SHAP chart */}
            <Card title="Feature Importance (SHAP)" className="h-[360px]">
              <div className="p-2 h-[310px]">
                <ShapChart data={shapData} />
              </div>
            </Card>
          </div>

          {/* Similar days sidebar */}
          <div className="col-span-4">
            <Card title="Similar Historical Days" className="h-full">
              <div className="p-3 flex flex-col gap-2 overflow-y-auto max-h-[600px]">
                {similarDays.length === 0 ? (
                  <div className="text-center text-gray-600 py-8">
                    <Clock className="w-8 h-8 mx-auto mb-2" />
                    <div className="text-sm">No similar days found</div>
                  </div>
                ) : (
                  similarDays.map((day) => (
                    <div
                      key={day.date}
                      className="p-3 bg-dark-bg rounded-sm border border-dark-border animate-slide-in"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-mono text-gray-300">{day.date}</span>
                        <Badge variant={day.price_change >= 0 ? 'success' : 'danger'}>
                          {day.price_change >= 0 ? '+' : ''}
                          {day.price_change.toFixed(2)}%
                        </Badge>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-gray-500">
                        <span>
                          Similarity:{' '}
                          <span className="font-mono text-accent">
                            {(day.similarity * 100).toFixed(0)}%
                          </span>
                        </span>
                        <span>{day.news_count} articles</span>
                        <span
                          className={`font-mono ${
                            day.sentiment_score > 0 ? 'text-bull' : 'text-bear'
                          }`}
                        >
                          {day.sentiment_score > 0 ? '+' : ''}
                          {day.sentiment_score.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!prediction && !loading && !error && (
        <div className="flex flex-col items-center justify-center py-24 text-gray-600">
          <div className="text-6xl mb-4">🎯</div>
          <div className="text-lg font-medium mb-1">Ready to Predict</div>
          <div className="text-sm">Click "Run Prediction" to generate T+1/T+3/T+5 forecasts</div>
        </div>
      )}
    </div>
  );
}
