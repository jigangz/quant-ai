import { useEffect, useState } from "react";
import { useMetaLabelModels } from "@/api/signalQueries";
import * as api from "@/api/client";

/** Last-7-days reliability score mini-line. */
export default function MetaSparkline({ ticker }) {
  const { data: models = [] } = useMetaLabelModels(ticker);
  const model = models[0];
  const [series, setSeries] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!model?.model_id) return;
    let cancelled = false;
    setLoading(true);
    (async () => {
      const now = new Date();
      const days = Array.from({ length: 7 }, (_, i) => {
        const d = new Date(now);
        d.setDate(now.getDate() - (6 - i));
        return d.toISOString().slice(0, 10);
      });
      const scores = await Promise.all(
        days.map((day) =>
          api
            .postSignalScore({
              ticker,
              meta_model_id: model.model_id,
              signal: 1,
              timestamp: day,
            })
            .then((r) => (r?.triggered ? r.reliability_score : null))
            .catch(() => null)
        ),
      );
      if (!cancelled) {
        setSeries(scores);
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [model?.model_id, ticker]);

  if (!model) return null;
  if (loading && series.length === 0) {
    return <div className="text-[10px] text-slate-500 mt-2">Loading signal quality...</div>;
  }

  const values = series.filter((v) => v !== null);
  if (values.length === 0) {
    return <div className="text-[10px] text-slate-500 mt-2">No recent triggers</div>;
  }

  // Simple inline SVG sparkline (no dep)
  const W = 120, H = 28;
  const min = Math.min(...values, 0.4);
  const max = Math.max(...values, 0.7);
  const range = Math.max(max - min, 0.01);
  const points = series
    .map((v, i) => {
      if (v === null) return null;
      const x = (i * (W - 6)) / 6 + 3;
      const y = H - 3 - ((v - min) / range) * (H - 6);
      return `${x},${y}`;
    })
    .filter(Boolean)
    .join(" ");

  return (
    <div className="mt-2 flex items-center gap-2">
      <div className="text-[10px] text-slate-400">7d signal quality:</div>
      <svg width={W} height={H} className="overflow-visible">
        <polyline points={points} fill="none" stroke="rgb(16 185 129)" strokeWidth="1.5" />
      </svg>
      <div className="text-[10px] text-slate-500">
        {values[values.length - 1].toFixed(2)}
      </div>
    </div>
  );
}
