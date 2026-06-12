import { useEffect, useRef, useState } from "react";
import { createChart } from "lightweight-charts";
import { PerformancePills } from "./PerformancePills";

// Hex colors for Lightweight Charts canvas — CSS vars not supported in canvas context
const CHART_COLORS = {
  bg: "transparent",
  text: "#a1a1aa",
  grid: "#27272a",
  border: "#3f3f46",
  up: "#10b981",
  down: "#f43f5e",
};

const RANGE_TO_DAYS = {
  "1D": 1,
  "5D": 5,
  "1M": 30,
  "6M": 180,
  "YTD": null,
  "1Y": 365,
  "5Y": 1825,
  "10Y": 3650,
  "ALL": null,
};

function filterCandlesByRange(candles, rangeKey) {
  if (!candles.length) return candles;
  const sorted = [...candles].sort((a, b) => new Date(a.date) - new Date(b.date));
  const last = sorted[sorted.length - 1];
  if (rangeKey === "ALL") return sorted;
  if (rangeKey === "YTD") {
    const start = new Date(new Date(last.date).getFullYear(), 0, 1);
    return sorted.filter((c) => new Date(c.date) >= start);
  }
  const days = RANGE_TO_DAYS[rangeKey];
  if (!days) return sorted;
  const cutoff = new Date(new Date(last.date).getTime() - days * 24 * 3600 * 1000);
  return sorted.filter((c) => new Date(c.date) >= cutoff);
}

export function ChartSection({ candles = [], isLoading = false }) {
  const containerRef = useRef(null);
  const [range, setRange] = useState("6M");

  useEffect(() => {
    if (!containerRef.current || !candles || candles.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 380,
      layout: {
        background: { color: CHART_COLORS.bg },
        textColor: CHART_COLORS.text,
        fontFamily: "Geist, system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: CHART_COLORS.grid },
        horzLines: { color: CHART_COLORS.grid },
      },
      rightPriceScale: { borderColor: CHART_COLORS.border },
      timeScale: { borderColor: CHART_COLORS.border },
    });

    const series = chart.addCandlestickSeries({
      upColor: CHART_COLORS.up,
      downColor: CHART_COLORS.down,
      borderVisible: false,
      wickUpColor: CHART_COLORS.up,
      wickDownColor: CHART_COLORS.down,
    });

    const filtered = filterCandlesByRange(candles, range);
    series.setData(
      filtered.map((c) => ({
        time: c.date,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
    );
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [candles, range]);

  return (
    <div className="bg-surface border border-surface-border rounded-md p-3 mb-4">
      <div className="flex justify-between items-center mb-2">
        <div className="text-sm font-bold text-foreground">Chart ›</div>
        <div className="text-[10px] text-muted">Full Chart · &lt;/&gt;</div>
      </div>
      {isLoading ? (
        <div className="h-[380px] bg-surface-muted rounded animate-pulse" />
      ) : (
        <div ref={containerRef} className="w-full h-[380px]" />
      )}
      <PerformancePills candles={candles} activeRange={range} onChange={setRange} />
    </div>
  );
}
