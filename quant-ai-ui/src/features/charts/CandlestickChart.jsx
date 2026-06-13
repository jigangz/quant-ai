import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

export default function CandlestickChart({ data, markers = [], height = 420 }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    // Lightweight Charts parser only accepts comma-separated rgb() or hex — not Tailwind's space-separated syntax
    const COLORS = {
      bg: "#ffffff",        // light page
      text: "#71717a",      // zinc-500
      grid: "#e5e7eb",      // gray-200
      border: "#d4d4d8",    // zinc-300
      up: "#059669",        // emerald-600
      down: "#e11d48",      // rose-600
    };

    const chart = createChart(containerRef.current, {
      autoSize: true,
      height,
      layout: {
        background: { type: ColorType.Solid, color: COLORS.bg },
        textColor: COLORS.text,
        fontFamily: "Geist, system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: COLORS.grid },
        horzLines: { color: COLORS.grid },
      },
      rightPriceScale: { borderColor: COLORS.border },
      timeScale: { borderColor: COLORS.border, timeVisible: true },
      crosshair: { mode: 1 },
    });
    chartRef.current = chart;
    const series = chart.addCandlestickSeries({
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderVisible: false,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
    });
    seriesRef.current = series;

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    if (!seriesRef.current || !data || data.length === 0) return;
    seriesRef.current.setData(data);
    if (chartRef.current) chartRef.current.timeScale().fitContent();
  }, [data]);

  useEffect(() => {
    if (!seriesRef.current || !markers) return;
    seriesRef.current.setMarkers(markers);
  }, [markers]);

  return <div ref={containerRef} style={{ width: "100%", height }} />;
}
