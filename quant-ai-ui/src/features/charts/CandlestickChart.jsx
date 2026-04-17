import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

export default function CandlestickChart({ data, markers = [], height = 420 }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      autoSize: true,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "rgb(24 24 27)" },
        textColor: "rgb(161 161 170)",
        fontFamily: "Geist, system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: "rgb(39 39 42)" },
        horzLines: { color: "rgb(39 39 42)" },
      },
      rightPriceScale: { borderColor: "rgb(63 63 70)" },
      timeScale: { borderColor: "rgb(63 63 70)", timeVisible: true },
      crosshair: { mode: 1 },
    });
    chartRef.current = chart;
    const series = chart.addCandlestickSeries({
      upColor: "rgb(16 185 129)",
      downColor: "rgb(244 63 94)",
      borderVisible: false,
      wickUpColor: "rgb(16 185 129)",
      wickDownColor: "rgb(244 63 94)",
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
