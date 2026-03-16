import { useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import type { MarketDataPoint, NewsItem } from '../../api';

interface NewsDot {
  date: string;
  count: number;
  avgSentiment: number;
}

interface Props {
  data: MarketDataPoint[];
  newsDots: NewsDot[];
  onDateClick: (date: string) => void;
  onHover: (date: string | null, ohlc?: MarketDataPoint) => void;
}

// Map sentiment score (-1..1) to color gradient (red..green)
function sentimentColor(score: number): string {
  if (score > 0.2) return '#10b981';
  if (score < -0.2) return '#ef4444';
  return '#6366f1';
}

export default function CandlestickChart({ data, newsDots, onDateClick, onHover }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const drawRef = useRef<() => void>(() => {});

  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const container = containerRef.current;
    if (!container || data.length === 0) return;

    const fullWidth = container.clientWidth;
    const fullHeight = container.clientHeight || 500;
    const margin = { top: 20, right: 50, bottom: 30, left: 60 };
    const width = fullWidth - margin.left - margin.right;
    const height = fullHeight - margin.top - margin.bottom;

    svg.attr('width', fullWidth).attr('height', fullHeight);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const parsedData = data.map((d, i) => ({
      ...d,
      dateObj: new Date(d.date),
      change: i > 0 ? ((d.close - data[i - 1].close) / data[i - 1].close) * 100 : 0,
    }));

    // Build news dot lookup
    const newsMap = new Map<string, NewsDot>();
    for (const nd of newsDots) {
      newsMap.set(nd.date, nd);
    }

    // Scales
    const x = d3.scaleTime()
      .domain(d3.extent(parsedData, (d) => d.dateObj) as [Date, Date])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([
        d3.min(parsedData, (d) => d.low)! * 0.995,
        d3.max(parsedData, (d) => d.high)! * 1.005,
      ])
      .range([height, 0]);

    // Grid
    g.append('g')
      .call(d3.axisLeft(y).ticks(6).tickSize(-width).tickFormat(() => ''))
      .selectAll('line')
      .style('stroke', '#1a1d27')
      .style('stroke-width', 1);
    g.selectAll('.domain').remove();

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(8).tickFormat(d3.timeFormat('%b %d') as any))
      .selectAll('text')
      .style('font-size', '11px')
      .style('fill', '#555')
      .style('font-family', 'JetBrains Mono, monospace');
    g.selectAll('.domain').style('stroke', '#2a2d3a');

    // Y axis
    g.append('g')
      .call(d3.axisRight(y).ticks(6).tickFormat((d) => `$${Number(d).toFixed(0)}`))
      .attr('transform', `translate(${width},0)`)
      .selectAll('text')
      .style('font-size', '11px')
      .style('fill', '#555')
      .style('font-family', 'JetBrains Mono, monospace');

    g.selectAll('.tick line').style('stroke', '#2a2d3a');

    const candleWidth = Math.max(2, (width / parsedData.length) * 0.6);

    // Candles
    const candles = g.selectAll('.candle')
      .data(parsedData)
      .enter()
      .append('g')
      .attr('class', 'candle');

    // Wicks
    candles.append('line')
      .attr('x1', (d) => x(d.dateObj))
      .attr('x2', (d) => x(d.dateObj))
      .attr('y1', (d) => y(d.high))
      .attr('y2', (d) => y(d.low))
      .attr('stroke', (d) => (d.close >= d.open ? '#10b981' : '#ef4444'))
      .attr('stroke-width', 1);

    // Bodies
    candles.append('rect')
      .attr('x', (d) => x(d.dateObj) - candleWidth / 2)
      .attr('y', (d) => y(Math.max(d.open, d.close)))
      .attr('width', candleWidth)
      .attr('height', (d) => Math.max(1, Math.abs(y(d.open) - y(d.close))))
      .attr('fill', (d) => (d.close >= d.open ? '#10b981' : '#ef4444'))
      .attr('rx', 1);

    // News dots on top of candles
    const dotsData = parsedData
      .filter((d) => newsMap.has(d.date))
      .map((d) => ({ ...d, news: newsMap.get(d.date)! }));

    const dots = g.selectAll('.news-dot')
      .data(dotsData)
      .enter()
      .append('g')
      .attr('class', 'news-dot')
      .style('cursor', 'pointer');

    dots.append('circle')
      .attr('cx', (d) => x(d.dateObj))
      .attr('cy', (d) => y(d.high) - 12)
      .attr('r', 4)
      .attr('fill', (d) => sentimentColor(d.news.avgSentiment))
      .attr('stroke', '#0f1117')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.9);

    // Dot count label
    dots.filter((d) => d.news.count > 1)
      .append('text')
      .attr('x', (d) => x(d.dateObj))
      .attr('y', (d) => y(d.high) - 22)
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .attr('fill', '#888')
      .attr('font-family', 'JetBrains Mono, monospace')
      .text((d) => d.news.count);

    dots.on('click', (_event, d) => {
      onDateClick(d.date);
    });

    // Crosshair + hover
    const crossV = g.append('line')
      .style('stroke', '#333')
      .style('stroke-width', 0.5)
      .style('stroke-dasharray', '4,3')
      .style('display', 'none')
      .style('pointer-events', 'none');

    const crossH = g.append('line')
      .style('stroke', '#333')
      .style('stroke-width', 0.5)
      .style('stroke-dasharray', '4,3')
      .style('display', 'none')
      .style('pointer-events', 'none');

    const bisect = d3.bisector<typeof parsedData[0], Date>((d) => d.dateObj).left;

    function snapToData(px: number) {
      const xDate = x.invert(px);
      const idx = bisect(parsedData, xDate, 1);
      const d0 = parsedData[idx - 1];
      const d1 = parsedData[idx];
      if (!d0) return parsedData[0];
      return d1 && xDate.getTime() - d0.dateObj.getTime() > d1.dateObj.getTime() - xDate.getTime()
        ? d1
        : d0;
    }

    // Hover overlay
    g.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'transparent')
      .style('cursor', 'crosshair')
      .on('mousemove', function (event) {
        const [mx, my] = d3.pointer(event);
        const d = snapToData(mx);
        const cx = x(d.dateObj);

        crossV.attr('x1', cx).attr('x2', cx).attr('y1', 0).attr('y2', height).style('display', null);
        crossH.attr('x1', 0).attr('x2', width).attr('y1', my).attr('y2', my).style('display', null);

        onHover(d.date, d);

        // Tooltip for news dots
        const nd = newsMap.get(d.date);
        const tooltip = tooltipRef.current;
        if (tooltip && nd) {
          tooltip.style.display = 'block';
          tooltip.innerHTML = `<div class="text-xs font-mono">${d.date}</div><div class="text-xs text-gray-400">${nd.count} news article${nd.count > 1 ? 's' : ''}</div>`;
          tooltip.style.left = `${margin.left + cx + 12}px`;
          tooltip.style.top = `${margin.top + y(d.high) - 30}px`;
        } else if (tooltip) {
          tooltip.style.display = 'none';
        }
      })
      .on('mouseleave', function () {
        crossV.style('display', 'none');
        crossH.style('display', 'none');
        onHover(null);
        const tooltip = tooltipRef.current;
        if (tooltip) tooltip.style.display = 'none';
      })
      .on('click', function (event) {
        const [mx] = d3.pointer(event);
        const d = snapToData(mx);
        onDateClick(d.date);
      });
  }, [data, newsDots, onDateClick, onHover]);

  drawRef.current = draw;

  useEffect(() => {
    draw();
    const handleResize = () => drawRef.current();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [draw]);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600">
        <div className="text-center">
          <div className="text-4xl mb-2">📈</div>
          <div className="text-sm">No market data available</div>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative w-full h-full">
      <svg ref={svgRef} className="w-full h-full" />
      <div
        ref={tooltipRef}
        className="absolute z-10 bg-dark-card border border-dark-border rounded-sm px-3 py-2 pointer-events-none shadow-lg"
        style={{ display: 'none' }}
      />
    </div>
  );
}
