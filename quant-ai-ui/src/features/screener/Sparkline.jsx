export default function Sparkline({ values = [], width = 88, height = 24 }) {
  if (!values || values.length < 2) {
    return <svg width={width} height={height} />;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / (values.length - 1);

  const points = values
    .map((v, i) => `${(i * step).toFixed(2)},${(height - ((v - min) / range) * height).toFixed(2)}`)
    .join(" ");

  const up = values[values.length - 1] >= values[0];
  const stroke = up ? "#10b981" : "#f43f5e";
  const fill = up ? "rgba(16,185,129,0.12)" : "rgba(244,63,94,0.12)";

  const areaPoints = `0,${height} ${points} ${width},${height}`;

  return (
    <svg width={width} height={height} className="block">
      <polyline points={areaPoints} fill={fill} stroke="none" />
      <polyline
        points={points}
        fill="none"
        stroke={stroke}
        strokeWidth={1.25}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
