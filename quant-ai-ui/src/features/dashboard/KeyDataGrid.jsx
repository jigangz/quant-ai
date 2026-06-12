function fmtVolume(v) {
  if (!v) return "—";
  if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return String(v);
}

export function KeyDataGrid({ latestCandle, prevClose }) {
  const items = [
    { label: "Volume", value: fmtVolume(latestCandle?.volume) },
    { label: "Previous Close", value: prevClose?.toFixed(2) ?? "—" },
    { label: "Open", value: latestCandle?.open?.toFixed(2) ?? "—" },
    { label: "Day's Range", value: latestCandle ? `${latestCandle.low?.toFixed(2)} — ${latestCandle.high?.toFixed(2)}` : "—" },
  ];
  return (
    <div className="mb-4">
      <h3 className="text-sm font-bold text-foreground mb-2">Key Stats</h3>
      <div className="grid grid-cols-4 gap-3">
        {items.map((i) => (
          <div key={i.label}>
            <div className="text-[10px] text-muted">{i.label}</div>
            <div className="text-sm font-mono text-foreground">{i.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
