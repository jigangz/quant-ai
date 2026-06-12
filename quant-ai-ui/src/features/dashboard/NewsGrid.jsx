import { useState } from "react";

function timeLabel(iso) {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diff / (24 * 3600 * 1000));
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days === 2) return "2 days ago";
  if (days < 30) return `${days} days ago`;
  return new Date(iso).toLocaleDateString("en-US");
}

export function NewsGrid({ items = [] }) {
  const [expanded, setExpanded] = useState(false);
  if (!items.length) {
    return (
      <section className="mb-4">
        <h3 className="text-sm font-bold text-foreground">News ›</h3>
        <p className="text-xs text-muted mt-2">News unavailable</p>
      </section>
    );
  }
  const visible = expanded ? items : items.slice(0, 8);
  return (
    <section className="mb-4">
      <h3 className="text-sm font-bold text-foreground mb-2">News ›</h3>
      <div className="grid grid-cols-4 gap-2.5">
        {visible.map((n, i) => (
          <a
            key={n.id ?? i}
            href={n.url ?? "#"}
            target="_blank"
            rel="noreferrer"
            className="text-[10px] hover:bg-surface-muted p-1 rounded"
          >
            <div className="text-muted text-[9px] mb-1">{timeLabel(n.published_at)} · {n.source ?? "Reuters"}</div>
            <div className="text-foreground line-clamp-2 leading-tight">{n.title}</div>
          </a>
        ))}
      </div>
      {items.length > 8 && !expanded && (
        <button onClick={() => setExpanded(true)} className="text-xs text-accent mt-2 hover:underline">
          Read more
        </button>
      )}
    </section>
  );
}
