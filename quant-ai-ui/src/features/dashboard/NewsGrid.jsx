import { useState } from "react";

function timeLabel(iso) {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diff / (24 * 3600 * 1000));
  if (days === 0) return "今天";
  if (days === 1) return "昨天";
  if (days === 2) return "前天";
  if (days < 30) return `${days} 天前`;
  return new Date(iso).toLocaleDateString("zh-CN");
}

export function NewsGrid({ items = [] }) {
  const [expanded, setExpanded] = useState(false);
  if (!items.length) {
    return (
      <section className="mb-4">
        <h3 className="text-sm font-bold text-foreground">新闻 ›</h3>
        <p className="text-xs text-muted mt-2">新闻暂不可用</p>
      </section>
    );
  }
  const visible = expanded ? items : items.slice(0, 8);
  return (
    <section className="mb-4">
      <h3 className="text-sm font-bold text-foreground mb-2">新闻 ›</h3>
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
          继续阅读
        </button>
      )}
    </section>
  );
}
