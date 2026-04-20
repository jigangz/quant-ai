const TABS = ["概览", "新闻", "社区", "技术指标", "模型历史", "预测记录"];

export function SymbolTabs({ active = "概览", onChange = () => {} }) {
  return (
    <div className="border-b border-surface-border flex gap-5 mb-4">
      {TABS.map((t) => (
        <button
          key={t}
          onClick={() => onChange(t)}
          className={`py-2.5 text-sm transition-colors ${
            active === t ? "text-foreground font-bold border-b-2 border-foreground" : "text-muted hover:text-foreground"
          }`}
        >
          {t}
        </button>
      ))}
    </div>
  );
}
