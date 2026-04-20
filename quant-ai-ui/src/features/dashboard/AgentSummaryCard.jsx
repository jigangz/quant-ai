export function AgentSummaryCard({ summary }) {
  return (
    <div className="bg-surface border border-surface-border border-l-[3px] border-l-accent rounded-md p-3">
      <div className="text-[9px] uppercase tracking-wide text-accent">⚡ 为什么这么说</div>
      <p className="italic text-xs text-foreground leading-relaxed mt-1.5">
        {summary ?? "AI 分析生成中..."}
      </p>
    </div>
  );
}
