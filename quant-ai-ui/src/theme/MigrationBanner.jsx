export function MigrationBanner({ currentPath, migratedPaths, allPaths = [] }) {
  if (migratedPaths.includes(currentPath)) return null;
  const unmigrated = allPaths
    .filter((p) => !migratedPaths.includes(p.path))
    .map((p) => p.label)
    .join(" · ");
  return (
    <div className="bg-accent/10 border-b border-accent/20 px-4 py-2 text-xs text-foreground">
      🎨 Quant AI 正在换新视觉。以下页面仍是旧样式：<span className="font-medium">{unmigrated}</span>。近期更新。
    </div>
  );
}
