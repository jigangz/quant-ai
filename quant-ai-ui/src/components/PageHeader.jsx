export default function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="flex items-start justify-between gap-4 mb-6 pb-4 border-b border-surface-border">
      <div className="min-w-0">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground truncate">
          {title}
        </h1>
        {subtitle && <p className="text-sm text-muted mt-1 break-words">{subtitle}</p>}
      </div>
      {actions && (
        <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">{actions}</div>
      )}
    </div>
  );
}
