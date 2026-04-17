export default function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="flex items-start justify-between mb-6 pb-4 border-b border-surface-border">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">{title}</h1>
        {subtitle && <p className="text-sm text-muted mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
