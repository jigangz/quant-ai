import { Inbox } from "lucide-react";
import { Button } from "./ui/button";

export default function EmptyState({
  icon: Icon = Inbox,
  title = "Nothing here yet",
  description,
  actionLabel,
  onAction,
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-surface-muted">
        <Icon className="h-8 w-8 text-muted" />
      </div>
      <h3 className="mt-4 text-lg font-semibold text-foreground">{title}</h3>
      {description && <p className="mt-1 text-sm text-muted max-w-sm">{description}</p>}
      {actionLabel && onAction && (
        <Button className="mt-4" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  );
}
