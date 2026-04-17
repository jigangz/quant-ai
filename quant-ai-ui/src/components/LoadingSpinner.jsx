import { cn } from "../lib/utils";

export default function LoadingSpinner({ size = "md", className }) {
  const sizes = { sm: "h-4 w-4", md: "h-6 w-6", lg: "h-10 w-10" };
  return (
    <div
      role="status"
      aria-label="Loading"
      className={cn(
        "animate-spin rounded-full border-2 border-surface-border border-t-accent",
        sizes[size],
        className
      )}
    />
  );
}

export function LoadingOverlay({ label = "Loading..." }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3">
      <LoadingSpinner size="lg" />
      <p className="text-sm text-muted">{label}</p>
    </div>
  );
}
