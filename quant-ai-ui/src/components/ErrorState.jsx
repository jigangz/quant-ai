import { AlertCircle } from "lucide-react";
import { Button } from "./ui/button";

export default function ErrorState({ error, onRetry, className = "" }) {
  const msg = error?.message || String(error || "Unknown error");
  return (
    <div className={`flex items-start gap-3 rounded-lg border border-down/40 bg-down/10 p-4 ${className}`}>
      <AlertCircle className="h-5 w-5 text-down flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <p className="text-sm font-medium text-down">Error</p>
        <p className="text-sm text-foreground mt-1">{msg}</p>
      </div>
      {onRetry && (
        <Button size="sm" variant="outline" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}
