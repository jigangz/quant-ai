import * as React from "react";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none",
  {
    variants: {
      variant: {
        default: "border-transparent bg-accent text-accent-foreground",
        secondary: "border-transparent bg-surface-muted text-foreground",
        outline: "border-surface-border text-foreground",
        success: "border-transparent bg-up/20 text-up",
        destructive: "border-transparent bg-down/20 text-down",
        warning: "border-transparent bg-warn/20 text-warn",
        info: "border-transparent bg-info/20 text-info",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

function Badge({ className, variant, ...props }) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
