import * as React from "react";
import * as ToastPrimitive from "@radix-ui/react-toast";
import { X } from "lucide-react";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

const ToastProvider = ToastPrimitive.Provider;
const ToastViewport = React.forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Viewport
    ref={ref}
    className={cn("fixed top-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px]", className)}
    {...props}
  />
));
ToastViewport.displayName = "ToastViewport";

const toastVariants = cva(
  "group pointer-events-auto relative flex w-full items-center justify-between space-x-4 overflow-hidden rounded-xl border p-6 pr-8 shadow-lg transition-all",
  {
    variants: {
      variant: {
        default: "border-surface-border bg-surface-card text-foreground",
        destructive: "border-down/50 bg-down/10 text-down",
        success: "border-up/50 bg-up/10 text-up",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

const Toast = React.forwardRef(({ className, variant, ...props }, ref) => (
  <ToastPrimitive.Root ref={ref} className={cn(toastVariants({ variant }), className)} {...props} />
));
Toast.displayName = "Toast";

const ToastAction = React.forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Action
    ref={ref}
    className={cn("inline-flex h-8 shrink-0 items-center justify-center rounded-md border border-surface-border bg-transparent px-3 text-sm font-medium transition-colors hover:bg-surface-hover focus:outline-none focus:ring-2 focus:ring-accent-ring", className)}
    {...props}
  />
));
ToastAction.displayName = "ToastAction";

const ToastClose = React.forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Close
    ref={ref}
    className={cn("absolute right-2 top-2 rounded-md p-1 opacity-0 transition-opacity hover:opacity-100 focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100", className)}
    toast-close=""
    {...props}
  >
    <X className="h-4 w-4" />
  </ToastPrimitive.Close>
));
ToastClose.displayName = "ToastClose";

const ToastTitle = React.forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Title ref={ref} className={cn("text-sm font-semibold", className)} {...props} />
));
ToastTitle.displayName = "ToastTitle";

const ToastDescription = React.forwardRef(({ className, ...props }, ref) => (
  <ToastPrimitive.Description ref={ref} className={cn("text-sm opacity-90", className)} {...props} />
));
ToastDescription.displayName = "ToastDescription";

export { ToastProvider, ToastViewport, Toast, ToastAction, ToastClose, ToastTitle, ToastDescription };
