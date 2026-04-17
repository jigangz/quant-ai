import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

const Sheet = DialogPrimitive.Root;
const SheetTrigger = DialogPrimitive.Trigger;
const SheetClose = DialogPrimitive.Close;
const SheetPortal = DialogPrimitive.Portal;

const SheetOverlay = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn("fixed inset-0 z-50 bg-background/80 backdrop-blur-sm", className)}
    {...props}
  />
));
SheetOverlay.displayName = "SheetOverlay";

const sheetVariants = cva(
  "fixed z-50 gap-4 bg-surface-card p-6 shadow-lg transition ease-in-out",
  {
    variants: {
      side: {
        top: "inset-x-0 top-0 border-b border-surface-border",
        bottom: "inset-x-0 bottom-0 border-t border-surface-border",
        left: "inset-y-0 left-0 h-full w-3/4 border-r border-surface-border sm:max-w-sm",
        right: "inset-y-0 right-0 h-full w-3/4 border-l border-surface-border sm:max-w-sm",
      },
    },
    defaultVariants: { side: "right" },
  }
);

const SheetContent = React.forwardRef(({ side = "right", className, children, ...props }, ref) => (
  <SheetPortal>
    <SheetOverlay />
    <DialogPrimitive.Content ref={ref} className={cn(sheetVariants({ side }), className)} {...props}>
      {children}
      <DialogPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-accent-ring">
        <X className="h-4 w-4" />
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </SheetPortal>
));
SheetContent.displayName = "SheetContent";

const SheetHeader = ({ className, ...props }) => (
  <div className={cn("flex flex-col gap-1.5 text-left", className)} {...props} />
);
const SheetFooter = ({ className, ...props }) => (
  <div className={cn("flex flex-col-reverse gap-2 sm:flex-row sm:justify-end", className)} {...props} />
);
const SheetTitle = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Title ref={ref} className={cn("text-lg font-semibold text-foreground", className)} {...props} />
));
SheetTitle.displayName = "SheetTitle";
const SheetDescription = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Description ref={ref} className={cn("text-sm text-muted", className)} {...props} />
));
SheetDescription.displayName = "SheetDescription";

export { Sheet, SheetPortal, SheetOverlay, SheetTrigger, SheetClose, SheetContent, SheetHeader, SheetFooter, SheetTitle, SheetDescription };
