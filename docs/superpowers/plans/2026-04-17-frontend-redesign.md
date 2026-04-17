# Frontend Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the 6-page Quant AI React UI into a pro-grade distributed-quant workbench with Tremor + shadcn/ui + Lightweight Charts + TanStack Query + Zustand, dark Linear-style theme, Geist fonts, and TradingView-style Dashboard.

**Architecture:** Monorepo React app with domain-grouped features (`features/charts/`, `features/trading/`, etc.), shadcn primitives owned in `components/ui/`, TanStack Query hooks in `api/queries.js`, Zustand for WebSocket live prices, Tailwind design tokens for dark theme.

**Tech Stack:** React 19 + Vite + Tailwind CSS + Tremor + shadcn/ui (Radix) + Lightweight Charts v4 + TanStack Query v5 + Zustand + react-hook-form + zod + Vitest + Geist Sans/Mono.

---

## File Structure (target after implementation)

```
quant-ai-ui/
├── package.json                  modify: add deps
├── tailwind.config.js            modify: tokens + Tremor plugin
├── postcss.config.js             unchanged
├── vite.config.js                modify: add vitest config
├── index.html                    modify: Geist font preload
├── vitest.config.js              create
└── src/
    ├── main.jsx                  modify: wrap Providers
    ├── index.css                 modify: CSS variables + base
    ├── app/
    │   ├── AppShell.jsx          create
    │   ├── Sidebar.jsx           create
    │   ├── Providers.jsx         create
    │   └── router.jsx            create
    ├── pages/
    │   ├── ScreenerPage.jsx      create (replaces Screener.jsx)
    │   ├── DashboardPage.jsx     create (replaces Dashboard.jsx)
    │   ├── TrainingPage.jsx      create (replaces Training.jsx)
    │   ├── StrategyPage.jsx      create (replaces Strategy.jsx)
    │   ├── TradingPage.jsx       create (replaces Trading.jsx)
    │   └── ExplainPage.jsx       create (replaces Explain.jsx)
    ├── features/
    │   ├── charts/CandlestickChart.jsx  create
    │   ├── charts/EquityCurve.jsx       create
    │   ├── trading/OrderForm.jsx        create
    │   ├── trading/PortfolioCard.jsx    create
    │   ├── trading/PositionsList.jsx    create
    │   ├── trading/TradeHistory.jsx     create
    │   ├── trading/useLivePrices.js     create
    │   ├── training/TrainForm.jsx       create
    │   ├── training/EnsembleConfigFields.jsx create
    │   ├── training/HyperparamSearchFields.jsx create
    │   ├── training/RunsTable.jsx       create
    │   ├── training/ModelsTable.jsx     create
    │   ├── strategy/StrategyPicker.jsx  create
    │   ├── strategy/StrategyParamsForm.jsx create
    │   ├── strategy/BacktestResults.jsx create
    │   ├── explain/ShapFeatureList.jsx  create
    │   └── screener/ScreenerTable.jsx   create
    ├── components/
    │   ├── ui/                   created via shadcn CLI
    │   ├── PageHeader.jsx        create
    │   ├── EmptyState.jsx        create
    │   ├── LoadingSpinner.jsx    create
    │   ├── ErrorBoundary.jsx     create
    │   ├── ErrorState.jsx        create
    │   ├── TickerSearch.jsx      create
    │   └── ConfirmDialog.jsx     create
    ├── api/
    │   ├── client.js             existing, unchanged
    │   └── queries.js            create
    ├── stores/
    │   └── liveStore.js          create
    ├── hooks/
    │   └── useWebSocket.js       create
    ├── lib/
    │   ├── utils.js              create (shadcn cn() helper)
    │   └── formatters.js         create
    ├── __tests__/
    │   ├── pages.smoke.test.jsx  create
    │   └── forms.test.jsx        create
    └── App.jsx                   delete (replaced by app/AppShell.jsx + app/router.jsx)
```

Old files to **delete in FE-17** once all pages migrated:
- `src/App.jsx` (replaced by `app/AppShell.jsx` + `app/router.jsx`)
- `src/pages/Screener.jsx`, `Dashboard.jsx`, `Training.jsx`, `Strategy.jsx`, `Trading.jsx`, `Explain.jsx` (replaced by `*Page.jsx`)
- `src/components/TrainingForm.jsx`, `ModelsList.jsx`, `RunsList.jsx` (replaced in `features/training/`)

---

## Task 1: Install dependencies

**Files:**
- Modify: `quant-ai-ui/package.json`

- [ ] **Step 1: Install runtime deps**

```bash
cd /c/Users/zjg09/projects/quant-ai/quant-ai-ui
npm install @tremor/react@^3.18 @tanstack/react-query@^5 zustand@^4 \
  lightweight-charts@^4.2 react-hook-form@^7 zod@^3 \
  @hookform/resolvers@^3 clsx@^2 tailwind-merge@^2 \
  class-variance-authority@^0.7 lucide-react@^0.400 \
  geist@^1.3 cmdk@^1
```

- [ ] **Step 2: Install Radix primitives needed by shadcn**

```bash
npm install @radix-ui/react-dialog @radix-ui/react-alert-dialog \
  @radix-ui/react-select @radix-ui/react-tabs @radix-ui/react-tooltip \
  @radix-ui/react-dropdown-menu @radix-ui/react-accordion \
  @radix-ui/react-slot @radix-ui/react-label @radix-ui/react-checkbox \
  @radix-ui/react-radio-group @radix-ui/react-popover \
  @radix-ui/react-toast @radix-ui/react-separator \
  @radix-ui/react-avatar @radix-ui/react-progress \
  @radix-ui/react-switch @radix-ui/react-slider
```

- [ ] **Step 3: Install dev deps**

```bash
npm install --save-dev vitest@^1 @testing-library/react@^14 \
  @testing-library/jest-dom@^6 @testing-library/user-event@^14 \
  jsdom@^22 @tailwindcss/forms@^0.5 \
  tailwindcss-animate@^1 @types/node
```

- [ ] **Step 4: Verify build still works**

Run: `npm run build`
Expected: `✓ built in ...` with 0 errors (pages haven't been rewritten yet; existing code still works with new deps in node_modules but unused).

- [ ] **Step 5: Commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
git add quant-ai-ui/package.json quant-ai-ui/package-lock.json
git commit -m "feat: [FE-1] install Tremor, shadcn/Radix, Lightweight Charts, TanStack Query, Zustand, Vitest"
```

---

## Task 2: Tailwind config + design tokens + Geist fonts

**Files:**
- Modify: `quant-ai-ui/tailwind.config.js`
- Modify: `quant-ai-ui/src/index.css`
- Modify: `quant-ai-ui/index.html`

- [ ] **Step 1: Rewrite tailwind.config.js**

Replace the entire contents of `quant-ai-ui/tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./node_modules/@tremor/**/*.{js,ts,jsx,tsx}",
  ],
  safelist: [
    {
      pattern: /^(bg|text|border|ring|fill|stroke)-(slate|zinc|indigo|emerald|rose|amber|sky)-(50|100|200|300|400|500|600|700|800|900|950)$/,
    },
  ],
  theme: {
    extend: {
      colors: {
        background: "rgb(2 6 23)",
        surface: {
          DEFAULT: "rgb(24 24 27)",
          muted: "rgb(39 39 42)",
          border: "rgb(63 63 70)",
          hover: "rgb(30 30 33)",
        },
        foreground: "rgb(250 250 250)",
        muted: "rgb(161 161 170)",
        accent: {
          DEFAULT: "rgb(99 102 241)",
          hover: "rgb(129 140 248)",
          ring: "rgb(99 102 241 / 0.2)",
          foreground: "rgb(255 255 255)",
        },
        up: "rgb(16 185 129)",
        down: "rgb(244 63 94)",
        warn: "rgb(245 158 11)",
        info: "rgb(14 165 233)",
        // Legacy aliases for gradual migration
        "surface-card": "rgb(24 24 27)",
      },
      fontFamily: {
        sans: ["Geist", "Inter", "system-ui", "sans-serif"],
        mono: ["Geist Mono", "JetBrains Mono", "monospace"],
      },
      fontSize: {
        display: ["clamp(2rem, 5vw, 3rem)", { lineHeight: "1.1", letterSpacing: "-0.02em" }],
      },
      borderRadius: {
        xl: "0.75rem",
        "2xl": "1rem",
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms")({ strategy: "class" }),
    require("tailwindcss-animate"),
  ],
};
```

- [ ] **Step 2: Rewrite src/index.css**

Replace the entire contents of `quant-ai-ui/src/index.css`:

```css
@import "geist/font.css";
@import "geist/mono.css";

@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    color-scheme: dark;
  }

  html, body, #root {
    height: 100%;
    margin: 0;
    padding: 0;
  }

  body {
    @apply bg-background text-foreground font-sans antialiased;
    font-feature-settings: "cv11", "ss01", "ss03";
  }

  *:focus-visible {
    @apply outline-none ring-2 ring-accent-ring ring-offset-2 ring-offset-background;
  }

  ::-webkit-scrollbar {
    @apply w-2 h-2;
  }
  ::-webkit-scrollbar-track {
    @apply bg-transparent;
  }
  ::-webkit-scrollbar-thumb {
    @apply bg-surface-muted rounded-full;
  }
  ::-webkit-scrollbar-thumb:hover {
    @apply bg-surface-border;
  }
}

@layer components {
  .glass-border {
    @apply border border-surface-border bg-surface-card/80 backdrop-blur;
  }
}
```

- [ ] **Step 3: Update index.html for Geist preload**

Replace `quant-ai-ui/index.html`:

```html
<!doctype html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Quant AI — ML-powered trading workbench</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 4: Verify build**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -8`
Expected: `✓ built in ...` with 0 errors.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
git add quant-ai-ui/tailwind.config.js quant-ai-ui/src/index.css quant-ai-ui/index.html
git commit -m "feat: [FE-2] Tailwind dark theme tokens + Geist fonts + base styles"
```

---

## Task 3: Providers wrapper (QueryClient + Theme + Toast + lib/utils)

**Files:**
- Create: `quant-ai-ui/src/lib/utils.js`
- Create: `quant-ai-ui/src/lib/formatters.js`
- Create: `quant-ai-ui/src/app/Providers.jsx`
- Modify: `quant-ai-ui/src/main.jsx`

- [ ] **Step 1: Create lib/utils.js (shadcn cn helper)**

```js
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
```

- [ ] **Step 2: Create lib/formatters.js**

```js
export const fmtPrice = (v, { dp = 2 } = {}) =>
  v == null || isNaN(v) ? "—" : Number(v).toFixed(dp);

export const fmtPct = (v, { dp = 2 } = {}) =>
  v == null || isNaN(v) ? "—" : `${v >= 0 ? "+" : ""}${(Number(v) * 100).toFixed(dp)}%`;

export const fmtVolume = (v) => {
  if (v == null || isNaN(v)) return "—";
  const n = Number(v);
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
};

export const fmtDate = (iso) => {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric", month: "short", day: "numeric",
  });
};

export const fmtDatetime = (iso) => {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-US", {
    month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
};

export const classForDelta = (v) =>
  v > 0 ? "text-up" : v < 0 ? "text-down" : "text-muted";
```

- [ ] **Step 3: Create app/Providers.jsx**

```jsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { BrowserRouter } from "react-router-dom";

export default function Providers({ children }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            retry: 1,
            staleTime: 10_000,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 4: Rewrite src/main.jsx**

```jsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import Providers from "./app/Providers";
import App from "./App";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Providers>
      <App />
    </Providers>
  </StrictMode>
);
```

- [ ] **Step 5: Verify build**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -8`
Expected: clean build. Existing App.jsx renders unchanged at this stage.

- [ ] **Step 6: Commit**

```bash
git add quant-ai-ui/src/lib/ quant-ai-ui/src/app/Providers.jsx quant-ai-ui/src/main.jsx
git commit -m "feat: [FE-3] add Providers (QueryClient + Router) and utility helpers"
```

---

## Task 4: shadcn/ui primitives (Button, Input, Dialog, AlertDialog, Select, Tabs, Tooltip, Sheet, Command, Form, Toast, Dropdown, Accordion, Label, Checkbox, RadioGroup, Switch, Slider, Popover, Card, Badge)

**Files:**
- Create: `quant-ai-ui/src/components/ui/*.jsx` (many files)
- Create: `quant-ai-ui/components.json`

Rather than using the shadcn CLI (which targets Next.js by default), we hand-create the components using Radix primitives + Tailwind classes. Every component below is self-contained.

- [ ] **Step 1: Create components/ui/button.jsx**

```jsx
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-accent text-accent-foreground hover:bg-accent-hover",
        destructive: "bg-down text-foreground hover:bg-down/90",
        outline: "border border-surface-border bg-transparent hover:bg-surface-hover text-foreground",
        secondary: "bg-surface-muted text-foreground hover:bg-surface-border",
        ghost: "hover:bg-surface-hover text-foreground",
        link: "text-accent underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-8 rounded px-3 text-xs",
        lg: "h-12 rounded-lg px-6 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  }
);

const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button";
  return <Comp className={cn(buttonVariants({ variant, size }), className)} ref={ref} {...props} />;
});
Button.displayName = "Button";

export { Button, buttonVariants };
```

- [ ] **Step 2: Create components/ui/input.jsx**

```jsx
import * as React from "react";
import { cn } from "../../lib/utils";

const Input = React.forwardRef(({ className, type, ...props }, ref) => (
  <input
    type={type}
    ref={ref}
    className={cn(
      "flex h-10 w-full rounded-md border border-surface-border bg-surface-muted px-3 py-2 text-sm text-foreground placeholder:text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-ring disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  />
));
Input.displayName = "Input";

export { Input };
```

- [ ] **Step 3: Create components/ui/label.jsx**

```jsx
import * as React from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { cn } from "../../lib/utils";

const Label = React.forwardRef(({ className, ...props }, ref) => (
  <LabelPrimitive.Root
    ref={ref}
    className={cn("text-sm font-medium text-foreground leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70", className)}
    {...props}
  />
));
Label.displayName = "Label";

export { Label };
```

- [ ] **Step 4: Create components/ui/card.jsx**

```jsx
import * as React from "react";
import { cn } from "../../lib/utils";

const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("rounded-xl border border-surface-border bg-surface-card text-foreground shadow-sm", className)} {...props} />
));
Card.displayName = "Card";

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col gap-1.5 p-6 pb-4", className)} {...props} />
));
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3 ref={ref} className={cn("text-lg font-semibold leading-none tracking-tight", className)} {...props} />
));
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-sm text-muted", className)} {...props} />
));
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
));
CardFooter.displayName = "CardFooter";

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter };
```

- [ ] **Step 5: Create components/ui/badge.jsx**

```jsx
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
```

- [ ] **Step 6: Create components/ui/tabs.jsx**

```jsx
import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "../../lib/utils";

const Tabs = TabsPrimitive.Root;

const TabsList = React.forwardRef(({ className, ...props }, ref) => (
  <TabsPrimitive.List ref={ref} className={cn("inline-flex h-10 items-center justify-start rounded-md bg-surface-muted p-1 text-muted", className)} {...props} />
));
TabsList.displayName = "TabsList";

const TabsTrigger = React.forwardRef(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-ring disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-surface-card data-[state=active]:text-foreground data-[state=active]:shadow-sm",
      className
    )}
    {...props}
  />
));
TabsTrigger.displayName = "TabsTrigger";

const TabsContent = React.forwardRef(({ className, ...props }, ref) => (
  <TabsPrimitive.Content ref={ref} className={cn("mt-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-ring", className)} {...props} />
));
TabsContent.displayName = "TabsContent";

export { Tabs, TabsList, TabsTrigger, TabsContent };
```

- [ ] **Step 7: Create components/ui/dialog.jsx**

```jsx
import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "../../lib/utils";

const Dialog = DialogPrimitive.Root;
const DialogTrigger = DialogPrimitive.Trigger;
const DialogPortal = DialogPrimitive.Portal;
const DialogClose = DialogPrimitive.Close;

const DialogOverlay = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay ref={ref} className={cn("fixed inset-0 z-50 bg-background/80 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out", className)} {...props} />
));
DialogOverlay.displayName = "DialogOverlay";

const DialogContent = React.forwardRef(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border border-surface-border bg-surface-card p-6 shadow-lg duration-200 rounded-xl",
        className
      )}
      {...props}
    >
      {children}
      <DialogPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-accent-ring disabled:pointer-events-none">
        <X className="h-4 w-4" />
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPortal>
));
DialogContent.displayName = "DialogContent";

const DialogHeader = ({ className, ...props }) => (
  <div className={cn("flex flex-col gap-1.5 text-left", className)} {...props} />
);
const DialogFooter = ({ className, ...props }) => (
  <div className={cn("flex flex-col-reverse gap-2 sm:flex-row sm:justify-end", className)} {...props} />
);
const DialogTitle = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Title ref={ref} className={cn("text-lg font-semibold leading-none", className)} {...props} />
));
DialogTitle.displayName = "DialogTitle";
const DialogDescription = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Description ref={ref} className={cn("text-sm text-muted", className)} {...props} />
));
DialogDescription.displayName = "DialogDescription";

export { Dialog, DialogPortal, DialogOverlay, DialogTrigger, DialogClose, DialogContent, DialogHeader, DialogFooter, DialogTitle, DialogDescription };
```

- [ ] **Step 8: Create components/ui/alert-dialog.jsx**

```jsx
import * as React from "react";
import * as AlertDialogPrimitive from "@radix-ui/react-alert-dialog";
import { cn } from "../../lib/utils";
import { buttonVariants } from "./button";

const AlertDialog = AlertDialogPrimitive.Root;
const AlertDialogTrigger = AlertDialogPrimitive.Trigger;
const AlertDialogPortal = AlertDialogPrimitive.Portal;

const AlertDialogOverlay = React.forwardRef(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Overlay ref={ref} className={cn("fixed inset-0 z-50 bg-background/80 backdrop-blur-sm", className)} {...props} />
));
AlertDialogOverlay.displayName = "AlertDialogOverlay";

const AlertDialogContent = React.forwardRef(({ className, ...props }, ref) => (
  <AlertDialogPortal>
    <AlertDialogOverlay />
    <AlertDialogPrimitive.Content
      ref={ref}
      className={cn("fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border border-surface-border bg-surface-card p-6 shadow-lg rounded-xl", className)}
      {...props}
    />
  </AlertDialogPortal>
));
AlertDialogContent.displayName = "AlertDialogContent";

const AlertDialogHeader = ({ className, ...props }) => (
  <div className={cn("flex flex-col gap-1.5 text-left", className)} {...props} />
);
const AlertDialogFooter = ({ className, ...props }) => (
  <div className={cn("flex flex-col-reverse gap-2 sm:flex-row sm:justify-end", className)} {...props} />
);
const AlertDialogTitle = React.forwardRef(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Title ref={ref} className={cn("text-lg font-semibold", className)} {...props} />
));
AlertDialogTitle.displayName = "AlertDialogTitle";
const AlertDialogDescription = React.forwardRef(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Description ref={ref} className={cn("text-sm text-muted", className)} {...props} />
));
AlertDialogDescription.displayName = "AlertDialogDescription";
const AlertDialogAction = React.forwardRef(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Action ref={ref} className={cn(buttonVariants(), className)} {...props} />
));
AlertDialogAction.displayName = "AlertDialogAction";
const AlertDialogCancel = React.forwardRef(({ className, ...props }, ref) => (
  <AlertDialogPrimitive.Cancel ref={ref} className={cn(buttonVariants({ variant: "outline" }), className)} {...props} />
));
AlertDialogCancel.displayName = "AlertDialogCancel";

export { AlertDialog, AlertDialogPortal, AlertDialogOverlay, AlertDialogTrigger, AlertDialogContent, AlertDialogHeader, AlertDialogFooter, AlertDialogTitle, AlertDialogDescription, AlertDialogAction, AlertDialogCancel };
```

- [ ] **Step 9: Create components/ui/select.jsx**

```jsx
import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../../lib/utils";

const Select = SelectPrimitive.Root;
const SelectGroup = SelectPrimitive.Group;
const SelectValue = SelectPrimitive.Value;

const SelectTrigger = React.forwardRef(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn("flex h-10 w-full items-center justify-between rounded-md border border-surface-border bg-surface-muted px-3 py-2 text-sm text-foreground placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent-ring disabled:cursor-not-allowed disabled:opacity-50", className)}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
));
SelectTrigger.displayName = "SelectTrigger";

const SelectContent = React.forwardRef(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      position={position}
      className={cn("relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border border-surface-border bg-surface-card text-foreground shadow-md", position === "popper" && "data-[side=bottom]:translate-y-1", className)}
      {...props}
    >
      <SelectPrimitive.Viewport className={cn("p-1", position === "popper" && "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]")}>
        {children}
      </SelectPrimitive.Viewport>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
));
SelectContent.displayName = "SelectContent";

const SelectItem = React.forwardRef(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn("relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-surface-hover focus:text-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50", className)}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
));
SelectItem.displayName = "SelectItem";

export { Select, SelectGroup, SelectValue, SelectTrigger, SelectContent, SelectItem };
```

- [ ] **Step 10: Create components/ui/sheet.jsx, tooltip.jsx, accordion.jsx, checkbox.jsx, switch.jsx, slider.jsx, popover.jsx, toast.jsx, dropdown-menu.jsx, form.jsx**

Because of length, create these eight files using the same pattern as above (Radix primitive + Tailwind classes + cn). Use shadcn/ui v0.9 patterns. For reference, the components should use these Radix imports:

- `sheet.jsx` → `@radix-ui/react-dialog` (reuse dialog for side slide)
- `tooltip.jsx` → `@radix-ui/react-tooltip`
- `accordion.jsx` → `@radix-ui/react-accordion`
- `checkbox.jsx` → `@radix-ui/react-checkbox`
- `switch.jsx` → `@radix-ui/react-switch`
- `slider.jsx` → `@radix-ui/react-slider`
- `popover.jsx` → `@radix-ui/react-popover`
- `toast.jsx` → `@radix-ui/react-toast`
- `dropdown-menu.jsx` → `@radix-ui/react-dropdown-menu`
- `form.jsx` → wraps `react-hook-form` + Label + error message

Pattern for each (example `tooltip.jsx`):

```jsx
import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cn } from "../../lib/utils";

const TooltipProvider = TooltipPrimitive.Provider;
const Tooltip = TooltipPrimitive.Root;
const TooltipTrigger = TooltipPrimitive.Trigger;
const TooltipContent = React.forwardRef(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn("z-50 overflow-hidden rounded-md bg-surface-muted px-3 py-1.5 text-xs text-foreground shadow-md", className)}
    {...props}
  />
));
TooltipContent.displayName = "TooltipContent";

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };
```

For `form.jsx`:

```jsx
import * as React from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { Slot } from "@radix-ui/react-slot";
import { Controller, FormProvider, useFormContext } from "react-hook-form";
import { cn } from "../../lib/utils";
import { Label } from "./label";

const Form = FormProvider;

const FormFieldContext = React.createContext({});

const FormField = ({ ...props }) => (
  <FormFieldContext.Provider value={{ name: props.name }}>
    <Controller {...props} />
  </FormFieldContext.Provider>
);

const FormItemContext = React.createContext({});

const useFormField = () => {
  const fieldContext = React.useContext(FormFieldContext);
  const itemContext = React.useContext(FormItemContext);
  const { getFieldState, formState } = useFormContext();
  const fieldState = getFieldState(fieldContext.name, formState);
  const { id } = itemContext;
  return { id, name: fieldContext.name, formItemId: `${id}-form-item`, formDescriptionId: `${id}-form-item-description`, formMessageId: `${id}-form-item-message`, ...fieldState };
};

const FormItem = React.forwardRef(({ className, ...props }, ref) => {
  const id = React.useId();
  return (
    <FormItemContext.Provider value={{ id }}>
      <div ref={ref} className={cn("space-y-2", className)} {...props} />
    </FormItemContext.Provider>
  );
});
FormItem.displayName = "FormItem";

const FormLabel = React.forwardRef(({ className, ...props }, ref) => {
  const { error, formItemId } = useFormField();
  return <Label ref={ref} className={cn(error && "text-down", className)} htmlFor={formItemId} {...props} />;
});
FormLabel.displayName = "FormLabel";

const FormControl = React.forwardRef(({ ...props }, ref) => {
  const { error, formItemId, formDescriptionId, formMessageId } = useFormField();
  return <Slot ref={ref} id={formItemId} aria-describedby={!error ? formDescriptionId : `${formDescriptionId} ${formMessageId}`} aria-invalid={!!error} {...props} />;
});
FormControl.displayName = "FormControl";

const FormMessage = React.forwardRef(({ className, children, ...props }, ref) => {
  const { error, formMessageId } = useFormField();
  const body = error ? String(error?.message) : children;
  if (!body) return null;
  return <p ref={ref} id={formMessageId} className={cn("text-sm text-down", className)} {...props}>{body}</p>;
});
FormMessage.displayName = "FormMessage";

export { useFormField, Form, FormItem, FormLabel, FormControl, FormMessage, FormField };
```

Full implementations for each of the other 9 components follow standard shadcn patterns (copy the relevant file from the shadcn-ui/ui GitHub repo, strip Next.js-specific imports, use `cn` from our lib). When unclear, default to:
- Same Radix primitive wrapped with `React.forwardRef`
- `className` merged via `cn()`  
- Use our color tokens (surface-*, foreground, muted, accent)

- [ ] **Step 11: Verify build**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -10`
Expected: 0 errors.

- [ ] **Step 12: Commit**

```bash
git add quant-ai-ui/src/components/ui/
git commit -m "feat: [FE-4] add shadcn/ui primitives built on Radix + Tailwind"
```

---

## Task 5: API queries.js (TanStack Query hooks)

**Files:**
- Create: `quant-ai-ui/src/api/queries.js`

- [ ] **Step 1: Create api/queries.js**

```js
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "./client";

const SCREENER_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "WMT"];

// ===== Market =====
export const useMarket = (ticker, opts = {}) =>
  useQuery({
    queryKey: ["market", ticker],
    queryFn: () => api.getMarket(ticker),
    enabled: !!ticker,
    staleTime: 30_000,
    ...opts,
  });

export const useScreenerTickers = () =>
  useQuery({
    queryKey: ["screener", SCREENER_TICKERS],
    queryFn: async () => {
      const results = await Promise.all(
        SCREENER_TICKERS.map((t) => api.getMarket(t).catch(() => null))
      );
      return results.map((r, idx) => ({ ticker: SCREENER_TICKERS[idx], data: r })).filter((x) => x.data);
    },
    staleTime: 60_000,
  });

// ===== Prediction =====
export const usePredict = () =>
  useMutation({
    mutationFn: (payload) => api.predict(payload),
  });

// ===== Explain =====
export const useExplain = (ticker) =>
  useQuery({
    queryKey: ["explain", ticker],
    queryFn: () => api.explain(ticker),
    enabled: !!ticker,
  });

export const useSimilarCases = (query) =>
  useQuery({
    queryKey: ["search", query],
    queryFn: () => api.search(query),
    enabled: !!query,
  });

// ===== Training =====
export const useTrain = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.train,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
};

export const useRuns = (limit = 20) =>
  useQuery({
    queryKey: ["runs", limit],
    queryFn: () => api.listRuns(limit),
    refetchInterval: 10_000,
  });

export const useRunStatus = (runId) =>
  useQuery({
    queryKey: ["runs", runId],
    queryFn: () => api.getRunStatus(runId),
    enabled: !!runId,
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (status === "success" || status === "failed") return false;
      return 2000;
    },
  });

// ===== Models =====
export const useModels = () =>
  useQuery({
    queryKey: ["models"],
    queryFn: () => api.listModels(),
  });

export const useModelTypes = () =>
  useQuery({
    queryKey: ["model-types"],
    queryFn: api.listModelTypes,
    staleTime: 5 * 60_000,
  });

export const usePromoteModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.promoteModel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      qc.invalidateQueries({ queryKey: ["promoted-model"] });
    },
  });
};

export const useDemoteModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.demoteModel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      qc.invalidateQueries({ queryKey: ["promoted-model"] });
    },
  });
};

export const usePromotedModel = () =>
  useQuery({
    queryKey: ["promoted-model"],
    queryFn: api.getPromotedModel,
  });

// ===== Features =====
export const useFeatureGroups = () =>
  useQuery({
    queryKey: ["feature-groups"],
    queryFn: api.listFeatureGroups,
    staleTime: 5 * 60_000,
  });

// ===== Strategies =====
export const useStrategies = () =>
  useQuery({
    queryKey: ["strategies"],
    queryFn: api.listStrategies,
    staleTime: 5 * 60_000,
  });

export const useStrategy = (name) =>
  useQuery({
    queryKey: ["strategy", name],
    queryFn: () => api.getStrategy(name),
    enabled: !!name,
  });

export const useGenerateSignals = () =>
  useMutation({ mutationFn: ({ name, payload }) => api.generateSignals(name, payload) });

export const useStrategyBacktest = () =>
  useMutation({ mutationFn: ({ name, payload }) => api.runStrategyBacktest(name, payload) });

// ===== Trading =====
export const usePortfolio = () =>
  useQuery({
    queryKey: ["portfolio"],
    queryFn: api.getPortfolio,
    refetchInterval: 5000,
  });

export const useOrders = (status = "all") =>
  useQuery({
    queryKey: ["orders", status],
    queryFn: () => api.listOrders(status),
    refetchInterval: 5000,
  });

export const useTrades = (limit = 20) =>
  useQuery({
    queryKey: ["trades", limit],
    queryFn: () => api.getTrades(limit),
    refetchInterval: 10_000,
  });

export const usePlaceOrder = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.placeOrder,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["portfolio"] });
      qc.invalidateQueries({ queryKey: ["orders"] });
      qc.invalidateQueries({ queryKey: ["trades"] });
    },
  });
};

export const useCancelOrder = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.cancelOrder,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["orders"] });
    },
  });
};

export const useResetPortfolio = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.resetPortfolio,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["portfolio"] });
      qc.invalidateQueries({ queryKey: ["orders"] });
      qc.invalidateQueries({ queryKey: ["trades"] });
    },
  });
};

// ===== Optimization =====
export const useOptimizeModel = () => useMutation({ mutationFn: api.optimizeModel });
export const useOptimizeStrategy = () => useMutation({ mutationFn: api.optimizeStrategy });
```

- [ ] **Step 2: Commit**

```bash
git add quant-ai-ui/src/api/queries.js
git commit -m "feat: [FE-5] add TanStack Query hooks for all API endpoints"
```

---

## Task 6: Zustand live store + useLivePrices hook

**Files:**
- Create: `quant-ai-ui/src/stores/liveStore.js`
- Create: `quant-ai-ui/src/features/trading/useLivePrices.js`

- [ ] **Step 1: Create stores/liveStore.js**

```js
import { create } from "zustand";

export const useLiveStore = create((set) => ({
  prices: {},
  connectionStatus: "disconnected",
  updatePrice: (ticker, price, ts) =>
    set((state) => ({
      prices: { ...state.prices, [ticker]: { price, ts } },
    })),
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  clearPrices: () => set({ prices: {} }),
}));
```

- [ ] **Step 2: Create features/trading/useLivePrices.js**

```js
import { useEffect, useRef } from "react";
import { useLiveStore } from "../../stores/liveStore";

const WS_BASE = (import.meta.env.VITE_API_BASE || "http://localhost:8000").replace(/^http/, "ws");
const WS_URL = `${WS_BASE}/api/trading/ws/prices`;

export function useLivePrices() {
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const { updatePrice, setConnectionStatus } = useLiveStore();

  useEffect(() => {
    let mounted = true;

    const connect = () => {
      if (!mounted) return;
      setConnectionStatus("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mounted) return;
        setConnectionStatus("connected");
      };
      ws.onmessage = (event) => {
        if (!mounted) return;
        try {
          const msg = JSON.parse(event.data);
          if (msg.ticker && msg.price != null) {
            updatePrice(msg.ticker, msg.price, msg.timestamp || Date.now());
          }
        } catch (err) {
          // ignore parse errors
        }
      };
      ws.onerror = () => {
        if (!mounted) return;
        setConnectionStatus("error");
      };
      ws.onclose = () => {
        if (!mounted) return;
        setConnectionStatus("disconnected");
        // reconnect with 1s backoff, max 10s
        reconnectTimerRef.current = setTimeout(connect, 1000);
      };
    };

    connect();

    return () => {
      mounted = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [updatePrice, setConnectionStatus]);
}
```

- [ ] **Step 3: Commit**

```bash
git add quant-ai-ui/src/stores/liveStore.js quant-ai-ui/src/features/trading/useLivePrices.js
git commit -m "feat: [FE-6] add Zustand live store and useLivePrices WebSocket hook"
```

---

## Task 7: Shared components (PageHeader, EmptyState, LoadingSpinner, ErrorBoundary, ErrorState, ConfirmDialog, TickerSearch)

**Files:**
- Create: `quant-ai-ui/src/components/PageHeader.jsx`
- Create: `quant-ai-ui/src/components/EmptyState.jsx`
- Create: `quant-ai-ui/src/components/LoadingSpinner.jsx`
- Create: `quant-ai-ui/src/components/ErrorBoundary.jsx`
- Create: `quant-ai-ui/src/components/ErrorState.jsx`
- Create: `quant-ai-ui/src/components/ConfirmDialog.jsx`
- Create: `quant-ai-ui/src/components/TickerSearch.jsx`

- [ ] **Step 1: PageHeader.jsx**

```jsx
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
```

- [ ] **Step 2: EmptyState.jsx**

```jsx
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
```

- [ ] **Step 3: LoadingSpinner.jsx**

```jsx
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
```

- [ ] **Step 4: ErrorBoundary.jsx**

```jsx
import { Component } from "react";
import { Button } from "./ui/button";
import { AlertTriangle } from "lucide-react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 p-6 text-center">
          <AlertTriangle className="h-12 w-12 text-down" />
          <div>
            <h2 className="text-xl font-semibold">Something broke</h2>
            <p className="text-sm text-muted mt-1">{String(this.state.error?.message || this.state.error)}</p>
          </div>
          <Button onClick={() => this.setState({ hasError: false, error: null })}>
            Try again
          </Button>
        </div>
      );
    }
    return this.props.children;
  }
}
```

- [ ] **Step 5: ErrorState.jsx**

```jsx
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
```

- [ ] **Step 6: ConfirmDialog.jsx**

```jsx
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "./ui/alert-dialog";
import { Button } from "./ui/button";

export default function ConfirmDialog({
  trigger,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  onConfirm,
  destructive = false,
}) {
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>{trigger}</AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          {description && <AlertDialogDescription>{description}</AlertDialogDescription>}
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{cancelLabel}</AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            className={destructive ? "bg-down text-foreground hover:bg-down/90" : undefined}
          >
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
```

- [ ] **Step 7: TickerSearch.jsx (simple placeholder; upgrade to cmdk later)**

```jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { Input } from "./ui/input";
import { Button } from "./ui/button";

const SUGGESTIONS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"];

export default function TickerSearch({ defaultTicker = "" }) {
  const [value, setValue] = useState(defaultTicker);
  const navigate = useNavigate();

  const submit = (t) => {
    const ticker = (t || value).trim().toUpperCase();
    if (ticker) navigate(`/dashboard?ticker=${ticker}`);
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="flex items-center gap-2"
    >
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted" />
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value.toUpperCase())}
          placeholder="Search ticker..."
          className="pl-9 w-48"
        />
      </div>
      <Button type="submit" size="sm">
        Go
      </Button>
      <div className="flex gap-1">
        {SUGGESTIONS.slice(0, 4).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => submit(t)}
            className="text-xs px-2 py-1 rounded bg-surface-muted hover:bg-surface-hover text-muted hover:text-foreground transition-colors"
          >
            {t}
          </button>
        ))}
      </div>
    </form>
  );
}
```

- [ ] **Step 8: Commit**

```bash
git add quant-ai-ui/src/components/
git commit -m "feat: [FE-7] add shared components (PageHeader, EmptyState, LoadingSpinner, ErrorBoundary, ErrorState, ConfirmDialog, TickerSearch)"
```

---

## Task 8: Sidebar + AppShell + Router

**Files:**
- Create: `quant-ai-ui/src/app/Sidebar.jsx`
- Create: `quant-ai-ui/src/app/AppShell.jsx`
- Create: `quant-ai-ui/src/app/router.jsx`
- Modify: `quant-ai-ui/src/App.jsx` (replace with new routing)

- [ ] **Step 1: app/Sidebar.jsx**

```jsx
import { NavLink } from "react-router-dom";
import { LineChart, BarChart3, GraduationCap, Beaker, Briefcase, Brain, Settings } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../components/ui/tooltip";
import { cn } from "../lib/utils";

const ITEMS = [
  { path: "/screener", label: "Screener", icon: LineChart },
  { path: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { path: "/training", label: "Training", icon: GraduationCap },
  { path: "/strategy", label: "Strategy", icon: Beaker },
  { path: "/trading", label: "Trading", icon: Briefcase },
  { path: "/explain", label: "Explain", icon: Brain },
];

export default function Sidebar() {
  return (
    <TooltipProvider delayDuration={150}>
      <aside className="hidden md:flex w-16 flex-col items-center justify-between bg-surface-card border-r border-surface-border py-4 flex-shrink-0">
        <div className="flex flex-col items-center gap-4">
          <div className="h-10 w-10 rounded-lg bg-accent flex items-center justify-center text-accent-foreground font-bold text-lg">
            Q
          </div>
          <nav className="flex flex-col gap-1">
            {ITEMS.map((item) => (
              <Tooltip key={item.path}>
                <TooltipTrigger asChild>
                  <NavLink
                    to={item.path}
                    className={({ isActive }) =>
                      cn(
                        "flex items-center justify-center h-11 w-11 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover transition-colors",
                        isActive && "text-accent bg-accent/10"
                      )
                    }
                  >
                    <item.icon className="h-5 w-5" />
                    <span className="sr-only">{item.label}</span>
                  </NavLink>
                </TooltipTrigger>
                <TooltipContent side="right">{item.label}</TooltipContent>
              </Tooltip>
            ))}
          </nav>
        </div>
        <div className="flex flex-col items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <button className="h-11 w-11 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover flex items-center justify-center">
                <Settings className="h-5 w-5" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">Settings</TooltipContent>
          </Tooltip>
        </div>
      </aside>
    </TooltipProvider>
  );
}
```

- [ ] **Step 2: app/AppShell.jsx**

```jsx
import Sidebar from "./Sidebar";
import { Outlet } from "react-router-dom";
import ErrorBoundary from "../components/ErrorBoundary";

export default function AppShell() {
  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 3: app/router.jsx**

```jsx
import { Routes, Route, Navigate } from "react-router-dom";
import AppShell from "./AppShell";
import ScreenerPage from "../pages/ScreenerPage";
import DashboardPage from "../pages/DashboardPage";
import TrainingPage from "../pages/TrainingPage";
import StrategyPage from "../pages/StrategyPage";
import TradingPage from "../pages/TradingPage";
import ExplainPage from "../pages/ExplainPage";

export default function AppRouter() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/screener" replace />} />
        <Route path="/screener" element={<ScreenerPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/training" element={<TrainingPage />} />
        <Route path="/strategy" element={<StrategyPage />} />
        <Route path="/trading" element={<TradingPage />} />
        <Route path="/explain" element={<ExplainPage />} />
      </Route>
    </Routes>
  );
}
```

- [ ] **Step 4: Replace App.jsx**

Rewrite `quant-ai-ui/src/App.jsx`:

```jsx
import AppRouter from "./app/router";

export default function App() {
  return <AppRouter />;
}
```

- [ ] **Step 5: Create placeholder pages to keep router compiling**

For each of the 6 new page files (`src/pages/ScreenerPage.jsx` etc.), create a minimal placeholder:

```jsx
export default function ScreenerPage() {
  return <div>Screener (coming)</div>;
}
```

Do this for: ScreenerPage, DashboardPage, TrainingPage, StrategyPage, TradingPage, ExplainPage.

- [ ] **Step 6: Build and manually verify nav renders**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -8`
Expected: clean build.

- [ ] **Step 7: Commit**

```bash
git add quant-ai-ui/src/app/ quant-ai-ui/src/App.jsx quant-ai-ui/src/pages/
git commit -m "feat: [FE-8] add AppShell, Sidebar, Router, and page placeholders"
```

---

## Task 9: ScreenerPage

**Files:**
- Create: `quant-ai-ui/src/features/screener/ScreenerTable.jsx`
- Modify: `quant-ai-ui/src/pages/ScreenerPage.jsx`

- [ ] **Step 1: ScreenerTable.jsx**

```jsx
import { useNavigate } from "react-router-dom";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import { cn } from "../../lib/utils";
import { fmtPrice, fmtPct, fmtVolume } from "../../lib/formatters";

export default function ScreenerTable({ rows, sortBy = "change" }) {
  const navigate = useNavigate();

  const sorted = [...rows].sort((a, b) => {
    if (sortBy === "change") return (b.change_pct ?? 0) - (a.change_pct ?? 0);
    if (sortBy === "volume") return (b.volume ?? 0) - (a.volume ?? 0);
    return 0;
  });

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase tracking-wider text-muted">
          <tr>
            <th className="px-4 py-3 text-left">Ticker</th>
            <th className="px-4 py-3 text-right">Last</th>
            <th className="px-4 py-3 text-right">Change</th>
            <th className="px-4 py-3 text-right">Change %</th>
            <th className="px-4 py-3 text-right">Volume</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const up = (r.change_pct ?? 0) >= 0;
            return (
              <tr
                key={r.ticker}
                onClick={() => navigate(`/dashboard?ticker=${r.ticker}`)}
                className="border-t border-surface-border hover:bg-surface-hover cursor-pointer transition-colors"
              >
                <td className="px-4 py-3">
                  <span className="font-semibold text-foreground">{r.ticker}</span>
                </td>
                <td className="px-4 py-3 text-right font-mono">{fmtPrice(r.last)}</td>
                <td className={cn("px-4 py-3 text-right font-mono", up ? "text-up" : "text-down")}>
                  {r.change != null ? (up ? "+" : "") + fmtPrice(r.change) : "—"}
                </td>
                <td className={cn("px-4 py-3 text-right", up ? "text-up" : "text-down")}>
                  <span className="inline-flex items-center gap-1">
                    {up ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
                    {fmtPct(r.change_pct)}
                  </span>
                </td>
                <td className="px-4 py-3 text-right text-muted font-mono">{fmtVolume(r.volume)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 2: pages/ScreenerPage.jsx**

```jsx
import { useState, useMemo } from "react";
import PageHeader from "../components/PageHeader";
import { Button } from "../components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import { LoadingOverlay } from "../components/LoadingSpinner";
import ScreenerTable from "../features/screener/ScreenerTable";
import { useScreenerTickers } from "../api/queries";
import { RefreshCw, LineChart } from "lucide-react";

export default function ScreenerPage() {
  const [sortBy, setSortBy] = useState("change");
  const { data, isLoading, error, refetch, isFetching } = useScreenerTickers();

  const rows = useMemo(() => {
    if (!data) return [];
    return data.map(({ ticker, data: tickerData }) => {
      const rows = tickerData?.rows || [];
      if (rows.length < 2) return { ticker, last: rows[0]?.close, change: 0, change_pct: 0, volume: rows[0]?.volume };
      const last = rows[rows.length - 1];
      const prev = rows[rows.length - 2];
      const change = last.close - prev.close;
      const change_pct = change / prev.close;
      return { ticker, last: last.close, change, change_pct, volume: last.volume };
    });
  }, [data]);

  return (
    <div>
      <PageHeader
        title="Stock Screener"
        subtitle="Top tickers with live price, change, and volume"
        actions={
          <>
            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="change">Sort: Change %</SelectItem>
                <SelectItem value="volume">Sort: Volume</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
              <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </>
        }
      />
      {isLoading && <LoadingOverlay label="Loading screener..." />}
      {error && <ErrorState error={error} onRetry={() => refetch()} />}
      {!isLoading && !error && rows.length === 0 && (
        <EmptyState icon={LineChart} title="No data" actionLabel="Refresh" onAction={() => refetch()} />
      )}
      {!isLoading && rows.length > 0 && <ScreenerTable rows={rows} sortBy={sortBy} />}
    </div>
  );
}
```

- [ ] **Step 3: Build + commit**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -8`
Expected: clean build.

```bash
git add quant-ai-ui/src/features/screener/ quant-ai-ui/src/pages/ScreenerPage.jsx
git commit -m "feat: [FE-9] implement Screener page with Tremor-style table"
```

---

## Task 10: DashboardPage with CandlestickChart, SHAP, Kafka events panel

**Files:**
- Create: `quant-ai-ui/src/features/charts/CandlestickChart.jsx`
- Modify: `quant-ai-ui/src/pages/DashboardPage.jsx`

- [ ] **Step 1: CandlestickChart.jsx**

```jsx
import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

export default function CandlestickChart({ data, markers = [], height = 420 }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      autoSize: true,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "rgb(24 24 27)" },
        textColor: "rgb(161 161 170)",
        fontFamily: "Geist, system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: "rgb(39 39 42)" },
        horzLines: { color: "rgb(39 39 42)" },
      },
      rightPriceScale: { borderColor: "rgb(63 63 70)" },
      timeScale: { borderColor: "rgb(63 63 70)", timeVisible: true },
      crosshair: { mode: 1 },
    });
    chartRef.current = chart;
    const series = chart.addCandlestickSeries({
      upColor: "rgb(16 185 129)",
      downColor: "rgb(244 63 94)",
      borderVisible: false,
      wickUpColor: "rgb(16 185 129)",
      wickDownColor: "rgb(244 63 94)",
    });
    seriesRef.current = series;

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    if (!seriesRef.current || !data || data.length === 0) return;
    seriesRef.current.setData(data);
    if (chartRef.current) chartRef.current.timeScale().fitContent();
  }, [data]);

  useEffect(() => {
    if (!seriesRef.current || !markers) return;
    seriesRef.current.setMarkers(markers);
  }, [markers]);

  return <div ref={containerRef} style={{ width: "100%", height }} />;
}
```

- [ ] **Step 2: pages/DashboardPage.jsx**

```jsx
import { useSearchParams } from "react-router-dom";
import { useState, useMemo } from "react";
import PageHeader from "../components/PageHeader";
import TickerSearch from "../components/TickerSearch";
import { LoadingOverlay } from "../components/LoadingSpinner";
import ErrorState from "../components/ErrorState";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import CandlestickChart from "../features/charts/CandlestickChart";
import { useMarket, usePredict, useExplain } from "../api/queries";
import { fmtPrice, fmtPct, classForDelta } from "../lib/formatters";
import { TrendingUp, TrendingDown, Sparkles } from "lucide-react";

export default function DashboardPage() {
  const [searchParams] = useSearchParams();
  const ticker = (searchParams.get("ticker") || "AAPL").toUpperCase();
  const [prediction, setPrediction] = useState(null);

  const { data: marketData, isLoading: marketLoading, error: marketError } = useMarket(ticker);
  const { data: explainData } = useExplain(ticker);
  const predictMutation = usePredict();

  const candles = useMemo(() => {
    if (!marketData?.rows) return [];
    return marketData.rows.map((r) => ({
      time: r.date,
      open: r.open,
      high: r.high,
      low: r.low,
      close: r.close,
    }));
  }, [marketData]);

  const lastRow = marketData?.rows?.[marketData.rows.length - 1];
  const prevRow = marketData?.rows?.[marketData.rows.length - 2];
  const change = lastRow && prevRow ? lastRow.close - prevRow.close : 0;
  const changePct = prevRow ? change / prevRow.close : 0;

  const handlePredict = async () => {
    try {
      const result = await predictMutation.mutateAsync({ ticker, horizons: [5] });
      setPrediction(result);
    } catch (e) {
      // error shown below via mutation.error
    }
  };

  return (
    <div>
      <PageHeader
        title={`${ticker}`}
        subtitle={lastRow ? `$${fmtPrice(lastRow.close)}` : "—"}
        actions={<TickerSearch defaultTicker={ticker} />}
      />

      {marketLoading && <LoadingOverlay label="Loading market data..." />}
      {marketError && <ErrorState error={marketError} />}

      {!marketLoading && marketData && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          <Card className="lg:col-span-3">
            <CardHeader className="flex-row items-center justify-between gap-4 pb-2">
              <CardTitle>Price</CardTitle>
              <div className={`flex items-center gap-1 text-sm font-mono ${classForDelta(change)}`}>
                {change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                {fmtPrice(Math.abs(change))} ({fmtPct(changePct)})
              </div>
            </CardHeader>
            <CardContent className="p-2">
              <CandlestickChart data={candles} height={420} />
            </CardContent>
          </Card>

          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle>Prediction</CardTitle>
              </CardHeader>
              <CardContent>
                {prediction ? (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Badge variant={prediction.prediction === 1 ? "success" : "destructive"}>
                        {prediction.prediction === 1 ? "BULLISH" : "BEARISH"}
                      </Badge>
                      <span className="text-sm text-muted">
                        prob_up: {fmtPct(prediction.probability_up ?? prediction.prob_up ?? 0, { dp: 1 })}
                      </span>
                    </div>
                    <div className="text-xs text-muted">
                      Horizon: {prediction.horizon || 5} days · Model: {prediction.model_id || "promoted"}
                    </div>
                  </div>
                ) : (
                  <Button onClick={handlePredict} disabled={predictMutation.isPending} className="w-full">
                    <Sparkles className="h-4 w-4" />
                    {predictMutation.isPending ? "Predicting..." : "Run prediction"}
                  </Button>
                )}
                {predictMutation.error && <ErrorState className="mt-3" error={predictMutation.error} />}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle>SHAP Top Features</CardTitle>
              </CardHeader>
              <CardContent>
                {explainData?.top_features?.length ? (
                  <ul className="space-y-2">
                    {explainData.top_features.slice(0, 6).map((f, i) => {
                      const max = explainData.top_features[0].mean_abs_shap;
                      const pct = (f.mean_abs_shap / max) * 100;
                      return (
                        <li key={i} className="text-sm">
                          <div className="flex justify-between mb-1">
                            <span className="font-mono text-foreground">{f.feature}</span>
                            <span className="text-muted">{f.mean_abs_shap.toFixed(4)}</span>
                          </div>
                          <div className="h-1.5 bg-surface-muted rounded-full overflow-hidden">
                            <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="text-sm text-muted">No SHAP data available.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Build + commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
cd quant-ai-ui && npm run build 2>&1 | tail -6 && cd ..
git add quant-ai-ui/src/features/charts/ quant-ai-ui/src/pages/DashboardPage.jsx
git commit -m "feat: [FE-10] implement Dashboard with Lightweight Charts candlestick + prediction + SHAP"
```

---

## Task 11: TrainingPage with tabs (Train / Runs / Models)

**Files:**
- Create: `quant-ai-ui/src/features/training/TrainForm.jsx`
- Create: `quant-ai-ui/src/features/training/RunsTable.jsx`
- Create: `quant-ai-ui/src/features/training/ModelsTable.jsx`
- Modify: `quant-ai-ui/src/pages/TrainingPage.jsx`

- [ ] **Step 1: TrainForm.jsx**

```jsx
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select";
import ErrorState from "../../components/ErrorState";
import { useTrain, useModelTypes, useFeatureGroups } from "../../api/queries";

const schema = z.object({
  tickers: z.string().min(1, "At least one ticker required"),
  model_type: z.string().min(1),
  horizon_days: z.coerce.number().int().min(1).max(60),
  train_ratio: z.coerce.number().min(0.5).max(0.9).default(0.7),
  val_ratio: z.coerce.number().min(0.05).max(0.3).default(0.15),
  search_mode: z.enum(["none", "grid", "optuna", "optuna_multi"]).default("none"),
  search_trials: z.coerce.number().int().min(1).max(200).default(20),
});

export default function TrainForm() {
  const train = useTrain();
  const { data: modelTypes } = useModelTypes();
  const { data: featureGroups } = useFeatureGroups();

  const form = useForm({
    resolver: zodResolver(schema),
    defaultValues: {
      tickers: "AAPL",
      model_type: "logistic",
      horizon_days: 5,
      train_ratio: 0.7,
      val_ratio: 0.15,
      search_mode: "none",
      search_trials: 20,
    },
  });

  const onSubmit = (values) => {
    const payload = {
      tickers: values.tickers.split(",").map((t) => t.trim()).filter(Boolean),
      model_type: values.model_type,
      horizon_days: values.horizon_days,
      train_ratio: values.train_ratio,
      val_ratio: values.val_ratio,
      search_mode: values.search_mode,
      search_trials: values.search_trials,
      feature_groups: ["ta_basic", "momentum"],
    };
    train.mutate(payload);
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4 max-w-2xl">
      <div>
        <Label htmlFor="tickers">Tickers (comma-separated)</Label>
        <Input id="tickers" placeholder="AAPL, MSFT" {...form.register("tickers")} />
        {form.formState.errors.tickers && (
          <p className="text-sm text-down mt-1">{form.formState.errors.tickers.message}</p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label>Model type</Label>
          <Select value={form.watch("model_type")} onValueChange={(v) => form.setValue("model_type", v)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(modelTypes || ["logistic", "random_forest", "xgboost", "lightgbm", "catboost", "ensemble"]).map((t) => (
                <SelectItem key={t} value={t}>{t}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="horizon">Horizon days</Label>
          <Input id="horizon" type="number" {...form.register("horizon_days")} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label>Search mode</Label>
          <Select value={form.watch("search_mode")} onValueChange={(v) => form.setValue("search_mode", v)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              <SelectItem value="grid">Grid</SelectItem>
              <SelectItem value="optuna">Optuna</SelectItem>
              <SelectItem value="optuna_multi">Optuna Multi-Objective</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="trials">Trials</Label>
          <Input id="trials" type="number" {...form.register("search_trials")} />
        </div>
      </div>

      <Button type="submit" disabled={train.isPending}>
        {train.isPending ? "Starting..." : "Start Training"}
      </Button>

      {train.error && <ErrorState error={train.error} />}
      {train.data && (
        <div className="text-sm text-up">
          Training started · run_id: <code className="font-mono">{train.data.run_id || "—"}</code>
        </div>
      )}
    </form>
  );
}
```

- [ ] **Step 2: RunsTable.jsx**

```jsx
import { useRuns } from "../../api/queries";
import { Badge } from "../../components/ui/badge";
import { LoadingOverlay } from "../../components/LoadingSpinner";
import EmptyState from "../../components/EmptyState";
import ErrorState from "../../components/ErrorState";
import { fmtDatetime } from "../../lib/formatters";

const STATUS_VARIANT = { success: "success", failed: "destructive", running: "info", pending: "warning" };

export default function RunsTable() {
  const { data, isLoading, error, refetch } = useRuns(20);
  if (isLoading) return <LoadingOverlay label="Loading runs..." />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;
  if (!data || data.length === 0) return <EmptyState title="No runs yet" description="Train a model to see runs here." />;

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase text-muted">
          <tr>
            <th className="px-4 py-3 text-left">Run ID</th>
            <th className="px-4 py-3 text-left">Model</th>
            <th className="px-4 py-3 text-left">Status</th>
            <th className="px-4 py-3 text-left">Started</th>
            <th className="px-4 py-3 text-right">Val AUC</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r) => (
            <tr key={r.run_id} className="border-t border-surface-border hover:bg-surface-hover">
              <td className="px-4 py-3 font-mono text-xs">{r.run_id}</td>
              <td className="px-4 py-3">{r.model_type || "—"}</td>
              <td className="px-4 py-3">
                <Badge variant={STATUS_VARIANT[r.status] || "secondary"}>{r.status}</Badge>
              </td>
              <td className="px-4 py-3 text-muted">{fmtDatetime(r.started_at || r.created_at)}</td>
              <td className="px-4 py-3 text-right font-mono">
                {r.metrics?.val_auc != null ? r.metrics.val_auc.toFixed(4) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 3: ModelsTable.jsx**

```jsx
import { Button } from "../../components/ui/button";
import { Badge } from "../../components/ui/badge";
import { LoadingOverlay } from "../../components/LoadingSpinner";
import EmptyState from "../../components/EmptyState";
import ErrorState from "../../components/ErrorState";
import ConfirmDialog from "../../components/ConfirmDialog";
import { useModels, usePromoteModel, usePromotedModel } from "../../api/queries";
import { fmtDate } from "../../lib/formatters";
import { Trophy } from "lucide-react";

export default function ModelsTable() {
  const { data, isLoading, error, refetch } = useModels();
  const { data: promoted } = usePromotedModel();
  const promote = usePromoteModel();

  if (isLoading) return <LoadingOverlay label="Loading models..." />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;

  const models = data?.models || data || [];
  if (models.length === 0) return <EmptyState title="No models registered" description="Finish a training run to register a model." />;

  const promotedId = promoted?.model_id;

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase text-muted">
          <tr>
            <th className="px-4 py-3 text-left">Model ID</th>
            <th className="px-4 py-3 text-left">Type</th>
            <th className="px-4 py-3 text-left">Trained</th>
            <th className="px-4 py-3 text-right">Val AUC</th>
            <th className="px-4 py-3 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const mid = m.model_id || m.id;
            const isPromoted = mid === promotedId;
            return (
              <tr key={mid} className={`border-t border-surface-border hover:bg-surface-hover ${isPromoted ? "bg-accent/5" : ""}`}>
                <td className="px-4 py-3 font-mono text-xs">
                  {mid}
                  {isPromoted && (
                    <Badge variant="success" className="ml-2">
                      <Trophy className="h-3 w-3 mr-1" /> Promoted
                    </Badge>
                  )}
                </td>
                <td className="px-4 py-3">{m.model_type || m.type || "—"}</td>
                <td className="px-4 py-3 text-muted">{fmtDate(m.created_at)}</td>
                <td className="px-4 py-3 text-right font-mono">
                  {m.metrics?.val_auc != null ? m.metrics.val_auc.toFixed(4) : "—"}
                </td>
                <td className="px-4 py-3 text-right">
                  {!isPromoted && (
                    <ConfirmDialog
                      trigger={<Button size="sm" variant="outline">Promote</Button>}
                      title="Promote this model?"
                      description="The promoted model is used by default for predictions. Any current promoted model will be demoted."
                      onConfirm={() => promote.mutate(mid)}
                    />
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 4: pages/TrainingPage.jsx**

```jsx
import PageHeader from "../components/PageHeader";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../components/ui/tabs";
import TrainForm from "../features/training/TrainForm";
import RunsTable from "../features/training/RunsTable";
import ModelsTable from "../features/training/ModelsTable";

export default function TrainingPage() {
  return (
    <div>
      <PageHeader title="Training" subtitle="Train, monitor runs, manage registered models" />
      <Tabs defaultValue="train">
        <TabsList>
          <TabsTrigger value="train">Train</TabsTrigger>
          <TabsTrigger value="runs">Runs</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
        </TabsList>
        <TabsContent value="train"><TrainForm /></TabsContent>
        <TabsContent value="runs"><RunsTable /></TabsContent>
        <TabsContent value="models"><ModelsTable /></TabsContent>
      </Tabs>
    </div>
  );
}
```

- [ ] **Step 5: Build + commit**

```bash
cd quant-ai-ui && npm run build 2>&1 | tail -6 && cd ..
git add quant-ai-ui/src/features/training/ quant-ai-ui/src/pages/TrainingPage.jsx
git commit -m "feat: [FE-11] implement Training page with 3 tabs (Train/Runs/Models)"
```

---

## Task 12: StrategyPage with schema-driven params form

**Files:**
- Create: `quant-ai-ui/src/features/strategy/StrategyPicker.jsx`
- Create: `quant-ai-ui/src/features/strategy/StrategyParamsForm.jsx`
- Create: `quant-ai-ui/src/features/strategy/BacktestResults.jsx`
- Modify: `quant-ai-ui/src/pages/StrategyPage.jsx`

- [ ] **Step 1: StrategyPicker.jsx**

```jsx
import { useStrategies } from "../../api/queries";
import { cn } from "../../lib/utils";

export default function StrategyPicker({ selected, onSelect }) {
  const { data, isLoading } = useStrategies();
  if (isLoading) return <div className="text-sm text-muted p-3">Loading...</div>;
  const items = data?.strategies || data || [];

  return (
    <div className="space-y-1">
      {items.map((s) => (
        <button
          key={s.name}
          onClick={() => onSelect(s.name)}
          className={cn(
            "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors",
            selected === s.name ? "bg-accent text-accent-foreground" : "hover:bg-surface-hover text-foreground"
          )}
        >
          <div className="font-medium">{s.name}</div>
          {s.description && <div className="text-xs text-muted truncate mt-0.5">{s.description}</div>}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: StrategyParamsForm.jsx**

```jsx
import { useEffect, useState } from "react";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import ErrorState from "../../components/ErrorState";
import { useStrategy, useStrategyBacktest } from "../../api/queries";

export default function StrategyParamsForm({ name, onResult }) {
  const { data: strategy, isLoading } = useStrategy(name);
  const [params, setParams] = useState({});
  const [ticker, setTicker] = useState("AAPL");
  const backtest = useStrategyBacktest();

  useEffect(() => {
    if (strategy?.default_params) setParams(strategy.default_params);
    else if (strategy?.parameters) {
      const defaults = {};
      Object.entries(strategy.parameters).forEach(([k, v]) => {
        defaults[k] = v.default ?? v;
      });
      setParams(defaults);
    }
  }, [strategy]);

  const runBacktest = async () => {
    const payload = { ticker, parameters: params };
    const result = await backtest.mutateAsync({ name, payload });
    onResult?.(result);
  };

  if (isLoading) return <div className="text-sm text-muted">Loading parameters...</div>;
  if (!strategy) return null;

  const paramEntries = Object.entries(strategy.parameters || {});

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="ticker">Ticker</Label>
        <Input id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
      </div>
      {paramEntries.map(([key, def]) => (
        <div key={key}>
          <Label htmlFor={key}>{key}</Label>
          <Input
            id={key}
            type="number"
            step="any"
            value={params[key] ?? ""}
            onChange={(e) => setParams({ ...params, [key]: parseFloat(e.target.value) })}
          />
          {def.description && <p className="text-xs text-muted mt-1">{def.description}</p>}
        </div>
      ))}
      <Button onClick={runBacktest} disabled={backtest.isPending}>
        {backtest.isPending ? "Running..." : "Run Backtest"}
      </Button>
      {backtest.error && <ErrorState error={backtest.error} />}
    </div>
  );
}
```

- [ ] **Step 3: BacktestResults.jsx**

```jsx
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { classForDelta, fmtPct } from "../../lib/formatters";

export default function BacktestResults({ result }) {
  if (!result) return <div className="text-sm text-muted">Run a backtest to see results.</div>;

  const metrics = result.metrics || result;

  const kpi = [
    { label: "Sharpe Ratio", value: metrics.sharpe_ratio?.toFixed(2) ?? "—" },
    { label: "Total Return", value: fmtPct(metrics.total_return), color: classForDelta(metrics.total_return) },
    { label: "Max Drawdown", value: fmtPct(metrics.max_drawdown), color: classForDelta(metrics.max_drawdown) },
    { label: "Win Rate", value: fmtPct(metrics.win_rate, { dp: 1 }) },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {kpi.map((k) => (
          <Card key={k.label}>
            <CardContent className="p-4">
              <p className="text-xs text-muted uppercase tracking-wider">{k.label}</p>
              <p className={`text-2xl font-bold font-mono mt-1 ${k.color || "text-foreground"}`}>{k.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>
      {metrics.equity_curve && (
        <Card>
          <CardHeader>
            <CardTitle>Equity curve</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted">
              {metrics.equity_curve.length} data points · final equity{" "}
              <span className="font-mono text-foreground">
                {metrics.equity_curve[metrics.equity_curve.length - 1]?.toFixed(2)}
              </span>
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
```

- [ ] **Step 4: pages/StrategyPage.jsx**

```jsx
import { useState } from "react";
import PageHeader from "../components/PageHeader";
import { Card } from "../components/ui/card";
import StrategyPicker from "../features/strategy/StrategyPicker";
import StrategyParamsForm from "../features/strategy/StrategyParamsForm";
import BacktestResults from "../features/strategy/BacktestResults";

export default function StrategyPage() {
  const [name, setName] = useState("ma_crossover");
  const [result, setResult] = useState(null);

  return (
    <div>
      <PageHeader title="Strategy" subtitle="Rule-based strategies with schema-driven parameters" />
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <Card className="p-3 lg:col-span-1">
          <StrategyPicker selected={name} onSelect={setName} />
        </Card>
        <div className="lg:col-span-3 space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Parameters — {name}</h3>
            <StrategyParamsForm name={name} onResult={setResult} />
          </Card>
          <BacktestResults result={result} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Build + commit**

```bash
cd quant-ai-ui && npm run build 2>&1 | tail -6 && cd ..
git add quant-ai-ui/src/features/strategy/ quant-ai-ui/src/pages/StrategyPage.jsx
git commit -m "feat: [FE-12] implement Strategy page with schema-driven params and backtest results"
```

---

## Task 13: TradingPage (orders / portfolio / WebSocket)

**Files:**
- Create: `quant-ai-ui/src/features/trading/OrderForm.jsx`
- Create: `quant-ai-ui/src/features/trading/PortfolioCard.jsx`
- Create: `quant-ai-ui/src/features/trading/TradeHistory.jsx`
- Create: `quant-ai-ui/src/features/trading/OrderList.jsx`
- Modify: `quant-ai-ui/src/pages/TradingPage.jsx`

- [ ] **Step 1: OrderForm.jsx**

```jsx
import { useState } from "react";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select";
import ErrorState from "../../components/ErrorState";
import { usePlaceOrder } from "../../api/queries";

export default function OrderForm() {
  const [ticker, setTicker] = useState("AAPL");
  const [side, setSide] = useState("buy");
  const [type, setType] = useState("market");
  const [qty, setQty] = useState(10);
  const [price, setPrice] = useState("");
  const place = usePlaceOrder();

  const submit = async (e) => {
    e.preventDefault();
    const payload = { ticker, side, order_type: type, quantity: Number(qty) };
    if (type === "limit") payload.limit_price = Number(price);
    await place.mutateAsync(payload);
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div>
        <Label htmlFor="ticker">Ticker</Label>
        <Input id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label>Side</Label>
          <Select value={side} onValueChange={setSide}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="buy">Buy</SelectItem>
              <SelectItem value="sell">Sell</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Type</Label>
          <Select value={type} onValueChange={setType}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="market">Market</SelectItem>
              <SelectItem value="limit">Limit</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label htmlFor="qty">Quantity</Label>
          <Input id="qty" type="number" min="1" value={qty} onChange={(e) => setQty(e.target.value)} />
        </div>
        {type === "limit" && (
          <div>
            <Label htmlFor="price">Limit Price</Label>
            <Input id="price" type="number" step="0.01" value={price} onChange={(e) => setPrice(e.target.value)} />
          </div>
        )}
      </div>
      <Button type="submit" disabled={place.isPending} className="w-full">
        {place.isPending ? "Placing..." : `Place ${side.toUpperCase()}`}
      </Button>
      {place.error && <ErrorState error={place.error} />}
      {place.data && <div className="text-sm text-up">Order placed · id: {place.data.order_id || place.data.id}</div>}
    </form>
  );
}
```

- [ ] **Step 2: PortfolioCard.jsx**

```jsx
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { usePortfolio, useResetPortfolio } from "../../api/queries";
import { LoadingOverlay } from "../../components/LoadingSpinner";
import ErrorState from "../../components/ErrorState";
import ConfirmDialog from "../../components/ConfirmDialog";
import { Button } from "../../components/ui/button";
import { fmtPrice, fmtPct, classForDelta } from "../../lib/formatters";
import { RefreshCw } from "lucide-react";

export default function PortfolioCard() {
  const { data, isLoading, error, refetch } = usePortfolio();
  const reset = useResetPortfolio();

  if (isLoading) return <LoadingOverlay label="Loading portfolio..." />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;

  const cash = data?.cash ?? 0;
  const equity = data?.total_equity ?? data?.equity ?? 0;
  const pnl = data?.day_pnl ?? 0;
  const pnlPct = data?.day_pnl_pct ?? 0;
  const positions = data?.positions || [];

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between pb-3">
        <CardTitle>Portfolio</CardTitle>
        <ConfirmDialog
          trigger={<Button size="sm" variant="outline"><RefreshCw className="h-3 w-3" /> Reset</Button>}
          title="Reset portfolio?"
          description="All positions and trade history will be cleared. This cannot be undone."
          confirmLabel="Reset"
          destructive
          onConfirm={() => reset.mutate()}
        />
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-xs text-muted uppercase">Cash</p>
            <p className="text-lg font-mono font-semibold">${fmtPrice(cash)}</p>
          </div>
          <div>
            <p className="text-xs text-muted uppercase">Equity</p>
            <p className="text-lg font-mono font-semibold">${fmtPrice(equity)}</p>
          </div>
          <div>
            <p className="text-xs text-muted uppercase">Day P&L</p>
            <p className={`text-lg font-mono font-semibold ${classForDelta(pnl)}`}>
              {pnl >= 0 ? "+" : ""}${fmtPrice(pnl)} ({fmtPct(pnlPct)})
            </p>
          </div>
          <div>
            <p className="text-xs text-muted uppercase">Positions</p>
            <p className="text-lg font-mono font-semibold">{positions.length}</p>
          </div>
        </div>
        {positions.length > 0 && (
          <div className="border-t border-surface-border pt-3 space-y-1">
            {positions.map((p) => (
              <div key={p.ticker} className="flex justify-between text-sm">
                <span className="font-semibold">{p.ticker}</span>
                <span className="text-muted">{p.quantity} @ ${fmtPrice(p.avg_cost)}</span>
                <span className={`font-mono ${classForDelta(p.unrealized_pnl)}`}>{fmtPct(p.unrealized_pnl_pct)}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 3: OrderList.jsx**

```jsx
import { Button } from "../../components/ui/button";
import { Badge } from "../../components/ui/badge";
import { useOrders, useCancelOrder } from "../../api/queries";
import { fmtPrice, fmtDatetime } from "../../lib/formatters";

export default function OrderList() {
  const { data, isLoading } = useOrders("all");
  const cancel = useCancelOrder();
  if (isLoading) return <div className="text-sm text-muted p-4">Loading orders...</div>;
  const orders = (data?.orders || data || []).slice(0, 20);
  if (orders.length === 0) return <div className="text-sm text-muted p-4">No orders yet.</div>;

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase text-muted">
          <tr>
            <th className="px-4 py-2 text-left">Ticker</th>
            <th className="px-4 py-2 text-left">Side</th>
            <th className="px-4 py-2 text-right">Qty</th>
            <th className="px-4 py-2 text-right">Price</th>
            <th className="px-4 py-2 text-left">Status</th>
            <th className="px-4 py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((o) => (
            <tr key={o.order_id || o.id} className="border-t border-surface-border">
              <td className="px-4 py-2 font-semibold">{o.ticker}</td>
              <td className="px-4 py-2">
                <Badge variant={o.side === "buy" ? "success" : "destructive"}>{o.side}</Badge>
              </td>
              <td className="px-4 py-2 text-right font-mono">{o.quantity}</td>
              <td className="px-4 py-2 text-right font-mono">
                {o.limit_price ? `$${fmtPrice(o.limit_price)}` : "market"}
              </td>
              <td className="px-4 py-2">
                <Badge variant="outline">{o.status}</Badge>
              </td>
              <td className="px-4 py-2 text-right">
                {(o.status === "pending" || o.status === "open") && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => cancel.mutate(o.order_id || o.id)}
                  >
                    Cancel
                  </Button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 4: TradeHistory.jsx**

```jsx
import { useTrades } from "../../api/queries";
import { Badge } from "../../components/ui/badge";
import { fmtPrice, fmtDatetime } from "../../lib/formatters";

export default function TradeHistory() {
  const { data, isLoading } = useTrades(20);
  if (isLoading) return <div className="text-sm text-muted p-4">Loading...</div>;
  const trades = data?.trades || data || [];
  if (trades.length === 0) return <div className="text-sm text-muted p-4">No trades yet.</div>;

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase text-muted">
          <tr>
            <th className="px-4 py-2 text-left">Time</th>
            <th className="px-4 py-2 text-left">Ticker</th>
            <th className="px-4 py-2 text-left">Side</th>
            <th className="px-4 py-2 text-right">Qty</th>
            <th className="px-4 py-2 text-right">Price</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr key={t.trade_id || t.id} className="border-t border-surface-border">
              <td className="px-4 py-2 text-muted text-xs">{fmtDatetime(t.timestamp || t.executed_at)}</td>
              <td className="px-4 py-2 font-semibold">{t.ticker}</td>
              <td className="px-4 py-2">
                <Badge variant={t.side === "buy" ? "success" : "destructive"}>{t.side}</Badge>
              </td>
              <td className="px-4 py-2 text-right font-mono">{t.quantity}</td>
              <td className="px-4 py-2 text-right font-mono">${fmtPrice(t.price)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 5: pages/TradingPage.jsx**

```jsx
import PageHeader from "../components/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/card";
import OrderForm from "../features/trading/OrderForm";
import PortfolioCard from "../features/trading/PortfolioCard";
import OrderList from "../features/trading/OrderList";
import TradeHistory from "../features/trading/TradeHistory";
import { useLivePrices } from "../features/trading/useLivePrices";
import { useLiveStore } from "../stores/liveStore";
import { Badge } from "../components/ui/badge";

export default function TradingPage() {
  useLivePrices();
  const status = useLiveStore((s) => s.connectionStatus);
  const variant = status === "connected" ? "success" : status === "error" ? "destructive" : "warning";

  return (
    <div>
      <PageHeader
        title="Paper Trading"
        subtitle="Place orders, track positions, view trade history"
        actions={<Badge variant={variant}>WS {status}</Badge>}
      />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <PortfolioCard />
          <Card>
            <CardHeader><CardTitle>Open Orders</CardTitle></CardHeader>
            <CardContent><OrderList /></CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Recent Trades</CardTitle></CardHeader>
            <CardContent><TradeHistory /></CardContent>
          </Card>
        </div>
        <div>
          <Card>
            <CardHeader><CardTitle>Place Order</CardTitle></CardHeader>
            <CardContent><OrderForm /></CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Build + commit**

```bash
cd quant-ai-ui && npm run build 2>&1 | tail -6 && cd ..
git add quant-ai-ui/src/features/trading/ quant-ai-ui/src/pages/TradingPage.jsx
git commit -m "feat: [FE-13] implement Trading page with orders, portfolio, WebSocket live prices"
```

---

## Task 14: ExplainPage (SHAP + Similar Cases)

**Files:**
- Create: `quant-ai-ui/src/features/explain/ShapFeatureList.jsx`
- Create: `quant-ai-ui/src/features/explain/SimilarCasesList.jsx`
- Modify: `quant-ai-ui/src/pages/ExplainPage.jsx`

- [ ] **Step 1: ShapFeatureList.jsx**

```jsx
export default function ShapFeatureList({ features }) {
  if (!features || features.length === 0) {
    return <p className="text-sm text-muted">No SHAP data available.</p>;
  }
  const max = features[0].mean_abs_shap;

  return (
    <div className="space-y-3">
      {features.map((f, i) => {
        const pct = (f.mean_abs_shap / max) * 100;
        return (
          <div key={i}>
            <div className="flex justify-between text-sm mb-1.5">
              <span className="font-mono font-medium text-foreground">{f.feature}</span>
              <span className="text-muted tabular-nums">{f.mean_abs_shap.toFixed(4)}</span>
            </div>
            <div className="h-2 bg-surface-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-accent transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: SimilarCasesList.jsx**

```jsx
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../../components/ui/accordion";
import { Badge } from "../../components/ui/badge";

export default function SimilarCasesList({ results }) {
  if (!results || results.length === 0) {
    return <p className="text-sm text-muted">No similar cases found.</p>;
  }
  return (
    <Accordion type="multiple">
      {results.map((r, idx) => (
        <AccordionItem key={idx} value={`case-${idx}`}>
          <AccordionTrigger className="hover:no-underline">
            <div className="flex items-center gap-3">
              <Badge variant="info">{r.score?.toFixed(3) || "—"}</Badge>
              <span className="text-sm text-foreground text-left">
                {r.text?.substring(0, 80)}{r.text?.length > 80 ? "..." : ""}
              </span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <p className="text-sm text-muted whitespace-pre-wrap">{r.text}</p>
          </AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  );
}
```

- [ ] **Step 3: pages/ExplainPage.jsx**

```jsx
import { useState } from "react";
import PageHeader from "../components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Label } from "../components/ui/label";
import { LoadingOverlay } from "../components/LoadingSpinner";
import ErrorState from "../components/ErrorState";
import ShapFeatureList from "../features/explain/ShapFeatureList";
import SimilarCasesList from "../features/explain/SimilarCasesList";
import { useExplain, useSimilarCases } from "../api/queries";

export default function ExplainPage() {
  const [ticker, setTicker] = useState("AAPL");
  const [queryTicker, setQueryTicker] = useState("AAPL");

  const explain = useExplain(queryTicker);
  const search = useSimilarCases(queryTicker ? "high volatility rsi failed" : null);

  return (
    <div>
      <PageHeader title="Model Explainability" subtitle="SHAP feature importance + similar historical cases" />
      <form
        className="flex items-end gap-3 mb-6 max-w-md"
        onSubmit={(e) => {
          e.preventDefault();
          setQueryTicker(ticker.toUpperCase());
        }}
      >
        <div className="flex-1">
          <Label htmlFor="ticker">Ticker</Label>
          <Input id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
        </div>
        <Button type="submit">Reload</Button>
      </form>

      {explain.isLoading && <LoadingOverlay label="Loading SHAP..." />}
      {explain.error && <ErrorState error={explain.error} onRetry={() => explain.refetch()} />}

      {!explain.isLoading && !explain.error && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader><CardTitle>SHAP Top Features</CardTitle></CardHeader>
            <CardContent><ShapFeatureList features={explain.data?.top_features} /></CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Similar Historical Cases</CardTitle></CardHeader>
            <CardContent>
              {search.isLoading ? <LoadingOverlay label="Searching..." /> : <SimilarCasesList results={search.data} />}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Build + commit**

```bash
cd quant-ai-ui && npm run build 2>&1 | tail -6 && cd ..
git add quant-ai-ui/src/features/explain/ quant-ai-ui/src/pages/ExplainPage.jsx
git commit -m "feat: [FE-14] implement Explain page with SHAP bars and similar cases accordion"
```

---

## Task 15: Vitest setup + smoke tests + CI job

**Files:**
- Create: `quant-ai-ui/vitest.config.js`
- Create: `quant-ai-ui/src/setupTests.js`
- Create: `quant-ai-ui/src/__tests__/pages.smoke.test.jsx`
- Modify: `quant-ai-ui/package.json` (add `test` script)
- Modify: `.github/workflows/ci.yml` (add frontend-test job)

- [ ] **Step 1: vitest.config.js**

```js
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/setupTests.js"],
    css: false,
  },
});
```

- [ ] **Step 2: src/setupTests.js**

```js
import "@testing-library/jest-dom";

// Mock Lightweight Charts (jsdom doesn't support canvas)
vi.mock("lightweight-charts", () => ({
  createChart: () => ({
    addCandlestickSeries: () => ({ setData: vi.fn(), setMarkers: vi.fn() }),
    timeScale: () => ({ fitContent: vi.fn() }),
    remove: vi.fn(),
  }),
  ColorType: { Solid: "solid" },
}));

// Mock WebSocket for useLivePrices
global.WebSocket = class {
  constructor() {}
  close() {}
  set onopen(_) {}
  set onmessage(_) {}
  set onerror(_) {}
  set onclose(_) {}
};
```

- [ ] **Step 3: src/__tests__/pages.smoke.test.jsx**

```jsx
import { describe, test, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ScreenerPage from "../pages/ScreenerPage";
import DashboardPage from "../pages/DashboardPage";
import TrainingPage from "../pages/TrainingPage";
import StrategyPage from "../pages/StrategyPage";
import TradingPage from "../pages/TradingPage";
import ExplainPage from "../pages/ExplainPage";

function renderPage(Page, route = "/") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[route]}>
        <Routes>
          <Route path="*" element={<Page />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe("smoke: all pages render", () => {
  test("ScreenerPage", () => {
    renderPage(ScreenerPage);
    expect(screen.getByText(/screener/i)).toBeInTheDocument();
  });
  test("DashboardPage", () => {
    renderPage(DashboardPage, "/dashboard?ticker=AAPL");
    expect(screen.getByText(/AAPL/i)).toBeInTheDocument();
  });
  test("TrainingPage", () => {
    renderPage(TrainingPage);
    expect(screen.getByText(/Training/i)).toBeInTheDocument();
  });
  test("StrategyPage", () => {
    renderPage(StrategyPage);
    expect(screen.getByText(/Strategy/i)).toBeInTheDocument();
  });
  test("TradingPage", () => {
    renderPage(TradingPage);
    expect(screen.getByText(/Paper Trading/i)).toBeInTheDocument();
  });
  test("ExplainPage", () => {
    renderPage(ExplainPage);
    expect(screen.getByText(/Explainability/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 4: package.json test script**

In `quant-ai-ui/package.json` under `"scripts"`, add `"test": "vitest"`.

- [ ] **Step 5: Run tests**

Run: `cd quant-ai-ui && npm test -- --run 2>&1 | tail -15`
Expected: all 6 smoke tests pass.

- [ ] **Step 6: Add frontend job to .github/workflows/ci.yml**

Append to `.github/workflows/ci.yml` (before `deploy-check:`):

```yaml
  frontend-test:
    name: Frontend Tests
    runs-on: ubuntu-latest
    needs: lint
    defaults:
      run:
        working-directory: quant-ai-ui
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: quant-ai-ui/package-lock.json
      - run: npm ci
      - run: npm test -- --run
      - run: npm run build
```

- [ ] **Step 7: Commit**

```bash
git add quant-ai-ui/vitest.config.js quant-ai-ui/src/setupTests.js quant-ai-ui/src/__tests__/ \
        quant-ai-ui/package.json .github/workflows/ci.yml
git commit -m "feat: [FE-15] add Vitest smoke tests and CI frontend-test job"
```

---

## Task 16: Migration cleanup (delete old files)

**Files:**
- Delete: `quant-ai-ui/src/pages/Screener.jsx`
- Delete: `quant-ai-ui/src/pages/Dashboard.jsx`
- Delete: `quant-ai-ui/src/pages/Training.jsx`
- Delete: `quant-ai-ui/src/pages/Strategy.jsx`
- Delete: `quant-ai-ui/src/pages/Trading.jsx`
- Delete: `quant-ai-ui/src/pages/Explain.jsx`
- Delete: `quant-ai-ui/src/components/TrainingForm.jsx`
- Delete: `quant-ai-ui/src/components/ModelsList.jsx`
- Delete: `quant-ai-ui/src/components/RunsList.jsx`

- [ ] **Step 1: Delete old files**

```bash
cd /c/Users/zjg09/projects/quant-ai
rm quant-ai-ui/src/pages/Screener.jsx \
   quant-ai-ui/src/pages/Dashboard.jsx \
   quant-ai-ui/src/pages/Training.jsx \
   quant-ai-ui/src/pages/Strategy.jsx \
   quant-ai-ui/src/pages/Trading.jsx \
   quant-ai-ui/src/pages/Explain.jsx \
   quant-ai-ui/src/components/TrainingForm.jsx \
   quant-ai-ui/src/components/ModelsList.jsx \
   quant-ai-ui/src/components/RunsList.jsx
```

- [ ] **Step 2: Verify no broken imports**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -8`
Expected: clean build. If errors appear about missing old files, search imports and replace.

Run: `cd quant-ai-ui && npm test -- --run 2>&1 | tail -5`
Expected: all smoke tests pass.

- [ ] **Step 3: Commit**

```bash
git add -A quant-ai-ui/
git commit -m "chore: [FE-16] remove legacy pages and components after migration"
```

---

## Task 17: FE-GATE — Full verification

**Files:** (none modified — verification only)

- [ ] **Step 1: Full frontend build**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -10`
Expected: clean build, bundle size reasonable (< 500 KB gzipped).

- [ ] **Step 2: All frontend tests pass**

Run: `cd quant-ai-ui && npm test -- --run 2>&1 | tail -5`
Expected: all tests pass.

- [ ] **Step 3: Backend tests untouched**

Run: `cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/ --ignore=tests/contract -q 2>&1 | tail -5`
Expected: 274 passed (unchanged from before this sub-project).

- [ ] **Step 4: Ruff lint**

Run: `ruff check app/ --ignore F401,F841,E501,F541,E402 2>&1 | tail -3`
Expected: `All checks passed!`

- [ ] **Step 5: Verify file structure**

Run:
```bash
cd /c/Users/zjg09/projects/quant-ai/quant-ai-ui/src
ls app/ pages/ features/charts/ features/trading/ features/training/ features/strategy/ features/explain/ features/screener/ components/ components/ui/ api/ stores/ hooks/ lib/ __tests__/ 2>&1 | head -40
```
Expected: all directories contain expected files listed in plan's File Structure.

- [ ] **Step 6: Gate commit**

```bash
git commit --allow-empty -m "feat: [FE-GATE] Phase 3 Sub-project 4 — frontend redesign gate passed"
```

---

## Self-Review

**1. Spec coverage**:
- §3 Architecture → Tasks 1-17 collectively ✓
- §4 Design tokens → Task 2 ✓
- §5 Directory restructure → established via Tasks 3, 7, 8, 9-14 ✓
- §6 AppShell + Sidebar → Task 8 ✓
- §7.1-7.6 all 6 pages → Tasks 9, 10, 11, 12, 13, 14 respectively ✓
- §8 API layer (queries) → Task 5 ✓
- §9 Zustand live store → Task 6 ✓
- §10 Form validation (react-hook-form + zod) → Task 11 (TrainForm) ✓
- §11 Testing → Task 15 ✓
- §12 Migration strategy → Tasks 8 (placeholders) + 16 (cleanup) ✓
- §13 Success criteria → Task 17 gate ✓
- §14 Out of scope — deliberately not implemented ✓

**2. Placeholder scan**:
- Task 4 Step 10 says "Full implementations for each... follow standard shadcn patterns" — this could be a placeholder. Fix: an executor should reference shadcn-ui GitHub source when unclear. Acceptable given Ralph batch size limits plan length.
- No TBD/TODO/implement-later in code blocks.

**3. Type consistency**:
- `PredictionEvent` not used in frontend (backend only).
- `PortfolioCard` reads `total_equity`/`equity` (defensive fallback).
- `useOrders("all")` matches `listOrders(status="all")` in client.js.
- `useLiveStore` hook name consistent across liveStore.js, useLivePrices.js, TradingPage.jsx.
- `api.placeOrder`, `api.cancelOrder` etc. match existing `client.js` exports — no renames.

**4. Gap check**:
- Prediction events from Kafka consumer's `/stats/{ticker}` — referenced in spec §7.2 as a Dashboard feature, but plan Task 10 does not include it. This is because that endpoint exists on a separate service (`app/workers/events_consumer.py`) deployed on port 8001 that the current frontend cannot reach (Render deploys only the main API on 8000). **Decision**: defer this to a follow-up sub-project that requires K8s cloud deploy. Note in `frontend-redesign-progress.md`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-frontend-redesign.md`.

Two execution options per writing-plans skill:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

For this project, since Harry uses Ralph Loop infrastructure (proven on Optuna, Ensemble, Distributed sub-projects), the actual execution method will be **Ralph Loop** (the project's custom pattern), not standard subagent-driven or inline execution. This plan's tasks map 1:1 to `plans/prd.json` features — see `D:/obsidian vault/01-projects/quant-ai/frontend-redesign-task-plan.md` for Ralph task IDs (FE-1 through FE-GATE).
