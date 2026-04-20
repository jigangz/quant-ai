# Guardrails

### SIGN-001: Verify Before Complete
**Trigger:** About to output completion promise
**Instruction:** Run verify command first. Never mark a task as passes:true without green verify.

### SIGN-002: Read Before Write
**Trigger:** About to modify any existing file
**Instruction:** Read the file first. Check current state. Don't assume.

### SIGN-003: Check Actual API/Types
**Trigger:** Using a library you haven't verified
**Instruction:** Check node_modules or actual installed version for API shape. Don't rely on memory.

### SIGN-004: Normalize Data at Boundary
**Trigger:** Receiving data from API/external source
**Instruction:** Normalize field names at the adapter layer, not deep in components.

### SIGN-005: One Concern Per Mechanism
**Trigger:** Adding automatic behavior (resize, sync, etc.)
**Instruction:** Never duplicate: e.g. autoSize + ResizeObserver = infinite loop. Pick one.

### SIGN-006: Cleanup in Effects
**Trigger:** Creating resources in useEffect / onMount
**Instruction:** Always return cleanup function. React StrictMode double-mounts will expose leaks.

### SIGN-010: Phase Gate Is Mandatory
**Trigger:** Last task in a phase (GATE task)
**Instruction:** ALL criteria must pass before marking gate. Gates run solo, never batched.

### SIGN-011: Batch ≠ Skip
**Trigger:** Working on batch of tasks
**Instruction:** Each task must meet its own acceptance criteria. Don't shortcut because you're in a batch.

### SIGN-012: Never Skip to Next Phase
**Trigger:** Phase gate fails
**Instruction:** Fix current phase first. Do not proceed.

### SIGN-013: Commit Per Task
**Trigger:** Completing a task in a batch
**Instruction:** One commit per task with "feat: [task-id] - description" format.

# Project-Specific Guardrails

### SIGN-020: Use Plan File for Code
**Trigger:** Implementing any task
**Instruction:** Read docs/superpowers/plans/2026-04-19-dashboard-productization.md for exact code. Don't improvise. Task IDs FE-DASH-N in PRD map to Task N in the plan.

### SIGN-021: Frontend Work from quant-ai-ui/
**Trigger:** Any npm or frontend command
**Instruction:** Always cd to quant-ai-ui/ first. The frontend is a sub-directory, not the project root.

### SIGN-022: Mock at Router Boundary
**Trigger:** Writing API tests
**Instruction:** Patch the function the router imports (e.g. app.api.market.get_prices), not deep internal functions.

### SIGN-023: Page-Scoped Theme (migration in progress)
**Trigger:** Any UI styling
**Instruction:** Theme is managed via `<ThemeScope value="light|dark">` at the page level (Sub 1 = `/dashboard` uses light; other 5 pages stay dark until their sub migrates them). Always use semantic Tailwind tokens: bg-background, bg-surface, text-foreground, text-muted, bg-accent, text-up, text-down, text-warn — these resolve per-theme automatically via CSS variables. No hardcoded hex and no inline styles.

### SIGN-024: Python 3.9 Compat
**Trigger:** Writing Python code
**Instruction:** Every file must have `from __future__ import annotations` at top. Use Optional[str] not str | None in runtime contexts.

### SIGN-025: API Contract Pre-Check (Boundary Safety)
**Trigger:** Adding a new client function in src/api/client.js OR consuming a backend endpoint for the first time
**Instruction:** BEFORE writing the client function, curl the endpoint against https://quant-ai-qzrg.onrender.com (live backend) to confirm response shape. If the actual shape differs from what the plan expects, normalize at the client.js boundary (per SIGN-004) and document the mismatch in docs/backend-gaps.md. NEVER let components deal with shape mismatches. Prior incident: `/data/market` returned a flat array while frontend expected `{rows: [...]}` — caused empty Screener/Dashboard until normalization was added. Prevent repeats.

### SIGN-026: Page-Scope Theme Wrapper
**Trigger:** Creating or modifying a migrated-to-light page
**Instruction:** Wrap the page root in `<ThemeScope value="light">`. Do not change the global `<html>` data-theme. Other pages remain dark until their sub-project migrates them.
