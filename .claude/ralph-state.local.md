---
iteration: 8
max_iterations: 20
run_id: "20260413-202348"
mode: fresh
---

## Guardrails (Signs)

Follow these learned constraints to avoid repeated failures:

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
**Instruction:** Read docs/superpowers/plans/2026-04-13-phase1-2.5-verification.md for exact code. Don't improvise.

### SIGN-021: Frontend Work from quant-ai-ui/
**Trigger:** Any npm or frontend command
**Instruction:** Always cd to quant-ai-ui/ first. The frontend is a sub-directory, not the project root.

### SIGN-022: Mock at Router Boundary
**Trigger:** Writing API tests
**Instruction:** Patch the function the router imports (e.g. app.api.market.get_prices), not deep internal functions.

### SIGN-023: Tailwind Dark Theme
**Trigger:** Any UI styling
**Instruction:** Use the custom colors: bg-surface, bg-surface-card, text-up, text-down, bg-accent. No inline styles.

### SIGN-024: Python 3.9 Compat
**Trigger:** Writing Python code
**Instruction:** Every file must have `from __future__ import annotations` at top. Use Optional[str] not str | None in runtime contexts.

---

## Instructions

You are in a Ralph loop (fresh-context mode). **Each iteration = up to 3 related tasks in the same phase.**

1. Read plans/prd.json and find the pending tasks where passes: false
2. Read plans/progress.md for context from previous iterations
3. Read and follow the Guardrails above - they prevent repeated mistakes
4. Work on these tasks for this iteration: **P3-1,P3-2,P3-3**
   - Complete each task until its acceptance criteria are met
   - If there are multiple tasks, do them sequentially (they are in the same phase)
   - GATE tasks are always done solo — verify ALL criteria thoroughly
5. Run verification after completing the batch: cd /c/Users/zjg09/projects/quant-ai && python -m pytest tests/ -v --tb=short --ignore=tests/contract -p no:cacheprovider 2>&1 | tail -5
6. For each completed task:
   - Update prd.json: set passes: true and add completed_at
   - Update progress.md with what you learned
   - **COMMIT your changes** with message: "feat: [task-id] - description"
7. After completing ALL assigned tasks, check prd.json:
   - If ALL tasks in the entire project pass: output <promise>COMPLETE</promise>
   - If tasks remain: **EXIT immediately** - do NOT continue to unassigned tasks

**Critical:** This is fresh-context mode. Complete ONLY the assigned tasks (P3-1,P3-2,P3-3), commit, then EXIT.
The bash loop will spawn a fresh session for the next batch. Do NOT work on tasks outside this batch.
