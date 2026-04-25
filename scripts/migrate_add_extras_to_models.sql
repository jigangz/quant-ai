-- V4 P4 migration · Add extras JSONB column to model_registry
-- Date: 2026-04-24
-- Safe to run multiple times (IF NOT EXISTS).
--
-- Why: P4-1 /api/meta-label/coverage filters meta-models by their primary
-- strategy (extras.meta_label.primary.strategy_name). Without this column
-- _register_meta_model can't persist primary/barrier/cv config, and
-- compute_coverage always returns count:0 even when meta-models exist.

-- Supabase / Postgres
ALTER TABLE IF EXISTS model_registry
    ADD COLUMN IF NOT EXISTS extras JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Optional: GIN index for querying inside extras (e.g. by
-- extras->'meta_label'->'primary'->>'strategy_name').
CREATE INDEX IF NOT EXISTS idx_model_registry_extras_meta_strategy
    ON model_registry((extras->'meta_label'->'primary'->>'strategy_name'));

-- Rollback (if ever needed):
-- ALTER TABLE model_registry DROP COLUMN IF EXISTS extras;
-- DROP INDEX IF EXISTS idx_model_registry_extras_meta_strategy;
