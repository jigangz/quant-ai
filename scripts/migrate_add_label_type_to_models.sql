-- V4 Pivot migration · Add label_type + horizon_days to model_registry
-- Date: 2026-04-22
-- Safe to run multiple times (IF NOT EXISTS).

-- Supabase / Postgres
ALTER TABLE IF EXISTS model_registry
    ADD COLUMN IF NOT EXISTS label_type TEXT NOT NULL DEFAULT 'direction';

ALTER TABLE IF EXISTS model_registry
    ADD COLUMN IF NOT EXISTS horizon_days INTEGER NOT NULL DEFAULT 5;

-- Optional: index for filtering models by label_type (UI Leaderboard filter)
CREATE INDEX IF NOT EXISTS idx_model_registry_label_type
    ON model_registry(label_type);

-- Rollback (if ever needed):
-- ALTER TABLE model_registry DROP COLUMN IF EXISTS label_type;
-- ALTER TABLE model_registry DROP COLUMN IF EXISTS horizon_days;
-- DROP INDEX IF EXISTS idx_model_registry_label_type;
