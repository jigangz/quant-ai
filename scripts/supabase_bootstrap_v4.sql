-- ============================================================================
-- Quant AI · Supabase Bootstrap + V4 Pivot Migration (2026-04-21)
-- ============================================================================
-- Safe to run multiple times (all CREATE IF NOT EXISTS / ADD IF NOT EXISTS).
-- Supersedes scripts/init.sql (for V4) and scripts/migrate_add_label_type_to_models.sql.
--
-- Run this ONCE in Supabase SQL Editor:
--   https://supabase.com/dashboard/project/_/sql
--
-- What this does:
--   1. Creates model_registry + training_runs tables (V4-ready schema)
--   2. Adds V4 columns (label_type, horizon_days) if tables already existed
--   3. Creates indexes incl. V4 label_type filter
-- ============================================================================

-- ============================================================================
-- 1. model_registry (V4 schema)
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    model_type      VARCHAR(50)  NOT NULL,
    version         INTEGER      NOT NULL DEFAULT 1,
    status          VARCHAR(20)  NOT NULL DEFAULT 'active',

    -- V4 Pivot · Multi-task target metadata
    label_type      TEXT         NOT NULL DEFAULT 'direction',
    horizon_days    INTEGER      NOT NULL DEFAULT 5,

    -- Dataset info (Pydantic ModelRecord fields)
    tickers         TEXT[],
    feature_groups  TEXT[],
    feature_names   TEXT[],
    n_features      INTEGER      DEFAULT 0,
    train_samples   INTEGER      DEFAULT 0,
    val_samples     INTEGER      DEFAULT 0,
    test_samples    INTEGER      DEFAULT 0,
    train_date_range TEXT[],

    -- Metrics + artifacts
    metrics         JSONB        DEFAULT '{}'::jsonb,
    artifact_path   TEXT,
    storage_backend VARCHAR(20)  DEFAULT 'local',

    -- Timestamps
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- If table already existed from pre-V4 era, add V4 columns idempotently:
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS label_type TEXT NOT NULL DEFAULT 'direction';
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS horizon_days INTEGER NOT NULL DEFAULT 5;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS feature_names TEXT[];
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS n_features INTEGER DEFAULT 0;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS train_samples INTEGER DEFAULT 0;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS val_samples INTEGER DEFAULT 0;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS test_samples INTEGER DEFAULT 0;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS train_date_range TEXT[];
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS storage_backend VARCHAR(20) DEFAULT 'local';


-- ============================================================================
-- 2. training_runs (full experiment tracking)
-- ============================================================================
CREATE TABLE IF NOT EXISTS training_runs (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id             UUID REFERENCES model_registry(id) ON DELETE SET NULL,

    -- Reproducibility
    git_sha              TEXT,
    git_branch           TEXT,
    git_dirty            BOOLEAN   DEFAULT FALSE,
    data_hash            TEXT,
    config_hash          TEXT,

    -- Request params (V4: label_type tracked here since V3)
    tickers              TEXT[],
    model_type           VARCHAR(50),
    feature_groups       TEXT[],
    model_params         JSONB     DEFAULT '{}'::jsonb,
    horizon_days         INTEGER   DEFAULT 5,
    label_type           TEXT      DEFAULT 'direction',
    train_ratio          FLOAT     DEFAULT 0.7,
    val_ratio            FLOAT     DEFAULT 0.15,

    -- Data info
    start_date           TEXT,
    end_date             TEXT,
    train_samples        INTEGER   DEFAULT 0,
    val_samples          INTEGER   DEFAULT 0,
    test_samples         INTEGER   DEFAULT 0,
    train_date_range     TEXT[],
    val_date_range       TEXT[],
    test_date_range      TEXT[],

    -- Results
    success              BOOLEAN   DEFAULT FALSE,
    error                TEXT,
    metrics              JSONB     DEFAULT '{}'::jsonb,
    training_time_seconds FLOAT    DEFAULT 0.0,

    -- Environment
    python_version       TEXT,
    packages             JSONB     DEFAULT '{}'::jsonb,

    -- Timestamps
    started_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at         TIMESTAMP
);

-- Safety: add V4 columns if table already existed
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS label_type TEXT DEFAULT 'direction';
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS horizon_days INTEGER DEFAULT 5;

-- ============================================================================
-- 3. Indexes
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_model_registry_status       ON model_registry(status);
CREATE INDEX IF NOT EXISTS idx_model_registry_model_type   ON model_registry(model_type);
CREATE INDEX IF NOT EXISTS idx_model_registry_label_type   ON model_registry(label_type);   -- V4
CREATE INDEX IF NOT EXISTS idx_training_runs_model_id      ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_label_type    ON training_runs(label_type);    -- V4

-- ============================================================================
-- 4. Verify (prints row with column names)
-- ============================================================================
-- After running, paste this in SQL editor to confirm columns exist:
--
--   SELECT column_name, data_type, column_default
--   FROM information_schema.columns
--   WHERE table_name = 'model_registry'
--   ORDER BY ordinal_position;
--
-- You should see label_type + horizon_days in the list.

-- Rollback (only if ever needed):
-- ALTER TABLE model_registry DROP COLUMN IF EXISTS label_type;
-- ALTER TABLE model_registry DROP COLUMN IF EXISTS horizon_days;
-- DROP INDEX IF EXISTS idx_model_registry_label_type;
-- DROP INDEX IF EXISTS idx_training_runs_label_type;
