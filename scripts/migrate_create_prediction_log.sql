-- V4 P5 migration · Create prediction_log table for live accuracy tracking
-- Date: 2026-04-25
-- Safe to run multiple times (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS prediction_log (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id        TEXT NOT NULL,
  ticker          TEXT NOT NULL,
  label_type      TEXT NOT NULL CHECK (label_type IN ('direction','volatility','meta_label')),

  predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  horizon_days    INTEGER NOT NULL,
  predicted_value NUMERIC NOT NULL,
  predicted_signal INTEGER,
  predicted_extras JSONB NOT NULL DEFAULT '{}'::jsonb,

  resolve_at      TIMESTAMPTZ NOT NULL,
  actual_value    NUMERIC,
  actual_return   NUMERIC,
  is_correct      BOOLEAN,
  realized_R      NUMERIC,
  resolved_at     TIMESTAMPTZ,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_log_model_id ON prediction_log(model_id);
CREATE INDEX IF NOT EXISTS idx_pred_log_resolve_pending
  ON prediction_log(resolve_at) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_pred_log_ticker_label
  ON prediction_log(ticker, label_type);

-- Rollback (if ever needed):
-- DROP TABLE IF EXISTS prediction_log;
