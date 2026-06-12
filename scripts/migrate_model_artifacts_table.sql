-- Persist model artifacts + promotion in the DB (2026-06-12)
--
-- Render free tier has an ephemeral filesystem: local artifacts/*.pkl and the
-- artifacts/.promoted_model file are wiped on every spin-down, so models and
-- the promotion pointer vanished. Move both into Supabase Postgres:
--   * model_artifacts.blob  — the zipped saved-model dir (kept in a SEPARATE
--     table so list_models' SELECT * never drags binary over the wire)
--   * model_registry.is_promoted — the production pointer, was a local file
-- Idempotent.

CREATE TABLE IF NOT EXISTS model_artifacts (
    model_id   text PRIMARY KEY REFERENCES model_registry(id) ON DELETE CASCADE,
    blob       bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE model_registry
    ADD COLUMN IF NOT EXISTS is_promoted boolean NOT NULL DEFAULT false;
