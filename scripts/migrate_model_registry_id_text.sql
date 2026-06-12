-- model_registry.id: UUID -> TEXT (2026-06-12)
--
-- The app's identity system uses human-readable string model ids
-- ("logistic_AAPL_20260612_073908") everywhere: SupabaseModelRegistry
-- .insert_model(), /models/{id} routes, artifact paths, frontend deep
-- links. The bootstrap SQL declared id as UUID, so registry inserts
-- failed with 22P02 'invalid input syntax for type uuid' (POST /train
-- returned 500). training_runs.model_id references it and converts in
-- the same transaction. Idempotent: safe to re-run.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'model_registry'
          AND column_name = 'id'
          AND data_type = 'uuid'
    ) THEN
        ALTER TABLE training_runs DROP CONSTRAINT IF EXISTS training_runs_model_id_fkey;

        ALTER TABLE model_registry ALTER COLUMN id DROP DEFAULT;
        ALTER TABLE model_registry ALTER COLUMN id TYPE text USING id::text;

        ALTER TABLE training_runs ALTER COLUMN model_id TYPE text USING model_id::text;
        ALTER TABLE training_runs
            ADD CONSTRAINT training_runs_model_id_fkey
            FOREIGN KEY (model_id) REFERENCES model_registry(id);
    END IF;
END $$;
