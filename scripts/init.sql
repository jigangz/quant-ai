-- ===================================
-- Quant-AI Database Initialization
-- ===================================

-- Prices table (existing from V1)
CREATE TABLE IF NOT EXISTS prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);

-- ===================================
-- V2 Tables (Model Registry)
-- ===================================

-- Model Registry: stores trained model metadata
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- active, archived, promoted
    artifact_path TEXT,
    metrics JSONB,
    feature_groups TEXT[],
    tickers TEXT[],
    train_start_date DATE,
    train_end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training Runs: records each training execution
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES model_registry(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, running, success, failed
    params JSONB,
    metrics JSONB,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status);
CREATE INDEX IF NOT EXISTS idx_model_registry_model_type ON model_registry(model_type);
CREATE INDEX IF NOT EXISTS idx_training_runs_model_id ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);

-- ===================================
-- V3 Tables (Experiment Tracking) - Placeholder
-- ===================================
-- Will be added in V3:
-- - experiments
-- - experiment_runs
-- - artifacts
