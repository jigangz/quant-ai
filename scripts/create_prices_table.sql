-- ===================================
-- Quant AI — Prices Table + RLS
-- ===================================
-- Run this in Supabase SQL Editor:
-- Dashboard → quant-ai → SQL Editor → paste → Run

-- 1. Table
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

-- 2. Index for fast per-ticker lookups
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);

-- 3. Enable Row-Level Security
ALTER TABLE prices ENABLE ROW LEVEL SECURITY;

-- 4. Allow service_role full access (backend writes)
CREATE POLICY "Service role full access" ON prices
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- 5. Allow public read (frontend screener doesn't auth; open reads OK for this data)
CREATE POLICY "Public read access" ON prices
    FOR SELECT
    TO anon, authenticated
    USING (true);

-- Verify
SELECT COUNT(*) AS existing_rows FROM prices;
