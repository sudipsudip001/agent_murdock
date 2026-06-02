CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    metadata    JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS user_preferences (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category            TEXT NOT NULL,
    key                 TEXT NOT NULL,
    value               TEXT NOT NULL,
    confidence          FLOAT DEFAULT 1.0,
    source              TEXT DEFAULT 'inferred',
    first_seen_at       TIMESTAMPTZ DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ DEFAULT NOW(),
    update_count        INT DEFAULT 1,
    embedding           vector(1536),
    UNIQUE (user_id, category, key)
);

CREATE INDEX IF NOT EXISTS idx_prefs_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_prefs_category ON user_preferences(user_id, category);
CREATE INDEX IF NOT EXISTS idx_prefs_confidence ON user_preferences(user_id, confidence DESC);

CREATE TABLE IF NOT EXISTS user_entities (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    entity_type         TEXT NOT NULL,
    entity_name         TEXT NOT NULL,
    attributes          JSONB DEFAULT '{}',
    last_seen_at        TIMESTAMPTZ DEFAULT NOW(),
    mention_count       INT DEFAULT 1,

    UNIQUE (user_id, entity_type, entity_name)
);

CREATE INDEX IF NOT EXISTS idx_entities_user_id ON user_entities(user_id);
CREATE INDEX IF NOT EXISTS idx_entities_type    ON user_entities(user_id, entity_type);

CREATE TABLE IF NOT EXISTS memory_extraction_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL REFERENCES users(user_id),
    session_id      TEXT NOT NULL,           -- LangGraph thread_id
    extracted_at    TIMESTAMPTZ DEFAULT NOW(),
    turns_processed INT,
    prefs_upserted  INT DEFAULT 0,
    entities_upserted INT DEFAULT 0,
    raw_llm_output  JSONB                    -- store the raw extraction for debugging
);
