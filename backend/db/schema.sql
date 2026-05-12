CREATE TABLE IF NOT EXISTS items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,           -- Publication name: "arXiv", "The Gradient", "Reddit", etc.
    title           TEXT NOT NULL,
    url             TEXT NOT NULL UNIQUE,    -- Canonical link; enforces global dedup
    abstract        TEXT NOT NULL DEFAULT '',-- Raw abstract, truncated to 500 chars
    authors         TEXT NOT NULL DEFAULT '[]', -- JSON array: '["Author One", "Author Two"]'
    published_date  TEXT NOT NULL,           -- ISO date YYYY-MM-DD
    topic_bucket    TEXT NOT NULL,           -- From topic_taxonomy in config.yaml
    summary         TEXT,                    -- 2–3 sentence LLM summary. NULL until scored/summarised
    embedding       BLOB,                    -- float32 numpy array (384,) as raw bytes via ndarray.tobytes()
    raw_id          TEXT NOT NULL,           -- Source-native ID for ingest-time dedup
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS digests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start      TEXT NOT NULL UNIQUE,    -- ISO date of the Friday this digest covers
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS digest_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_id       INTEGER NOT NULL REFERENCES digests(id),
    item_id         INTEGER NOT NULL REFERENCES items(id),
    relevance_score REAL NOT NULL DEFAULT -1.0, -- 0.0–1.0; -1.0 sentinel for wildcard slots
    is_wildcard     INTEGER NOT NULL DEFAULT 0,  -- 0 = exploit, 1 = wildcard
    position        INTEGER NOT NULL,            -- 0–4; 0–2 exploit, 3–4 wildcard
    UNIQUE(digest_id, item_id),                  -- Prevents duplicate picks in one digest
    UNIQUE(digest_id, position)                  -- Enforces exactly one item per position
);

CREATE TABLE IF NOT EXISTS ratings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         INTEGER NOT NULL REFERENCES items(id) UNIQUE, -- One rating per paper; upsert replaces
    rating          TEXT NOT NULL CHECK(rating IN ('up', 'down')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prose           TEXT NOT NULL,           -- Full interest profile prose text
    mode            TEXT NOT NULL CHECK(mode IN ('incremental', 'full_rebuild')),
    rated_items_count INTEGER NOT NULL,      -- Total ratings in DB at generation time
    llm_provider    TEXT NOT NULL,           -- e.g. "claude", "gemini", "ollama"
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    -- Append-only. Newest row by created_at is the active profile.
    -- Never UPDATE or DELETE rows. Full history is intentional.
);

CREATE TABLE IF NOT EXISTS topic_exposure (
    topic_bucket    TEXT PRIMARY KEY,
    seen_count      INTEGER NOT NULL DEFAULT 0,  -- Times appeared in any digest
    up_count        INTEGER NOT NULL DEFAULT 0,  -- Times a paper in bucket was rated up
    down_count      INTEGER NOT NULL DEFAULT 0,  -- Times a paper in bucket was rated down
    last_seen       TEXT                         -- ISO date of most recent digest appearance
);
