# Project Specification: `paper-scout`

## 0. Purpose of This Document

This document is the single source of truth for the coding agent implementing
`paper-scout`. Read it in full before writing any code. Follow every explicit
decision stated here rather than making assumptions or applying defaults.
Where this document is silent, ask — do not invent.

---

## 1. Project Overview

`paper-scout` is a weekly AI paper and idea discovery agent. It ingests
content from arXiv, Semantic Scholar, HuggingFace Papers, RSS feeds (The
Gradient, Distill), and Reddit (r/MachineLearning, r/artificial). Each week
it selects 5 items: 3 exploit picks (highest relevance to a learned interest
profile) and 2 wildcard picks (structured exploration from underrepresented
topic buckets). Picks are delivered via Gmail on Friday morning with a 2–3
sentence plain-English summary per item. The user rates each item via signed
one-click thumbs-up/down links in the email. Ratings are stored in SQLite and
drive incremental refinement of the user's interest profile, improving future
picks.

Sentence Transformers (`all-MiniLM-L6-v2`, Apache 2.0) run unconditionally
for deduplication, topic classification, and candidate pre-filtering —
minimising LLM API calls. One LLM provider is active at runtime (Claude,
Gemini, or Ollama), configured via environment variable, used only for:
(a) combined relevance scoring + summary generation per candidate, and
(b) interest profile generation and incremental refinement.

**Interfaces:**
- **CLI** — primary interface. Commands: `digest`, `history`, `profile`.
  Used locally and by GitHub Actions cron.
- **Minimal FastAPI** — 2 routes only (`GET /rate`, `GET /health`).
  Deployed on Railway. Exists solely to handle signed one-click rating
  links from email. No frontend. No Streamlit.

**Deployment:**
- Railway: hosts the minimal FastAPI service with a persistent volume
  for SQLite.
- GitHub Actions: runs the weekly digest cron (Friday 09:00 UTC) and CI.
- Local: CLI for manual digest runs, history browsing, profile inspection.

**No Streamlit. No extra frontend. No LangGraph. No full REST API.**
The email is the weekly UI. The CLI is the management UI.

---

## 2. Bugs / Issues in Existing Code

_Greenfield project. No existing code to audit._

---

## 3. Architecture

### 3.1 Separation of Concerns

**Canonical rule: The FastAPI service handles only rating submissions and
health checks. All business logic lives in CLI-callable Python modules.
The API imports from those modules — not the reverse. No module outside
`backend/db/repository.py` writes to the database directly.**

| Layer | Owns | Must NOT contain |
|---|---|---|
| `backend/api/` | Token verification, rating DB write, profile regen trigger | Scoring, ingestion, email, exploration logic |
| `backend/cli.py` | CLI entry points, argument parsing | Business logic (delegates to modules) |
| `backend/digest.py` | Weekly pipeline orchestration | HTTP handling, email formatting |
| `backend/agents/scorer.py` | LLM scoring + summary (one call) | DB writes, HTTP |
| `backend/agents/profiler.py` | LLM profile generation + incremental refinement | DB writes, HTTP |
| `backend/agents/explorer.py` | Wildcard selection strategies | DB writes, LLM calls |
| `backend/ingestors/` | Per-source fetch, normalize to `RawItem` | DB writes, scoring, embedding |
| `backend/embeddings.py` | Model load, encode, deduplicate, classify | DB writes, LLM calls |
| `backend/llm/` | Provider abstraction + 3 adapters | DB access, business logic |
| `backend/db/repository.py` | All DB read/write operations | LLM calls, HTTP, embedding generation |
| `backend/email_sender.py` | Format + send Gmail digest | DB writes, scoring |
| `backend/security.py` | HMAC token generation + verification | Everything else |

### 3.2 Directory Structure

```
paper-scout/
├── backend/
│   ├── cli.py                       # Typer CLI: digest, history, profile subcommands
│   ├── config.py                    # Loads config.yaml + .env → typed Settings dataclass
│   ├── digest.py                    # Weekly orchestrator
│   ├── email_sender.py              # Gmail SMTP via smtplib + ssl
│   ├── embeddings.py                # SentenceTransformer: encode, deduplicate, classify
│   ├── security.py                  # HMAC token generation + verification
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app: lifespan, route registration
│   │   ├── schemas.py               # Pydantic models for API request/response
│   │   └── routes_rate.py           # GET /rate, GET /health
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── scorer.py                # LLM: relevance score + summary in one call
│   │   ├── profiler.py              # LLM: profile generation + incremental refinement
│   │   └── explorer.py              # ExplorationStrategy ABC + 3 implementations
│   ├── ingestors/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseIngestor ABC + RawItem dataclass
│   │   ├── arxiv.py
│   │   ├── semantic_scholar.py
│   │   ├── huggingface.py
│   │   ├── rss.py                   # feedparser, feed list from config
│   │   └── reddit.py                # PRAW
│   └── db/
│       ├── __init__.py
│       ├── schema.sql               # Authoritative DDL
│       ├── migrations.py            # Run-once migration runner
│       └── repository.py            # All DB read/write functions
├── tests/
│   ├── conftest.py                  # Fixtures: tmp_db, mock_llm, mock_ingestors, fixed_embeddings
│   ├── test_ingestors.py
│   ├── test_embeddings.py
│   ├── test_scorer.py
│   ├── test_profiler.py
│   ├── test_explorer.py
│   ├── test_digest_orchestrator.py
│   ├── test_repository.py
│   ├── test_security.py
│   └── test_api_routes.py
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Lint + test on push/PR
│       └── weekly_digest.yml        # Cron: Friday 09:00 UTC
├── config.yaml                      # Non-secret runtime config (committed)
├── .env.example                     # Secrets template — never commit .env
├── railway.toml                     # Railway service config
├── Procfile                         # Railway process: uvicorn backend.api.main:app
├── pyproject.toml
├── Makefile
└── README.md
```

---

## 4. Dependency Inventory

All licenses are permissive. No GPL. No proprietary.

| Package | License | Purpose |
|---|---|---|
| `fastapi>=0.111.0` | MIT | Minimal rating API |
| `uvicorn[standard]>=0.29.0` | BSD-3 | ASGI server (Railway) |
| `typer>=0.12.0` | MIT | CLI framework |
| `pydantic>=2.7.0` | MIT | Data validation, API schemas |
| `python-dotenv>=1.0.0` | BSD-3 | `.env` loading |
| `pyyaml>=6.0.1` | MIT | `config.yaml` parsing |
| `sentence-transformers>=2.7.0` | Apache 2.0 | Embeddings (always-on, free) |
| `numpy>=1.26.0` | BSD-3 | Vector math |
| `torch` (transitive) | BSD-3 | Pulled in by sentence-transformers (~2GB) |
| `anthropic>=0.25.0` | MIT | Claude LLM adapter |
| `google-generativeai>=0.5.0` | Apache 2.0 | Gemini LLM adapter |
| `ollama>=0.1.0` | MIT | Ollama local LLM adapter |
| `arxiv>=2.1.0` | MIT | arXiv ingestor |
| `feedparser>=6.0.11` | BSD-2 | RSS + HuggingFace ingestor |
| `praw>=7.7.1` | BSD-2 | Reddit ingestor |
| `aiohttp>=3.9.0` | Apache 2.0 | Semantic Scholar HTTP |
| `pytest>=8.0.0` | MIT | Test runner |
| `pytest-asyncio>=0.23.0` | Apache 2.0 | Async test support |
| `pytest-cov>=5.0.0` | MIT | Coverage |
| `respx>=0.21.0` | BSD-3 | Mock HTTP in tests |
| `ruff>=0.4.0` | MIT | Lint + format |

`smtplib`, `ssl`, `sqlite3`, `hmac`, `hashlib` — Python stdlib (PSF license).

---

## 5. Database Schema

### `schema.sql` (authoritative DDL)

```sql
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
```

### Key design decisions

- **Ratings on items, not digest slots.** Profile generation queries
  `ratings JOIN items` for `(title, summary, source, url, rating)` — no
  three-way join needed.
- **Summaries on `items`, not `digest_items`.** A summary describes the paper,
  not the slot. Re-surfacing an old paper reuses its existing summary.
- **`profiles` is append-only.** Never UPDATE or DELETE. Full history of how
  preferences evolved; `paper-scout profile --history` reads this table.
- **RSS source stores publication name** (`"The Gradient"`, `"Distill"`), not
  the generic string `"rss"`. Set in `config.yaml` feed list, passed through
  the ingestor to `RawItem.source`.
- **No `user_id` anywhere.** Solo tool. One migration adds it if ever needed.

---

## 6. Module Specifications

### 6.1 `backend/security.py`

```python
def generate_rating_token(item_id: int, rating: str, secret: str) -> str:
    """HMAC-SHA256 over f"{item_id}:{rating}" using secret. Return first 32 hex chars."""

def verify_rating_token(item_id: int, rating: str, token: str, secret: str) -> bool:
    """Recompute expected token. Compare with hmac.compare_digest (timing-safe).
    Return True if match, False otherwise. Never raise."""
```

`secret` is read from `PAPER_SCOUT_HMAC_SECRET` env var. This value must be
set identically in Railway environment variables and GitHub Actions secrets —
links are generated during the Actions cron run and verified by the Railway
service.

Rating link format in email:
```
{RAILWAY_API_URL}/rate?item_id={id}&rating=up&token={token}
{RAILWAY_API_URL}/rate?item_id={id}&rating=down&token={token}
```

---

### 6.2 `backend/api/routes_rate.py`

Two routes. No business logic beyond token verification and DB write.

```
GET /rate
    Query params: item_id: int, rating: str, token: str
    Steps:
      1. verify_rating_token() → return 400 HTML on failure
      2. repository.upsert_rating(item_id, rating)
      3. count = repository.count_ratings_since_last_profile()
         if count >= settings.profile.incremental_every_n:
             rated = repository.get_rated_items_with_summaries()
             current = repository.get_latest_profile()
             prose = await profiler.update_profile(rated, current, provider)
             repository.insert_profile(prose, mode, len(rated), provider_name)
      4. Return minimal HTML confirmation:
         "<html><body><p>✓ Rated '<title>' as 👍/👎. Thanks.</p></body></html>"

GET /health
    Returns: {"ok": true}
```

HTML response (not JSON) so clicking the link in any email client renders a
readable confirmation page. No CSS frameworks — plain `<body>` tag only.

---

### 6.3 `backend/llm/`

```python
# backend/llm/base.py
class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Return assistant text. Raise LLMError on failure."""

# backend/llm/factory.py
def get_provider(settings: Settings) -> BaseLLMProvider:
    """Read settings.llm_provider. Return adapter. Raise ValueError on unknown."""
```

Adapters: `claude_provider.py` (anthropic), `gemini_provider.py`
(google-generativeai), `ollama_provider.py` (ollama). Each maps
provider-specific exceptions to `LLMError(message: str)`. No other module
imports a provider SDK directly.

Model names from env vars: `ANTHROPIC_MODEL` (default `claude-sonnet-4-20250514`),
`GEMINI_MODEL` (default `gemini-1.5-flash`), `OLLAMA_MODEL` (default `llama3`).

---

### 6.4 `backend/embeddings.py`

```python
def load_model() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 once. Module-level cache. Called from cli.py
    and api/main.py at startup."""

def encode(texts: list[str]) -> np.ndarray:
    """Return (N, 384) float32. Input: title + ' ' + abstract[:200] per item."""

def deduplicate(
    candidates: list[RawItem],
    existing_vectors: np.ndarray,  # (M, 384) — from repository.get_all_embeddings()
    threshold: float,              # from settings.embeddings.dedup_threshold
) -> list[RawItem]:
    """Remove candidates with cosine similarity > threshold to any existing vector.
    Also deduplicate within the candidates batch (keep first occurrence)."""

def classify_topic(text: str, taxonomy: list[str]) -> str:
    """Return closest taxonomy label by cosine similarity.
    Cache taxonomy label embeddings on first call."""
```

---

### 6.5 `backend/ingestors/base.py`

```python
@dataclass
class RawItem:
    source: str          # Publication name: "arXiv", "The Gradient", "Reddit", etc.
    title: str
    url: str
    abstract: str        # Truncated to 500 chars; empty string if unavailable
    authors: list[str]
    published_date: date
    raw_id: str          # Source-native ID for dedup

class BaseIngestor(ABC):
    @abstractmethod
    async def fetch(self, since: date) -> list[RawItem]:
        """Fetch items published/posted since `since`.
        Never raise — catch all exceptions, log WARNING, return partial list."""
```

| Ingestor | Method | Auth |
|---|---|---|
| `arxiv.py` | `arxiv` PyPI; categories from config | None |
| `semantic_scholar.py` | `aiohttp` REST; keywords from config | `S2_API_KEY` env (optional) |
| `huggingface.py` | `feedparser` on HuggingFace Papers RSS | None |
| `rss.py` | `feedparser`; feed list from config; `source` = feed name from config | None |
| `reddit.py` | `praw`; subreddits from config | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` |

All ingestors run concurrently in `digest.py` via
`asyncio.gather(..., return_exceptions=True)`. One ingestor failure never
aborts the digest.

---

### 6.6 `backend/agents/scorer.py`

One LLM call per item returns both relevance score and summary.
Wildcard items receive a summary-only call (score not needed).

```python
@dataclass
class ScoreResult:
    score: float    # 0.0–1.0; -1.0 for wildcard (summary-only call)
    summary: str    # 2–3 sentences, plain English, written for a technical reader

async def score_and_summarise(
    item: RawItem,
    profile: str,
    provider: BaseLLMProvider,
    wildcard: bool = False,
) -> ScoreResult:
    """
    If wildcard=False (exploit):
      Prompt instructs LLM to return only JSON:
      {"score": 0.0-1.0, "summary": "2-3 sentence plain English summary"}
      Parse JSON. On parse failure or LLMError: log WARNING, return
      ScoreResult(score=0.0, summary="").

    If wildcard=True:
      Prompt requests summary only (no score). Return ScoreResult(score=-1.0,
      summary=...). On failure: return ScoreResult(score=-1.0, summary="").

    Never raise.
    """
```

**Concurrency in `digest.py`:** `asyncio.gather` across all candidates with
`asyncio.Semaphore(5)` to avoid provider rate limits.

**Embedding pre-filter (applied before scoring in `digest.py`):**
Compute centroid of all thumbs-up item embeddings. Discard exploit candidates
with cosine similarity to centroid below `settings.scoring.prefilter_threshold`
(default 0.2). Skip pre-filter entirely if fewer than 3 thumbs-up ratings
exist in DB.

---

### 6.7 `backend/agents/profiler.py`

Two-mode profile update strategy. Input always uses stored summaries —
never raw abstracts — keeping token cost flat regardless of history size.

```python
async def update_profile(
    rated_items: list[tuple[dict, str]],  # (item_row, "up"|"down") — item_row includes summary
    current_profile: str | None,
    provider: BaseLLMProvider,
    mode: Literal["incremental", "full_rebuild"],
) -> str:
    """
    incremental mode:
      Input: current_profile prose + N most recent rated items (title + summary + rating).
      Prompt: "Here is my current interest profile. Here are N newly rated papers.
               Refine the profile to reflect these new ratings."
      Used when: ratings since last profile >= profile.incremental_every_n (default 5).

    full_rebuild mode:
      Input: up to 30 most recent thumbs-up items + 20 most recent thumbs-down items
             (title + summary only — no raw abstracts).
      Prompt: "Based on these rated papers, write a 2–4 paragraph research interest
               profile. Emphasise what I find valuable in thumbs-up papers and what
               I find uninteresting in thumbs-down papers."
      Used when: total ratings >= profile.full_rebuild_every_n (default 50)
                 AND total ratings % full_rebuild_every_n == 0.

    Returns prose string. Raises LLMError on provider failure (caller handles).
    """
```

**Trigger logic lives in two places:**
- `backend/api/routes_rate.py` — checks after every rating submission
  (incremental trigger only; full rebuild can also trigger here).
- `backend/cli.py` `profile` command — manual trigger for either mode.

---

### 6.8 `backend/agents/explorer.py`

```python
class ExplorationStrategy(ABC):
    @abstractmethod
    def select_wildcards(
        self,
        candidates: list[RawItem],
        topic_exposure: dict[str, int],  # topic_bucket → seen_count over lookback window
        n: int = 2,
    ) -> list[RawItem]:
        """Return exactly n wildcard items. Never raise."""

class TopicDiversityExplorer(ExplorationStrategy):
    """DEFAULT. Rank buckets by ascending seen_count. Select one candidate
    from each of the n least-seen buckets. Tiebreak: random seeded on
    digest week_start date for reproducibility."""

class AdjacentScoreExplorer(ExplorationStrategy):
    """Requires pre-scored candidates. Select n items with relevance_score
    in [0.3, 0.6]. Fill remaining slots from TopicDiversityExplorer if
    fewer than n items fall in range."""

class ThompsonExplorer(ExplorationStrategy):
    """Sample Beta(up_count+1, down_count+1) per topic bucket. Select
    candidates from the n highest-sampled buckets. Falls back to
    TopicDiversityExplorer if any bucket has fewer than 3 total ratings."""

def get_explorer(strategy_name: str) -> ExplorationStrategy:
    """Factory. Maps config name to class. Raise ValueError on unknown."""
```

---

### 6.9 `backend/digest.py`

```python
async def run_weekly_digest() -> int:
    """
    Full weekly pipeline. Returns digest_id. Raises DigestError on
    unrecoverable failure (all ingestors failed, DB unavailable).

    Steps (in order):
    1.  week_start = date of the coming/current Friday.
    2.  Abort (log + return) if digest for this week_start already in DB.
    3.  Run all enabled ingestors concurrently via asyncio.gather(return_exceptions=True).
        Log ingestor exceptions at WARNING. Raise DigestError if zero items total.
    4.  Load existing embeddings from DB (ids + vectors).
    5.  Encode all candidates.
    6.  Deduplicate candidates against existing DB vectors + within batch.
    7.  Classify topic bucket for each candidate.
    8.  Load current interest profile from DB (may be None on first run).
    9.  If profile and >=3 thumbs-up items: apply embedding pre-filter to
        exploit candidates.
    10. Score + summarise exploit candidates concurrently (Semaphore=5).
        Assign ScoreResult(score=0.0, summary="") to any item that fails.
    11. Sort by score descending. Select top exploit_count (3) as exploit picks.
    12. Remove exploit picks from candidate pool.
    13. Instantiate explorer from config. Select wildcard_count (2) wildcards.
    14. Summarise wildcard items (summary-only LLM calls, Semaphore=5).
    15. Combine: exploit picks (positions 0–2) + wildcard picks (positions 3–4).
    16. Write digest + all 5 items (with embeddings + summaries) to DB in a
        single transaction.
    17. Update topic_exposure counts.
    18. Send email via email_sender.send_digest_email().
    19. Return digest_id.
    """

def run_cli() -> None:
    """Sync entry point for CLI and GitHub Actions. Calls asyncio.run(run_weekly_digest())."""
```

---

### 6.10 `backend/db/repository.py`

Module-level `sqlite3.Connection` initialised once from `settings.db_path`.
All multi-statement writes use explicit transactions. Embeddings stored as
`BLOB` via `ndarray.tobytes()`, rehydrated with
`np.frombuffer(blob, dtype=np.float32).reshape(384)`.

Required functions:

```python
# Items
def insert_items(items: list[dict]) -> list[int]          # returns inserted ids
def get_all_embeddings() -> tuple[list[int], np.ndarray]  # (ids, (N,384))
def get_rated_embeddings(rating: str) -> np.ndarray       # (N,384) for "up" or "down"

# Digests
def insert_digest(week_start: date, items: list[dict]) -> int   # single transaction
def get_latest_digest() -> dict | None
def get_digest_by_id(digest_id: int) -> dict | None
def get_digest_history(limit: int = 10) -> list[dict]

# Ratings
def upsert_rating(item_id: int, rating: str) -> None      # INSERT OR REPLACE
def get_all_ratings() -> list[tuple[dict, str]]           # (item_dict, rating)
def get_rated_items_with_summaries() -> list[tuple[dict, str]]  # (item_dict, rating)
def count_ratings_since_last_profile() -> int

# Profiles
def insert_profile(prose: str, mode: str, rated_count: int, provider: str) -> int
def get_latest_profile() -> dict | None
def get_profile_history(limit: int = 10) -> list[dict]

# Topic exposure
def get_topic_exposure(lookback_weeks: int) -> dict[str, int]
def update_topic_exposure(buckets_seen: list[str], ratings: dict[str, str]) -> None
```

---

### 6.11 `backend/email_sender.py`

```python
def send_digest_email(digest_id: int, items: list[dict], recipient: str) -> None:
    """
    Build plain-text + HTML email. For each of the 5 items include:
      - Position number and label ("Exploit Pick 1" or "Wildcard 1")
      - Title
      - Source (e.g. "arXiv", "The Gradient", "Reddit")
      - Topic bucket
      - Published date
      - 2–3 sentence summary (from items[n]["summary"])
      - URL
      - Thumbs-up link: {RAILWAY_API_URL}/rate?item_id={id}&rating=up&token={token}
      - Thumbs-down link: {RAILWAY_API_URL}/rate?item_id={id}&rating=down&token={token}

    Tokens generated via security.generate_rating_token().
    Send via smtplib.SMTP_SSL("smtp.gmail.com", 465).
    Authenticate with GMAIL_ADDRESS + GMAIL_APP_PASSWORD from env.
    Raise EmailError on SMTP failure.
    """
```

---

### 6.12 `backend/cli.py`

Built with `typer`. Three subcommands:

```
paper-scout digest
    Runs run_weekly_digest(). Prints digest_id and item titles on success.
    Accepts --dry-run flag: runs full pipeline but skips DB write and email.

paper-scout history [--limit 10]
    Prints past N digests: week_start, item titles, ratings if submitted.
    Reads from repository.get_digest_history().

paper-scout profile [--history] [--rebuild]
    Default: prints current profile prose + metadata.
    --history: prints last 5 profiles with timestamps and mode.
    --rebuild: triggers full_rebuild profile regeneration immediately.
```

---

## 7. Configuration

### `config.yaml` (non-secret — commit this file)

```yaml
digest:
  exploit_count: 3
  wildcard_count: 2
  exploration_strategy: "topic_diversity"   # "topic_diversity"|"adjacent_score"|"thompson"

scoring:
  prefilter_threshold: 0.2        # Min cosine sim to thumbs-up centroid before LLM scoring
  llm_concurrency: 5              # Max concurrent LLM calls

profile:
  incremental_every_n: 5          # Trigger incremental update after this many new ratings
  full_rebuild_every_n: 50        # Trigger full rebuild at multiples of this total rating count
  full_rebuild_up_limit: 30       # Max thumbs-up items used in full rebuild
  full_rebuild_down_limit: 20     # Max thumbs-down items used in full rebuild

embeddings:
  model: "all-MiniLM-L6-v2"
  dedup_threshold: 0.92

topic_exposure:
  lookback_weeks: 4

topic_taxonomy:
  - "reinforcement_learning"
  - "large_language_models"
  - "computer_vision"
  - "interpretability"
  - "efficient_training"
  - "multimodal"
  - "agents_and_planning"
  - "theory_and_optimization"
  - "robotics"
  - "applications"
  - "other"

ingestors:
  arxiv:
    enabled: true
    categories: ["cs.LG", "cs.AI", "cs.CL", "stat.ML"]
    max_results: 50
  semantic_scholar:
    enabled: true
    keywords: ["machine learning", "deep learning", "reinforcement learning", "LLM agents"]
    max_results: 30
  huggingface:
    enabled: true
  rss:
    enabled: true
    feeds:
      - url: "https://thegradient.pub/rss/"
        name: "The Gradient"
      - url: "https://distill.pub/rss.xml"
        name: "Distill"
  reddit:
    enabled: true
    subreddits: ["MachineLearning", "artificial"]
    post_limit: 25

email:
  subject_template: "paper-scout: your {date} reading list"
```

### `.env.example` (never commit `.env`)

```bash
# LLM Provider — set exactly one
LLM_PROVIDER=claude                          # "claude" | "gemini" | "ollama"
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514
GEMINI_API_KEY=
GEMINI_MODEL=gemini-1.5-flash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Gmail (App Password — not your real Gmail password)
GMAIL_ADDRESS=you@gmail.com
GMAIL_APP_PASSWORD=                          # 16-char App Password from Google Account settings
PAPER_SCOUT_EMAIL_TO=you@gmail.com

# Security
PAPER_SCOUT_HMAC_SECRET=                     # Any long random string — must match in Railway + Actions

# Railway API URL (used to build rating links in email)
RAILWAY_API_URL=https://your-app.railway.app

# Database
PAPER_SCOUT_DB_PATH=paper_scout.db

# Optional: Semantic Scholar (raises rate limit without)
S2_API_KEY=

# Reddit
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=paper-scout/0.1
```

---

## 8. GitHub Actions Workflows

### `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check backend/ tests/
      - run: pytest tests/ --cov=backend --cov-report=term-missing --cov-fail-under=80
    env:
      LLM_PROVIDER: claude
      ANTHROPIC_API_KEY: test-key-not-used    # LLM mocked in all tests
      PAPER_SCOUT_DB_PATH: ":memory:"
      PAPER_SCOUT_HMAC_SECRET: test-secret
```

### `.github/workflows/weekly_digest.yml`

```yaml
name: Weekly Digest
on:
  schedule:
    - cron: "0 9 * * 5"     # Friday 09:00 UTC
  workflow_dispatch:          # Manual trigger via GitHub UI for testing
jobs:
  digest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .
      - name: Download database from Railway volume
        run: |
          # Railway exposes the volume at /data — copy db to workspace before run
          # This step is a placeholder: actual sync method depends on Railway CLI setup
          # See README for Railway volume sync instructions
          echo "Ensure paper_scout.db is available at $PAPER_SCOUT_DB_PATH"
      - run: paper-scout digest
    env:
      LLM_PROVIDER: ${{ secrets.LLM_PROVIDER }}
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      ANTHROPIC_MODEL: ${{ secrets.ANTHROPIC_MODEL }}
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      GMAIL_ADDRESS: ${{ secrets.GMAIL_ADDRESS }}
      GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}
      PAPER_SCOUT_EMAIL_TO: ${{ secrets.PAPER_SCOUT_EMAIL_TO }}
      PAPER_SCOUT_HMAC_SECRET: ${{ secrets.PAPER_SCOUT_HMAC_SECRET }}
      RAILWAY_API_URL: ${{ secrets.RAILWAY_API_URL }}
      PAPER_SCOUT_DB_PATH: paper_scout.db
      REDDIT_CLIENT_ID: ${{ secrets.REDDIT_CLIENT_ID }}
      REDDIT_CLIENT_SECRET: ${{ secrets.REDDIT_CLIENT_SECRET }}
      REDDIT_USER_AGENT: paper-scout/0.1
      S2_API_KEY: ${{ secrets.S2_API_KEY }}
```

**Note on DB + GitHub Actions:** The SQLite database lives on a Railway
persistent volume (mounted at `/data/paper_scout.db` in the Railway service).
The GitHub Actions digest job needs read/write access to this same database.
Two approaches — choose one and document in README:

1. **Railway CLI sync:** Use `railway run` to execute the digest command
   directly inside the Railway environment, bypassing the volume sync problem
   entirely. Simplest approach.
2. **Commit DB to repo:** Store `paper_scout.db` in the private repo, commit
   it back after each digest run (same pattern as the original design). Works
   but grows repo size over time.

The spec does not mandate which — document the chosen approach in README
before first deploy.

---

## 9. Railway Deployment

### `railway.toml`

```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 10
restartPolicyType = "on_failure"
```

### `Procfile` (fallback)

```
web: uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

Railway environment variables to set (mirrors `.env.example` secrets):
`LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `GMAIL_ADDRESS`,
`GMAIL_APP_PASSWORD`, `PAPER_SCOUT_EMAIL_TO`, `PAPER_SCOUT_HMAC_SECRET`,
`PAPER_SCOUT_DB_PATH` (set to `/data/paper_scout.db` — Railway volume mount),
`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `S2_API_KEY`.

---

## 10. Build System (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "paper-scout"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "typer>=0.12.0",
    "pydantic>=2.7.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    "sentence-transformers>=2.7.0",
    "numpy>=1.26.0",
    "anthropic>=0.25.0",
    "google-generativeai>=0.5.0",
    "ollama>=0.1.0",
    "arxiv>=2.1.0",
    "feedparser>=6.0.11",
    "praw>=7.7.1",
    "aiohttp>=3.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "respx>=0.21.0",
    "ruff>=0.4.0",
]

[project.scripts]
paper-scout = "backend.cli:app"
```

---

## 11. Makefile

```makefile
.PHONY: install dev-install migrate run-api run-digest test lint format

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

migrate:
	python -c "from backend.db.migrations import run; run()"

run-api:
	uvicorn backend.api.main:app --reload --port 8000

run-digest:
	paper-scout digest

run-digest-dry:
	paper-scout digest --dry-run

test:
	pytest tests/ --cov=backend --cov-report=term-missing

lint:
	ruff check backend/ tests/

format:
	ruff format backend/ tests/
```

---

## 12. Testing

**Coverage target: 80%.** Excluded: `email_sender.py` (SMTP mocked),
ingestor network calls (mocked via `respx`).

### `conftest.py` fixtures
- `tmp_db` — in-memory SQLite, runs migrations, patches repository connection
- `mock_llm` — `BaseLLMProvider` returning canned `{"score": 0.8, "summary": "Test summary."}` for scorer and prose string for profiler
- `mock_ingestors` — returns fixed list of 10 `RawItem` with varied topic buckets
- `fixed_embeddings` — monkeypatches `embeddings.encode` to return `np.random.default_rng(42).random((N, 384)).astype(np.float32)`

### Test coverage by module
- `test_security.py` — token round-trip, tampered token rejected, timing-safe compare
- `test_embeddings.py` — encode shape, dedup removes above threshold, dedup keeps below, classify returns taxonomy value
- `test_ingestors.py` — each ingestor returns `RawItem` list; HTTP failure returns `[]` without raising; RSS stores feed name as source
- `test_scorer.py` — returns `ScoreResult` with float in `[0,1]` and non-empty summary; returns `ScoreResult(0.0, "")` on LLM failure without raising; wildcard=True returns score=-1.0
- `test_profiler.py` — incremental prompt includes existing profile; full_rebuild prompt does not exceed item limits; raises `LLMError` on provider failure
- `test_explorer.py` — all three strategies return exactly n items; ThompsonExplorer falls back to diversity; AdjacentScoreExplorer fills from diversity when range empty; `get_explorer` raises on unknown name
- `test_digest_orchestrator.py` — full pipeline produces 5 items (positions 0–2 exploit, 3–4 wildcard); aborts on duplicate week_start; continues when one ingestor fails
- `test_repository.py` — insert/retrieve round-trips for all tables; upsert_rating replaces on duplicate item_id; count_ratings_since_last_profile returns correct delta
- `test_api_routes.py` — valid token + known item_id → 200 HTML; invalid token → 400; unknown item_id → 400; `/health` → `{"ok": true}`

---

## 13. Implementation Order

Build in dependency order. Do not skip ahead.

1. `backend/db/schema.sql` + `migrations.py` + `repository.py` — data layer first
2. `backend/config.py` — `Settings` dataclass from `config.yaml` + `.env`
3. `backend/security.py` — HMAC token, test immediately
4. `backend/embeddings.py` — encode, dedupe, classify; test with synthetic data
5. `backend/llm/` — base class, all three adapters, factory; test with `mock_llm`
6. `backend/ingestors/` — `arxiv.py` first, then `rss.py`, `huggingface.py`, `semantic_scholar.py`, `reddit.py`
7. `backend/agents/scorer.py` + `profiler.py` — against `mock_llm`
8. `backend/agents/explorer.py` — all three strategies + factory
9. `backend/digest.py` — full orchestrator; integration test with all mocks
10. `backend/email_sender.py` — SMTP; mock in tests
11. `backend/api/` — routes, schemas, lifespan; test all routes
12. `backend/cli.py` — typer commands wiring to digest/repository/profiler
13. CI workflow — get `ci.yml` green
14. Railway deploy — `railway.toml`, verify `/health`, verify rating link end-to-end
15. `weekly_digest.yml` — validate via `workflow_dispatch`; confirm DB strategy; enable cron

---

## 14. Definition of Done

- [ ] `make migrate` completes on a fresh DB path
- [ ] `make lint` passes with zero errors
- [ ] `make test` passes with ≥80% coverage, zero failures
- [ ] `make run-api` starts; `GET /health` returns `{"ok": true}`
- [ ] `make run-digest-dry` completes full pipeline without writing to DB or sending email
- [ ] `make run-digest` completes; digest row visible via `paper-scout history`
- [ ] `paper-scout profile` returns non-empty prose after 5 ratings
- [ ] `paper-scout profile --rebuild` triggers full rebuild and updates profile
- [ ] Rating link in email (or constructed manually) records rating and returns HTML confirmation
- [ ] Invalid/tampered rating token returns 400
- [ ] Email arrives in Gmail inbox with 5 items, summaries, and working rating links
- [ ] Switching `LLM_PROVIDER=gemini` in `.env` runs digest end-to-end without error
- [ ] All three exploration strategies selectable via `config.yaml` without code changes
- [ ] Railway service live; `/health` reachable at `RAILWAY_API_URL`
- [ ] `workflow_dispatch` trigger in GitHub UI completes successfully
- [ ] `README.md` covers: prerequisites, Gmail App Password setup, Reddit app registration, Railway deploy steps, `.env` configuration, `make` targets, first-run instructions
