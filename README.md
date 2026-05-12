# paper-scout

A weekly AI paper discovery agent. Ingests papers from arXiv, Papers With Code,
Semantic Scholar, HuggingFace Papers, RSS feeds, and Reddit. Selects 5 items per
week (3 exploit + 2 wildcard), delivers them via Gmail with one-click rating
links, and refines your interest profile from feedback.

## Prerequisites

- Docker + Docker Compose
- A Gmail account with an App Password (see below)
- One LLM API key: Anthropic, Google Gemini, or a local Ollama instance
- Reddit app credentials (optional — see below)

## Quick start

```bash
# 1. Clone and configure secrets
git clone <repo>
cd paper-scout
cp .env.example .env
# Edit .env — fill in LLM key, Gmail credentials, HMAC secret

# 2. Build the image (downloads deps + embedding model — takes a few minutes once)
docker compose build

# 3. Dry-run digest — fetches papers, scores them, logs top 5 (no email, no DB write)
docker compose run --rm app paper-scout digest --dry-run

# 4. When ready, run for real
docker compose run --rm app paper-scout digest

# 5. Start the API (for receiving rating clicks from email)
docker compose up
```

## Gmail App Password setup

1. Go to your Google Account → Security → 2-Step Verification (must be on)
2. Search for "App passwords" → create one for "Mail"
3. Copy the 16-character password into `GMAIL_APP_PASSWORD` in `.env`

## Configuring sources

Each ingestor can be independently enabled or disabled in `config.yaml`:

```yaml
ingestors:
  arxiv:
    enabled: true
  papers_with_code:
    enabled: true      # no credentials required
  semantic_scholar:
    enabled: true      # optional S2_API_KEY for higher rate limits
  huggingface:
    enabled: true
  rss:
    enabled: true
  reddit:
    enabled: false     # requires app credentials; set true once configured
```

Sources that are disabled are skipped entirely. Sources that fail (network error,
auth failure, etc.) log a warning and return an empty list — the digest continues
with whatever other sources returned.

## Reddit app registration (optional)

Reddit API access requires a registered app. Note that Reddit has been
progressively tightening API access; if registration is unavailable, set
`enabled: false` for the Reddit source in `config.yaml` — paper-scout works fine
without it.

1. Go to https://www.reddit.com/prefs/apps → "create another app"
2. Choose **script** type; read and accept the Responsible Builder Policy
3. Copy `client_id` (short string under app name) and `client_secret` into `.env`

## `.env` configuration

See `.env.example` for all required and optional variables. Required:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `claude`, `gemini`, or `ollama` |
| `ANTHROPIC_API_KEY` | Required if `LLM_PROVIDER=claude` |
| `GMAIL_ADDRESS` | Sender email |
| `GMAIL_APP_PASSWORD` | 16-char App Password |
| `PAPER_SCOUT_EMAIL_TO` | Recipient email |
| `PAPER_SCOUT_HMAC_SECRET` | Random string — must match in Railway |
| `RAILWAY_API_URL` | Your Railway service URL |

## Docker commands

| Command | Description |
|---|---|
| `docker compose build` | Build the image |
| `docker compose up` | Start the API on port 8000 |
| `docker compose run --rm app paper-scout digest --dry-run` | Dry-run digest |
| `docker compose run --rm app paper-scout digest` | Run weekly digest |
| `docker compose run --rm app paper-scout history` | Show past digests |
| `docker compose run --rm app paper-scout profile` | Show interest profile |

The DB is stored in a named Docker volume (`paper_scout_data`) and persists across runs.

## Local dev (without Docker)

```bash
pip install -e ".[dev]"
make test      # run tests
make lint      # ruff check
make format    # ruff format
```

## Railway deployment

1. Create a Railway project → add a service from this repo
2. Add a Persistent Volume mounted at `/data`
3. Set `PAPER_SCOUT_DB_PATH=/data/paper_scout.db` in Railway environment variables
4. Set all other secrets from `.env.example` in Railway environment variables
5. Deploy — `/health` should return `{"ok": true}`

### DB sync strategy (chosen: Railway CLI)

The recommended approach is to run the weekly digest **inside the Railway
environment** using `railway run paper-scout digest`. This runs the job in the
same container as the Railway service, giving it direct read/write access to the
persistent volume without any sync step.

To use this approach in `weekly_digest.yml`:
1. Install the Railway CLI in GitHub Actions
2. Authenticate with `railway login --browserless` using `RAILWAY_TOKEN`
3. Replace `paper-scout digest` with `railway run paper-scout digest`

See: https://docs.railway.app/guides/cli

## CLI reference

```bash
paper-scout digest [--dry-run]      # Run weekly pipeline
paper-scout history [--limit N]     # Show past digests
paper-scout profile                 # Show current interest profile
paper-scout profile --history       # Show last 5 profiles
paper-scout profile --rebuild       # Trigger full profile rebuild
```

## First run

On first run there is no interest profile, so all papers are scored without a
profile baseline. After you rate 5 papers from the first digest email, an
incremental profile is generated automatically. After 50 total ratings a full
rebuild runs, producing a richer profile.
