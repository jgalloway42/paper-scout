"""Loads config.yaml + .env → typed Settings dataclass."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(os.environ.get("PAPER_SCOUT_CONFIG") or Path(__file__).parent.parent / "config.yaml")


@dataclass
class DigestConfig:
    exploit_count: int = 3
    wildcard_count: int = 2
    exploration_strategy: str = "topic_diversity"


@dataclass
class ScoringConfig:
    prefilter_threshold: float = 0.2
    llm_concurrency: int = 5


@dataclass
class ProfileConfig:
    incremental_every_n: int = 5
    full_rebuild_every_n: int = 50
    full_rebuild_up_limit: int = 30
    full_rebuild_down_limit: int = 20


@dataclass
class EmbeddingsConfig:
    model: str = "all-MiniLM-L6-v2"
    dedup_threshold: float = 0.92


@dataclass
class TopicExposureConfig:
    lookback_weeks: int = 4


@dataclass
class ArxivConfig:
    enabled: bool = True
    categories: list[str] = field(default_factory=lambda: ["cs.LG", "cs.AI"])
    max_results: int = 50


@dataclass
class SemanticScholarConfig:
    enabled: bool = True
    keywords: list[str] = field(default_factory=list)
    max_results: int = 30


@dataclass
class HuggingFaceConfig:
    enabled: bool = True


@dataclass
class RssFeedConfig:
    url: str = ""
    name: str = ""


@dataclass
class RssConfig:
    enabled: bool = True
    feeds: list[RssFeedConfig] = field(default_factory=list)


@dataclass
class RedditConfig:
    enabled: bool = True
    subreddits: list[str] = field(default_factory=list)
    post_limit: int = 25


@dataclass
class PapersWithCodeConfig:
    enabled: bool = True
    max_results: int = 50


@dataclass
class IngestorsConfig:
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    semantic_scholar: SemanticScholarConfig = field(default_factory=SemanticScholarConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    rss: RssConfig = field(default_factory=RssConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    papers_with_code: PapersWithCodeConfig = field(default_factory=PapersWithCodeConfig)


@dataclass
class EmailConfig:
    subject_template: str = "paper-scout: your {date} reading list"


@dataclass
class Settings:
    digest: DigestConfig
    scoring: ScoringConfig
    profile: ProfileConfig
    embeddings: EmbeddingsConfig
    topic_exposure: TopicExposureConfig
    topic_taxonomy: list[str]
    ingestors: IngestorsConfig
    email: EmailConfig

    # Secrets / env-only fields
    llm_provider: str = "claude"
    db_path: str = "paper_scout.db"
    hmac_secret: str = ""
    railway_api_url: str = ""
    gmail_address: str = ""
    gmail_app_password: str = ""
    email_to: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    s2_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "paper-scout/0.1"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_settings(config_path: Path | None = None) -> Settings:
    """Parse config.yaml and overlay environment variables."""
    path = config_path or _CONFIG_PATH
    raw = _load_yaml(path)

    dig = raw.get("digest", {})
    sc = raw.get("scoring", {})
    pr = raw.get("profile", {})
    emb = raw.get("embeddings", {})
    te = raw.get("topic_exposure", {})
    ing = raw.get("ingestors", {})
    eml = raw.get("email", {})

    rss_raw = ing.get("rss", {})
    rss_feeds = [
        RssFeedConfig(url=f["url"], name=f["name"])
        for f in rss_raw.get("feeds", [])
    ]

    return Settings(
        digest=DigestConfig(
            exploit_count=dig.get("exploit_count", 3),
            wildcard_count=dig.get("wildcard_count", 2),
            exploration_strategy=dig.get("exploration_strategy", "topic_diversity"),
        ),
        scoring=ScoringConfig(
            prefilter_threshold=sc.get("prefilter_threshold", 0.2),
            llm_concurrency=sc.get("llm_concurrency", 5),
        ),
        profile=ProfileConfig(
            incremental_every_n=pr.get("incremental_every_n", 5),
            full_rebuild_every_n=pr.get("full_rebuild_every_n", 50),
            full_rebuild_up_limit=pr.get("full_rebuild_up_limit", 30),
            full_rebuild_down_limit=pr.get("full_rebuild_down_limit", 20),
        ),
        embeddings=EmbeddingsConfig(
            model=emb.get("model", "all-MiniLM-L6-v2"),
            dedup_threshold=emb.get("dedup_threshold", 0.92),
        ),
        topic_exposure=TopicExposureConfig(
            lookback_weeks=te.get("lookback_weeks", 4),
        ),
        topic_taxonomy=raw.get("topic_taxonomy", []),
        ingestors=IngestorsConfig(
            arxiv=ArxivConfig(
                enabled=ing.get("arxiv", {}).get("enabled", True),
                categories=ing.get("arxiv", {}).get("categories", []),
                max_results=ing.get("arxiv", {}).get("max_results", 50),
            ),
            semantic_scholar=SemanticScholarConfig(
                enabled=ing.get("semantic_scholar", {}).get("enabled", True),
                keywords=ing.get("semantic_scholar", {}).get("keywords", []),
                max_results=ing.get("semantic_scholar", {}).get("max_results", 30),
            ),
            huggingface=HuggingFaceConfig(
                enabled=ing.get("huggingface", {}).get("enabled", True),
            ),
            rss=RssConfig(
                enabled=rss_raw.get("enabled", True),
                feeds=rss_feeds,
            ),
            reddit=RedditConfig(
                enabled=ing.get("reddit", {}).get("enabled", True),
                subreddits=ing.get("reddit", {}).get("subreddits", []),
                post_limit=ing.get("reddit", {}).get("post_limit", 25),
            ),
            papers_with_code=PapersWithCodeConfig(
                enabled=ing.get("papers_with_code", {}).get("enabled", True),
                max_results=ing.get("papers_with_code", {}).get("max_results", 50),
            ),
        ),
        email=EmailConfig(
            subject_template=eml.get("subject_template", "paper-scout: your {date} reading list"),
        ),
        # Env vars
        llm_provider=os.environ.get("LLM_PROVIDER", "claude"),
        db_path=os.environ.get("PAPER_SCOUT_DB_PATH", "paper_scout.db"),
        hmac_secret=os.environ.get("PAPER_SCOUT_HMAC_SECRET", ""),
        railway_api_url=os.environ.get("RAILWAY_API_URL", ""),
        gmail_address=os.environ.get("GMAIL_ADDRESS", ""),
        gmail_app_password=os.environ.get("GMAIL_APP_PASSWORD", ""),
        email_to=os.environ.get("PAPER_SCOUT_EMAIL_TO", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3"),
        s2_api_key=os.environ.get("S2_API_KEY", ""),
        reddit_client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
        reddit_user_agent=os.environ.get("REDDIT_USER_AGENT", "paper-scout/0.1"),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached Settings instance (loaded once at first call)."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
