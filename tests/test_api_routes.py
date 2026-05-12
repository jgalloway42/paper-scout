"""Tests for backend/api/routes_rate.py."""

import pytest
from fastapi.testclient import TestClient

import backend.db.repository as repo
from backend.api.main import app
from backend.security import generate_rating_token

SECRET = "test-api-secret"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    """Override settings so tests use the tmp_db secret and skip email/profile."""
    from backend.config import Settings, DigestConfig, ScoringConfig, ProfileConfig
    from backend.config import EmbeddingsConfig, TopicExposureConfig, IngestorsConfig, EmailConfig
    from backend.config import ArxivConfig, SemanticScholarConfig, HuggingFaceConfig, RssConfig, RedditConfig

    mock_settings = Settings(
        digest=DigestConfig(),
        scoring=ScoringConfig(),
        profile=ProfileConfig(incremental_every_n=999),  # prevent auto-trigger in tests
        embeddings=EmbeddingsConfig(),
        topic_exposure=TopicExposureConfig(),
        topic_taxonomy=["large_language_models", "other"],
        ingestors=IngestorsConfig(
            arxiv=ArxivConfig(),
            semantic_scholar=SemanticScholarConfig(),
            huggingface=HuggingFaceConfig(),
            rss=RssConfig(),
            reddit=RedditConfig(),
        ),
        email=EmailConfig(),
        hmac_secret=SECRET,
        llm_provider="claude",
    )
    monkeypatch.setattr("backend.api.routes_rate.get_settings", lambda: mock_settings)
    monkeypatch.setattr("backend.config.get_settings", lambda: mock_settings)


@pytest.fixture()
def client(tmp_db):
    """TestClient with tmp_db injected. No context manager = no lifespan."""
    yield TestClient(app, raise_server_exceptions=True)


def _insert_item(title: str = "Test Paper") -> int:
    ids = repo.insert_items([{
        "source": "arXiv",
        "title": title,
        "url": f"https://arxiv.org/abs/{hash(title) % 100000}",
        "abstract": "An abstract.",
        "authors": ["A"],
        "published_date": "2024-01-15",
        "topic_bucket": "large_language_models",
        "summary": "A short summary.",
        "raw_id": str(hash(title) % 100000),
    }])
    return ids[0]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ---------------------------------------------------------------------------
# /rate — valid token
# ---------------------------------------------------------------------------


def test_rate_valid_token_returns_200_html(client, tmp_db):
    item_id = _insert_item()
    token = generate_rating_token(item_id, "up", SECRET)
    resp = client.get(f"/rate?item_id={item_id}&rating=up&token={token}")
    assert resp.status_code == 200
    assert "html" in resp.headers["content-type"]
    assert "&#x2713;" in resp.text or "✓" in resp.text


def test_rate_records_rating_in_db(client, tmp_db):
    item_id = _insert_item("Paper B")
    token = generate_rating_token(item_id, "down", SECRET)
    client.get(f"/rate?item_id={item_id}&rating=down&token={token}")
    ratings = repo.get_all_ratings()
    assert len(ratings) == 1
    assert ratings[0][1] == "down"


# ---------------------------------------------------------------------------
# /rate — invalid token → 400
# ---------------------------------------------------------------------------


def test_rate_invalid_token_returns_400(client, tmp_db):
    item_id = _insert_item()
    resp = client.get(f"/rate?item_id={item_id}&rating=up&token=badtoken")
    assert resp.status_code == 400


def test_rate_tampered_item_id_returns_400(client, tmp_db):
    item_id = _insert_item()
    token = generate_rating_token(item_id, "up", SECRET)
    resp = client.get(f"/rate?item_id=9999&rating=up&token={token}")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /rate — unknown item_id → 400
# ---------------------------------------------------------------------------


def test_rate_unknown_item_id_returns_400(client, tmp_db):
    token = generate_rating_token(9999, "up", SECRET)
    resp = client.get(f"/rate?item_id=9999&rating=up&token={token}")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /rate — invalid rating value → 400
# ---------------------------------------------------------------------------


def test_rate_invalid_rating_value_returns_400(client, tmp_db):
    item_id = _insert_item("Rating Value Test")
    token = generate_rating_token(item_id, "meh", SECRET)
    resp = client.get(f"/rate?item_id={item_id}&rating=meh&token={token}")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /rate — profile update triggered when threshold met
# ---------------------------------------------------------------------------


def test_rate_triggers_incremental_profile_update(tmp_db, monkeypatch):
    """When count_ratings_since_last_profile >= incremental_every_n, profile is updated."""
    from backend.config import Settings, DigestConfig, ScoringConfig, ProfileConfig
    from backend.config import EmbeddingsConfig, TopicExposureConfig, IngestorsConfig, EmailConfig
    from backend.config import ArxivConfig, SemanticScholarConfig, HuggingFaceConfig, RssConfig, RedditConfig
    from backend.llm.base import BaseLLMProvider

    class _MockLLM(BaseLLMProvider):
        async def complete(self, system, user, max_tokens=1000):
            return "Updated profile prose."

    settings_with_low_threshold = Settings(
        digest=DigestConfig(),
        scoring=ScoringConfig(),
        profile=ProfileConfig(incremental_every_n=1),  # trigger after 1 rating
        embeddings=EmbeddingsConfig(),
        topic_exposure=TopicExposureConfig(),
        topic_taxonomy=["large_language_models", "other"],
        ingestors=IngestorsConfig(
            arxiv=ArxivConfig(),
            semantic_scholar=SemanticScholarConfig(),
            huggingface=HuggingFaceConfig(),
            rss=RssConfig(),
            reddit=RedditConfig(),
        ),
        email=EmailConfig(),
        hmac_secret=SECRET,
        llm_provider="claude",
    )
    monkeypatch.setattr("backend.api.routes_rate.get_settings", lambda: settings_with_low_threshold)
    monkeypatch.setattr("backend.api.routes_rate.get_provider", lambda s: _MockLLM())

    # Add a test item with a summary (needed for get_rated_items_with_summaries)
    item_id = _insert_item("Profile Trigger Paper")
    # Manually update the summary so it's included in profile input
    tmp_db.execute("UPDATE items SET summary='A good summary.' WHERE id=?", (item_id,))
    tmp_db.commit()

    from backend.api.main import app
    client = TestClient(app, raise_server_exceptions=True)
    token = generate_rating_token(item_id, "up", SECRET)
    resp = client.get(f"/rate?item_id={item_id}&rating=up&token={token}")
    assert resp.status_code == 200

    # Profile should have been inserted
    profile = repo.get_latest_profile()
    assert profile is not None
    assert profile["prose"] == "Updated profile prose."
