"""Integration tests for backend/digest.py."""

from datetime import date, timedelta

import numpy as np
import pytest

import backend.db.repository as repo
from backend.digest import DigestError, _next_friday, run_weekly_digest
from backend.ingestors.base import RawItem


def _raw_items(n: int = 10) -> list[RawItem]:
    items = []
    for i in range(n):
        items.append(
            RawItem(
                source="arXiv",
                title=f"Paper {i}",
                url=f"https://arxiv.org/abs/{i:04d}",
                abstract=f"Abstract {i}. " * 5,
                authors=[f"Author {i}"],
                published_date=date.today() - timedelta(days=i),
                raw_id=f"{i:04d}",
            )
        )
    return items


def _setup_mocks(monkeypatch, tmp_db, items=None):
    """Patch ingestors, LLM, embeddings, email for a full orchestrator run."""
    if items is None:
        items = _raw_items()

    # Patch ingestors
    async def _fake_fetch(self, since):
        return items

    monkeypatch.setattr("backend.ingestors.arxiv.ArxivIngestor.fetch", _fake_fetch)
    monkeypatch.setattr("backend.ingestors.semantic_scholar.SemanticScholarIngestor.fetch", _fake_fetch)
    monkeypatch.setattr("backend.ingestors.huggingface.HuggingFaceIngestor.fetch", _fake_fetch)
    monkeypatch.setattr("backend.ingestors.rss.RssIngestor.fetch", _fake_fetch)
    monkeypatch.setattr("backend.ingestors.reddit.RedditIngestor.fetch", _fake_fetch)

    # Patch embeddings
    def _fake_encode_items(its):
        rng = np.random.default_rng(42)
        return rng.random((len(its), 384)).astype(np.float32)

    def _fake_classify(text, taxonomy):
        return taxonomy[hash(text) % len(taxonomy)]

    monkeypatch.setattr("backend.digest.encode_items", _fake_encode_items)
    monkeypatch.setattr("backend.digest.deduplicate", lambda candidates, vecs, thresh: candidates)
    monkeypatch.setattr("backend.digest.classify_topic", _fake_classify)

    # Patch LLM provider
    from backend.llm.base import BaseLLMProvider

    class _MockLLM(BaseLLMProvider):
        async def complete(self, system, user, max_tokens=1000):
            return '{"score": 0.8, "summary": "Test summary sentence."}'

    monkeypatch.setattr("backend.digest.get_provider", lambda settings: _MockLLM())

    # Patch email
    monkeypatch.setattr("backend.digest.send_digest_email", lambda *a, **kw: None)

    return items


@pytest.mark.asyncio
async def test_full_pipeline_produces_5_items(tmp_db, monkeypatch):
    _setup_mocks(monkeypatch, tmp_db)
    digest_id = await run_weekly_digest()
    assert digest_id > 0
    history = repo.get_digest_history(limit=1)
    assert len(history) == 1
    assert len(history[0]["items"]) == 5


@pytest.mark.asyncio
async def test_exploit_positions_0_2_wildcard_positions_3_4(tmp_db, monkeypatch):
    _setup_mocks(monkeypatch, tmp_db)
    await run_weekly_digest()
    history = repo.get_digest_history(limit=1)
    items = history[0]["items"]
    for item in items:
        if item["position"] in (0, 1, 2):
            assert item["is_wildcard"] == 0
        else:
            assert item["is_wildcard"] == 1


@pytest.mark.asyncio
async def test_aborts_on_duplicate_week_start(tmp_db, monkeypatch):
    _setup_mocks(monkeypatch, tmp_db)
    id1 = await run_weekly_digest()
    id2 = await run_weekly_digest()
    assert id1 > 0
    assert id2 == -1  # aborted
    # Only one digest should exist
    history = repo.get_digest_history()
    assert len(history) == 1


@pytest.mark.asyncio
async def test_continues_when_one_ingestor_fails(tmp_db, monkeypatch):
    items = _raw_items()

    async def _good_fetch(self, since):
        return items

    async def _bad_fetch(self, since):
        raise RuntimeError("network error")

    monkeypatch.setattr("backend.ingestors.arxiv.ArxivIngestor.fetch", _good_fetch)
    monkeypatch.setattr("backend.ingestors.semantic_scholar.SemanticScholarIngestor.fetch", _bad_fetch)
    monkeypatch.setattr("backend.ingestors.huggingface.HuggingFaceIngestor.fetch", _bad_fetch)
    monkeypatch.setattr("backend.ingestors.rss.RssIngestor.fetch", _bad_fetch)
    monkeypatch.setattr("backend.ingestors.reddit.RedditIngestor.fetch", _bad_fetch)

    def _fake_encode_items(its):
        rng = np.random.default_rng(42)
        return rng.random((len(its), 384)).astype(np.float32)

    monkeypatch.setattr("backend.digest.encode_items", _fake_encode_items)
    monkeypatch.setattr("backend.digest.deduplicate", lambda c, v, t: c)
    monkeypatch.setattr("backend.digest.classify_topic", lambda text, taxonomy: taxonomy[0])

    from backend.llm.base import BaseLLMProvider

    class _MockLLM(BaseLLMProvider):
        async def complete(self, system, user, max_tokens=1000):
            return '{"score": 0.7, "summary": "Summary."}'

    monkeypatch.setattr("backend.digest.get_provider", lambda s: _MockLLM())
    monkeypatch.setattr("backend.digest.send_digest_email", lambda *a, **kw: None)

    digest_id = await run_weekly_digest()
    assert digest_id > 0


@pytest.mark.asyncio
async def test_raises_digest_error_when_all_ingestors_fail(tmp_db, monkeypatch):
    async def _bad_fetch(self, since):
        raise RuntimeError("all down")

    for ingestor in [
        "backend.ingestors.arxiv.ArxivIngestor.fetch",
        "backend.ingestors.semantic_scholar.SemanticScholarIngestor.fetch",
        "backend.ingestors.huggingface.HuggingFaceIngestor.fetch",
        "backend.ingestors.rss.RssIngestor.fetch",
        "backend.ingestors.reddit.RedditIngestor.fetch",
    ]:
        monkeypatch.setattr(ingestor, _bad_fetch)

    from backend.llm.base import BaseLLMProvider

    class _MockLLM(BaseLLMProvider):
        async def complete(self, system, user, max_tokens=1000):
            return '{"score": 0.5, "summary": "S"}'

    monkeypatch.setattr("backend.digest.get_provider", lambda s: _MockLLM())

    with pytest.raises(DigestError):
        await run_weekly_digest()


@pytest.mark.asyncio
async def test_dry_run_does_not_write_to_db(tmp_db, monkeypatch):
    _setup_mocks(monkeypatch, tmp_db)
    result = await run_weekly_digest(dry_run=True)
    assert result == -1
    history = repo.get_digest_history()
    assert len(history) == 0


def test_next_friday():
    # Monday 2024-01-08 → Friday 2024-01-12
    assert _next_friday(date(2024, 1, 8)) == date(2024, 1, 12)
    # Friday 2024-01-12 → same day
    assert _next_friday(date(2024, 1, 12)) == date(2024, 1, 12)
    # Saturday 2024-01-13 → next Friday 2024-01-19
    assert _next_friday(date(2024, 1, 13)) == date(2024, 1, 19)
