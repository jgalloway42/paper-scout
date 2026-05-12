"""Tests for backend/ingestors/."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.ingestors.base import RawItem
from backend.ingestors.rss import RssIngestor
from backend.ingestors.huggingface import HuggingFaceIngestor
from backend.ingestors.arxiv import ArxivIngestor
from backend.ingestors.semantic_scholar import SemanticScholarIngestor
from backend.ingestors.reddit import RedditIngestor


SINCE = date.today() - timedelta(days=7)

# ---------------------------------------------------------------------------
# RSS ingestor
# ---------------------------------------------------------------------------


def _rss_entry(title="Test Post", link="https://example.com/post", since_offset=0):
    from time import struct_time
    d = date.today() - timedelta(days=since_offset)
    t = struct_time((d.year, d.month, d.day, 0, 0, 0, 0, 0, 0))
    return MagicMock(
        title=title,
        link=link,
        id=link,
        summary="A summary.",
        published_parsed=t,
        updated_parsed=None,
        authors=[],
        author=None,
    )


@pytest.mark.asyncio
async def test_rss_ingestor_returns_items(monkeypatch):
    entry = _rss_entry()
    mock_parsed = MagicMock()
    mock_parsed.entries = [entry]
    monkeypatch.setattr("backend.ingestors.rss.feedparser.parse", lambda url: mock_parsed)

    ingestor = RssIngestor(feeds=[{"url": "https://example.com/rss", "name": "The Gradient"}])
    items = await ingestor.fetch(SINCE)

    assert len(items) == 1
    assert items[0].source == "The Gradient"
    assert isinstance(items[0], RawItem)


@pytest.mark.asyncio
async def test_rss_ingestor_stores_feed_name_as_source(monkeypatch):
    entry = _rss_entry()
    mock_parsed = MagicMock()
    mock_parsed.entries = [entry]
    monkeypatch.setattr("backend.ingestors.rss.feedparser.parse", lambda url: mock_parsed)

    ingestor = RssIngestor(feeds=[{"url": "https://distill.pub/rss.xml", "name": "Distill"}])
    items = await ingestor.fetch(SINCE)
    assert items[0].source == "Distill"


@pytest.mark.asyncio
async def test_rss_ingestor_http_failure_returns_empty(monkeypatch):
    def _boom(url):
        raise ConnectionError("network error")

    monkeypatch.setattr("backend.ingestors.rss.feedparser.parse", _boom)
    ingestor = RssIngestor(feeds=[{"url": "https://bad.url/rss", "name": "Bad Feed"}])
    items = await ingestor.fetch(SINCE)
    assert items == []


@pytest.mark.asyncio
async def test_rss_ingestor_filters_old_entries(monkeypatch):
    old_entry = _rss_entry(since_offset=30)  # 30 days ago
    mock_parsed = MagicMock()
    mock_parsed.entries = [old_entry]
    monkeypatch.setattr("backend.ingestors.rss.feedparser.parse", lambda url: mock_parsed)

    ingestor = RssIngestor(feeds=[{"url": "https://example.com/rss", "name": "Feed"}])
    items = await ingestor.fetch(SINCE)
    assert items == []


# ---------------------------------------------------------------------------
# HuggingFace ingestor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_huggingface_ingestor_returns_items(monkeypatch):
    entry = _rss_entry(title="HF Paper", link="https://huggingface.co/papers/1234")
    mock_parsed = MagicMock()
    mock_parsed.entries = [entry]
    monkeypatch.setattr("backend.ingestors.huggingface.feedparser.parse", lambda url: mock_parsed)

    ingestor = HuggingFaceIngestor()
    items = await ingestor.fetch(SINCE)
    assert len(items) == 1
    assert items[0].source == "HuggingFace Papers"


@pytest.mark.asyncio
async def test_huggingface_ingestor_http_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "backend.ingestors.huggingface.feedparser.parse", lambda url: (_ for _ in ()).throw(Exception("fail"))
    )
    ingestor = HuggingFaceIngestor()
    items = await ingestor.fetch(SINCE)
    assert items == []


# ---------------------------------------------------------------------------
# arXiv ingestor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arxiv_ingestor_returns_items(monkeypatch):
    from datetime import datetime, timezone

    mock_result = MagicMock()
    mock_result.title = "An arXiv Paper"
    mock_result.entry_id = "https://arxiv.org/abs/2401.00001"
    mock_result.summary = "Summary text."
    mock_result.authors = [MagicMock(__str__=lambda s: "Author A")]
    mock_result.published = datetime(2024, 1, 15, tzinfo=timezone.utc)
    mock_result.get_short_id.return_value = "2401.00001"

    mock_client = MagicMock()
    mock_client.results.return_value = [mock_result]

    monkeypatch.setattr("backend.ingestors.arxiv.arxiv.Client", lambda: mock_client)

    ingestor = ArxivIngestor(categories=["cs.LG"], max_results=10)
    items = await ingestor.fetch(date(2024, 1, 1))

    assert len(items) == 1
    assert items[0].source == "arXiv"
    assert items[0].raw_id == "2401.00001"


@pytest.mark.asyncio
async def test_arxiv_ingestor_failure_returns_empty(monkeypatch):
    def _boom():
        raise RuntimeError("arxiv down")

    monkeypatch.setattr("backend.ingestors.arxiv.arxiv.Client", _boom)
    ingestor = ArxivIngestor(categories=["cs.LG"])
    items = await ingestor.fetch(SINCE)
    assert items == []


# ---------------------------------------------------------------------------
# Semantic Scholar ingestor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_scholar_returns_items(monkeypatch):
    payload = {
        "data": [
            {
                "paperId": "abc123",
                "title": "S2 Paper",
                "abstract": "Great work.",
                "authors": [{"name": "Dr. Smith"}],
                "publicationDate": date.today().isoformat(),
                "url": "https://www.semanticscholar.org/paper/abc123",
            }
        ]
    }
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=payload)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr("backend.ingestors.semantic_scholar.aiohttp.ClientSession", lambda **kw: mock_session)

    ingestor = SemanticScholarIngestor(keywords=["machine learning"])
    items = await ingestor.fetch(SINCE)

    assert len(items) == 1
    assert items[0].source == "Semantic Scholar"


@pytest.mark.asyncio
async def test_semantic_scholar_failure_returns_empty(monkeypatch):
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(side_effect=Exception("connection error"))
    mock_session.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr(
        "backend.ingestors.semantic_scholar.aiohttp.ClientSession",
        lambda **kw: mock_session,
    )

    ingestor = SemanticScholarIngestor(keywords=["ml"])
    items = await ingestor.fetch(SINCE)
    assert items == []


# ---------------------------------------------------------------------------
# Reddit ingestor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reddit_ingestor_returns_items(monkeypatch):
    from datetime import datetime, timezone

    mock_post = MagicMock()
    mock_post.title = "Cool ML post"
    mock_post.permalink = "/r/MachineLearning/comments/abc/cool_post/"
    mock_post.selftext = "Some post body."
    mock_post.created_utc = datetime.now(timezone.utc).timestamp()
    mock_post.id = "abc"
    mock_post.author = MagicMock(name="user123")
    mock_post.author.name = "user123"

    mock_sub = MagicMock()
    mock_sub.new.return_value = [mock_post]

    mock_reddit = MagicMock()
    mock_reddit.subreddit.return_value = mock_sub

    monkeypatch.setattr("backend.ingestors.reddit.praw.Reddit", lambda **kw: mock_reddit)

    ingestor = RedditIngestor(
        client_id="cid",
        client_secret="csec",
        user_agent="test",
        subreddits=["MachineLearning"],
        post_limit=25,
    )
    items = await ingestor.fetch(SINCE)
    assert len(items) == 1
    assert items[0].source == "Reddit"


@pytest.mark.asyncio
async def test_reddit_ingestor_failure_returns_empty(monkeypatch):
    mock_reddit = MagicMock()
    mock_reddit.subreddit.side_effect = Exception("auth failed")

    monkeypatch.setattr("backend.ingestors.reddit.praw.Reddit", lambda **kw: mock_reddit)

    ingestor = RedditIngestor("", "", "", ["MachineLearning"])
    items = await ingestor.fetch(SINCE)
    assert items == []
