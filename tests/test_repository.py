"""Tests for backend/db/repository.py."""

import json
from datetime import date

import numpy as np

import backend.db.repository as repo


def _make_item(**kwargs):
    defaults = {
        "source": "arXiv",
        "title": "Test Paper",
        "url": "https://arxiv.org/abs/1234.5678",
        "abstract": "An abstract.",
        "authors": ["Alice", "Bob"],
        "published_date": "2024-01-15",
        "topic_bucket": "large_language_models",
        "raw_id": "1234.5678",
    }
    defaults.update(kwargs)
    return defaults


def _make_embedding() -> bytes:
    return np.ones(384, dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def test_insert_and_retrieve_item(tmp_db):
    item = _make_item()
    ids = repo.insert_items([item])
    assert len(ids) == 1
    retrieved = repo.get_item_by_id(ids[0])
    assert retrieved["title"] == "Test Paper"
    assert retrieved["source"] == "arXiv"
    assert json.loads(retrieved["authors"]) == ["Alice", "Bob"]


def test_insert_duplicate_url_skipped(tmp_db):
    item = _make_item()
    ids1 = repo.insert_items([item])
    ids2 = repo.insert_items([item])
    assert len(ids1) == 1
    assert ids2 == []


def test_insert_items_returns_only_new_ids(tmp_db):
    item_a = _make_item(url="https://arxiv.org/abs/aaaa", raw_id="aaaa", title="A")
    item_b = _make_item(url="https://arxiv.org/abs/bbbb", raw_id="bbbb", title="B")
    ids = repo.insert_items([item_a, item_b])
    assert len(ids) == 2


def test_get_all_embeddings_empty(tmp_db):
    ids, vecs = repo.get_all_embeddings()
    assert ids == []
    assert vecs.shape == (0, 384)


def test_get_all_embeddings_with_data(tmp_db):
    emb = _make_embedding()
    item = _make_item(embedding=emb)
    repo.insert_items([item])
    ids, vecs = repo.get_all_embeddings()
    assert len(ids) == 1
    assert vecs.shape == (1, 384)
    np.testing.assert_array_almost_equal(vecs[0], np.ones(384, dtype=np.float32))


def test_get_rated_embeddings(tmp_db):
    emb = _make_embedding()
    item = _make_item(embedding=emb)
    [item_id] = repo.insert_items([item])
    repo.upsert_rating(item_id, "up")
    up_vecs = repo.get_rated_embeddings("up")
    down_vecs = repo.get_rated_embeddings("down")
    assert up_vecs.shape == (1, 384)
    assert down_vecs.shape == (0, 384)


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def test_insert_and_get_digest(tmp_db):
    item = _make_item()
    [item_id] = repo.insert_items([item])
    week = date(2024, 1, 12)
    digest_items = [
        {"item_id": item_id, "relevance_score": 0.9, "is_wildcard": 0, "position": 0}
    ]
    digest_id = repo.insert_digest(week, digest_items)
    assert digest_id == 1
    d = repo.get_digest_by_id(digest_id)
    assert d["week_start"] == "2024-01-12"


def test_get_latest_digest(tmp_db):
    assert repo.get_latest_digest() is None
    item = _make_item()
    [item_id] = repo.insert_items([item])
    repo.insert_digest(date(2024, 1, 12), [
        {"item_id": item_id, "relevance_score": 0.5, "is_wildcard": 0, "position": 0}
    ])
    d = repo.get_latest_digest()
    assert d is not None
    assert d["week_start"] == "2024-01-12"


def test_digest_exists(tmp_db):
    item = _make_item()
    [item_id] = repo.insert_items([item])
    week = date(2024, 1, 12)
    assert not repo.digest_exists(week)
    repo.insert_digest(week, [
        {"item_id": item_id, "relevance_score": 0.5, "is_wildcard": 0, "position": 0}
    ])
    assert repo.digest_exists(week)


def test_get_digest_history(tmp_db):
    items_data = [
        _make_item(url=f"https://arxiv.org/abs/{i}", raw_id=str(i), title=f"P{i}")
        for i in range(3)
    ]
    item_ids = repo.insert_items(items_data)
    for i, item_id in enumerate(item_ids):
        repo.insert_digest(date(2024, 1, 12 + i * 7), [
            {"item_id": item_id, "relevance_score": 0.5, "is_wildcard": 0, "position": 0}
        ])
    history = repo.get_digest_history(limit=2)
    assert len(history) == 2
    assert len(history[0]["items"]) == 1


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------


def test_upsert_rating_insert(tmp_db):
    [item_id] = repo.insert_items([_make_item()])
    repo.upsert_rating(item_id, "up")
    ratings = repo.get_all_ratings()
    assert len(ratings) == 1
    assert ratings[0][1] == "up"


def test_upsert_rating_replaces_on_duplicate(tmp_db):
    [item_id] = repo.insert_items([_make_item()])
    repo.upsert_rating(item_id, "up")
    repo.upsert_rating(item_id, "down")
    ratings = repo.get_all_ratings()
    assert len(ratings) == 1
    assert ratings[0][1] == "down"


def test_get_rated_items_with_summaries(tmp_db):
    item_with = _make_item(url="https://arxiv.org/abs/A", raw_id="A")
    item_with["summary"] = "A good summary."
    item_without = _make_item(url="https://arxiv.org/abs/B", raw_id="B")

    [id_with] = repo.insert_items([item_with])
    [id_without] = repo.insert_items([item_without])
    repo.upsert_rating(id_with, "up")
    repo.upsert_rating(id_without, "down")

    results = repo.get_rated_items_with_summaries()
    assert len(results) == 1
    assert results[0][1] == "up"


def test_count_ratings_since_last_profile_no_profile(tmp_db):
    assert repo.count_ratings_since_last_profile() == 0
    [id1] = repo.insert_items([_make_item(url="https://a", raw_id="a")])
    repo.upsert_rating(id1, "up")
    assert repo.count_ratings_since_last_profile() == 1


def test_count_ratings_since_last_profile_with_profile(tmp_db):
    [id1] = repo.insert_items([_make_item(url="https://a", raw_id="a")])
    repo.upsert_rating(id1, "up")
    # Insert a profile now
    repo.insert_profile("Some prose.", "full_rebuild", 1, "claude")
    # Add a new rating after profile
    [id2] = repo.insert_items([_make_item(url="https://b", raw_id="b", title="B")])
    repo.upsert_rating(id2, "down")
    count = repo.count_ratings_since_last_profile()
    assert count == 1


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def test_insert_and_get_profile(tmp_db):
    assert repo.get_latest_profile() is None
    pid = repo.insert_profile("My interests.", "full_rebuild", 10, "claude")
    assert pid == 1
    p = repo.get_latest_profile()
    assert p["prose"] == "My interests."
    assert p["mode"] == "full_rebuild"
    assert p["llm_provider"] == "claude"


def test_get_profile_history(tmp_db):
    repo.insert_profile("Profile 1.", "full_rebuild", 10, "claude")
    repo.insert_profile("Profile 2.", "incremental", 15, "gemini")
    history = repo.get_profile_history(limit=10)
    assert len(history) == 2
    assert history[0]["prose"] == "Profile 2."  # newest first


# ---------------------------------------------------------------------------
# Topic exposure
# ---------------------------------------------------------------------------


def test_update_and_get_topic_exposure(tmp_db):
    repo.update_topic_exposure(["large_language_models", "robotics"], {})
    exp = repo.get_topic_exposure(lookback_weeks=4)
    assert exp["large_language_models"] == 1
    assert exp["robotics"] == 1


def test_topic_exposure_up_down_counts(tmp_db):
    repo.update_topic_exposure(
        ["large_language_models"],
        {"large_language_models": "up"},
    )
    row = tmp_db.execute(
        "SELECT up_count, down_count FROM topic_exposure WHERE topic_bucket = ?",
        ("large_language_models",),
    ).fetchone()
    assert row["up_count"] == 1
    assert row["down_count"] == 0
