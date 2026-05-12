"""Tests for backend/embeddings.py."""

from datetime import date

import numpy as np

from backend.ingestors.base import RawItem
import backend.embeddings as embeddings_mod


def _item(i: int, title: str = "Test", abstract: str = "Abstract.") -> RawItem:
    return RawItem(
        source="arXiv",
        title=title,
        url=f"https://arxiv.org/abs/{i}",
        abstract=abstract,
        authors=[],
        published_date=date.today(),
        raw_id=str(i),
    )


TAXONOMY = [
    "reinforcement_learning",
    "large_language_models",
    "computer_vision",
    "interpretability",
    "other",
]


def test_encode_shape(fixed_embeddings):
    vecs = embeddings_mod.encode(["hello", "world", "foo"])
    assert vecs.shape == (3, 384)
    assert vecs.dtype == np.float32


def test_encode_empty(fixed_embeddings):
    vecs = embeddings_mod.encode([])
    assert vecs.shape == (0, 384)


def test_dedup_removes_above_threshold(fixed_embeddings, monkeypatch):
    """Items with sim > threshold to existing vectors are removed."""
    items = [_item(1), _item(2)]
    # Make encode return identical vectors for both items → sim = 1.0 to existing
    identical_vec = np.ones((1, 384), dtype=np.float32)
    existing = identical_vec.copy()

    def _fake_encode_items(its):
        return np.ones((len(its), 384), dtype=np.float32)

    monkeypatch.setattr(embeddings_mod, "encode_items", _fake_encode_items)

    result = embeddings_mod.deduplicate(items, existing, threshold=0.92)
    assert len(result) == 0


def test_dedup_keeps_below_threshold(fixed_embeddings, monkeypatch):
    """Items with sim ≤ threshold to existing vectors are kept."""
    items = [_item(1), _item(2)]
    # Existing is all-ones, candidates are all-zeros → sim = 0.0
    existing = np.ones((1, 384), dtype=np.float32)

    def _fake_encode_items(its):
        return np.zeros((len(its), 384), dtype=np.float32)

    monkeypatch.setattr(embeddings_mod, "encode_items", _fake_encode_items)

    result = embeddings_mod.deduplicate(items, existing, threshold=0.92)
    assert len(result) == 2


def test_dedup_within_batch(monkeypatch):
    """Near-duplicate candidates within the batch: only first is kept."""
    items = [_item(1), _item(2)]
    call_count = [0]

    def _fake_encode_items(its):
        call_count[0] += 1
        # Both items get identical vectors → they're near-duplicates
        return np.ones((len(its), 384), dtype=np.float32)

    monkeypatch.setattr(embeddings_mod, "encode_items", _fake_encode_items)

    result = embeddings_mod.deduplicate(
        items, np.empty((0, 384), dtype=np.float32), threshold=0.92
    )
    assert len(result) == 1
    assert result[0].url == items[0].url


def test_dedup_empty_candidates():
    result = embeddings_mod.deduplicate(
        [], np.empty((0, 384), dtype=np.float32), threshold=0.92
    )
    assert result == []


def test_classify_returns_taxonomy_value(fixed_embeddings, monkeypatch):
    """classify_topic returns a value from the taxonomy list."""
    # Real call uses actual SentenceTransformer; we monkeypatch encode to keep tests fast
    def _fake_encode(texts):
        # Return a unit vector per text; first taxonomy label wins for matching text
        vecs = np.zeros((len(texts), 384), dtype=np.float32)
        vecs[:, 0] = 1.0
        return vecs

    monkeypatch.setattr(embeddings_mod, "encode", _fake_encode)
    # Reset cached taxonomy embeddings to force rebuild with fake encoder
    embeddings_mod._taxonomy_embeddings = None
    embeddings_mod._taxonomy_labels = None

    result = embeddings_mod.classify_topic("some AI paper text", TAXONOMY)
    assert result in TAXONOMY
