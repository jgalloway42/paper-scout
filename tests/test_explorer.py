"""Tests for backend/agents/explorer.py."""

from datetime import date

import pytest

from backend.agents.explorer import (
    AdjacentScoreExplorer,
    ThompsonExplorer,
    TopicDiversityExplorer,
    get_explorer,
)
from backend.ingestors.base import RawItem


def _item(i: int, bucket: str = "large_language_models", score: float = 0.5) -> RawItem:
    item = RawItem(
        source="arXiv",
        title=f"Paper {i}",
        url=f"https://arxiv.org/abs/{i}",
        abstract="Abstract.",
        authors=[],
        published_date=date.today(),
        raw_id=str(i),
    )
    item.topic_bucket = bucket  # type: ignore[attr-defined]
    item._score = score  # type: ignore[attr-defined]
    return item


WEEK = date(2024, 1, 12)


def _diverse_candidates():
    buckets = [
        "reinforcement_learning",
        "large_language_models",
        "computer_vision",
        "interpretability",
        "efficient_training",
    ]
    return [_item(i, bucket=b) for i, b in enumerate(buckets)]


# ---------------------------------------------------------------------------
# TopicDiversityExplorer
# ---------------------------------------------------------------------------


def test_topic_diversity_returns_n_items():
    candidates = _diverse_candidates()
    result = TopicDiversityExplorer(WEEK).select_wildcards(candidates, {}, n=2)
    assert len(result) == 2


def test_topic_diversity_selects_least_seen():
    candidates = _diverse_candidates()
    exposure = {
        "reinforcement_learning": 10,
        "large_language_models": 5,
        "computer_vision": 1,
        "interpretability": 0,
        "efficient_training": 3,
    }
    result = TopicDiversityExplorer(WEEK).select_wildcards(candidates, exposure, n=2)
    buckets = {getattr(r, "topic_bucket", None) for r in result}
    # Should prefer interpretability (0) and computer_vision (1)
    assert "interpretability" in buckets


def test_topic_diversity_empty_candidates():
    result = TopicDiversityExplorer(WEEK).select_wildcards([], {}, n=2)
    assert result == []


def test_topic_diversity_reproducible():
    candidates = _diverse_candidates()
    r1 = TopicDiversityExplorer(WEEK).select_wildcards(candidates, {}, n=2)
    r2 = TopicDiversityExplorer(WEEK).select_wildcards(candidates, {}, n=2)
    assert [c.url for c in r1] == [c.url for c in r2]


# ---------------------------------------------------------------------------
# AdjacentScoreExplorer
# ---------------------------------------------------------------------------


def test_adjacent_score_returns_n_items():
    candidates = [
        _item(0, score=0.4),
        _item(1, score=0.5),
        _item(2, score=0.9),
        _item(3, score=0.1),
    ]
    result = AdjacentScoreExplorer(WEEK).select_wildcards(candidates, {}, n=2)
    assert len(result) == 2


def test_adjacent_score_prefers_mid_range():
    candidates = [
        _item(0, score=0.4),
        _item(1, score=0.55),
        _item(2, score=0.9),
        _item(3, score=0.05),
    ]
    result = AdjacentScoreExplorer(WEEK).select_wildcards(candidates, {}, n=2)
    scores = [getattr(r, "_score", -1.0) for r in result]
    # Both selected should be in [0.3, 0.6]
    for s in scores:
        assert 0.3 <= s <= 0.6


def test_adjacent_score_falls_back_to_diversity_when_range_empty():
    # All items outside [0.3, 0.6]
    candidates = [_item(i, score=0.95) for i in range(4)]
    for i, c in enumerate(candidates):
        c.topic_bucket = f"bucket_{i}"  # type: ignore[attr-defined]
    result = AdjacentScoreExplorer(WEEK).select_wildcards(candidates, {}, n=2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# ThompsonExplorer
# ---------------------------------------------------------------------------


def _bucket_ratings_sufficient():
    """All buckets have >= 3 total ratings."""
    return {
        "reinforcement_learning": {"up_count": 2, "down_count": 2},
        "large_language_models": {"up_count": 4, "down_count": 1},
        "computer_vision": {"up_count": 1, "down_count": 3},
        "interpretability": {"up_count": 3, "down_count": 2},
        "efficient_training": {"up_count": 2, "down_count": 3},
    }


def test_thompson_returns_n_items():
    candidates = _diverse_candidates()
    result = ThompsonExplorer(
        week_start=WEEK, bucket_ratings=_bucket_ratings_sufficient()
    ).select_wildcards(candidates, {}, n=2)
    assert len(result) == 2


def test_thompson_falls_back_when_any_bucket_has_few_ratings():
    candidates = _diverse_candidates()
    # One bucket has < 3 ratings
    sparse = {"reinforcement_learning": {"up_count": 1, "down_count": 0}}
    result = ThompsonExplorer(
        week_start=WEEK, bucket_ratings=sparse
    ).select_wildcards(candidates, {}, n=2)
    assert len(result) == 2  # falls back to diversity, still returns 2


def test_thompson_empty_candidates():
    result = ThompsonExplorer(WEEK).select_wildcards([], {}, n=2)
    assert result == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_get_explorer_topic_diversity():
    e = get_explorer("topic_diversity")
    assert isinstance(e, TopicDiversityExplorer)


def test_get_explorer_adjacent_score():
    e = get_explorer("adjacent_score")
    assert isinstance(e, AdjacentScoreExplorer)


def test_get_explorer_thompson():
    e = get_explorer("thompson")
    assert isinstance(e, ThompsonExplorer)


def test_get_explorer_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown exploration strategy"):
        get_explorer("nonexistent_strategy")
