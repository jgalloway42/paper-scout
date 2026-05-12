"""Wildcard exploration strategies."""

import random
from abc import ABC, abstractmethod
from datetime import date

from backend.ingestors.base import RawItem


class ExplorationStrategy(ABC):
    @abstractmethod
    def select_wildcards(
        self,
        candidates: list[RawItem],
        topic_exposure: dict[str, int],
        n: int = 2,
    ) -> list[RawItem]:
        """Return exactly n wildcard items. Never raise."""


class TopicDiversityExplorer(ExplorationStrategy):
    """DEFAULT. Select from least-seen topic buckets.

    Tiebreak: random seeded on digest week_start date for reproducibility.
    """

    def __init__(self, week_start: date | None = None) -> None:
        self._week_start = week_start or date.today()

    def select_wildcards(
        self,
        candidates: list[RawItem],
        topic_exposure: dict[str, int],
        n: int = 2,
    ) -> list[RawItem]:
        if not candidates:
            return []

        rng = random.Random(self._week_start.isoformat())

        # Group candidates by topic_bucket
        buckets: dict[str, list[RawItem]] = {}
        for item in candidates:
            bucket = getattr(item, "topic_bucket", "other")
            buckets.setdefault(bucket, []).append(item)

        # Sort buckets by ascending seen_count; tiebreak by rng
        def _key(bucket: str) -> tuple:
            count = topic_exposure.get(bucket, 0)
            return (count, rng.random())

        sorted_buckets = sorted(buckets.keys(), key=_key)

        selected: list[RawItem] = []
        for bucket in sorted_buckets:
            if len(selected) >= n:
                break
            pool = buckets[bucket]
            selected.append(rng.choice(pool))

        # Fill remaining slots from any candidates if not enough distinct buckets
        if len(selected) < n:
            remaining = [c for c in candidates if c not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: n - len(selected)])

        return selected[:n]


class AdjacentScoreExplorer(ExplorationStrategy):
    """Select items with relevance_score in [0.3, 0.6].

    Falls back to TopicDiversityExplorer when insufficient items in range.
    Items must have a `_score` attribute set by the caller.
    """

    def __init__(self, week_start: date | None = None) -> None:
        self._week_start = week_start or date.today()

    def select_wildcards(
        self,
        candidates: list[RawItem],
        topic_exposure: dict[str, int],
        n: int = 2,
    ) -> list[RawItem]:
        in_range = [
            c for c in candidates
            if 0.3 <= getattr(c, "_score", -1.0) <= 0.6
        ]
        selected = in_range[:n]
        if len(selected) < n:
            remaining = [c for c in candidates if c not in selected]
            fallback = TopicDiversityExplorer(self._week_start).select_wildcards(
                remaining, topic_exposure, n - len(selected)
            )
            selected.extend(fallback)
        return selected[:n]


class ThompsonExplorer(ExplorationStrategy):
    """Thompson sampling over topic buckets using Beta(up+1, down+1).

    Falls back to TopicDiversityExplorer if any relevant bucket has < 3 total ratings.
    topic_exposure must include 'up_count' and 'down_count' per bucket.
    """

    def __init__(
        self,
        week_start: date | None = None,
        bucket_ratings: dict[str, dict] | None = None,
    ) -> None:
        self._week_start = week_start or date.today()
        self._bucket_ratings = bucket_ratings or {}

    def select_wildcards(
        self,
        candidates: list[RawItem],
        topic_exposure: dict[str, int],
        n: int = 2,
    ) -> list[RawItem]:
        if not candidates:
            return []

        buckets_present = {getattr(c, "topic_bucket", "other") for c in candidates}

        # Check fallback condition: any bucket with < 3 total ratings
        for bucket in buckets_present:
            info = self._bucket_ratings.get(bucket, {})
            total = info.get("up_count", 0) + info.get("down_count", 0)
            if total < 3:
                return TopicDiversityExplorer(self._week_start).select_wildcards(
                    candidates, topic_exposure, n
                )

        rng = random.Random(self._week_start.isoformat())

        def _sample(bucket: str) -> float:
            info = self._bucket_ratings.get(bucket, {})
            alpha = info.get("up_count", 0) + 1
            beta = info.get("down_count", 0) + 1
            # Beta sample via ratio of gamma samples
            g1 = rng.gammavariate(alpha, 1.0)
            g2 = rng.gammavariate(beta, 1.0)
            return g1 / (g1 + g2) if (g1 + g2) > 0 else 0.5

        # Group by bucket and sample
        buckets: dict[str, list[RawItem]] = {}
        for item in candidates:
            b = getattr(item, "topic_bucket", "other")
            buckets.setdefault(b, []).append(item)

        scored_buckets = [(b, _sample(b)) for b in buckets]
        scored_buckets.sort(key=lambda x: x[1], reverse=True)

        selected: list[RawItem] = []
        for bucket, _ in scored_buckets:
            if len(selected) >= n:
                break
            pool = buckets[bucket]
            selected.append(rng.choice(pool))

        if len(selected) < n:
            remaining = [c for c in candidates if c not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: n - len(selected)])

        return selected[:n]


def get_explorer(strategy_name: str, **kwargs) -> ExplorationStrategy:
    """Factory. Maps config name to class. Raise ValueError on unknown."""
    strategies = {
        "topic_diversity": TopicDiversityExplorer,
        "adjacent_score": AdjacentScoreExplorer,
        "thompson": ThompsonExplorer,
    }
    if strategy_name not in strategies:
        raise ValueError(f"Unknown exploration strategy: {strategy_name!r}")
    return strategies[strategy_name](**kwargs)
