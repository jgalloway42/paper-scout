"""Shared pytest fixtures."""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import backend.db.repository as repo
from backend.ingestors.base import RawItem


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """In-memory SQLite with migrations applied. Patches repository connection."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Apply schema
    schema_path = Path(__file__).parent.parent / "backend" / "db" / "schema.sql"
    conn.executescript(schema_path.read_text())
    conn.commit()
    # Inject into repository
    repo.set_connection(conn)
    yield conn
    conn.close()
    # Reset so next test gets a fresh connection
    repo._conn = None


@pytest.fixture()
def mock_llm():
    """BaseLLMProvider that returns canned responses."""
    from backend.llm.base import BaseLLMProvider

    class _MockLLM(BaseLLMProvider):
        async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
            # Scorer expects JSON; profiler expects prose
            if '"score"' in system or "score" in user.lower():
                return '{"score": 0.8, "summary": "Test summary."}'
            return "This is a test interest profile prose."

    return _MockLLM()


@pytest.fixture()
def mock_ingestors():
    """Returns a fixed list of 10 RawItem with varied topic buckets."""
    buckets = [
        "reinforcement_learning",
        "large_language_models",
        "computer_vision",
        "interpretability",
        "efficient_training",
        "multimodal",
        "agents_and_planning",
        "theory_and_optimization",
        "robotics",
        "applications",
    ]
    items = []
    for i, bucket in enumerate(buckets):
        items.append(
            RawItem(
                source="arXiv",
                title=f"Test Paper {i}",
                url=f"https://arxiv.org/abs/test{i:04d}",
                abstract=f"Abstract for paper {i}." * 5,
                authors=[f"Author {i}"],
                published_date=date.today() - timedelta(days=i),
                raw_id=f"test{i:04d}",
            )
        )
    return items


@pytest.fixture()
def fixed_embeddings(monkeypatch):
    """Monkeypatches embeddings.encode to return deterministic (N, 384) float32."""
    import backend.embeddings as embeddings_mod

    def _fake_encode(texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.random((len(texts), 384)).astype(np.float32)

    monkeypatch.setattr(embeddings_mod, "encode", _fake_encode)
    return _fake_encode
