"""Coverage for factory, migrations, and other small modules."""

import sqlite3

import pytest

from backend.config import load_settings
from backend.db.migrations import run as run_migrations


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def test_migrations_run_on_memory_db():
    """run() applies schema without error."""
    import os
    old = os.environ.get("PAPER_SCOUT_DB_PATH")
    os.environ["PAPER_SCOUT_DB_PATH"] = ":memory:"
    try:
        run_migrations(":memory:")
    finally:
        if old is None:
            del os.environ["PAPER_SCOUT_DB_PATH"]
        else:
            os.environ["PAPER_SCOUT_DB_PATH"] = old


def test_migrations_idempotent(tmp_path):
    """Calling run() twice does not raise (IF NOT EXISTS)."""
    db_path = str(tmp_path / "test.db")
    run_migrations(db_path)
    run_migrations(db_path)  # second call should be a no-op


def test_migrations_uses_env_var(tmp_path, monkeypatch):
    db_path = str(tmp_path / "env_test.db")
    monkeypatch.setenv("PAPER_SCOUT_DB_PATH", db_path)
    run_migrations()  # no argument — reads from env
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "items" in tables
    assert "digests" in tables


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------


def test_factory_returns_claude_provider():
    from backend.llm.claude_provider import ClaudeProvider
    from backend.llm.factory import get_provider

    settings = load_settings()
    settings.llm_provider = "claude"
    settings.anthropic_api_key = "test-key"
    provider = get_provider(settings)
    assert isinstance(provider, ClaudeProvider)


def test_factory_returns_gemini_provider():
    from backend.llm.gemini_provider import GeminiProvider
    from backend.llm.factory import get_provider

    settings = load_settings()
    settings.llm_provider = "gemini"
    settings.gemini_api_key = "test-key"
    provider = get_provider(settings)
    assert isinstance(provider, GeminiProvider)


def test_factory_returns_ollama_provider():
    from backend.llm.ollama_provider import OllamaProvider
    from backend.llm.factory import get_provider

    settings = load_settings()
    settings.llm_provider = "ollama"
    provider = get_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_factory_raises_on_unknown_provider():
    from backend.llm.factory import get_provider

    settings = load_settings()
    settings.llm_provider = "nonexistent"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider(settings)


# ---------------------------------------------------------------------------
# Security — exception branch
# ---------------------------------------------------------------------------


def test_verify_token_returns_false_on_exception():
    from backend.security import verify_rating_token

    # Passing None values that would cause an exception in hmac.new
    result = verify_rating_token(0, "", "", "")
    assert result is False
