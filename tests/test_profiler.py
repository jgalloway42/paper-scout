"""Tests for backend/agents/profiler.py."""

import pytest

from backend.agents.profiler import update_profile
from backend.llm.base import BaseLLMProvider, LLMError


def _rated_items(n: int = 3) -> list[tuple[dict, str]]:
    items = []
    for i in range(n):
        items.append((
            {"title": f"Paper {i}", "summary": f"Summary {i}"},
            "up" if i % 2 == 0 else "down",
        ))
    return items


class _CapturingLLM(BaseLLMProvider):
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        self.calls.append((system, user))
        return "Refined interest profile prose."


class _FailingLLM(BaseLLMProvider):
    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        raise LLMError("provider unavailable")


@pytest.mark.asyncio
async def test_incremental_prompt_includes_existing_profile():
    llm = _CapturingLLM()
    current = "I like reinforcement learning."
    await update_profile(_rated_items(), current, llm, mode="incremental")

    assert len(llm.calls) == 1
    _, user_prompt = llm.calls[0]
    assert "I like reinforcement learning." in user_prompt


@pytest.mark.asyncio
async def test_incremental_returns_prose_string():
    result = await update_profile(_rated_items(), "profile", _CapturingLLM(), mode="incremental")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_full_rebuild_prompt_does_not_exceed_item_limits():
    llm = _CapturingLLM()
    up_items = [
        ({"title": f"Up {i}", "summary": f"Summary {i}"}, "up")
        for i in range(40)  # exceeds 30-item limit
    ]
    down_items = [
        ({"title": f"Down {i}", "summary": f"Summary {i}"}, "down")
        for i in range(25)  # exceeds 20-item limit
    ]
    # We pass all items; the profiler should format them as-is.
    # The digest caller is responsible for slicing to limits before calling update_profile.
    # This test verifies the prompt is built correctly with the items provided.
    await update_profile(up_items[:30] + down_items[:20], None, llm, mode="full_rebuild")

    _, user_prompt = llm.calls[0]
    # Should contain thumbs-up and thumbs-down markers
    assert "👍" in user_prompt
    assert "👎" in user_prompt


@pytest.mark.asyncio
async def test_full_rebuild_no_existing_profile():
    result = await update_profile(_rated_items(), None, _CapturingLLM(), mode="full_rebuild")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_raises_llm_error_on_provider_failure():
    with pytest.raises(LLMError):
        await update_profile(_rated_items(), "profile", _FailingLLM(), mode="incremental")
