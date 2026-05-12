"""Tests for backend/agents/scorer.py."""

from datetime import date

import pytest

from backend.agents.scorer import ScoreResult, score_and_summarise
from backend.ingestors.base import RawItem
from backend.llm.base import BaseLLMProvider, LLMError


def _item() -> RawItem:
    return RawItem(
        source="arXiv",
        title="Test Paper on LLMs",
        url="https://arxiv.org/abs/1234",
        abstract="We study large language models and their properties.",
        authors=["Author A"],
        published_date=date.today(),
        raw_id="1234",
    )


class _GoodLLM(BaseLLMProvider):
    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        return '{"score": 0.8, "summary": "This is a good summary of the paper."}'


class _WildcardLLM(BaseLLMProvider):
    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        return "This paper does interesting things with language models."


class _FailingLLM(BaseLLMProvider):
    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        raise LLMError("provider down")


class _BadJsonLLM(BaseLLMProvider):
    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        return "not valid json at all"


@pytest.mark.asyncio
async def test_score_returns_float_in_range():
    result = await score_and_summarise(_item(), "I like LLMs.", _GoodLLM())
    assert isinstance(result, ScoreResult)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_score_returns_non_empty_summary():
    result = await score_and_summarise(_item(), "I like LLMs.", _GoodLLM())
    assert result.summary != ""


@pytest.mark.asyncio
async def test_score_failure_returns_zero_score_empty_summary():
    result = await score_and_summarise(_item(), "profile", _FailingLLM())
    assert result.score == 0.0
    assert result.summary == ""


@pytest.mark.asyncio
async def test_score_bad_json_returns_zero_score():
    result = await score_and_summarise(_item(), "profile", _BadJsonLLM())
    assert result.score == 0.0
    assert result.summary == ""


@pytest.mark.asyncio
async def test_wildcard_returns_score_minus_one():
    result = await score_and_summarise(_item(), "profile", _WildcardLLM(), wildcard=True)
    assert result.score == -1.0
    assert result.summary != ""


@pytest.mark.asyncio
async def test_wildcard_failure_returns_minus_one_empty_summary():
    result = await score_and_summarise(_item(), "profile", _FailingLLM(), wildcard=True)
    assert result.score == -1.0
    assert result.summary == ""


@pytest.mark.asyncio
async def test_score_never_raises():
    # Even with a deeply broken provider, score_and_summarise should not raise
    class _CrashLLM(BaseLLMProvider):
        async def complete(self, system, user, max_tokens=1000):
            raise RuntimeError("crash")

    result = await score_and_summarise(_item(), "profile", _CrashLLM())
    assert isinstance(result, ScoreResult)
