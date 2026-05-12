"""LLM scoring + summary generation. One call per item."""

import json
import logging
from dataclasses import dataclass

from backend.ingestors.base import RawItem
from backend.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_EXPLOIT_SYSTEM = """\
You are a research assistant helping a technical reader discover relevant papers.
Given a paper and a user interest profile, return ONLY valid JSON in this exact format:
{"score": <float 0.0-1.0>, "summary": "<2-3 sentence plain English summary>"}
The score represents how relevant this paper is to the user's interests (0=irrelevant, 1=highly relevant).
The summary should highlight what the paper does and why it matters, written for a technical reader.
Do not include any text outside the JSON object."""

_WILDCARD_SYSTEM = """\
You are a research assistant. Write a 2-3 sentence plain English summary of the following paper,
highlighting what it does and why it might be interesting to a machine learning researcher.
Return only the summary text, no JSON."""


@dataclass
class ScoreResult:
    score: float   # 0.0–1.0; -1.0 for wildcard (summary-only call)
    summary: str   # 2–3 sentences, plain English


async def score_and_summarise(
    item: RawItem,
    profile: str,
    provider: BaseLLMProvider,
    wildcard: bool = False,
) -> ScoreResult:
    """Score and summarise a single item with one LLM call. Never raises."""
    if wildcard:
        return await _summarise_only(item, provider)
    return await _score_and_summarise(item, profile, provider)


async def _score_and_summarise(
    item: RawItem,
    profile: str,
    provider: BaseLLMProvider,
) -> ScoreResult:
    user_prompt = (
        f"User interest profile:\n{profile}\n\n"
        f"Paper title: {item.title}\n"
        f"Abstract: {item.abstract[:400]}\n"
        f"Source: {item.source}"
    )
    try:
        raw = await provider.complete(_EXPLOIT_SYSTEM, user_prompt, max_tokens=300)
        data = json.loads(raw.strip())
        score = float(data["score"])
        score = max(0.0, min(1.0, score))
        summary = str(data.get("summary", ""))
        return ScoreResult(score=score, summary=summary)
    except Exception as exc:
        logger.warning("score_and_summarise failed for %r: %s", item.title, exc)
        return ScoreResult(score=0.0, summary="")


async def _summarise_only(item: RawItem, provider: BaseLLMProvider) -> ScoreResult:
    user_prompt = (
        f"Paper title: {item.title}\n"
        f"Abstract: {item.abstract[:400]}\n"
        f"Source: {item.source}"
    )
    try:
        summary = await provider.complete(_WILDCARD_SYSTEM, user_prompt, max_tokens=200)
        return ScoreResult(score=-1.0, summary=summary.strip())
    except Exception as exc:
        logger.warning("summarise_only failed for %r: %s", item.title, exc)
        return ScoreResult(score=-1.0, summary="")
