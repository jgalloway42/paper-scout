"""LLM scoring + summary generation. One call per item."""

import asyncio
import json
import logging
import re
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

_daily_quota_exhausted = False


@dataclass
class ScoreResult:
    score: float   # 0.0–1.0; -1.0 for wildcard (summary-only call)
    summary: str   # 2–3 sentences, plain English


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that some models wrap around JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _classify_llm_error(exc: Exception) -> tuple[str, float | None]:
    """Return (kind, retry_delay_seconds) for a caught LLM exception.

    kind is one of: 'daily_quota', 'transient_rate_limit', 'other'
    """
    msg = str(exc)
    is_rate_limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg
    if not is_rate_limited:
        return ("other", None)
    if "GenerateRequestsPerDayPerProjectPerModel" in msg:
        return ("daily_quota", None)
    delay: float | None = None
    m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s", msg)
    if m:
        delay = float(m.group(1))
    return ("transient_rate_limit", delay)


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
    global _daily_quota_exhausted
    if _daily_quota_exhausted:
        return ScoreResult(score=0.0, summary="")

    user_prompt = (
        f"User interest profile:\n{profile}\n\n"
        f"Paper title: {item.title}\n"
        f"Abstract: {item.abstract[:400]}\n"
        f"Source: {item.source}"
    )
    for attempt in range(2):
        raw = ""
        try:
            raw = await provider.complete(_EXPLOIT_SYSTEM, user_prompt, max_tokens=300)
            data = json.loads(_strip_code_fences(raw))
            score = float(data["score"])
            score = max(0.0, min(1.0, score))
            summary = str(data.get("summary", ""))
            return ScoreResult(score=score, summary=summary)
        except Exception as exc:
            kind, delay = _classify_llm_error(exc)
            if kind == "daily_quota":
                _daily_quota_exhausted = True
                logger.warning("Gemini daily quota exhausted — skipping remaining scoring")
                return ScoreResult(score=0.0, summary="")
            elif kind == "transient_rate_limit" and attempt == 0:
                wait = delay or 30.0
                logger.info("Rate limited on %r, retrying in %.0fs", item.title, wait)
                await asyncio.sleep(wait)
                continue
            elif kind == "transient_rate_limit":
                logger.warning("Rate limited twice on %r, skipping", item.title)
                return ScoreResult(score=0.0, summary="")
            else:
                logger.warning("score failed for %r: %.200s | raw=%.100r", item.title, str(exc), raw)
                return ScoreResult(score=0.0, summary="")
    return ScoreResult(score=0.0, summary="")


async def _summarise_only(item: RawItem, provider: BaseLLMProvider) -> ScoreResult:
    global _daily_quota_exhausted
    if _daily_quota_exhausted:
        return ScoreResult(score=-1.0, summary="")

    user_prompt = (
        f"Paper title: {item.title}\n"
        f"Abstract: {item.abstract[:400]}\n"
        f"Source: {item.source}"
    )
    for attempt in range(2):
        try:
            summary = await provider.complete(_WILDCARD_SYSTEM, user_prompt, max_tokens=200)
            return ScoreResult(score=-1.0, summary=summary.strip())
        except Exception as exc:
            kind, delay = _classify_llm_error(exc)
            if kind == "daily_quota":
                _daily_quota_exhausted = True
                logger.warning("Gemini daily quota exhausted — skipping remaining scoring")
                return ScoreResult(score=-1.0, summary="")
            elif kind == "transient_rate_limit" and attempt == 0:
                wait = delay or 30.0
                logger.info("Rate limited on %r, retrying in %.0fs", item.title, wait)
                await asyncio.sleep(wait)
                continue
            elif kind == "transient_rate_limit":
                logger.warning("Rate limited twice on %r, skipping", item.title)
                return ScoreResult(score=-1.0, summary="")
            else:
                logger.warning("summarise failed for %r: %.200s", item.title, str(exc))
                return ScoreResult(score=-1.0, summary="")
    return ScoreResult(score=-1.0, summary="")
