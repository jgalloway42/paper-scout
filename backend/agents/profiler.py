"""LLM profile generation and incremental refinement."""

import logging
from typing import Literal

from backend.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_INCREMENTAL_SYSTEM = """\
You are helping a researcher maintain an up-to-date interest profile.
Given their current profile and newly rated papers, refine the profile to reflect the new ratings.
Return only the updated profile prose — no headers, no JSON, no preamble."""

_FULL_REBUILD_SYSTEM = """\
You are helping a researcher build an interest profile from their paper ratings.
Based on these rated papers, write a 2-4 paragraph research interest profile.
Emphasise what they find valuable in thumbs-up papers and what they find uninteresting in thumbs-down papers.
Return only the profile prose — no headers, no JSON, no preamble."""


async def update_profile(
    rated_items: list[tuple[dict, str]],
    current_profile: str | None,
    provider: BaseLLMProvider,
    mode: Literal["incremental", "full_rebuild"] = "incremental",
) -> str:
    """Generate or refine the interest profile. Raises LLMError on provider failure."""
    if mode == "incremental":
        return await _incremental(rated_items, current_profile or "", provider)
    return await _full_rebuild(rated_items, provider)


async def _incremental(
    rated_items: list[tuple[dict, str]],
    current_profile: str,
    provider: BaseLLMProvider,
) -> str:
    items_text = _format_items(rated_items)
    user_prompt = (
        f"Current interest profile:\n{current_profile}\n\n"
        f"Newly rated papers:\n{items_text}\n\n"
        "Refine the profile to incorporate these new ratings."
    )
    return await provider.complete(_INCREMENTAL_SYSTEM, user_prompt, max_tokens=800)


async def _full_rebuild(
    rated_items: list[tuple[dict, str]],
    provider: BaseLLMProvider,
) -> str:
    items_text = _format_items(rated_items)
    user_prompt = (
        f"Rated papers:\n{items_text}\n\n"
        "Write a 2-4 paragraph research interest profile based on these ratings."
    )
    return await provider.complete(_FULL_REBUILD_SYSTEM, user_prompt, max_tokens=1000)


def _format_items(rated_items: list[tuple[dict, str]]) -> str:
    lines = []
    for item_dict, rating in rated_items:
        emoji = "👍" if rating == "up" else "👎"
        title = item_dict.get("title", "")
        summary = item_dict.get("summary", "")
        lines.append(f"{emoji} {title}\n  {summary}")
    return "\n\n".join(lines)
