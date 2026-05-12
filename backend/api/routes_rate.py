"""GET /rate and GET /health routes."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

import backend.db.repository as repo
from backend.agents import profiler as profiler_mod
from backend.config import get_settings
from backend.llm.factory import get_provider
from backend.security import verify_rating_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    return {"ok": True}


@router.get("/rate", response_class=HTMLResponse)
async def rate(request: Request, item_id: int, rating: str, token: str):
    settings = get_settings()
    secret = settings.hmac_secret

    # Step 1: Verify HMAC token
    if not verify_rating_token(item_id, rating, token, secret):
        return HTMLResponse(
            "<html><body><p>Invalid or expired rating link.</p></body></html>",
            status_code=400,
        )

    # Validate rating value
    if rating not in ("up", "down"):
        return HTMLResponse(
            "<html><body><p>Invalid rating value.</p></body></html>",
            status_code=400,
        )

    # Look up item
    item = repo.get_item_by_id(item_id)
    if item is None:
        return HTMLResponse(
            "<html><body><p>Unknown item.</p></body></html>",
            status_code=400,
        )

    # Step 2: Record rating
    repo.upsert_rating(item_id, rating)

    # Step 3: Trigger incremental profile update if threshold reached
    count = repo.count_ratings_since_last_profile()
    if count >= settings.profile.incremental_every_n:
        try:
            provider = get_provider(settings)
            rated = repo.get_rated_items_with_summaries()
            current_row = repo.get_latest_profile()
            current_prose = current_row["prose"] if current_row else None
            all_ratings = repo.get_all_ratings()
            n_total = len(all_ratings)

            # Check full rebuild trigger
            if n_total >= settings.profile.full_rebuild_every_n and n_total % settings.profile.full_rebuild_every_n == 0:
                mode = "full_rebuild"
                up_items = [
                    (it, r) for it, r in rated if r == "up"
                ][: settings.profile.full_rebuild_up_limit]
                down_items = [
                    (it, r) for it, r in rated if r == "down"
                ][: settings.profile.full_rebuild_down_limit]
                profile_input = up_items + down_items
            else:
                mode = "incremental"
                profile_input = rated

            prose = await profiler_mod.update_profile(
                profile_input, current_prose, provider, mode=mode
            )
            repo.insert_profile(prose, mode, len(rated), settings.llm_provider)
            logger.info("Profile updated (mode=%s)", mode)
        except Exception as exc:
            logger.warning("Profile update failed: %s", exc)

    # Step 4: Return confirmation HTML
    title = item.get("title", str(item_id))
    emoji = "👍" if rating == "up" else "👎"
    return HTMLResponse(
        f"<html><body><p>&#x2713; Rated '{title}' as {emoji}. Thanks.</p></body></html>",
        status_code=200,
    )
