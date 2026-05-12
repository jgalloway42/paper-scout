"""Weekly pipeline orchestrator."""

import asyncio
import logging
from datetime import date, timedelta

import numpy as np

from backend.agents.explorer import get_explorer
from backend.agents.scorer import score_and_summarise
from backend.config import get_settings
from backend.email_sender import send_digest_email
from backend.embeddings import classify_topic, deduplicate, encode_items
from backend.ingestors.base import RawItem
from backend.llm.factory import get_provider

import backend.db.repository as repo

logger = logging.getLogger(__name__)


class DigestError(Exception):
    pass


def _next_friday(ref: date) -> date:
    """Return the coming (or today's) Friday relative to ref."""
    days_ahead = 4 - ref.weekday()  # Friday = weekday 4
    if days_ahead < 0:
        days_ahead += 7
    return ref + timedelta(days=days_ahead)


def _build_ingestors(settings):
    """Instantiate all enabled ingestors."""
    from backend.ingestors.arxiv import ArxivIngestor
    from backend.ingestors.huggingface import HuggingFaceIngestor
    from backend.ingestors.reddit import RedditIngestor
    from backend.ingestors.rss import RssIngestor
    from backend.ingestors.semantic_scholar import SemanticScholarIngestor

    ingestors = []
    cfg = settings.ingestors

    if cfg.arxiv.enabled:
        ingestors.append(
            ArxivIngestor(cfg.arxiv.categories, cfg.arxiv.max_results)
        )
    if cfg.semantic_scholar.enabled:
        ingestors.append(
            SemanticScholarIngestor(
                cfg.semantic_scholar.keywords,
                cfg.semantic_scholar.max_results,
                settings.s2_api_key,
            )
        )
    if cfg.huggingface.enabled:
        ingestors.append(HuggingFaceIngestor())
    if cfg.rss.enabled:
        ingestors.append(
            RssIngestor([{"url": f.url, "name": f.name} for f in cfg.rss.feeds])
        )
    if cfg.reddit.enabled:
        ingestors.append(
            RedditIngestor(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
                subreddits=cfg.reddit.subreddits,
                post_limit=cfg.reddit.post_limit,
            )
        )
    return ingestors


async def run_weekly_digest(dry_run: bool = False) -> int:
    """Full weekly pipeline. Returns digest_id. Raises DigestError on unrecoverable failure."""
    settings = get_settings()
    provider = get_provider(settings)

    # Step 1: Determine week_start
    week_start = _next_friday(date.today())
    logger.info("Running digest for week_start=%s", week_start)

    # Step 2: Abort if already done this week
    if repo.digest_exists(week_start):
        logger.info("Digest for %s already exists — aborting.", week_start)
        return -1

    # Step 3: Run ingestors concurrently
    ingestors = _build_ingestors(settings)
    since = week_start - timedelta(days=7)
    results = await asyncio.gather(
        *[ing.fetch(since) for ing in ingestors],
        return_exceptions=True,
    )
    all_items: list[RawItem] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Ingestor failed: %s", r)
        else:
            all_items.extend(r)

    if not all_items:
        raise DigestError("All ingestors failed — no items to process.")

    logger.info("Fetched %d raw items", len(all_items))

    # Step 4–6: Load existing embeddings, encode, deduplicate
    _, existing_vecs = repo.get_all_embeddings()
    deduped = deduplicate(all_items, existing_vecs, settings.embeddings.dedup_threshold)
    logger.info("%d items after deduplication", len(deduped))

    if not deduped:
        raise DigestError("No new items after deduplication.")

    # Step 7: Classify topic buckets
    taxonomy = settings.topic_taxonomy
    for item in deduped:
        text = f"{item.title} {item.abstract[:200]}"
        item.topic_bucket = classify_topic(text, taxonomy)  # type: ignore[attr-defined]

    # Step 8: Load current profile
    profile_row = repo.get_latest_profile()
    profile_text = profile_row["prose"] if profile_row else ""

    # Step 9: Embedding pre-filter for exploit candidates
    exploit_candidates = list(deduped)
    up_vecs = repo.get_rated_embeddings("up")
    if profile_text and up_vecs.shape[0] >= 3:
        cand_vecs = encode_items(exploit_candidates)
        centroid = up_vecs.mean(axis=0, keepdims=True)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        cand_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-10)
        sims = (cand_norm @ centroid_norm.T).flatten()
        threshold = settings.scoring.prefilter_threshold
        exploit_candidates = [c for c, s in zip(exploit_candidates, sims) if s >= threshold]
        logger.info("%d items after pre-filter", len(exploit_candidates))

    if not exploit_candidates:
        exploit_candidates = list(deduped)

    # Step 10: Score exploit candidates with concurrency limit
    semaphore = asyncio.Semaphore(settings.scoring.llm_concurrency)

    async def _score_one(item: RawItem):
        async with semaphore:
            return await score_and_summarise(item, profile_text, provider)

    score_tasks = [_score_one(item) for item in exploit_candidates]
    score_results = await asyncio.gather(*score_tasks)

    # Step 11: Sort, select top exploit_count
    scored = sorted(
        zip(exploit_candidates, score_results),
        key=lambda x: x[1].score,
        reverse=True,
    )
    exploit_count = settings.digest.exploit_count
    exploit_picks = [(item, sr) for item, sr in scored[:exploit_count]]

    # Step 12: Remove exploit picks from candidate pool for wildcards
    exploit_set = {id(item) for item, _ in exploit_picks}
    wildcard_pool = [c for c in deduped if id(c) not in exploit_set]

    # Step 13: Select wildcards
    topic_exposure = repo.get_topic_exposure(settings.topic_exposure.lookback_weeks)
    explorer = get_explorer(
        settings.digest.exploration_strategy,
        week_start=week_start,
    )
    wildcard_items = explorer.select_wildcards(
        wildcard_pool, topic_exposure, n=settings.digest.wildcard_count
    )

    # Step 14: Summarise wildcard items
    async def _summarise_one(item: RawItem):
        async with semaphore:
            return await score_and_summarise(item, profile_text, provider, wildcard=True)

    wildcard_results = await asyncio.gather(
        *[_summarise_one(item) for item in wildcard_items]
    )

    # Step 15: Combine all 5 picks
    all_picks: list[tuple[RawItem, float, bool, str]] = []  # (item, score, is_wildcard, summary)
    for i, (item, sr) in enumerate(exploit_picks):
        all_picks.append((item, sr.score, False, sr.summary))
    for item, sr in zip(wildcard_items, wildcard_results):
        all_picks.append((item, sr.score, True, sr.summary))

    if dry_run:
        logger.info("DRY RUN — skipping DB write and email send.")
        for pos, (item, score, is_wc, summary) in enumerate(all_picks):
            label = "Wildcard" if is_wc else "Exploit"
            logger.info("[%s %d] score=%.2f %s", label, pos, score, item.title)
        return -1

    # Step 16: Encode all picks and write to DB in single transaction
    pick_items = [p[0] for p in all_picks]
    pick_vecs = encode_items(pick_items)

    db_items = []
    for pos, ((item, score, is_wc, summary), vec) in enumerate(
        zip(all_picks, pick_vecs)
    ):
        db_items.append(
            {
                "source": item.source,
                "title": item.title,
                "url": item.url,
                "abstract": item.abstract,
                "authors": item.authors,
                "published_date": item.published_date.isoformat(),
                "topic_bucket": getattr(item, "topic_bucket", "other"),
                "summary": summary,
                "embedding": vec.tobytes(),
                "raw_id": item.raw_id,
            }
        )

    item_ids = repo.insert_items(db_items)

    digest_slot_items = []
    for pos, (item_id, (item, score, is_wc, summary)) in enumerate(
        zip(item_ids, all_picks)
    ):
        digest_slot_items.append(
            {
                "item_id": item_id,
                "relevance_score": score,
                "is_wildcard": 1 if is_wc else 0,
                "position": pos,
            }
        )

    digest_id = repo.insert_digest(week_start, digest_slot_items)

    # Step 17: Update topic exposure
    buckets_seen = [getattr(p[0], "topic_bucket", "other") for p in all_picks]
    repo.update_topic_exposure(buckets_seen, {})

    # Step 18: Send email
    email_items = []
    for pos, (item, score, is_wc, summary) in enumerate(all_picks):
        item_id = item_ids[pos]
        email_items.append(
            {
                "id": item_id,
                "title": item.title,
                "source": item.source,
                "url": item.url,
                "topic_bucket": getattr(item, "topic_bucket", "other"),
                "published_date": item.published_date.isoformat(),
                "summary": summary,
                "is_wildcard": is_wc,
                "position": pos,
                "relevance_score": score,
            }
        )

    send_digest_email(digest_id, email_items, settings.email_to)
    logger.info("Digest %d sent to %s", digest_id, settings.email_to)

    return digest_id


def run_cli(dry_run: bool = False) -> None:
    """Sync entry point for CLI and GitHub Actions."""
    asyncio.run(run_weekly_digest(dry_run=dry_run))
