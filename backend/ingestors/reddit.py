"""Reddit ingestor using PRAW."""

import logging
from datetime import date, datetime, timezone

import praw

from backend.ingestors.base import BaseIngestor, RawItem

logger = logging.getLogger(__name__)


class RedditIngestor(BaseIngestor):
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        subreddits: list[str],
        post_limit: int = 25,
    ) -> None:
        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self._subreddits = subreddits
        self._post_limit = post_limit

    async def fetch(self, since: date) -> list[RawItem]:
        items: list[RawItem] = []
        try:
            for sub_name in self._subreddits:
                try:
                    sub = self._reddit.subreddit(sub_name)
                    for post in sub.new(limit=self._post_limit):
                        pub = datetime.fromtimestamp(
                            post.created_utc, tz=timezone.utc
                        ).date()
                        if pub < since:
                            continue
                        url = f"https://www.reddit.com{post.permalink}"
                        abstract = (post.selftext or "")[:500]
                        items.append(
                            RawItem(
                                source="Reddit",
                                title=post.title,
                                url=url,
                                abstract=abstract,
                                authors=[post.author.name if post.author else "unknown"],
                                published_date=pub,
                                raw_id=post.id,
                            )
                        )
                except Exception as exc:
                    logger.warning("RedditIngestor failed for r/%s: %s", sub_name, exc)
        except Exception as exc:
            logger.warning("RedditIngestor session failed: %s", exc)
        return items
