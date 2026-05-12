"""RSS ingestor using feedparser. Source name comes from config feed list."""

import logging
from datetime import date, datetime

import feedparser

from backend.ingestors.base import BaseIngestor, RawItem

logger = logging.getLogger(__name__)


class RssIngestor(BaseIngestor):
    def __init__(self, feeds: list[dict]) -> None:
        """feeds: list of {"url": str, "name": str}"""
        self._feeds = feeds

    async def fetch(self, since: date) -> list[RawItem]:
        items: list[RawItem] = []
        for feed_cfg in self._feeds:
            url = feed_cfg["url"]
            name = feed_cfg["name"]
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries:
                    pub = _parse_date(entry)
                    if pub < since:
                        continue
                    link = entry.get("link", "")
                    if not link:
                        continue
                    abstract = (entry.get("summary") or "")[:500]
                    items.append(
                        RawItem(
                            source=name,
                            title=entry.get("title", "(no title)"),
                            url=link,
                            abstract=abstract,
                            authors=_parse_authors(entry),
                            published_date=pub,
                            raw_id=entry.get("id") or link,
                        )
                    )
            except Exception as exc:
                logger.warning("RssIngestor failed for %s: %s", url, exc)
        return items


def _parse_date(entry) -> date:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6]).date()
            except Exception:
                pass
    return date.today()


def _parse_authors(entry) -> list[str]:
    authors = []
    for a in entry.get("authors", []):
        name = a.get("name", "")
        if name:
            authors.append(name)
    if not authors and entry.get("author"):
        authors = [entry["author"]]
    return authors
