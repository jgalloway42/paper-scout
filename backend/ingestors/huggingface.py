"""HuggingFace Papers RSS ingestor."""

import logging
from datetime import date, datetime

import feedparser

from backend.ingestors.base import BaseIngestor, RawItem

logger = logging.getLogger(__name__)

_HF_RSS_URL = "https://huggingface.co/papers.rss"


class HuggingFaceIngestor(BaseIngestor):
    def __init__(self, rss_url: str = _HF_RSS_URL) -> None:
        self._url = rss_url

    async def fetch(self, since: date) -> list[RawItem]:
        items: list[RawItem] = []
        try:
            parsed = feedparser.parse(self._url)
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
                        source="HuggingFace Papers",
                        title=entry.get("title", "(no title)"),
                        url=link,
                        abstract=abstract,
                        authors=[],
                        published_date=pub,
                        raw_id=entry.get("id") or link,
                    )
                )
        except Exception as exc:
            logger.warning("HuggingFaceIngestor failed: %s", exc)
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
