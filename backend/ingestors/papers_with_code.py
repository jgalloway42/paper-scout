"""HuggingFace daily papers ingestor (replaces Papers With Code, which now redirects here)."""

import logging
from datetime import date, timedelta

import aiohttp

from backend.ingestors.base import BaseIngestor, RawItem

logger = logging.getLogger(__name__)

_BASE_URL = "https://huggingface.co/api/daily_papers"


class PapersWithCodeIngestor(BaseIngestor):
    def __init__(self, max_results: int = 50) -> None:
        self._max_results = max_results

    async def fetch(self, since: date) -> list[RawItem]:
        items: list[RawItem] = []
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch one day at a time from since → today
                current = since
                today = date.today()
                while current <= today and len(items) < self._max_results:
                    await self._fetch_date(session, current, since, items)
                    current += timedelta(days=1)
        except Exception as exc:
            logger.warning("PapersWithCodeIngestor failed: %s", exc)
        return items[: self._max_results]

    async def _fetch_date(
        self,
        session: aiohttp.ClientSession,
        for_date: date,
        since: date,
        items: list[RawItem],
    ) -> None:
        params = {"date": for_date.isoformat()}
        try:
            async with session.get(
                _BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 404:
                    return
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logger.debug("HF daily papers %s failed: %s", for_date, exc)
            return

        seen_urls: set[str] = {it.url for it in items}
        for entry in data:
            paper = entry.get("paper") or entry
            arxiv_id = paper.get("id") or ""
            url = (
                f"https://arxiv.org/abs/{arxiv_id}"
                if arxiv_id
                else paper.get("url", "")
            )
            if not url or url in seen_urls:
                continue

            pub_str = paper.get("publishedAt") or ""
            try:
                pub = date.fromisoformat(pub_str[:10]) if pub_str else for_date
            except ValueError:
                pub = for_date

            if pub < since:
                continue
            seen_urls.add(url)

            raw_authors = paper.get("authors") or []
            if raw_authors and isinstance(raw_authors[0], dict):
                authors = [a.get("name", "") for a in raw_authors if a.get("name")]
            else:
                authors = [str(a) for a in raw_authors if a]

            items.append(
                RawItem(
                    source="HuggingFace Daily",
                    title=paper.get("title") or "(no title)",
                    url=url,
                    abstract=(paper.get("summary") or paper.get("abstract") or "")[:500],
                    authors=authors,
                    published_date=pub,
                    raw_id=arxiv_id or url,
                )
            )
