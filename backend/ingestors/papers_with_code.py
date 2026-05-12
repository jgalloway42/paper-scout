"""Papers With Code ingestor via public REST API (no auth required)."""

import logging
from datetime import date

import aiohttp

from backend.ingestors.base import BaseIngestor, RawItem

logger = logging.getLogger(__name__)

_BASE_URL = "https://paperswithcode.com/api/v1/papers/"


class PapersWithCodeIngestor(BaseIngestor):
    def __init__(self, max_results: int = 50) -> None:
        self._max_results = max_results

    async def fetch(self, since: date) -> list[RawItem]:
        items: list[RawItem] = []
        try:
            async with aiohttp.ClientSession() as session:
                await self._fetch_page(session, since, items)
        except Exception as exc:
            logger.warning("PapersWithCodeIngestor failed: %s", exc)
        return items

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        since: date,
        items: list[RawItem],
    ) -> None:
        params = {
            "ordering": "-published",
            "items_per_page": self._max_results,
            "page": 1,
        }
        async with session.get(
            _BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        seen_urls: set[str] = set()
        for paper in data.get("results", []):
            pub_str = paper.get("published") or ""
            try:
                pub = date.fromisoformat(pub_str) if pub_str else date.today()
            except ValueError:
                pub = date.today()
            if pub < since:
                continue

            arxiv_id = paper.get("arxiv_id") or ""
            url = (
                paper.get("url_abs")
                or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
                or f"https://paperswithcode.com/paper/{paper.get('id', '')}"
            )
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            raw_authors = paper.get("authors") or []
            if raw_authors and isinstance(raw_authors[0], dict):
                authors = [a.get("name", "") for a in raw_authors if a.get("name")]
            else:
                authors = [str(a) for a in raw_authors if a]

            abstract = (paper.get("abstract") or "")[:500]
            items.append(
                RawItem(
                    source="Papers With Code",
                    title=paper.get("title") or "(no title)",
                    url=url,
                    abstract=abstract,
                    authors=authors,
                    published_date=pub,
                    raw_id=arxiv_id or paper.get("id", url),
                )
            )
