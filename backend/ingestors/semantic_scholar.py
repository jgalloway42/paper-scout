"""Semantic Scholar ingestor via aiohttp REST."""

import asyncio
import logging
from datetime import date

import aiohttp

from backend.ingestors.base import BaseIngestor, RawItem

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,title,abstract,authors,publicationDate,url,externalIds"


class SemanticScholarIngestor(BaseIngestor):
    def __init__(
        self,
        keywords: list[str],
        max_results: int = 30,
        api_key: str = "",
    ) -> None:
        self._keywords = keywords
        self._max_results = max_results
        self._api_key = api_key

    async def fetch(self, since: date) -> list[RawItem]:
        items: list[RawItem] = []
        headers = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                for i, keyword in enumerate(self._keywords):
                    if i > 0:
                        await asyncio.sleep(1)
                    try:
                        await self._fetch_keyword(session, keyword, since, items)
                    except Exception as exc:
                        logger.warning(
                            "SemanticScholar keyword %r failed: %s", keyword, exc
                        )
        except Exception as exc:
            logger.warning("SemanticScholarIngestor session failed: %s", exc)
        return items

    async def _fetch_keyword(
        self,
        session: aiohttp.ClientSession,
        keyword: str,
        since: date,
        items: list[RawItem],
    ) -> None:
        params = {
            "query": keyword,
            "limit": self._max_results,
            "fields": _FIELDS,
        }
        async with session.get(
            f"{_BASE_URL}/paper/search", params=params, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        seen_urls: set[str] = {it.url for it in items}
        for paper in data.get("data", []):
            pub_str = paper.get("publicationDate") or ""
            try:
                pub = date.fromisoformat(pub_str) if pub_str else date.today()
            except ValueError:
                pub = date.today()
            if pub < since:
                continue
            url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            abstract = (paper.get("abstract") or "")[:500]
            authors = [a["name"] for a in paper.get("authors", []) if a.get("name")]
            items.append(
                RawItem(
                    source="Semantic Scholar",
                    title=paper.get("title", "(no title)"),
                    url=url,
                    abstract=abstract,
                    authors=authors,
                    published_date=pub,
                    raw_id=paper.get("paperId", url),
                )
            )
