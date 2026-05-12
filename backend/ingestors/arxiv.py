"""arXiv ingestor using the arxiv PyPI package."""

import logging
from datetime import date

import arxiv

from backend.ingestors.base import BaseIngestor, RawItem

logger = logging.getLogger(__name__)


class ArxivIngestor(BaseIngestor):
    def __init__(self, categories: list[str], max_results: int = 50) -> None:
        self._categories = categories
        self._max_results = max_results

    async def fetch(self, since: date) -> list[RawItem]:
        items: list[RawItem] = []
        try:
            query = " OR ".join(f"cat:{c}" for c in self._categories)
            client = arxiv.Client(delay_seconds=5, num_retries=5)
            search = arxiv.Search(
                query=query,
                max_results=self._max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            for result in client.results(search):
                pub = result.published.date() if result.published else date.today()
                if pub < since:
                    continue
                abstract = (result.summary or "")[:500]
                authors = [str(a) for a in result.authors]
                items.append(
                    RawItem(
                        source="arXiv",
                        title=result.title,
                        url=result.entry_id,
                        abstract=abstract,
                        authors=authors,
                        published_date=pub,
                        raw_id=result.get_short_id(),
                    )
                )
        except Exception as exc:
            logger.warning("ArxivIngestor failed: %s", exc)
        return items
