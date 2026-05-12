"""Base types for all ingestors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


@dataclass
class RawItem:
    source: str           # Publication name: "arXiv", "The Gradient", "Reddit", etc.
    title: str
    url: str
    abstract: str         # Truncated to 500 chars; empty string if unavailable
    authors: list[str]
    published_date: date
    raw_id: str           # Source-native ID for dedup


class BaseIngestor(ABC):
    @abstractmethod
    async def fetch(self, since: date) -> list[RawItem]:
        """Fetch items published/posted since `since`.

        Never raise — catch all exceptions, log WARNING, return partial list.
        """
