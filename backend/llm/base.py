"""Base class and error type for LLM provider adapters."""

from abc import ABC, abstractmethod


class LLMError(Exception):
    """Raised by provider adapters on unrecoverable LLM failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Return assistant text. Raise LLMError on failure."""
