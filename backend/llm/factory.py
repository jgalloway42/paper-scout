"""LLM provider factory."""

from backend.llm.base import BaseLLMProvider
from backend.config import Settings


def get_provider(settings: Settings) -> BaseLLMProvider:
    """Return the configured LLM adapter. Raise ValueError on unknown provider."""
    provider = settings.llm_provider.lower()

    if provider == "claude":
        from backend.llm.claude_provider import ClaudeProvider
        return ClaudeProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
    elif provider == "gemini":
        from backend.llm.gemini_provider import GeminiProvider
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
    elif provider == "ollama":
        from backend.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
