"""Anthropic Claude LLM adapter."""

import anthropic

from backend.llm.base import BaseLLMProvider, LLMError


class ClaudeProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        try:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = msg.content[0].text if msg.content else ""
            if not text:
                raise LLMError(f"empty response (stop_reason={msg.stop_reason})")
            return text
        except anthropic.APIError as exc:
            raise LLMError(str(exc)) from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc
