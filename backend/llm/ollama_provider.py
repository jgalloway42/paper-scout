"""Ollama local LLM adapter."""

import ollama as ollama_lib

from backend.llm.base import BaseLLMProvider, LLMError


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3") -> None:
        self._client = ollama_lib.AsyncClient(host=base_url)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        try:
            response = await self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"num_predict": max_tokens},
            )
            return response["message"]["content"]
        except Exception as exc:
            raise LLMError(str(exc)) from exc
