"""Google Gemini LLM adapter (google-genai SDK)."""

from google import genai
from google.genai import types

from backend.llm.base import BaseLLMProvider, LLMError


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as exc:
            raise LLMError(str(exc)) from exc
