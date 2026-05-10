"""
Ollama provider — fully local, no API key required.
Best for development and privacy-sensitive deployments.

Install: https://ollama.ai
Then: ollama pull llama3
"""
import time

import httpx

from config import settings
from llm.base import LLMProvider, LLMResponse, Message


class OllamaProvider(LLMProvider):
    """
    Ollama local inference server.

    Calls the /api/chat endpoint with streaming disabled.
    Zero cost, zero data leaving the machine.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for m in messages:
            ollama_messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        latency_ms = (time.monotonic() - t0) * 1000

        data = response.json()
        content = data.get("message", {}).get("content", "")

        # Ollama eval counts
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            latency_ms=latency_ms,
            raw=data,
        )

    async def list_models(self) -> list[str]:
        """Return available model names from local Ollama installation."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
