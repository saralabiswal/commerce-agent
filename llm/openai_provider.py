"""
OpenAI provider — GPT-4 family.
Optional provider for teams already in the OpenAI ecosystem.

Owner: Sarala Biswal
"""
import time

from config import settings
from llm.base import LLMProvider, LLMResponse, Message

_COST_PER_M_INPUT = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4-turbo": 10.00,
}
_COST_PER_M_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-4-turbo": 30.00,
}


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider via the official Python SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the OpenAI async client for the selected chat model."""
        from openai import AsyncOpenAI

        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_model
        self._client = AsyncOpenAI(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        """Return the provider identifier used in configuration and logs."""
        return "openai"

    @property
    def model_name(self) -> str:
        """Return the active OpenAI model name."""
        return self._model

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate an OpenAI chat completion for normalized messages."""
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for m in messages:
            openai_messages.append({"role": m.role, "content": m.content})

        t0 = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            raw=response,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate OpenAI request cost from input and output token counts."""
        in_rate = _COST_PER_M_INPUT.get(self._model, 2.50)
        out_rate = _COST_PER_M_OUTPUT.get(self._model, 10.00)
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
