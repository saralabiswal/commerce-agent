"""
Anthropic Claude provider.
Production default — best reasoning quality for content generation.

Owner: Sarala Biswal
"""
import time
from typing import Any

import anthropic

from config import settings
from llm.base import LLMProvider, LLMResponse, Message

# Pricing per million tokens (as of May 2026 — update as needed)
_COST_PER_M_INPUT = {
    "claude-sonnet-4-20250514": 3.00,
    "claude-opus-4-20250514": 15.00,
    "claude-haiku-4-5-20251001": 0.25,
}
_COST_PER_M_OUTPUT = {
    "claude-sonnet-4-20250514": 15.00,
    "claude-opus-4-20250514": 75.00,
    "claude-haiku-4-5-20251001": 1.25,
}


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude via the official Python SDK.

    Uses async client for non-blocking I/O in FastAPI / LangGraph contexts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the Anthropic async client for the selected model."""
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model or settings.anthropic_model
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        """Return the provider identifier used in configuration and logs."""
        return "anthropic"

    @property
    def model_name(self) -> str:
        """Return the active Claude model name."""
        return self._model

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a Claude completion for normalized chat messages."""
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system

        t0 = time.monotonic()
        response = await self._client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            raw=response,
        )

    @property
    def estimated_cost_usd(self) -> float:
        """Return the default estimate when per-call token counts are unavailable."""
        return 0.0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate Anthropic request cost from input and output token counts."""
        in_rate = _COST_PER_M_INPUT.get(self._model, 3.00)
        out_rate = _COST_PER_M_OUTPUT.get(self._model, 15.00)
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
