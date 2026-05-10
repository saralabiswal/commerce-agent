"""
Abstract base class for all LLM providers.
Every provider implements this interface — zero agent code changes when switching models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate — override per provider for accuracy."""
        return 0.0


class LLMProvider(ABC):
    """
    Model-agnostic LLM interface. Implement this to add a new provider.

    Design principle: every agent receives the provider via dependency injection.
    The agent never imports a concrete provider class — only this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. 'anthropic', 'ollama'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate a completion from a list of messages.

        Args:
            messages: Conversation history (user/assistant turns).
            system: System prompt injected before messages.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMResponse with content, token counts, and latency.
        """
        ...

    async def complete_json(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Complete with a strong JSON output instruction appended to system prompt.
        Override in providers that support native JSON mode.
        """
        json_system = (system or "") + (
            "\n\nIMPORTANT: Respond with valid JSON only. "
            "No markdown fences, no explanation, no preamble. "
            "Output must be parseable by json.loads()."
        )
        return await self.complete(
            messages=messages,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def health_check(self) -> dict[str, Any]:
        """
        Verify the provider is reachable and the model is available.
        Returns a status dict: {"status": "ok"|"error", "provider": ..., "model": ...}
        """
        try:
            response = await self.complete(
                messages=[Message(role="user", content="Reply with: OK")],
                max_tokens=10,
            )
            return {
                "status": "ok",
                "provider": self.provider_name,
                "model": self.model_name,
                "latency_ms": response.latency_ms,
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": self.provider_name,
                "model": self.model_name,
                "error": str(e),
            }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
