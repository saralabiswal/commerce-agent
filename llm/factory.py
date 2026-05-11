"""
LLM provider factory.
Single entry point for creating the configured provider.

Usage:
    from llm.factory import get_llm_provider
    provider = get_llm_provider()   # reads LLM_PROVIDER from .env

Owner: Sarala Biswal
"""
from functools import lru_cache

from config import settings
from llm.base import LLMProvider


def get_llm_provider(provider_name: str | None = None) -> LLMProvider:
    """
    Instantiate and return the configured LLM provider.

    Args:
        provider_name: Override the LLM_PROVIDER env var. Useful in tests.

    Returns:
        A concrete LLMProvider implementation.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    name = (provider_name or settings.llm_provider).lower().strip()

    if name == "anthropic":
        from llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    elif name == "ollama":
        from llm.ollama_provider import OllamaProvider
        return OllamaProvider()

    elif name == "huggingface_local":
        from llm.huggingface_provider import HuggingFaceLocalProvider
        return HuggingFaceLocalProvider()

    elif name == "huggingface_api":
        from llm.huggingface_provider import HuggingFaceAPIProvider
        return HuggingFaceAPIProvider()

    elif name == "openai":
        from llm.openai_provider import OpenAIProvider
        return OpenAIProvider()

    else:
        supported = ["anthropic", "ollama", "huggingface_local", "huggingface_api", "openai"]
        raise ValueError(
            f"Unknown LLM provider: '{name}'. "
            f"Supported providers: {supported}. "
            f"Set LLM_PROVIDER in your .env file."
        )


@lru_cache(maxsize=1)
def get_cached_provider() -> LLMProvider:
    """
    Cached singleton provider for use in long-running services.
    Use get_llm_provider() in tests where you need fresh instances.
    """
    return get_llm_provider()
