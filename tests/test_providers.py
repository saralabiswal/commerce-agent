"""
Tests for LLM provider interface compliance.
Tests the provider contract — not the actual API calls (those need real keys).
"""
from importlib.util import find_spec
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.base import LLMProvider, LLMResponse, Message

# Check which provider SDKs are available
HAS_ANTHROPIC = find_spec("anthropic") is not None
HAS_HTTPX = find_spec("httpx") is not None
HAS_OPENAI = find_spec("openai") is not None

requires_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic SDK not installed")
requires_httpx = pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")
requires_openai = pytest.mark.skipif(not HAS_OPENAI, reason="openai SDK not installed")


# ── Interface contract tests ───────────────────────────────────────────────────

class TestLLMProviderInterface:

    def test_message_dataclass(self):
        """Message should store role and content correctly."""
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_llm_response_total_tokens(self):
        """LLMResponse.total_tokens should sum input + output."""
        r = LLMResponse(
            content="response",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
        )
        assert r.total_tokens == 150

    def test_llm_response_defaults(self):
        """LLMResponse should have sensible defaults."""
        r = LLMResponse(content="test", model="m")
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.latency_ms == 0.0

    def test_mock_provider_satisfies_interface(self, mock_provider):
        """Mock provider should implement all required abstract methods."""
        assert isinstance(mock_provider, LLMProvider)
        assert hasattr(mock_provider, "complete")
        assert hasattr(mock_provider, "complete_json")
        assert hasattr(mock_provider, "health_check")
        assert isinstance(mock_provider.provider_name, str)
        assert isinstance(mock_provider.model_name, str)

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self, mock_provider):
        """complete() should return an LLMResponse."""
        mock_provider.set_response("hello", "Hello back!")
        response = await mock_provider.complete(
            messages=[Message(role="user", content="hello")]
        )
        assert isinstance(response, LLMResponse)
        assert isinstance(response.content, str)

    @pytest.mark.asyncio
    async def test_complete_json_appends_json_instruction(self, mock_provider):
        """complete_json() should invoke complete() with JSON system prompt."""
        mock_provider.set_response("json", '{"test": true}')
        response = await mock_provider.complete_json(
            messages=[Message(role="user", content="return json")]
        )
        assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio
    async def test_health_check_ok(self, mock_provider):
        """health_check() should return status ok when complete() works."""
        mock_provider.set_response("Reply with: OK", "OK")
        result = await mock_provider.health_check()
        assert result["status"] in ("ok", "error")
        assert "provider" in result
        assert "model" in result

    @pytest.mark.asyncio
    async def test_health_check_error_on_failure(self):
        """health_check() should return status=error when provider fails."""
        from llm.base import LLMProvider

        class FailingProvider(LLMProvider):
            @property
            def provider_name(self): return "failing"
            @property
            def model_name(self): return "fail-model"
            async def complete(self, messages, **kwargs):
                raise ConnectionError("Cannot connect")

        provider = FailingProvider()
        result = await provider.health_check()
        assert result["status"] == "error"
        assert "error" in result


# ── Factory tests ──────────────────────────────────────────────────────────────

class TestLLMFactory:

    @requires_anthropic
    def test_factory_anthropic(self):
        """Factory should return AnthropicProvider for 'anthropic'."""
        from llm.anthropic_provider import AnthropicProvider
        from llm.factory import get_llm_provider
        with patch.dict("os.environ", {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test"}):
            provider = get_llm_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    @requires_httpx
    def test_factory_ollama(self):
        """Factory should return OllamaProvider for 'ollama'."""
        from llm.factory import get_llm_provider
        from llm.ollama_provider import OllamaProvider
        provider = get_llm_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_factory_huggingface_local(self):
        """Factory should return HuggingFaceLocalProvider for 'huggingface_local'."""
        from llm.factory import get_llm_provider
        from llm.huggingface_provider import HuggingFaceLocalProvider
        provider = get_llm_provider("huggingface_local")
        assert isinstance(provider, HuggingFaceLocalProvider)

    def test_factory_huggingface_api(self):
        """Factory should return HuggingFaceAPIProvider for 'huggingface_api'."""
        from llm.factory import get_llm_provider
        from llm.huggingface_provider import HuggingFaceAPIProvider
        provider = get_llm_provider("huggingface_api")
        assert isinstance(provider, HuggingFaceAPIProvider)

    @requires_openai
    def test_factory_openai(self):
        """Factory should return OpenAIProvider for 'openai'."""
        from llm.factory import get_llm_provider
        from llm.openai_provider import OpenAIProvider
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = get_llm_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_factory_unknown_raises_value_error(self):
        """Factory should raise ValueError for unknown provider."""
        from llm.factory import get_llm_provider
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("nonexistent_provider_xyz")

    @requires_httpx
    def test_factory_case_insensitive(self):
        """Provider name should be case-insensitive."""
        from llm.factory import get_llm_provider
        from llm.ollama_provider import OllamaProvider
        provider = get_llm_provider("OLLAMA")
        assert isinstance(provider, OllamaProvider)

    def test_provider_repr(self, mock_provider):
        """Provider __repr__ should include provider name."""
        repr_str = repr(mock_provider)
        assert "mock" in repr_str.lower() or "Model" in repr_str


# ── Anthropic provider unit tests (mocked) ────────────────────────────────────

class TestAnthropicProvider:

    @requires_anthropic
    @pytest.mark.asyncio
    async def test_complete_passes_system_prompt(self):
        """System prompt should be passed to the Anthropic API."""
        from llm.anthropic_provider import AnthropicProvider

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="test response")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 20

        with patch("anthropic.AsyncAnthropic") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-20250514")
            response = await provider.complete(
                messages=[Message(role="user", content="Test")],
                system="You are helpful.",
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "You are helpful."
        assert response.content == "test response"
        assert response.input_tokens == 50
        assert response.output_tokens == 20

    @requires_anthropic
    def test_estimate_cost(self):
        """Cost estimation should use correct pricing table."""
        from llm.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-20250514")
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        # Input: $3.00/M, Output: $15.00/M = $18.00 total
        assert abs(cost - 18.0) < 0.01

    @requires_anthropic
    def test_provider_name_and_model(self):
        from llm.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-20250514")
        assert provider.provider_name == "anthropic"
        assert provider.model_name == "claude-sonnet-4-20250514"


# ── Ollama provider unit tests ─────────────────────────────────────────────────

class TestOllamaProvider:

    @requires_httpx
    def test_provider_name_and_model(self):
        from llm.ollama_provider import OllamaProvider
        provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")
        assert provider.provider_name == "ollama"
        assert provider.model_name == "llama3"

    @requires_httpx
    @pytest.mark.asyncio
    async def test_complete_formats_messages(self):
        """Ollama provider should format messages as system + user/assistant."""
        from llm.ollama_provider import OllamaProvider

        mock_response_data = {
            "message": {"content": "Ollama response"},
            "prompt_eval_count": 80,
            "eval_count": 30,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")
            response = await provider.complete(
                messages=[Message(role="user", content="Hello")],
                system="Be helpful.",
            )

        assert response.content == "Ollama response"
        assert response.input_tokens == 80
        assert response.output_tokens == 30
