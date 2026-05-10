"""
HuggingFace providers — local model weights and Inference API.

huggingface_local: Downloads and runs model weights locally.
                   Good for air-gapped or GPU-equipped environments.
huggingface_api:   Calls HF Inference API. No GPU needed, but requires API key.
"""
import time
from typing import Any

from config import settings
from llm.base import LLMProvider, LLMResponse, Message


class HuggingFaceLocalProvider(LLMProvider):
    """
    Runs a HuggingFace model locally using transformers + pipeline.
    Downloads model weights on first run (~1-4GB depending on model).

    Best models for this use case:
    - microsoft/Phi-3-mini-4k-instruct  (fast, small)
    - mistralai/Mistral-7B-Instruct-v0.2 (better quality, needs more RAM)
    """

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
    ):
        self._model_name = model or settings.hf_model
        self._device = device or settings.hf_device
        self._pipeline: Any = None  # lazy-loaded on first call

    def _load_pipeline(self):
        """Lazy-load the model pipeline on first inference call."""
        import torch
        from transformers import pipeline

        torch_dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._pipeline = pipeline(
            "text-generation",
            model=self._model_name,
            device=self._device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    @property
    def provider_name(self) -> str:
        return "huggingface_local"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        import asyncio

        if self._pipeline is None:
            self._load_pipeline()

        # Build prompt from messages
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        for m in messages:
            chat_messages.append({"role": m.role, "content": m.content})

        # Run in thread pool to avoid blocking the event loop
        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._pipeline(
                chat_messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
            ),
        )
        latency_ms = (time.monotonic() - t0) * 1000

        content = result[0]["generated_text"]
        if isinstance(content, list):
            content = content[-1].get("content", "")

        return LLMResponse(
            content=content,
            model=self._model_name,
            latency_ms=latency_ms,
            raw=result,
        )


class HuggingFaceAPIProvider(LLMProvider):
    """
    HuggingFace Inference API — no local GPU, just an API key.
    Calls the serverless inference endpoint via httpx.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._api_key = api_key or settings.hf_api_key
        self._model_name = model or settings.hf_api_model
        self._base_url = "https://api-inference.huggingface.co/models"

    @property
    def provider_name(self) -> str:
        return "huggingface_api"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        import httpx

        # Build text prompt (HF Inference API uses raw text, not chat format)
        parts = []
        if system:
            parts.append(f"<|system|>\n{system}")
        for m in messages:
            tag = "<|user|>" if m.role == "user" else "<|assistant|>"
            parts.append(f"{tag}\n{m.content}")
        parts.append("<|assistant|>")
        prompt = "\n".join(parts)

        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/{self._model_name}",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        latency_ms = (time.monotonic() - t0) * 1000

        data = response.json()
        content = data[0].get("generated_text", "") if isinstance(data, list) else ""

        return LLMResponse(
            content=content,
            model=self._model_name,
            latency_ms=latency_ms,
            raw=data,
        )
