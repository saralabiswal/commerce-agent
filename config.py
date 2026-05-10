"""
Central configuration — reads from .env via pydantic-settings.
Import `settings` anywhere to get typed, validated config.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")

    # Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514", alias="ANTHROPIC_MODEL"
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")

    # HuggingFace
    hf_model: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct", alias="HF_MODEL"
    )
    hf_device: str = Field(default="cpu", alias="HF_DEVICE")
    hf_api_key: str = Field(default="", alias="HF_API_KEY")
    hf_api_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3", alias="HF_API_MODEL"
    )

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    # ── Observability ─────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="commerceagent", alias="LANGCHAIN_PROJECT")

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_mode: str = Field(default="vector", alias="RAG_MODE")
    chroma_persist_dir: str = Field(
        default="./data/chroma_db", alias="CHROMA_PERSIST_DIR"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/runs.db", alias="DATABASE_URL"
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_debug: bool = Field(default=False, alias="API_DEBUG")

    # ── Quality Gate ──────────────────────────────────────────────────────────
    quality_gate_threshold: float = Field(
        default=70.0, alias="QUALITY_GATE_THRESHOLD"
    )
    quality_gate_max_retries: int = Field(
        default=3, alias="QUALITY_GATE_MAX_RETRIES"
    )

    # ── MCP ───────────────────────────────────────────────────────────────────
    retailer_mcp_port: int = Field(default=8001, alias="RETAILER_MCP_PORT")
    catalog_mcp_port: int = Field(default=8002, alias="CATALOG_MCP_PORT")
    scoring_mcp_port: int = Field(default=8003, alias="SCORING_MCP_PORT")

    # ── Guardrails ────────────────────────────────────────────────────────────
    guardrails_enabled: bool = Field(default=True, alias="GUARDRAILS_ENABLED")
    hallucination_detection: bool = Field(
        default=True, alias="HALLUCINATION_DETECTION"
    )
    brand_safety_enabled: bool = Field(default=True, alias="BRAND_SAFETY_ENABLED")
    pii_detection_enabled: bool = Field(default=True, alias="PII_DETECTION_ENABLED")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
