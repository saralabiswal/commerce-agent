"""
CommerceAgent FastAPI application.
Exposes the agent pipeline as a REST API.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import audit, metrics, optimize
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up RAG retriever and verify DB schema."""
    logger.info("CommerceAgent API starting up...")

    # Warm up RAG (triggers ChromaDB connection + ingestion if needed)
    try:
        from rag.retrieval import get_retriever
        retriever = get_retriever()
        count = retriever.document_count
        logger.info(f"RAG ready: {count} chunks in ChromaDB")
    except Exception as e:
        logger.warning(f"RAG warmup failed (will retry on first request): {e}")

    # Verify DB exists
    try:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.init_db import ensure_db
        await ensure_db()
        logger.info("Database ready")
    except Exception as e:
        logger.warning(f"DB init failed: {e}")

    model_name = (
        settings.anthropic_model
        if settings.llm_provider == "anthropic"
        else settings.llm_provider
    )
    logger.info(f"Provider: {settings.llm_provider} | Model: {model_name}")
    logger.info("CommerceAgent API ready")

    yield

    logger.info("CommerceAgent API shutting down")


app = FastAPI(
    title="CommerceAgent API",
    description="Production-grade agentic AI platform for e-commerce content optimization",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(optimize.router, tags=["optimization"])
app.include_router(audit.router, tags=["audit"])
app.include_router(metrics.router, tags=["metrics"])


@app.get("/health", tags=["system"])
async def health():
    """Check provider connectivity and system status."""
    from api.schemas import HealthResponse
    from llm.factory import get_llm_provider

    provider = get_llm_provider()
    health_check = await provider.health_check()

    rag_count = 0
    try:
        from rag.retrieval import get_retriever
        rag_count = get_retriever().document_count
    except Exception:
        pass

    db_ok = True
    try:
        from observability.tracing import get_tracer
        await get_tracer().get_aggregate_metrics()
    except Exception:
        db_ok = False

    return HealthResponse(
        status=health_check["status"],
        provider=health_check["provider"],
        model=health_check["model"],
        provider_latency_ms=health_check.get("latency_ms"),
        rag_document_count=rag_count,
        database_ok=db_ok,
    )
