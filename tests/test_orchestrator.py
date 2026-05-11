"""
Tests for the LangGraph orchestrator.

The goal is to prove graph-level control flow terminates safely even when one
agent keeps failing, because that is the highest-risk behavior in production.

Owner: Sarala Biswal
"""
from __future__ import annotations

import json

import pytest

from agents.models import AuditReport, CompetitorReport, GeneratedContent
from llm.base import LLMProvider, LLMResponse


@pytest.fixture(autouse=True)
def disable_langsmith_tracing(monkeypatch):
    """Keep graph tests local even when a developer .env enables LangSmith."""
    from config import settings
    from observability.tracing import get_tracer

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setattr(settings, "langchain_tracing_v2", False)
    monkeypatch.setattr(settings, "langchain_api_key", "")
    get_tracer.cache_clear()
    yield
    get_tracer.cache_clear()


class StaticProvider(LLMProvider):
    """Minimal provider used only to satisfy orchestrator construction."""

    @property
    def provider_name(self) -> str:
        """Return the static provider name for orchestrator tests."""
        return "test"

    @property
    def model_name(self) -> str:
        """Return the static model name for orchestrator tests."""
        return "test-model"

    async def complete(self, messages, system=None, temperature=0.3, max_tokens=2048):
        """Return an empty JSON response for unused LLM calls."""
        return LLMResponse(content=json.dumps({}), model=self.model_name)


class StaticAuditAgent:
    """Returns the smallest audit payload needed by downstream nodes."""

    async def run(self, sku: str, retailer: str) -> AuditReport:
        """Return a deterministic audit report."""
        return AuditReport(
            sku=sku,
            retailer=retailer,
            current_score=58.0,
            gap_analysis=[],
            priority_improvements=["Add Bluetooth version"],
            retailer_compliance={"compliant": True},
            character_counts={},
            score_breakdown={"title_compliance": 60.0},
            listing_snapshot={"category": "electronics"},
        )


class StaticCompetitorAgent:
    """Returns a valid competitor report without calling the LLM."""

    async def run(self, category: str, retailer: str, current_listing=None) -> CompetitorReport:
        """Return a deterministic competitor report."""
        return CompetitorReport(
            category=category,
            top_keywords=[],
            winning_patterns=["Use concrete specs"],
            content_gaps=["Missing Bluetooth version"],
            benchmark_scores={},
            competitor_count=1,
        )


class FailingGenerationAgent:
    """Always raises to exercise the quality gate failure path."""

    async def run(self, **kwargs):
        """Raise a generation failure for error-path testing."""
        raise RuntimeError("generation unavailable")


class LowQualityGenerationAgent:
    """Returns low-scoring content so the quality gate retries and then exits."""

    async def run(self, sku: str, retailer: str, retry_count: int = 0, **kwargs):
        """Return deterministic low-quality content for retry-path testing."""
        return GeneratedContent(
            sku=sku,
            retailer=retailer,
            title="Weak title",
            bullet_points=[],
            description="",
            backend_keywords="",
            quality_score=10.0,
            score_breakdown={"title_compliance": 10.0},
            compliance_check={"compliant": True},
            brand_safety={"passed": True},
            previous_score=58.0,
            improvement_delta=-48.0,
            reasoning="Low score test fixture",
            retry_count=retry_count,
        )


@pytest.mark.asyncio
async def test_graph_stops_after_generation_failures(monkeypatch):
    """A failing generation node should return an error result, not recurse forever."""
    from agents.orchestrator import CommerceAgentOrchestrator

    orchestrator = CommerceAgentOrchestrator(StaticProvider())
    orchestrator.audit_agent = StaticAuditAgent()
    orchestrator.competitor_agent = StaticCompetitorAgent()
    orchestrator.generation_agent = FailingGenerationAgent()

    async def noop_end_run(*args, **kwargs):
        """Disable persistence for this orchestrator test."""
        return None

    monkeypatch.setattr(orchestrator.tracer, "end_run", noop_end_run)

    result = await orchestrator.run("DEMO-SKU-001", "amazon", run_id="fail-generation")

    assert result.content.title == ""
    assert result.content.quality_score == 0
    assert any("generation_failed" in warning for warning in result.content.warnings)


@pytest.mark.asyncio
async def test_graph_exits_after_quality_gate_max_retries(monkeypatch):
    """Low-quality generated content should retry up to the configured cap and stop."""
    from agents.orchestrator import CommerceAgentOrchestrator
    from config import settings

    orchestrator = CommerceAgentOrchestrator(StaticProvider())
    orchestrator.audit_agent = StaticAuditAgent()
    orchestrator.competitor_agent = StaticCompetitorAgent()
    orchestrator.generation_agent = LowQualityGenerationAgent()

    async def noop_end_run(*args, **kwargs):
        """Disable persistence for this orchestrator test."""
        return None

    monkeypatch.setattr(orchestrator.tracer, "end_run", noop_end_run)

    result = await orchestrator.run("DEMO-SKU-001", "amazon", run_id="low-quality")

    assert result.content.retry_count == settings.quality_gate_max_retries
    assert result.content.quality_score == 10.0
    assert any("Quality gate not met" in warning for warning in result.content.warnings)


@pytest.mark.asyncio
async def test_progress_callback_reports_pipeline_stages(monkeypatch):
    """UI callers should receive meaningful progress events while the graph runs."""
    from agents.orchestrator import CommerceAgentOrchestrator

    events = []
    orchestrator = CommerceAgentOrchestrator(StaticProvider())
    orchestrator.audit_agent = StaticAuditAgent()
    orchestrator.competitor_agent = StaticCompetitorAgent()
    orchestrator.generation_agent = LowQualityGenerationAgent()

    async def noop_end_run(*args, **kwargs):
        """Disable persistence for this orchestrator test."""
        return None

    monkeypatch.setattr(orchestrator.tracer, "end_run", noop_end_run)

    await orchestrator.run(
        "DEMO-SKU-001",
        "amazon",
        run_id="progress",
        progress_callback=lambda stage, message: events.append((stage, message)),
    )

    stages = {stage for stage, _ in events}
    assert {"Pipeline", "ContentAuditAgent", "CompetitorAnalysisAgent"}.issubset(stages)
    assert "ContentGenerationAgent" in stages
    assert "Quality Gate" in stages
