"""
CommerceAgent Orchestrator — LangGraph multi-agent workflow.

Workflow graph:
  START → [ContentAudit] → [CompetitorAnalysis] → [ContentGeneration] → [QualityGate]
                                                           ↑                    |
                                                           └── score < 70 ──────┘
                                                                       ↓
                                                                     [END]

Why LangGraph over chains: the quality gate loop and parallel-ready structure require
a graph, not a linear chain. State is explicit, transitions are debuggable (ADR-002).
"""
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from inspect import isawaitable
from typing import Any, TypedDict

from agents.competitor_analysis_agent import CompetitorAnalysisAgent
from agents.content_audit_agent import ContentAuditAgent
from agents.content_generation_agent import ContentGenerationAgent
from agents.models import AuditReport, CompetitorReport, GeneratedContent, OptimizationResult
from config import settings
from llm.base import LLMProvider
from observability.cost_tracker import CostTracker
from observability.tracing import get_tracer

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, str], Any]


class GraphState(TypedDict):
    """LangGraph requires a TypedDict for state typing."""
    sku: str
    retailer: str
    category: str
    audit_report: Any
    competitor_report: Any
    generated_content: Any
    quality_passed: bool
    retry_count: int
    retry_feedback: str
    run_id: str
    errors: list
    total_tokens: int
    total_cost_usd: float


class CommerceAgentOrchestrator:
    """
    Coordinates the three agents through a LangGraph stateful workflow.

    The orchestrator is the only component that knows about the full pipeline.
    Each agent only knows about its own inputs and outputs.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.audit_agent = ContentAuditAgent(provider)
        self.competitor_agent = CompetitorAnalysisAgent(provider)
        self.generation_agent = ContentGenerationAgent(provider)
        self.tracer = get_tracer()
        self.cost_tracker = CostTracker(provider)
        self._graph = self._build_graph()
        self._progress_callback: ProgressCallback | None = None

    def _build_graph(self):
        """
        Build the LangGraph workflow.

        Nodes: content_audit, competitor_analysis, content_generation, quality_gate
        Edges: conditional on quality_gate result
        """
        try:
            from langgraph.graph import END, StateGraph

            graph = StateGraph(GraphState)

            # Add nodes
            graph.add_node("content_audit", self._node_content_audit)
            graph.add_node("competitor_analysis", self._node_competitor_analysis)
            graph.add_node("content_generation", self._node_content_generation)
            graph.add_node("quality_gate", self._node_quality_gate)

            # Set entry point
            graph.set_entry_point("content_audit")

            # Linear edges
            graph.add_edge("content_audit", "competitor_analysis")
            graph.add_edge("competitor_analysis", "content_generation")
            graph.add_edge("content_generation", "quality_gate")

            # Conditional edge from quality gate: pass → END, fail → retry
            graph.add_conditional_edges(
                "quality_gate",
                self._should_retry,
                {
                    "retry": "content_generation",
                    "end": END,
                },
            )

            return graph.compile()

        except ImportError:
            logger.warning("LangGraph not available — falling back to sequential execution")
            return None

    async def run(
        self,
        sku: str,
        retailer: str = "amazon",
        run_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> OptimizationResult:
        """
        Run the full optimization pipeline for a SKU.

        Args:
            sku: Product SKU to optimize
            retailer: Target retailer (amazon | walmart)
            run_id: Optional run ID for tracing. Auto-generated if not provided.

        Returns:
            OptimizationResult with audit, competitor analysis, and generated content.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        start_time = datetime.now(UTC)
        previous_progress_callback = self._progress_callback
        self._progress_callback = progress_callback

        logger.info(f"[{run_id}] Starting optimization: {sku} on {retailer}")
        self.tracer.start_run(run_id, sku, retailer, self.provider.provider_name)

        initial_state: GraphState = {
            "sku": sku,
            "retailer": retailer,
            "category": "electronics",  # Default; ContentAuditAgent will update
            "audit_report": None,
            "competitor_report": None,
            "generated_content": None,
            "quality_passed": False,
            "retry_count": 0,
            "retry_feedback": "",
            "run_id": run_id,
            "errors": [],
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

        try:
            await self._emit_progress("Pipeline", f"Started run {run_id} for {sku} on {retailer}")
            if self._graph is not None:
                # Use LangGraph execution
                final_state = await self._graph.ainvoke(initial_state)
            else:
                # Fallback sequential execution (same logic, no graph)
                final_state = await self._run_sequential(initial_state)

        except Exception as e:
            logger.error(f"[{run_id}] Pipeline failed: {e}", exc_info=True)
            final_state = {**initial_state, "errors": [str(e)]}
            await self._emit_progress("Pipeline", f"Failed: {e}")

        end_time = datetime.now(UTC)
        latency_ms = (end_time - start_time).total_seconds() * 1000

        result = self._build_result(final_state, run_id, latency_ms)
        await self._emit_progress("Pipeline", f"Finished in {latency_ms / 1000:.1f}s")

        # Persist run to database
        await self.tracer.end_run(run_id, result)
        self._progress_callback = previous_progress_callback

        return result

    async def _emit_progress(self, stage: str, message: str) -> None:
        """Send a progress event to optional UI callers without coupling to Streamlit."""
        if self._progress_callback is None:
            return
        try:
            result = self._progress_callback(stage, message)
            if isawaitable(result):
                await result
        except Exception as e:
            logger.debug(f"Progress callback failed: {e}")

    async def _node_content_audit(self, state: GraphState) -> GraphState:
        """LangGraph node: run ContentAuditAgent."""
        try:
            await self._emit_progress(
                "ContentAuditAgent",
                "Fetching the current listing and scoring it against retailer rules",
            )
            audit = await self.audit_agent.run(state["sku"], state["retailer"])
            # Update category from listing data
            category = audit.listing_snapshot.get("category", state["category"])
            await self._emit_progress(
                "ContentAuditAgent",
                f"Audit complete: score {audit.current_score:.1f}/100, category {category}",
            )
            return {
                **state,
                "audit_report": audit,
                "category": category,
            }
        except Exception as e:
            logger.error(f"ContentAuditAgent failed: {e}", exc_info=True)
            return {**state, "errors": state["errors"] + [f"audit_failed: {e}"]}

    async def _node_competitor_analysis(self, state: GraphState) -> GraphState:
        """LangGraph node: run CompetitorAnalysisAgent."""
        try:
            await self._emit_progress(
                "CompetitorAnalysisAgent",
                "Loading competitor listings, keywords, and search-volume signals",
            )
            audit: AuditReport = state["audit_report"]
            current_listing = audit.listing_snapshot if audit else {}

            competitors = await self.competitor_agent.run(
                category=state["category"],
                retailer=state["retailer"],
                current_listing=current_listing,
            )
            await self._emit_progress(
                "CompetitorAnalysisAgent",
                f"Competitor analysis complete: {competitors.competitor_count} listings analyzed",
            )
            return {**state, "competitor_report": competitors}
        except Exception as e:
            logger.error(f"CompetitorAnalysisAgent failed: {e}", exc_info=True)
            return {**state, "errors": state["errors"] + [f"competitor_failed: {e}"]}

    async def _node_content_generation(self, state: GraphState) -> GraphState:
        """LangGraph node: run ContentGenerationAgent."""
        try:
            await self._emit_progress(
                "ContentGenerationAgent",
                (
                    "Retrieving product specs and RAG guidance, then generating content "
                    f"(attempt {state['retry_count'] + 1})"
                ),
            )
            audit: AuditReport = state["audit_report"]
            competitors: CompetitorReport = state["competitor_report"]

            if not audit or not competitors:
                raise ValueError("Missing audit or competitor report for generation")

            content = await self.generation_agent.run(
                sku=state["sku"],
                retailer=state["retailer"],
                audit=audit,
                competitors=competitors,
                retry_feedback=state["retry_feedback"],
                retry_count=state["retry_count"],
            )
            await self._emit_progress(
                "ContentGenerationAgent",
                f"Generated candidate content with score {content.quality_score:.1f}/100",
            )
            return {**state, "generated_content": content}
        except Exception as e:
            logger.error(f"ContentGenerationAgent failed: {e}", exc_info=True)
            return {**state, "errors": state["errors"] + [f"generation_failed: {e}"]}

    async def _node_quality_gate(self, state: GraphState) -> GraphState:
        """
        LangGraph node: quality gate check.

        If content score >= threshold → pass.
        If score < threshold and retries remaining → build feedback for retry.
        If max retries reached → pass with warning.
        """
        content: GeneratedContent | None = state.get("generated_content")
        await self._emit_progress("Quality Gate", "Checking score, compliance, and retry policy")
        if not content:
            retry_count = state["retry_count"]
            max_retries = settings.quality_gate_max_retries
            if retry_count >= max_retries:
                logger.warning(
                    f"[{state['run_id']}] Quality gate stopping after "
                    f"{max_retries} failed generation attempts"
                )
                return {**state, "quality_passed": True}

            logger.warning(
                f"[{state['run_id']}] No generated content available; "
                f"retrying generation ({retry_count + 1}/{max_retries})"
            )
            return {
                **state,
                "quality_passed": False,
                "retry_count": retry_count + 1,
                "retry_feedback": (
                    "Previous generation attempt failed. Return valid listing content."
                ),
            }

        threshold = settings.quality_gate_threshold
        score = content.quality_score
        retry_count = state["retry_count"]
        max_retries = settings.quality_gate_max_retries

        logger.info(
            f"[{state['run_id']}] Quality gate: score={score:.1f}, "
            f"threshold={threshold}, retries={retry_count}/{max_retries}"
        )

        if score >= threshold:
            logger.info(f"[{state['run_id']}] Quality gate PASSED (score={score:.1f})")
            await self._emit_progress(
                "Quality Gate",
                f"Passed: score {score:.1f}/100 meets threshold {threshold}",
            )
            return {**state, "quality_passed": True}

        if retry_count >= max_retries:
            logger.warning(
                f"[{state['run_id']}] Max retries ({max_retries}) reached. "
                f"Returning best available output (score={score:.1f})"
            )
            if content.warnings is None:
                content.warnings = []
            content.warnings.append(
                f"Quality gate not met after {max_retries} attempts. "
                f"Score: {score:.1f}/{threshold}"
            )
            await self._emit_progress(
                "Quality Gate",
                f"Max retries reached; returning best score {score:.1f}/100",
            )
            return {**state, "quality_passed": True}  # Force exit

        # Build specific improvement feedback for next attempt
        feedback = self._build_retry_feedback(content)
        logger.info(f"[{state['run_id']}] Quality gate FAILED — retry {retry_count + 1}")
        await self._emit_progress(
            "Quality Gate",
            f"Score {score:.1f}/100 is below {threshold}; retrying with targeted feedback",
        )

        return {
            **state,
            "quality_passed": False,
            "retry_count": retry_count + 1,
            "retry_feedback": feedback,
        }

    def _should_retry(self, state: GraphState) -> str:
        """LangGraph conditional edge: 'end' or 'retry'."""
        if state.get("quality_passed", False):
            return "end"
        return "retry"

    def _build_retry_feedback(self, content: GeneratedContent) -> str:
        """Build specific improvement instructions for the retry attempt."""
        lines = [
            (
                f"Previous score: {content.quality_score:.1f}/100 "
                f"(target: {settings.quality_gate_threshold})"
            ),
            "",
            "Specific issues to fix:",
        ]

        # Score-based feedback
        for dim, score in sorted(content.score_breakdown.items(), key=lambda x: x[1]):
            if score < 70:
                lines.append(f"  - {dim.replace('_', ' ')}: {score:.0f}/100 — needs improvement")

        # Compliance feedback
        if not content.compliance_check.get("compliant"):
            for v in content.compliance_check.get("violations", []):
                lines.append(f"  - COMPLIANCE VIOLATION: {v}")

        # Brand safety feedback
        if not content.brand_safety.get("passed"):
            for flag in content.brand_safety.get("flags", []):
                lines.append(f"  - BRAND SAFETY: Remove '{flag['term']}' ({flag['severity']})")

        return "\n".join(lines)

    async def _run_sequential(self, state: GraphState) -> GraphState:
        """
        Fallback sequential execution when LangGraph is not available.
        Identical logic to the graph — just called directly.
        """
        state = await self._node_content_audit(state)
        state = await self._node_competitor_analysis(state)

        max_retries = settings.quality_gate_max_retries
        for attempt in range(max_retries + 1):
            state = await self._node_content_generation(state)
            state = await self._node_quality_gate(state)
            if self._should_retry(state) == "end":
                break

        return state

    def _build_result(
        self, state: GraphState, run_id: str, latency_ms: float
    ) -> OptimizationResult:
        """Assemble the final OptimizationResult from graph state."""
        audit: AuditReport = state.get("audit_report")
        competitors: CompetitorReport = state.get("competitor_report")
        content: GeneratedContent = state.get("generated_content")

        # Fallback objects if agents failed
        if not audit:
            audit = AuditReport(
                sku=state["sku"], retailer=state["retailer"],
                current_score=0, gap_analysis=[], priority_improvements=[],
                retailer_compliance={}, character_counts={}, score_breakdown={},
                listing_snapshot={},
            )
        if not competitors:
            competitors = CompetitorReport(
                category=state["category"], top_keywords=[], winning_patterns=[],
                content_gaps=[], benchmark_scores={}, competitor_count=0,
            )
        if not content:
            content = GeneratedContent(
                sku=state["sku"], retailer=state["retailer"],
                title="", bullet_points=[], description="", backend_keywords="",
                quality_score=0, score_breakdown={}, compliance_check={},
                brand_safety={}, previous_score=0, improvement_delta=0,
                reasoning="Pipeline failed — see errors",
                warnings=state.get("errors", []),
            )

        estimated_cost = self.cost_tracker.estimate_run_cost(
            state.get("total_tokens", 0)
        )

        return OptimizationResult(
            run_id=run_id,
            sku=state["sku"],
            retailer=state["retailer"],
            audit=audit,
            competitors=competitors,
            content=content,
            total_latency_ms=latency_ms,
            total_tokens=state.get("total_tokens", 0),
            estimated_cost_usd=estimated_cost,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            timestamp=datetime.now(UTC).isoformat(),
        )
