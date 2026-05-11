"""
Shared Pydantic models for agent inputs, outputs, and orchestrator state.
These are the contracts between agents — changing them requires updating all consumers.

Owner: Sarala Biswal
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Gap:
    """Single content issue found during listing audit."""
    field: str          # "title" | "bullets" | "description" | "keywords"
    issue: str          # Human-readable description of the issue
    severity: str       # "critical" | "high" | "medium" | "low"
    current_value: str  # What's there now
    recommendation: str # What should be there


@dataclass
class AuditReport:
    """Structured audit output for one SKU and retailer."""
    sku: str
    retailer: str
    current_score: float
    gap_analysis: list[Gap]
    priority_improvements: list[str]
    retailer_compliance: dict[str, Any]
    character_counts: dict[str, dict[str, int]]   # {field: {current, allowed}}
    score_breakdown: dict[str, float]
    listing_snapshot: dict[str, Any]


@dataclass
class Keyword:
    """Keyword candidate with search demand and listing presence metadata."""
    term: str
    monthly_volume: int
    competition: str        # "high" | "medium" | "low"
    present_in_listing: bool


@dataclass
class CompetitorReport:
    """Competitive benchmark findings for a product category."""
    category: str
    top_keywords: list[Keyword]
    winning_patterns: list[str]     # Title/bullet patterns that work
    content_gaps: list[str]         # Things competitors do that we don't
    benchmark_scores: dict[str, float]
    competitor_count: int


@dataclass
class GeneratedContent:
    """Optimized listing content plus validation and scoring metadata."""
    sku: str
    retailer: str
    title: str
    bullet_points: list[str]
    description: str
    backend_keywords: str
    quality_score: float
    score_breakdown: dict[str, float]
    compliance_check: dict[str, Any]
    brand_safety: dict[str, Any]
    previous_score: float
    improvement_delta: float
    reasoning: str
    retry_count: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Full pipeline result returned by the orchestrator."""
    run_id: str
    sku: str
    retailer: str
    audit: AuditReport
    competitors: CompetitorReport
    content: GeneratedContent
    total_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    provider: str
    model: str
    timestamp: str


# ── LangGraph state ───────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """
    Shared state passed between nodes in the LangGraph workflow.
    LangGraph passes this dict through every node — nodes read what they need
    and write their outputs back to the same dict.
    """
    sku: str = ""
    retailer: str = "amazon"
    category: str = "electronics"

    # Populated by ContentAuditAgent
    audit_report: AuditReport | None = None

    # Populated by CompetitorAnalysisAgent
    competitor_report: CompetitorReport | None = None

    # Populated by ContentGenerationAgent
    generated_content: GeneratedContent | None = None

    # Quality gate state
    quality_passed: bool = False
    retry_count: int = 0
    retry_feedback: str = ""

    # Run metadata
    run_id: str = ""
    errors: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
