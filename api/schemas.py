"""
Pydantic request/response models for the FastAPI layer.
These are the public API contracts — versioned separately from internal agent models.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request models ─────────────────────────────────────────────────────────────

class AuditRequest(BaseModel):
    sku: str = Field(..., description="Product SKU or ASIN to audit")
    retailer: str = Field(default="amazon", description="Target retailer: amazon | walmart")


class AnalyzeRequest(BaseModel):
    sku: str = Field(..., description="Product SKU")
    category: str = Field(default="electronics", description="Product category")
    retailer: str = Field(default="amazon")


class GenerateRequest(BaseModel):
    sku: str
    retailer: str = "amazon"
    mode: str = Field(
        default="full",
        description="Generation mode: full (audit + competitors + generate) | generate_only"
    )


class OptimizeRequest(BaseModel):
    sku: str = Field(..., description="Product SKU to fully optimize")
    retailer: str = Field(default="amazon")


class FeedbackRequest(BaseModel):
    run_id: str
    sku: str
    rating: int = Field(..., ge=-1, le=1, description="1=thumbs up, -1=thumbs down")
    comment: str = Field(default="")
    field: str = Field(default="overall")


# ── Response models ────────────────────────────────────────────────────────────

class GapResponse(BaseModel):
    field: str
    issue: str
    severity: str
    current_value: str
    recommendation: str


class AuditResponse(BaseModel):
    sku: str
    retailer: str
    current_score: float
    grade: str
    gap_count: int
    priority_improvements: list[str]
    compliance_passed: bool
    character_counts: dict
    score_breakdown: dict


class KeywordResponse(BaseModel):
    term: str
    monthly_volume: int
    competition: str
    present_in_listing: bool


class CompetitorResponse(BaseModel):
    category: str
    top_keywords: list[KeywordResponse]
    winning_patterns: list[str]
    content_gaps: list[str]
    benchmark_scores: dict
    competitor_count: int


class ContentResponse(BaseModel):
    sku: str
    retailer: str
    title: str
    bullet_points: list[str]
    description: str
    backend_keywords: str
    quality_score: float
    grade: str
    previous_score: float
    improvement_delta: float
    compliance_passed: bool
    brand_safety_passed: bool
    retry_count: int
    warnings: list[str]
    reasoning: str


class OptimizeResponse(BaseModel):
    run_id: str
    sku: str
    retailer: str
    audit: AuditResponse
    competitors: CompetitorResponse
    content: ContentResponse
    total_latency_ms: float
    estimated_cost_usd: float
    provider: str
    model: str
    timestamp: str


class RunHistoryItem(BaseModel):
    run_id: str
    sku: str
    retailer: str
    provider: str
    model: str
    score_before: float
    score_after: float
    improvement_delta: float
    latency_ms: float
    estimated_cost_usd: float
    quality_passed: bool
    timestamp: str


class MetricsResponse(BaseModel):
    total_runs: int
    avg_quality_score: float
    avg_improvement: float
    avg_latency_ms: float
    total_cost_usd: float
    quality_gate_pass_rate: float


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    provider_latency_ms: float | None
    rag_document_count: int
    database_ok: bool


# ── Conversion helpers ─────────────────────────────────────────────────────────

def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def audit_to_response(audit) -> AuditResponse:
    return AuditResponse(
        sku=audit.sku,
        retailer=audit.retailer,
        current_score=audit.current_score,
        grade=_score_to_grade(audit.current_score),
        gap_count=len(audit.gap_analysis),
        priority_improvements=audit.priority_improvements,
        compliance_passed=audit.retailer_compliance.get("compliant", True),
        character_counts=audit.character_counts,
        score_breakdown=audit.score_breakdown,
    )


def competitor_to_response(competitors) -> CompetitorResponse:
    return CompetitorResponse(
        category=competitors.category,
        top_keywords=[
            KeywordResponse(
                term=k.term,
                monthly_volume=k.monthly_volume,
                competition=k.competition,
                present_in_listing=k.present_in_listing,
            )
            for k in competitors.top_keywords
        ],
        winning_patterns=competitors.winning_patterns,
        content_gaps=competitors.content_gaps,
        benchmark_scores=competitors.benchmark_scores,
        competitor_count=competitors.competitor_count,
    )


def content_to_response(content) -> ContentResponse:
    return ContentResponse(
        sku=content.sku,
        retailer=content.retailer,
        title=content.title,
        bullet_points=content.bullet_points,
        description=content.description,
        backend_keywords=content.backend_keywords,
        quality_score=content.quality_score,
        grade=_score_to_grade(content.quality_score),
        previous_score=content.previous_score,
        improvement_delta=content.improvement_delta,
        compliance_passed=content.compliance_check.get("compliant", True),
        brand_safety_passed=content.brand_safety.get("passed", True),
        retry_count=content.retry_count,
        warnings=content.warnings or [],
        reasoning=content.reasoning,
    )


def result_to_response(result) -> OptimizeResponse:
    return OptimizeResponse(
        run_id=result.run_id,
        sku=result.sku,
        retailer=result.retailer,
        audit=audit_to_response(result.audit),
        competitors=competitor_to_response(result.competitors),
        content=content_to_response(result.content),
        total_latency_ms=result.total_latency_ms,
        estimated_cost_usd=result.estimated_cost_usd,
        provider=result.provider,
        model=result.model,
        timestamp=result.timestamp,
    )
