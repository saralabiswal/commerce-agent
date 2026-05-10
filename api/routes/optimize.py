"""
/optimize — full pipeline: audit + competitor analysis + content generation.
The primary endpoint for the CommerceAgent platform.
"""
from fastapi import APIRouter, HTTPException

from agents.orchestrator import CommerceAgentOrchestrator
from api.schemas import OptimizeRequest, OptimizeResponse, result_to_response
from llm.factory import get_cached_provider

router = APIRouter()


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_listing(request: OptimizeRequest):
    """
    Run the full CommerceAgent pipeline for a product SKU.

    Executes in sequence:
    1. ContentAuditAgent — analyzes current listing quality
    2. CompetitorAnalysisAgent — extracts winning patterns
    3. ContentGenerationAgent — generates optimized content (with quality gate loop)

    Returns audit findings, competitor intelligence, and generated content
    with quality scores and compliance checks.
    """
    try:
        provider = get_cached_provider()
        orchestrator = CommerceAgentOrchestrator(provider)
        result = await orchestrator.run(sku=request.sku, retailer=request.retailer)
        return result_to_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Retrieve a specific run by ID."""
    from observability.tracing import get_tracer
    run = await get_tracer().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run
