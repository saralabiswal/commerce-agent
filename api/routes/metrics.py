"""
/metrics — aggregate quality metrics and run history.

Owner: Sarala Biswal
"""
from fastapi import APIRouter, HTTPException

from api.schemas import FeedbackRequest, MetricsResponse, RunHistoryItem

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Aggregate quality and performance metrics across all runs."""
    from observability.tracing import get_tracer
    try:
        data = await get_tracer().get_aggregate_metrics()
        return MetricsResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs", response_model=list[RunHistoryItem])
async def list_runs(limit: int = 20):
    """List recent optimization runs."""
    from observability.tracing import get_tracer
    try:
        runs = await get_tracer().get_recent_runs(limit=limit)
        return [
            RunHistoryItem(
                run_id=r["run_id"],
                sku=r["sku"],
                retailer=r["retailer"],
                provider=r["provider"],
                model=r["model"],
                score_before=r["score_before"],
                score_after=r["score_after"],
                improvement_delta=r["improvement_delta"],
                latency_ms=r["latency_ms"],
                estimated_cost_usd=r["estimated_cost_usd"],
                quality_passed=bool(r["quality_passed"]),
                timestamp=r["timestamp"],
            )
            for r in runs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit human feedback (thumbs up/down) for a run."""
    from evaluation.human_feedback import HumanFeedbackStore
    try:
        store = HumanFeedbackStore()
        feedback_id = await store.record_feedback(
            run_id=request.run_id,
            sku=request.sku,
            rating=request.rating,
            comment=request.comment,
            field=request.field,
        )
        return {"feedback_id": feedback_id, "status": "recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
