"""
/audit — standalone content audit endpoint.

Owner: Sarala Biswal
"""
from fastapi import APIRouter, HTTPException

from agents.content_audit_agent import ContentAuditAgent
from api.schemas import AuditRequest, AuditResponse, audit_to_response
from llm.factory import get_cached_provider

router = APIRouter()


@router.post("/audit", response_model=AuditResponse)
async def audit_listing(request: AuditRequest):
    """
    Audit a product listing's content quality.
    Returns gap analysis, compliance check, and priority improvements.
    """
    try:
        provider = get_cached_provider()
        agent = ContentAuditAgent(provider)
        audit = await agent.run(sku=request.sku, retailer=request.retailer)
        return audit_to_response(audit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
