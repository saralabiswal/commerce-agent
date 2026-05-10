"""
ContentAuditAgent — analyzes current listing quality and produces a gap analysis.

Input:  Product SKU + retailer
Output: AuditReport with scores, gaps, and priority improvements

Tools used:
  RetailerMCP.get_listing
  RetailerMCP.get_retailer_requirements
  ScoringMCP.score_content
"""
import json
import logging

from agents.models import AuditReport, Gap
from llm.base import LLMProvider, Message
from mcp_servers.retailer_mcp_server import (
    get_category_benchmarks,
    get_listing,
    get_retailer_requirements,
)
from mcp_servers.scoring_mcp_server import check_compliance, score_content

logger = logging.getLogger(__name__)


class ContentAuditAgent:
    """
    Analyzes a product listing against retailer requirements and returns
    a structured audit report with prioritized improvement recommendations.
    """

    SYSTEM_PROMPT = """You are a senior e-commerce content strategist specializing in
Amazon and Walmart product listing optimization. You analyze listings with precision,
identifying specific compliance issues, content gaps, and improvement opportunities.

You always base your analysis on actual data — character counts, missing keywords,
compliance violations — not subjective opinions. Your recommendations are specific and
actionable, not vague suggestions.

Respond only with valid JSON. No markdown fences, no explanation."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, sku: str, retailer: str = "amazon") -> AuditReport:
        """
        Run the content audit for a SKU on a given retailer.

        1. Fetch current listing from RetailerMCP
        2. Fetch retailer requirements from RetailerMCP
        3. Score with ScoringMCP
        4. Use LLM for gap analysis and priority recommendations
        """
        logger.info(f"ContentAuditAgent: auditing {sku} on {retailer}")

        # ── Step 1: Gather data ───────────────────────────────────────────────
        listing = get_listing(sku, retailer)
        requirements = get_retailer_requirements(
            listing.get("category", "electronics"), retailer
        )
        benchmarks = get_category_benchmarks(
            listing.get("category", "electronics"), retailer
        )

        # ── Step 2: Automated scoring ─────────────────────────────────────────
        score_result = score_content(listing, requirements)
        compliance_result = check_compliance(listing, retailer)

        # ── Step 3: Character count analysis ──────────────────────────────────
        character_counts = self._compute_character_counts(listing, requirements)

        # ── Step 4: LLM gap analysis ──────────────────────────────────────────
        gap_analysis = await self._llm_gap_analysis(
            listing, requirements, score_result, compliance_result, benchmarks
        )

        # ── Step 5: Priority improvements ─────────────────────────────────────
        priority_improvements = self._derive_priorities(
            score_result, compliance_result, gap_analysis
        )

        return AuditReport(
            sku=sku,
            retailer=retailer,
            current_score=score_result["total_score"],
            gap_analysis=gap_analysis,
            priority_improvements=priority_improvements,
            retailer_compliance=compliance_result,
            character_counts=character_counts,
            score_breakdown=score_result["scores"],
            listing_snapshot=listing,
        )

    async def _llm_gap_analysis(
        self,
        listing: dict,
        requirements: dict,
        score_result: dict,
        compliance_result: dict,
        benchmarks: dict,
    ) -> list[Gap]:
        """Use LLM to identify nuanced content gaps beyond rule-based checks."""

        prompt = f"""Analyze this product listing and identify specific content gaps.

CURRENT LISTING:
Title: {listing.get('title', '')}
Bullets: {json.dumps(listing.get('bullet_points', []), indent=2)}
Description: {listing.get('description', '')}

SCORING RESULTS:
Total Score: {score_result['total_score']}/100
Breakdown: {json.dumps(score_result['scores'], indent=2)}
Issues Found: {json.dumps(score_result['issues'], indent=2)}

RETAILER REQUIREMENTS:
{json.dumps(requirements, indent=2)}

CATEGORY BENCHMARKS:
{json.dumps(benchmarks, indent=2)}

Identify the most impactful gaps. For each gap, return a JSON object.

Return a JSON array of gap objects with this exact structure:
[
  {{
    "field": "title|bullets|description|backend_keywords",
    "issue": "specific description of the problem",
    "severity": "critical|high|medium|low",
    "current_value": "the problematic text or stat",
    "recommendation": "specific actionable fix"
  }}
]

Focus on: keyword gaps, specification gaps, weak/vague language, compliance violations,
missed opportunities vs category benchmarks. Be specific — not 'improve bullets' but
'Bullet 3 uses vague language (good sound) instead of a spec (40mm dynamic drivers)'."""

        response = await self.provider.complete_json(
            messages=[Message(role="user", content=prompt)],
            system=self.SYSTEM_PROMPT,
            temperature=0.1,
        )

        try:
            gaps_data = json.loads(response.content)
            return [Gap(**g) for g in gaps_data if isinstance(g, dict)]
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse gap analysis JSON: {e}")
            # Fallback: derive gaps from score issues
            return [
                Gap(
                    field="general",
                    issue=issue,
                    severity="medium",
                    current_value="",
                    recommendation="See score breakdown for details",
                )
                for issue in score_result.get("issues", [])
            ]

    def _compute_character_counts(
        self, listing: dict, requirements: dict
    ) -> dict[str, dict]:
        counts = {}

        title = listing.get("title", "")
        title_rules = requirements.get("title", {})
        counts["title"] = {
            "current": len(title),
            "allowed": title_rules.get("max_chars", 200),
            "utilization_pct": round(len(title) / title_rules.get("max_chars", 200) * 100, 1),
        }

        bullets = listing.get("bullet_points", [])
        bullet_rules = requirements.get("bullet_points", {})
        counts["bullets"] = {
            "count": len(bullets),
            "expected": bullet_rules.get("count", 5),
            "max_each": bullet_rules.get("max_chars_each", 255),
            "lengths": [len(b) for b in bullets],
        }

        desc = listing.get("description", "")
        desc_rules = requirements.get("description", {})
        counts["description"] = {
            "current": len(desc),
            "allowed": desc_rules.get("max_chars", 2000),
            "utilization_pct": round(len(desc) / desc_rules.get("max_chars", 2000) * 100, 1),
        }

        kw = listing.get("backend_keywords", "")
        kw_rules = requirements.get("backend_keywords", {})
        counts["backend_keywords"] = {
            "current": len(kw),
            "allowed": kw_rules.get("max_chars", 250),
        }

        return counts

    def _derive_priorities(
        self, score_result: dict, compliance_result: dict, gap_analysis: list[Gap]
    ) -> list[str]:
        """Order improvements by expected impact on score."""
        priorities = []

        # Compliance violations first — they can suppress listings
        if not compliance_result["compliant"]:
            for v in compliance_result["violations"][:2]:
                priorities.append(f"[COMPLIANCE] {v}")

        # Critical gaps next
        critical_gaps = [g for g in gap_analysis if g.severity == "critical"]
        for gap in critical_gaps[:2]:
            priorities.append(f"[CRITICAL] {gap.recommendation}")

        # Low-scoring dimensions
        scores = score_result.get("scores", {})
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        for dim, score in sorted_scores[:2]:
            if score < 70:
                label = dim.replace("_", " ")
                priorities.append(f"[SCORE] Improve {label}: currently {score:.0f}/100")

        # High gaps
        high_gaps = [g for g in gap_analysis if g.severity == "high"]
        for gap in high_gaps[:2]:
            priorities.append(f"[HIGH] {gap.recommendation}")

        return priorities[:6]  # Return top 6 priorities
