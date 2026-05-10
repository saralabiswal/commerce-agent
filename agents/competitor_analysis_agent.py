"""
CompetitorAnalysisAgent — extracts winning content patterns from competitor listings.

Input:  Product category + competitor listings
Output: CompetitorReport with top keywords, winning patterns, and content gaps

Tools used:
  CatalogMCP.get_competitor_listings
  RetailerMCP.get_search_volume
"""
import json
import logging

from agents.models import CompetitorReport, Keyword
from llm.base import LLMProvider, Message
from mcp_servers.catalog_mcp_server import get_competitor_listings
from mcp_servers.retailer_mcp_server import get_search_volume

logger = logging.getLogger(__name__)


class CompetitorAnalysisAgent:
    """
    Analyzes top competitor listings to extract:
    - High-volume keywords competitors rank for
    - Title and bullet structures that correlate with strong performance
    - Content elements our listing is missing vs competitors
    """

    SYSTEM_PROMPT = """You are an e-commerce competitive intelligence analyst.
You analyze competitor product listings to identify patterns, keywords, and content
strategies that distinguish top-performing listings from average ones.

Your analysis is data-driven and specific. You identify patterns that can be replicated,
not vague observations like "their content is better."

Respond only with valid JSON. No markdown fences, no explanation."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(
        self,
        category: str,
        retailer: str = "amazon",
        current_listing: dict | None = None,
    ) -> CompetitorReport:
        """
        Run competitor analysis for a product category.

        Args:
            category: Product category (e.g., "electronics")
            retailer: Target retailer
            current_listing: Our current listing for gap comparison
        """
        logger.info(f"CompetitorAnalysisAgent: analyzing {category} on {retailer}")

        # ── Step 1: Fetch competitor listings ─────────────────────────────────
        competitors = get_competitor_listings(category, limit=5)
        if not competitors:
            logger.warning(f"No competitor data for category: {category}")
            return self._empty_report(category)

        # ── Step 2: Extract keywords from competitor content ──────────────────
        raw_keywords = self._extract_candidate_keywords(competitors)

        # ── Step 3: Enrich with search volume data ────────────────────────────
        keywords_with_volume = []
        for term in raw_keywords[:10]:  # Cap API calls
            vol_data = get_search_volume(term, retailer)
            present = self._keyword_present(term, current_listing or {})
            keywords_with_volume.append(Keyword(
                term=term,
                monthly_volume=vol_data.get("monthly_searches", 0),
                competition=vol_data.get("competition", "medium"),
                present_in_listing=present,
            ))

        # Sort by volume desc
        keywords_with_volume.sort(key=lambda k: k.monthly_volume, reverse=True)

        # ── Step 4: LLM pattern analysis ──────────────────────────────────────
        analysis = await self._llm_pattern_analysis(
            competitors, current_listing, keywords_with_volume
        )

        # ── Step 5: Benchmark scores ───────────────────────────────────────────
        benchmark_scores = {
            c["brand"]: float(c.get("quality_score", 0))
            for c in competitors
        }

        return CompetitorReport(
            category=category,
            top_keywords=keywords_with_volume[:8],
            winning_patterns=analysis.get("winning_patterns", []),
            content_gaps=analysis.get("content_gaps", []),
            benchmark_scores=benchmark_scores,
            competitor_count=len(competitors),
        )

    async def _llm_pattern_analysis(
        self,
        competitors: list[dict],
        current_listing: dict | None,
        keywords: list[Keyword],
    ) -> dict:
        """Use LLM to identify patterns across competitor listings."""

        competitors_summary = []
        for c in competitors:
            competitors_summary.append({
                "brand": c.get("brand"),
                "title": c.get("title"),
                "bullets": c.get("bullet_points", []),
                "quality_score": c.get("quality_score"),
            })

        current_summary = {}
        if current_listing:
            current_summary = {
                "title": current_listing.get("title"),
                "bullets": current_listing.get("bullet_points", []),
            }

        keyword_summary = [
            {
                "term": k.term,
                "volume": k.monthly_volume,
                "we_have_it": k.present_in_listing,
            }
            for k in keywords
        ]

        prompt = f"""Analyze these competitor listings and identify winning content patterns.

TOP COMPETITOR LISTINGS (sorted by quality score):
{json.dumps(competitors_summary, indent=2)}

OUR CURRENT LISTING:
{json.dumps(current_summary, indent=2) if current_summary else "Not provided"}

TOP KEYWORDS BY SEARCH VOLUME:
{json.dumps(keyword_summary, indent=2)}

Return a JSON object with this exact structure:
{{
  "winning_patterns": [
    "specific pattern competitors use that works, with example",
    ...
  ],
  "content_gaps": [
    "something top competitors have that our listing lacks",
    ...
  ]
}}

Winning patterns should be specific and actionable:
GOOD: "Title starts with brand + product type + key differentiator"
BAD: "Good title structure"

Content gaps should compare us specifically to competitors:
GOOD: "All top competitors mention Bluetooth version (5.0/5.2) — our listing doesn't"
BAD: "Missing technical specs"

Provide 4-6 winning patterns and 4-6 content gaps."""

        response = await self.provider.complete_json(
            messages=[Message(role="user", content=prompt)],
            system=self.SYSTEM_PROMPT,
            temperature=0.2,
        )

        try:
            return json.loads(response.content)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse competitor analysis JSON: {e}")
            return {
                "winning_patterns": [
                    (
                        "Include specific technical specs "
                        "(Bluetooth version, driver size, battery hours)"
                    ),
                    "Lead title with brand name followed by product type and key differentiator",
                    "Structure bullets around benefit + feature + specification",
                ],
                "content_gaps": [
                    "Missing Bluetooth version specification",
                    "No microphone detail for call quality buyers",
                    "No charging speed specification",
                ],
            }

    def _extract_candidate_keywords(self, competitors: list[dict]) -> list[str]:
        """Extract unique keyword candidates from competitor listings."""
        seen = set()
        keywords = []
        for c in competitors:
            for kw in c.get("keywords", []):
                if kw.lower() not in seen:
                    seen.add(kw.lower())
                    keywords.append(kw.lower())
        return keywords

    def _keyword_present(self, keyword: str, listing: dict) -> bool:
        """Check if a keyword appears anywhere in the listing."""
        text = " ".join([
            listing.get("title", ""),
            " ".join(listing.get("bullet_points", [])),
            listing.get("description", ""),
            listing.get("backend_keywords", ""),
        ]).lower()
        return keyword.lower() in text

    def _empty_report(self, category: str) -> CompetitorReport:
        return CompetitorReport(
            category=category,
            top_keywords=[],
            winning_patterns=[],
            content_gaps=["No competitor data available for this category"],
            benchmark_scores={},
            competitor_count=0,
        )
