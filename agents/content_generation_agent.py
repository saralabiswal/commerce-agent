"""
ContentGenerationAgent — generates optimized listing content grounded in RAG context.

Input:  AuditReport + CompetitorReport + ProductSpecs
Output: GeneratedContent with quality score and compliance check

Tools used:
  CatalogMCP.get_product_specs
  CatalogMCP.get_brand_guidelines
  RAG retrieval (retailer rules + category requirements)
  ScoringMCP.score_content
  ScoringMCP.check_compliance
  ScoringMCP.check_brand_safety

Owner: Sarala Biswal
"""
import json
import logging

from agents.models import AuditReport, CompetitorReport, GeneratedContent
from llm.base import LLMProvider, Message
from mcp_servers.catalog_mcp_server import get_brand_guidelines, get_product_specs
from mcp_servers.retailer_mcp_server import get_retailer_requirements
from mcp_servers.scoring_mcp_server import check_brand_safety, check_compliance, score_content
from rag.retrieval import get_retriever

logger = logging.getLogger(__name__)


class ContentGenerationAgent:
    """
    Generates optimized product listing content grounded in:
    - Retrieved retailer style guide rules (RAG)
    - Authoritative product specifications (CatalogMCP)
    - Brand guidelines (CatalogMCP)
    - Audit findings (from ContentAuditAgent)
    - Competitor intelligence (from CompetitorAnalysisAgent)

    The quality gate in the orchestrator calls this agent up to N times
    if initial content doesn't meet the threshold.
    """

    SYSTEM_PROMPT = """You are an expert e-commerce content writer specializing in
Amazon and Walmart product listings. You write content that:

1. Strictly complies with retailer character limits and rules
2. Leads with customer benefits before specifications
3. Includes specific, measurable claims (never vague language like "great quality")
4. Naturally incorporates high-volume search keywords
5. Reflects the brand's tone of voice
6. Never invents specifications not present in the product data provided

You are given authoritative product specs — only use facts from these specs.
Never hallucinate product features, specifications, or claims.

    Respond only with valid JSON. No markdown fences, no explanation."""

    def __init__(self, provider: LLMProvider):
        """Initialize generation dependencies for LLM and RAG retrieval."""
        self.provider = provider
        self.retriever = get_retriever()

    async def run(
        self,
        sku: str,
        retailer: str,
        audit: AuditReport,
        competitors: CompetitorReport,
        retry_feedback: str = "",
        retry_count: int = 0,
    ) -> GeneratedContent:
        """
        Generate optimized content for a product listing.

        Args:
            sku: Product SKU
            retailer: Target retailer
            audit: Audit report from ContentAuditAgent
            competitors: Competitor analysis from CompetitorAnalysisAgent
            retry_feedback: Specific improvement instructions on retry (from quality gate)
            retry_count: Current retry attempt number
        """
        logger.info(
            f"ContentGenerationAgent: generating for {sku} on {retailer} "
            f"(attempt {retry_count + 1})"
        )

        # ── Step 1: Gather all context ─────────────────────────────────────────
        category = audit.listing_snapshot.get("category", "electronics")
        product_specs = get_product_specs(sku)
        brand_guidelines = get_brand_guidelines(product_specs.get("brand", ""))
        requirements = get_retailer_requirements(category, retailer)

        # Retrieve relevant rules from RAG
        rag_context = self.retriever.retrieve_for_generation(
            product_category=category,
            retailer=retailer,
            product_type=product_specs.get("product_type", ""),
        )

        # ── Step 2: Build generation prompt ───────────────────────────────────
        prompt = self._build_prompt(
            product_specs=product_specs,
            brand_guidelines=brand_guidelines,
            requirements=requirements,
            audit=audit,
            competitors=competitors,
            rag_context=rag_context,
            retry_feedback=retry_feedback,
        )

        # ── Step 3: Generate ──────────────────────────────────────────────────
        response = await self.provider.complete_json(
            messages=[Message(role="user", content=prompt)],
            system=self.SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=2048,
        )

        # ── Step 4: Parse and validate ────────────────────────────────────────
        content_dict = self._parse_response(response.content, sku, retailer)
        content_dict["category"] = category

        # ── Step 5: Score and compliance check ────────────────────────────────
        score_result = score_content(content_dict, requirements)
        compliance_result = check_compliance(content_dict, retailer)
        safety_result = check_brand_safety(content_dict, brand_guidelines)

        previous_score = audit.current_score
        improvement_delta = score_result["total_score"] - previous_score

        warnings = []
        if not compliance_result["compliant"]:
            warnings.extend([f"Compliance: {v}" for v in compliance_result["violations"]])
        if not safety_result["passed"]:
            warnings.extend([
                f"Brand safety: {f['term']} ({f['severity']})"
                for f in safety_result["flags"]
            ])

        return GeneratedContent(
            sku=sku,
            retailer=retailer,
            title=content_dict.get("title", ""),
            bullet_points=content_dict.get("bullet_points", []),
            description=content_dict.get("description", ""),
            backend_keywords=content_dict.get("backend_keywords", ""),
            quality_score=score_result["total_score"],
            score_breakdown=score_result["scores"],
            compliance_check=compliance_result,
            brand_safety=safety_result,
            previous_score=previous_score,
            improvement_delta=improvement_delta,
            reasoning=content_dict.get("reasoning", ""),
            retry_count=retry_count,
            warnings=warnings,
        )

    def _build_prompt(
        self,
        product_specs: dict,
        brand_guidelines: dict,
        requirements: dict,
        audit: AuditReport,
        competitors: CompetitorReport,
        rag_context: str,
        retry_feedback: str,
    ) -> str:
        """Assemble the grounded prompt used to generate optimized content."""
        # Format top keywords
        keyword_list = ", ".join([
            f"{k.term} ({k.monthly_volume:,}/mo)"
            for k in competitors.top_keywords[:6]
        ])

        # Format priority improvements from audit
        priorities = "\n".join([f"- {p}" for p in audit.priority_improvements])

        # Format competitor patterns
        patterns = "\n".join([f"- {p}" for p in competitors.winning_patterns[:4]])

        # Format content gaps to address
        gaps = "\n".join([f"- {g}" for g in competitors.content_gaps[:4]])

        retry_section = ""
        if retry_feedback:
            retry_section = f"""
⚠️ RETRY INSTRUCTIONS (Previous attempt did not pass quality gate):
{retry_feedback}
Address ALL of these specific issues in this attempt.
"""

        # Get character limits
        title_max = requirements.get("title", {}).get("max_chars", 200)
        bullet_max = requirements.get("bullet_points", {}).get("max_chars_each", 255)
        desc_max = requirements.get("description", {}).get("max_chars", 2000)
        kw_max = requirements.get("backend_keywords", {}).get("max_chars", 250)
        bullet_count = requirements.get("bullet_points", {}).get("count", 5)

        return f"""Generate optimized product listing content. Use ONLY the product specs provided.
{retry_section}
=== PRODUCT SPECIFICATIONS (source of truth — never invent facts) ===
{json.dumps(product_specs.get("specifications", {}), indent=2)}
Brand: {product_specs.get("brand", "")}
Product Name: {product_specs.get("product_name", "")}
Product Type: {product_specs.get("product_type", "")}
Model: {product_specs.get("model_number", "")}

=== BRAND GUIDELINES ===
Tone: {brand_guidelines.get("tone", "professional")}
Prohibited claims: {', '.join(brand_guidelines.get("prohibited_claims", []))}
Key differentiators to highlight: {', '.join(brand_guidelines.get("key_differentiators", []))}
Target audience: {brand_guidelines.get("target_audience", "general consumer")}

{rag_context}

=== AUDIT FINDINGS (what to fix) ===
Current score: {audit.current_score}/100
Top priorities:
{priorities}

=== COMPETITOR INTELLIGENCE ===
Top keywords by search volume: {keyword_list}
Winning patterns from top competitors:
{patterns}
Content gaps vs competitors:
{gaps}

=== STRICT REQUIREMENTS ===
Retailer: {audit.retailer}
- Title: max {title_max} chars (target 80%+ utilization)
- Bullets: exactly {bullet_count} bullets, max {bullet_max} chars each
- Description: max {desc_max} chars
- Backend keywords: max {kw_max} chars, no title word repetition

Return a JSON object with EXACTLY this structure:
{{
  "title": "optimized title string",
  "bullet_points": ["bullet 1", "bullet 2", "bullet 3", "bullet 4", "bullet 5"],
  "description": "optimized description",
  "backend_keywords": "keyword1 keyword2 keyword3 ...",
  "reasoning": "1-2 sentences explaining the key content decisions made"
}}"""

    def _parse_response(
        self, raw_response: str, sku: str, retailer: str
    ) -> dict:
        """Parse and validate the LLM JSON response."""
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON from potential surrounding text
            import re
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        # Ensure all required fields exist with sensible defaults
        return {
            "title": data.get("title", f"Product {sku}"),
            "bullet_points": data.get("bullet_points", [])[:5],
            "description": data.get("description", ""),
            "backend_keywords": data.get("backend_keywords", ""),
            "reasoning": data.get("reasoning", ""),
        }
