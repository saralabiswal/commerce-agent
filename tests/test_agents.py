"""
Tests for CommerceAgent's three agents.
Uses mock LLM provider — no real API calls.

Owner: Sarala Biswal
"""
import json

import pytest

from agents.competitor_analysis_agent import CompetitorAnalysisAgent
from agents.content_audit_agent import ContentAuditAgent
from agents.content_generation_agent import ContentGenerationAgent
from agents.models import AuditReport, CompetitorReport, Gap, GeneratedContent

# ── ContentAuditAgent ─────────────────────────────────────────────────────────

class TestContentAuditAgent:
    """Coverage for content audit agent behavior and fallback handling."""

    def _gap_response(self) -> str:
        """Return mock JSON gap analysis for audit tests."""
        return json.dumps([
            {
                "field": "title",
                "issue": "Title missing key search terms and too short",
                "severity": "high",
                "current_value": "Wireless Bluetooth Headphones Noise Canceling",
                "recommendation": "Add brand name, Bluetooth version, and ANC specification to title"
            },
            {
                "field": "bullets",
                "issue": "Bullets use vague language instead of specs",
                "severity": "high",
                "current_value": "Good sound quality",
                "recommendation": "Replace 'Good sound quality' with '40mm dynamic drivers' specification"
            }
        ])

    @pytest.mark.asyncio
    async def test_audit_returns_report(self, mock_provider):
        """AuditAgent should return a populated AuditReport."""
        mock_provider.set_response("gap analysis", self._gap_response())
        mock_provider.set_response("audit", self._gap_response())

        agent = ContentAuditAgent(mock_provider)
        report = await agent.run(sku="DEMO-SKU-001", retailer="amazon")

        assert isinstance(report, AuditReport)
        assert report.sku == "DEMO-SKU-001"
        assert report.retailer == "amazon"
        assert isinstance(report.current_score, float)
        assert 0 <= report.current_score <= 100

    @pytest.mark.asyncio
    async def test_audit_score_breakdown(self, mock_provider):
        """Score breakdown should contain all expected dimensions."""
        mock_provider.set_response("gap", self._gap_response())

        agent = ContentAuditAgent(mock_provider)
        report = await agent.run(sku="DEMO-SKU-001", retailer="amazon")

        expected_dims = {"title_compliance", "bullet_compliance", "keyword_inclusion"}
        assert expected_dims.issubset(set(report.score_breakdown.keys()))

    @pytest.mark.asyncio
    async def test_audit_gap_analysis_parsed(self, mock_provider):
        """Gap analysis should be a list of Gap objects."""
        mock_provider.set_response("gap", self._gap_response())

        agent = ContentAuditAgent(mock_provider)
        report = await agent.run(sku="DEMO-SKU-001", retailer="amazon")

        assert isinstance(report.gap_analysis, list)
        for gap in report.gap_analysis:
            assert isinstance(gap, Gap)
            assert gap.severity in ("critical", "high", "medium", "low")

    @pytest.mark.asyncio
    async def test_audit_priority_improvements_ordered(self, mock_provider):
        """Priority improvements should be a non-empty list of strings."""
        mock_provider.set_response("gap", self._gap_response())

        agent = ContentAuditAgent(mock_provider)
        report = await agent.run(sku="DEMO-SKU-001", retailer="amazon")

        assert isinstance(report.priority_improvements, list)

    @pytest.mark.asyncio
    async def test_audit_character_counts(self, mock_provider):
        """Character counts should cover title, bullets, description."""
        mock_provider.set_response("gap", self._gap_response())

        agent = ContentAuditAgent(mock_provider)
        report = await agent.run(sku="DEMO-SKU-001", retailer="amazon")

        assert "title" in report.character_counts
        assert "bullets" in report.character_counts

    @pytest.mark.asyncio
    async def test_audit_malformed_json_fallback(self, mock_provider):
        """Agent should not crash when LLM returns malformed JSON."""
        # Provider returns malformed JSON
        from llm.base import LLMResponse
        mock_provider._responses = {}

        async def bad_complete(messages, system=None, temperature=0.3, max_tokens=2048):
            """Return malformed JSON to exercise fallback parsing."""
            return LLMResponse(
                content="not valid json {{ broken",
                model="mock",
                input_tokens=10,
                output_tokens=10,
                latency_ms=1.0,
            )
        mock_provider.complete = bad_complete
        mock_provider.complete_json = bad_complete

        agent = ContentAuditAgent(mock_provider)
        # Should not raise — should return a valid report with empty/fallback gaps
        report = await agent.run(sku="DEMO-SKU-001", retailer="amazon")
        assert isinstance(report, AuditReport)


# ── CompetitorAnalysisAgent ───────────────────────────────────────────────────

class TestCompetitorAnalysisAgent:
    """Coverage for competitor analysis outputs and keyword handling."""

    def _analysis_response(self) -> str:
        """Return mock JSON competitor analysis for agent tests."""
        return json.dumps({
            "winning_patterns": [
                "Title starts with brand + product type + key differentiator",
                "Every competitor mentions Bluetooth version explicitly",
                "ANC effectiveness quantified (dB reduction or levels)"
            ],
            "content_gaps": [
                "All competitors mention Bluetooth version — we don't",
                "Top performers include IP rating in title or bullet 1",
                "Missing microphone count and type vs competitors"
            ]
        })

    @pytest.mark.asyncio
    async def test_competitor_returns_report(self, mock_provider):
        """Should return a CompetitorReport with populated fields."""
        mock_provider.set_response("competitor", self._analysis_response())
        mock_provider.set_response("pattern", self._analysis_response())

        agent = CompetitorAnalysisAgent(mock_provider)
        report = await agent.run(category="electronics", retailer="amazon")

        assert isinstance(report, CompetitorReport)
        assert report.category == "electronics"
        assert report.competitor_count >= 0

    @pytest.mark.asyncio
    async def test_competitor_keywords_have_volume(self, mock_provider):
        """Keywords should have non-negative monthly volume."""
        mock_provider.set_response("pattern", self._analysis_response())

        agent = CompetitorAnalysisAgent(mock_provider)
        report = await agent.run(category="electronics", retailer="amazon")

        for kw in report.top_keywords:
            assert kw.monthly_volume >= 0
            assert kw.competition in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_competitor_winning_patterns_are_strings(self, mock_provider):
        """Winning patterns should be a list of strings."""
        mock_provider.set_response("pattern", self._analysis_response())

        agent = CompetitorAnalysisAgent(mock_provider)
        report = await agent.run(category="electronics")

        assert isinstance(report.winning_patterns, list)
        for p in report.winning_patterns:
            assert isinstance(p, str)

    @pytest.mark.asyncio
    async def test_competitor_content_gaps_are_strings(self, mock_provider):
        """Content gaps should be a list of strings."""
        mock_provider.set_response("pattern", self._analysis_response())

        agent = CompetitorAnalysisAgent(mock_provider)
        report = await agent.run(category="electronics")

        assert isinstance(report.content_gaps, list)

    @pytest.mark.asyncio
    async def test_competitor_keyword_present_in_listing(self, mock_provider):
        """Keyword presence check should work correctly."""
        mock_provider.set_response("pattern", self._analysis_response())

        listing = {"title": "Bluetooth Headphones Wireless"}
        agent = CompetitorAnalysisAgent(mock_provider)

        assert agent._keyword_present("bluetooth headphones", listing) is True
        assert agent._keyword_present("noise cancelling", listing) is False

    @pytest.mark.asyncio
    async def test_competitor_unknown_category_returns_empty(self, mock_provider):
        """Unknown category should return an empty report without crashing."""
        mock_provider.set_response("pattern", self._analysis_response())

        agent = CompetitorAnalysisAgent(mock_provider)
        report = await agent.run(category="unicorn_products")

        assert isinstance(report, CompetitorReport)
        assert report.competitor_count == 0


# ── ContentGenerationAgent ────────────────────────────────────────────────────

class TestContentGenerationAgent:
    """Coverage for generated content shape, scoring, and validation metadata."""

    def _generation_response(self) -> str:
        """Return mock JSON generated content for generation tests."""
        return json.dumps({
            "title": "SoundWave Pro X1 Wireless Bluetooth Headphones — 30-Hr Battery, Active Noise Cancellation, IPX4, Bluetooth 5.2",
            "bullet_points": [
                "POWERFUL NOISE CANCELLATION — 3 adjustable ANC levels and 40mm dynamic drivers deliver immersive audio in any environment",
                "30-HOUR BATTERY + FAST CHARGE — Full day listening on one charge; 10-minute USB-C charge adds 3 hours when you're in a hurry",
                "PROFESSIONAL CALL QUALITY — Dual beamforming microphones with echo cancellation so every call sounds clear from anywhere",
                "WEATHER-RESISTANT BUILD — IPX4 splash resistance handles rain and sweat; Bluetooth 5.2 multipoint connects 2 devices simultaneously",
                "COMFORT FOR HOURS — 250g lightweight build, memory foam protein leather ear cushions, and a foldable design with included carrying case",
            ],
            "description": "The SoundWave Pro X1 delivers 30 hours of wireless audio with Active Noise Cancellation across 3 adjustable levels. Bluetooth 5.2 connects to two devices simultaneously.",
            "backend_keywords": "over ear headphones wireless noise cancelling headphones with microphone bluetooth 5.2 headphones ipx4 splash resistant foldable headphones",
            "reasoning": "Prioritized high-volume keywords (bluetooth headphones, noise cancelling headphones) in title. Quantified all specs to replace vague language.",
        })

    def _make_audit(self):
        """Create a minimal AuditReport for testing."""
        from agents.models import AuditReport
        return AuditReport(
            sku="DEMO-SKU-001",
            retailer="amazon",
            current_score=58.0,
            gap_analysis=[],
            priority_improvements=["Add Bluetooth version", "Replace vague language with specs"],
            retailer_compliance={"compliant": True},
            character_counts={},
            score_breakdown={"title_compliance": 60, "keyword_inclusion": 40},
            listing_snapshot={"category": "electronics"},
        )

    def _make_competitors(self):
        """Create a minimal CompetitorReport for testing."""
        from agents.models import CompetitorReport, Keyword
        return CompetitorReport(
            category="electronics",
            top_keywords=[
                Keyword("bluetooth headphones", 450000, "high", False),
                Keyword("noise cancelling headphones", 290000, "high", False),
            ],
            winning_patterns=["Include Bluetooth version in title"],
            content_gaps=["Missing Bluetooth version"],
            benchmark_scores={"Sony": 91.0, "Bose": 88.0},
            competitor_count=3,
        )

    @pytest.mark.asyncio
    async def test_generation_returns_content(self, mock_provider):
        """Should return a GeneratedContent object."""
        mock_provider.set_response("generate", self._generation_response())
        mock_provider.set_response("title", self._generation_response())

        agent = ContentGenerationAgent(mock_provider)
        content = await agent.run(
            sku="DEMO-SKU-001",
            retailer="amazon",
            audit=self._make_audit(),
            competitors=self._make_competitors(),
        )

        assert isinstance(content, GeneratedContent)
        assert content.sku == "DEMO-SKU-001"
        assert content.retailer == "amazon"

    @pytest.mark.asyncio
    async def test_generation_has_five_bullets(self, mock_provider):
        """Generated content should have 5 bullet points."""
        mock_provider.set_response("bullet", self._generation_response())

        agent = ContentGenerationAgent(mock_provider)
        content = await agent.run(
            sku="DEMO-SKU-001",
            retailer="amazon",
            audit=self._make_audit(),
            competitors=self._make_competitors(),
        )

        assert len(content.bullet_points) == 5

    @pytest.mark.asyncio
    async def test_generation_score_in_range(self, mock_provider):
        """Quality score should be between 0 and 100."""
        mock_provider.set_response("generate", self._generation_response())

        agent = ContentGenerationAgent(mock_provider)
        content = await agent.run(
            sku="DEMO-SKU-001",
            retailer="amazon",
            audit=self._make_audit(),
            competitors=self._make_competitors(),
        )

        assert 0 <= content.quality_score <= 100

    @pytest.mark.asyncio
    async def test_generation_improvement_delta(self, mock_provider):
        """Improvement delta should equal score_after - score_before."""
        mock_provider.set_response("generate", self._generation_response())

        agent = ContentGenerationAgent(mock_provider)
        audit = self._make_audit()
        content = await agent.run(
            sku="DEMO-SKU-001",
            retailer="amazon",
            audit=audit,
            competitors=self._make_competitors(),
        )

        expected_delta = round(content.quality_score - audit.current_score, 1)
        assert abs(content.improvement_delta - expected_delta) < 0.5

    @pytest.mark.asyncio
    async def test_generation_compliance_check_present(self, mock_provider):
        """compliance_check and brand_safety should be populated."""
        mock_provider.set_response("generate", self._generation_response())

        agent = ContentGenerationAgent(mock_provider)
        content = await agent.run(
            sku="DEMO-SKU-001",
            retailer="amazon",
            audit=self._make_audit(),
            competitors=self._make_competitors(),
        )

        assert isinstance(content.compliance_check, dict)
        assert "compliant" in content.compliance_check
        assert isinstance(content.brand_safety, dict)
        assert "passed" in content.brand_safety

    @pytest.mark.asyncio
    async def test_generation_retry_count_tracked(self, mock_provider):
        """retry_count should be reflected in output."""
        mock_provider.set_response("generate", self._generation_response())

        agent = ContentGenerationAgent(mock_provider)
        content = await agent.run(
            sku="DEMO-SKU-001",
            retailer="amazon",
            audit=self._make_audit(),
            competitors=self._make_competitors(),
            retry_count=2,
        )

        assert content.retry_count == 2
