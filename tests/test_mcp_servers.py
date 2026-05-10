"""
Tests for all three MCP servers.
No LLM calls — pure data layer tests.
"""


# ── RetailerMCP ────────────────────────────────────────────────────────────────

class TestRetailerMCP:

    def test_get_listing_known_sku(self):
        from mcp_servers.retailer_mcp_server import get_listing
        listing = get_listing("ANKER-Q30-BLK", "amazon")
        assert listing["sku"] == "ANKER-Q30-BLK"
        assert listing["retailer"] == "amazon"
        assert "title" in listing
        assert "bullet_points" in listing
        assert isinstance(listing["bullet_points"], list)

    def test_get_listing_unknown_sku_returns_stub(self):
        """Unknown SKU should return a stub, not raise."""
        from mcp_servers.retailer_mcp_server import get_listing
        listing = get_listing("UNKNOWN-9999", "amazon")
        assert "error" in listing
        assert listing["asin"] == "UNKNOWN-9999"

    def test_get_listing_walmart(self):
        from mcp_servers.retailer_mcp_server import get_listing
        listing = get_listing("ANKER-Q30-BLK", "walmart")
        assert listing["retailer"] == "walmart"

    def test_get_listing_resolves_catalog_sku_alias(self):
        from mcp_servers.retailer_mcp_server import get_listing

        listing = get_listing("ECHO-DOT-5G-BLK", "amazon")

        assert listing["sku"] == "ECHO-DOT-5G-BLK"
        assert listing["brand"] == "Amazon"

    def test_get_retailer_requirements_electronics(self):
        from mcp_servers.retailer_mcp_server import get_retailer_requirements
        reqs = get_retailer_requirements("electronics", "amazon")
        assert "title" in reqs
        assert "bullet_points" in reqs
        assert "max_chars" in reqs["title"]
        assert reqs["title"]["max_chars"] == 200

    def test_get_retailer_requirements_unknown_category_fallback(self):
        """Unknown category should fall back to electronics without crashing."""
        from mcp_servers.retailer_mcp_server import get_retailer_requirements
        reqs = get_retailer_requirements("underwater_basket_weaving", "amazon")
        assert "title" in reqs  # fallback should still have structure
        assert reqs.get("_fallback") is True

    def test_get_retailer_requirements_walmart_grocery(self):
        from mcp_servers.retailer_mcp_server import get_retailer_requirements

        reqs = get_retailer_requirements("grocery", "walmart")

        assert reqs["category"] == "grocery"
        assert reqs["bullet_points"]["count"] == 5

    def test_get_retailer_requirements_prohibited_words(self):
        """Electronics title should have prohibited words list."""
        from mcp_servers.retailer_mcp_server import get_retailer_requirements
        reqs = get_retailer_requirements("electronics", "amazon")
        prohibited = reqs["title"].get("prohibited_words", [])
        assert "best" in prohibited or len(prohibited) > 0

    def test_get_search_volume_known_keyword(self):
        from mcp_servers.retailer_mcp_server import get_search_volume
        result = get_search_volume("bluetooth headphones", "amazon")
        assert result["keyword"] == "bluetooth headphones"
        assert result["monthly_searches"] > 0
        assert result["competition"] in ("high", "medium", "low")

    def test_get_search_volume_unknown_keyword_returns_estimate(self):
        """Unknown keyword should return a mock estimate, not raise."""
        from mcp_servers.retailer_mcp_server import get_search_volume
        result = get_search_volume("quantum-flux-capacitor-headphones")
        assert result["monthly_searches"] >= 0
        assert "note" in result  # flagged as simulated

    def test_get_search_volume_deterministic(self):
        """Same keyword should return same volume (seed-based mock)."""
        from mcp_servers.retailer_mcp_server import get_search_volume
        r1 = get_search_volume("some-unique-test-keyword")
        r2 = get_search_volume("some-unique-test-keyword")
        assert r1["monthly_searches"] == r2["monthly_searches"]

    def test_get_category_benchmarks_electronics(self):
        from mcp_servers.retailer_mcp_server import get_category_benchmarks
        bm = get_category_benchmarks("electronics", "amazon")
        assert "avg_quality_score" in bm
        assert "top_quartile_score" in bm
        assert bm["avg_quality_score"] > 0

    def test_get_category_benchmarks_unknown_fallback(self):
        """Unknown category should fall back without crashing."""
        from mcp_servers.retailer_mcp_server import get_category_benchmarks
        bm = get_category_benchmarks("unknown_category")
        assert isinstance(bm, dict)


# ── CatalogMCP ────────────────────────────────────────────────────────────────

class TestCatalogMCP:

    def test_list_products_returns_friendly_names(self):
        from mcp_servers.catalog_mcp_server import list_products

        products = list_products()
        product_names = {product["product_name"] for product in products}
        assert len(products) >= 3
        assert "Soundcore by Anker Life Q30 Hybrid Active Noise Cancelling Headphones" in product_names
        assert "Echo Dot (5th Gen) Smart Speaker with Alexa" in product_names
        assert "Tide PODS Spring Meadow Laundry Detergent Pacs, 96 Count" in product_names

    def test_get_product_specs_known_sku(self):
        from mcp_servers.catalog_mcp_server import get_product_specs
        specs = get_product_specs("ANKER-Q30-BLK")
        assert specs["sku"] == "ANKER-Q30-BLK"
        assert "specifications" in specs
        assert "brand_guidelines" in specs
        assert "battery_life" in specs["specifications"]

    def test_get_product_specs_real_world_catalog_skus(self):
        from mcp_servers.catalog_mcp_server import get_product_specs

        echo = get_product_specs("ECHO-DOT-5G-BLK")
        tide = get_product_specs("TIDE-POD-96CT")

        assert echo["brand"] == "Amazon"
        assert echo["product_type"] == "Smart Speaker"
        assert tide["brand"] == "Tide"
        assert tide["specifications"]["count"] == "96 laundry pacs"

    def test_get_product_specs_unknown_sku_returns_error(self):
        """Unknown SKU should return error dict, not raise."""
        from mcp_servers.catalog_mcp_server import get_product_specs
        specs = get_product_specs("UNKNOWN-SKU-999")
        assert "error" in specs

    def test_get_brand_guidelines_known_brand(self):
        from mcp_servers.catalog_mcp_server import get_brand_guidelines
        guidelines = get_brand_guidelines("Soundcore")
        assert "tone" in guidelines
        assert "prohibited_claims" in guidelines
        assert isinstance(guidelines["prohibited_claims"], list)

    def test_get_brand_guidelines_unknown_brand_fallback(self):
        """Unknown brand should return default guidelines."""
        from mcp_servers.catalog_mcp_server import get_brand_guidelines
        guidelines = get_brand_guidelines("BrandXYZ999")
        assert "tone" in guidelines
        assert guidelines.get("_fallback") is True

    def test_get_competitor_listings_electronics(self):
        from mcp_servers.catalog_mcp_server import get_competitor_listings
        competitors = get_competitor_listings("electronics", limit=3)
        assert isinstance(competitors, list)
        assert len(competitors) <= 3
        for c in competitors:
            assert "brand" in c
            assert "title" in c
            assert "bullet_points" in c

    def test_get_competitor_listings_sorted_by_score(self):
        """Competitors should be sorted by quality_score descending."""
        from mcp_servers.catalog_mcp_server import get_competitor_listings
        competitors = get_competitor_listings("electronics", limit=5)
        if len(competitors) >= 2:
            scores = [c.get("quality_score", 0) for c in competitors]
            assert scores == sorted(scores, reverse=True)

    def test_get_competitor_listings_unknown_category(self):
        """Unknown category should return empty list."""
        from mcp_servers.catalog_mcp_server import get_competitor_listings
        competitors = get_competitor_listings("snorkel_gear")
        assert isinstance(competitors, list)
        assert len(competitors) == 0

    def test_get_historical_content_known_sku(self):
        from mcp_servers.catalog_mcp_server import get_historical_content
        history = get_historical_content("ANKER-Q30-BLK")
        assert isinstance(history, list)
        assert len(history) > 0
        for entry in history:
            assert "version" in entry
            assert "quality_score" in entry

    def test_get_historical_content_unknown_sku(self):
        """Unknown SKU should return empty history."""
        from mcp_servers.catalog_mcp_server import get_historical_content
        history = get_historical_content("UNKNOWN-999")
        assert isinstance(history, list)
        assert len(history) == 0


# ── ScoringMCP ────────────────────────────────────────────────────────────────

class TestScoringMCP:

    def test_score_content_returns_0_to_100(self, sample_listing, sample_requirements):
        from mcp_servers.scoring_mcp_server import score_content
        result = score_content(sample_listing, sample_requirements)
        assert 0 <= result["total_score"] <= 100

    def test_score_content_has_breakdown(self, sample_listing, sample_requirements):
        from mcp_servers.scoring_mcp_server import score_content
        result = score_content(sample_listing, sample_requirements)
        assert "scores" in result
        expected_dims = {"title_compliance", "bullet_compliance", "keyword_inclusion"}
        assert expected_dims.issubset(set(result["scores"].keys()))

    def test_score_content_issues_is_list(self, sample_listing, sample_requirements):
        from mcp_servers.scoring_mcp_server import score_content
        result = score_content(sample_listing, sample_requirements)
        assert isinstance(result["issues"], list)
        assert isinstance(result["suggestions"], list)

    def test_score_content_grade(self, sample_listing, sample_requirements):
        from mcp_servers.scoring_mcp_server import score_content
        result = score_content(sample_listing, sample_requirements)
        assert result["grade"] in ("A", "B", "C", "D", "F")

    def test_score_content_title_too_long_penalized(self, sample_requirements):
        from mcp_servers.scoring_mcp_server import score_content
        content = {"title": "A" * 210, "bullet_points": [], "description": "", "backend_keywords": ""}
        result = score_content(content, sample_requirements)
        # Title exceeds 200 char limit — issues list should flag it
        issues_text = " ".join(result["issues"]).lower()
        assert "long" in issues_text or "title" in issues_text or result["total_score"] < 80

    def test_score_content_prohibited_word_in_title_penalized(self, sample_requirements):
        from mcp_servers.scoring_mcp_server import score_content
        content = {
            "title": "The Best Wireless Headphones Ever #1 Guaranteed",
            "bullet_points": [],
            "description": "",
            "backend_keywords": "",
        }
        result = score_content(content, sample_requirements)
        issues_text = " ".join(result["issues"]).lower()
        assert "prohibited" in issues_text or result["scores"]["title_compliance"] < 60

    def test_check_compliance_compliant_content(self, sample_requirements):
        from mcp_servers.scoring_mcp_server import check_compliance
        content = {
            "title": "SoundWave Pro X1 Wireless Bluetooth Headphones Active Noise Cancellation 30Hr",
            "bullet_points": [
                "Great sound quality and performance for everyday use",
                "Long-lasting 30-hour battery life for all-day listening",
                "Comfortable memory foam ear cushions reduce fatigue",
                "Bluetooth 5.2 with multipoint for seamless device switching",
                "IPX4 water resistance handles sweat and light rain",
            ],
            "description": "High-performance wireless headphones for work and commute.",
            "backend_keywords": "over ear headphones noise cancelling wireless",
            "category": "electronics",
        }
        result = check_compliance(content, "amazon")
        assert isinstance(result["compliant"], bool)
        assert isinstance(result["violations"], list)

    def test_check_compliance_too_many_bullets_fails(self):
        from mcp_servers.scoring_mcp_server import check_compliance
        content = {
            "title": "SoundWave Headphones",
            "bullet_points": ["Bullet 1", "Bullet 2", "Bullet 3", "Bullet 4", "Bullet 5", "Bullet 6"],
            "description": "Good headphones",
            "category": "electronics",
        }
        result = check_compliance(content, "amazon")
        assert result["compliant"] is False
        assert result["checks"]["bullet_count"] is False

    def test_check_brand_safety_clean_content(self):
        from mcp_servers.scoring_mcp_server import check_brand_safety
        content = {
            "title": "SoundWave Pro X1 Wireless Headphones",
            "bullet_points": ["30-hour battery", "Active noise cancellation"],
            "description": "Quality headphones for everyday use.",
        }
        result = check_brand_safety(content)
        assert result["passed"] is True
        assert result["flag_count"] == 0

    def test_check_brand_safety_medical_claim_flagged(self):
        from mcp_servers.scoring_mcp_server import check_brand_safety
        content = {
            "title": "Headphones that cures hearing fatigue",
            "bullet_points": [],
            "description": "These headphones treat tinnitus naturally.",
        }
        result = check_brand_safety(content)
        assert result["passed"] is False
        assert result["flag_count"] > 0

    def test_check_brand_safety_pii_flagged(self):
        from mcp_servers.scoring_mcp_server import check_brand_safety
        content = {
            "title": "SoundWave Headphones",
            "bullet_points": [],
            "description": "Contact us at support@example.com for help.",
        }
        result = check_brand_safety(content)
        assert result["flag_count"] > 0

    def test_diff_content_changed_fields(self):
        from mcp_servers.scoring_mcp_server import diff_content
        old = {
            "title": "Old title",
            "bullet_points": ["Bullet A", "Bullet B"],
            "description": "Old description",
            "backend_keywords": "old keywords",
        }
        new = {
            "title": "New improved title with keywords",
            "bullet_points": ["Better Bullet A", "Better Bullet B"],
            "description": "New description with more detail",
            "backend_keywords": "new keywords better",
        }
        result = diff_content(old, new)
        assert isinstance(result["field_diffs"], list)
        assert isinstance(result["bullets_changed"], int)
        assert result["bullets_changed"] == 2  # Both bullets changed

    def test_mcp_call_tool_dispatch(self):
        from mcp_servers.retailer_mcp_server import call_tool
        result = call_tool("get_listing", asin="ANKER-Q30-BLK", retailer="amazon")
        assert "title" in result

    def test_mcp_call_tool_unknown_tool(self):
        from mcp_servers.retailer_mcp_server import call_tool
        result = call_tool("nonexistent_tool")
        assert "error" in result
