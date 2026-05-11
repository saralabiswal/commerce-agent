"""
RetailerMCP — provides retailer-specific data and requirements to agents.

Tools:
  get_listing(asin, retailer)       → current listing data
  get_retailer_requirements(cat)    → character limits, prohibited words, rules
  get_search_volume(keyword)        → simulated keyword search volume
  get_category_benchmarks(cat)      → avg quality scores for category

Design note: MCP over direct function calls gives us versioning, access control,
and tool reusability across agents without tight coupling (see ADR-001).

Owner: Sarala Biswal
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "mock_data"
SKU_ALIASES = {
    "DEMO-SKU-001": "ANKER-Q30-BLK",
}


def _load_json(filename: str) -> dict:
    """Load a mock retailer data file from the MCP data directory."""
    with open(DATA_DIR / filename) as f:
        return json.load(f)


# ── Tool implementations ──────────────────────────────────────────────────────

def get_listing(asin: str, retailer: str = "amazon") -> dict:
    """
    Fetch the current product listing for an ASIN on a given retailer.

    Returns listing data including title, bullets, description, and performance
    signals (BSR, review count, rating).
    """
    asin = SKU_ALIASES.get(asin, asin)
    data = _load_json("listings.json")
    retailer_data = data.get(retailer.lower(), {})
    listing = retailer_data.get(asin)
    if not listing:
        listing = next(
            (
                item
                for item in retailer_data.values()
                if item.get("sku") == asin or item.get("asin") == asin
            ),
            None,
        )

    if not listing:
        # Return a minimal stub for unknown ASINs (simulates API behavior)
        return {
            "asin": asin,
            "sku": asin,
            "retailer": retailer,
            "title": f"[No listing found for {asin} on {retailer}]",
            "bullet_points": [],
            "description": "",
            "backend_keywords": "",
            "category": "electronics",
            "brand": "Unknown",
            "price": 0.0,
            "bsr": None,
            "review_count": 0,
            "review_rating": 0.0,
            "error": f"Listing not found: {asin} on {retailer}",
        }

    return listing


def get_retailer_requirements(category: str, retailer: str = "amazon") -> dict:
    """
    Fetch retailer-specific content requirements for a product category.

    Includes character limits, prohibited words, required elements, and
    category-specific rules for title, bullets, description, and backend keywords.
    """
    data = _load_json("retailer_requirements.json")
    retailer_data = data.get(retailer.lower(), {})
    requirements = retailer_data.get(category.lower())

    if not requirements:
        # Fall back to electronics as the most common category
        requirements = retailer_data.get("electronics", {})
        requirements = dict(requirements)  # copy
        requirements["_fallback"] = True
        requirements["_requested_category"] = category

    return requirements


def get_search_volume(keyword: str, retailer: str = "amazon") -> dict:
    """
    Return simulated monthly search volume for a keyword.

    In production this would call a keyword data API (e.g. Helium10, Jungle Scout).
    Here we return realistic mock data with deterministic results for known keywords.
    """
    data = _load_json("retailer_requirements.json")
    retailer_reqs = data.get(retailer.lower(), {})

    # Check all categories for benchmark data
    for category_data in retailer_reqs.values():
        benchmarks = category_data.get("search_volume_benchmarks", {})
        if keyword.lower() in benchmarks:
            volume = benchmarks[keyword.lower()]
            return {
                "keyword": keyword,
                "monthly_searches": volume,
                "trend": "stable",
                "competition": "high" if volume > 100000 else "medium" if volume > 30000 else "low",
                "retailer": retailer,
            }

    # Generate plausible mock data for unknown keywords
    seed = sum(ord(c) for c in keyword)
    random.seed(seed)
    volume = random.randint(1000, 500000)
    return {
        "keyword": keyword,
        "monthly_searches": volume,
        "trend": random.choice(["rising", "stable", "declining"]),
        "competition": "high" if volume > 100000 else "medium" if volume > 30000 else "low",
        "retailer": retailer,
        "note": "simulated data",
    }


def get_category_benchmarks(category: str, retailer: str = "amazon") -> dict:
    """
    Return average content quality scores and benchmarks for a category.

    Helps agents understand what "good" looks like for a given category
    before generating or auditing content.
    """
    benchmarks = {
        "amazon": {
            "electronics": {
                "avg_quality_score": 72,
                "top_quartile_score": 88,
                "avg_title_length": 148,
                "avg_bullet_length": 195,
                "avg_keyword_count": 8,
                "top_performers_have_anc": True,
                "category": "electronics",
                "retailer": "amazon",
            },
            "grocery": {
                "avg_quality_score": 68,
                "top_quartile_score": 84,
                "avg_title_length": 120,
                "avg_bullet_length": 180,
                "avg_keyword_count": 6,
                "category": "grocery",
                "retailer": "amazon",
            },
            "apparel": {
                "avg_quality_score": 65,
                "top_quartile_score": 82,
                "avg_title_length": 130,
                "avg_bullet_length": 170,
                "avg_keyword_count": 7,
                "category": "apparel",
                "retailer": "amazon",
            },
        },
        "walmart": {
            "electronics": {
                "avg_quality_score": 70,
                "top_quartile_score": 85,
                "avg_title_length": 135,
                "avg_bullet_length": 300,
                "avg_keyword_count": 9,
                "category": "electronics",
                "retailer": "walmart",
            }
        },
    }

    retailer_bm = benchmarks.get(retailer.lower(), benchmarks["amazon"])
    return retailer_bm.get(category.lower(), retailer_bm.get("electronics", {}))


# ── MCP Server entrypoint ─────────────────────────────────────────────────────

TOOLS = {
    "get_listing": get_listing,
    "get_retailer_requirements": get_retailer_requirements,
    "get_search_volume": get_search_volume,
    "get_category_benchmarks": get_category_benchmarks,
}


def call_tool(tool_name: str, **kwargs) -> dict:
    """Dispatch a tool call by name. Used by agents via MCP protocol."""
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}"}
    try:
        return TOOLS[tool_name](**kwargs)
    except Exception as e:
        return {"error": str(e), "tool": tool_name, "args": kwargs}


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    from config import settings

    app = FastAPI(title="RetailerMCP", version="1.0")

    @app.get("/tools")
    def list_tools():
        """Return the registered retailer tool names."""
        return {"tools": list(TOOLS.keys())}

    @app.post("/call/{tool_name}")
    def call(tool_name: str, args: dict):
        """Invoke a registered retailer tool from the HTTP wrapper."""
        return call_tool(tool_name, **args)

    uvicorn.run(app, host="0.0.0.0", port=settings.retailer_mcp_port)
