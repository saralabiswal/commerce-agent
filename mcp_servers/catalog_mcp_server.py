"""
CatalogMCP — provides product catalog and brand guideline data to agents.

Tools:
  get_product_specs(sku)           → authoritative product specifications
  get_brand_guidelines(brand)      → tone, prohibited claims, required disclaimers
  get_competitor_listings(asin)    → top 5 competitor listings for category
  get_historical_content(sku)      → previous content versions + performance
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "mock_data"
SKU_ALIASES = {
    "DEMO-SKU-001": "ANKER-Q30-BLK",
}


def _load_json(filename: str) -> dict:
    with open(DATA_DIR / filename) as f:
        return json.load(f)


# ── Tool implementations ──────────────────────────────────────────────────────

def list_products() -> list[dict]:
    """Return catalog products for UI selectors and product pickers."""
    data = _load_json("competitor_data.json")
    catalog = data.get("product_catalog", {})
    return [
        {
            "sku": sku,
            "product_name": product.get("product_name", sku),
            "brand": product.get("brand", ""),
            "product_type": product.get("product_type", ""),
        }
        for sku, product in sorted(catalog.items())
        if sku not in SKU_ALIASES
    ]


def get_product_specs(sku: str) -> dict:
    """
    Fetch authoritative product specifications for a SKU.

    This is the source of truth for product facts. ContentGenerationAgent
    must never invent specifications not present in this data (guardrail enforced).
    """
    data = _load_json("competitor_data.json")
    catalog = data.get("product_catalog", {})
    sku = SKU_ALIASES.get(sku, sku)
    product = catalog.get(sku)

    if not product:
        return {
            "sku": sku,
            "error": f"Product not found in catalog: {sku}",
            "note": "Cannot generate content without authoritative specs.",
        }

    return product


def get_brand_guidelines(brand: str) -> dict:
    """
    Fetch brand guidelines — tone of voice, prohibited claims, required disclaimers.

    In production this would query a DAM (Digital Asset Management) system.
    """
    data = _load_json("competitor_data.json")
    catalog = data.get("product_catalog", {})

    # Find guidelines for the first product matching this brand
    for product_data in catalog.values():
        if product_data.get("brand", "").lower() == brand.lower():
            guidelines = product_data.get("brand_guidelines", {})
            if guidelines:
                return {**guidelines, "brand": brand}

    # Default guidelines for unknown brands
    return {
        "brand": brand,
        "tone": "professional, benefit-focused",
        "prohibited_claims": ["guaranteed", "best in class (unsubstantiated)", "#1"],
        "required_disclaimers": [],
        "key_differentiators": [],
        "target_audience": "general consumer",
        "_fallback": True,
    }


def get_competitor_listings(
    category: str, limit: int = 5
) -> list[dict]:
    """
    Fetch top competitor listings for a product category.

    Returns listings ranked by quality score (proxy for content performance).
    CompetitorAnalysisAgent uses this to identify winning content patterns.
    """
    data = _load_json("competitor_data.json")
    competitors = data.get("competitor_listings", {}).get(category.lower(), [])

    # Sort by quality score descending, return top N
    competitors_sorted = sorted(
        competitors, key=lambda x: x.get("quality_score", 0), reverse=True
    )
    return competitors_sorted[:limit]


def get_historical_content(sku: str) -> list[dict]:
    """
    Fetch historical content versions and their performance signals for a SKU.

    In production this would query a content versioning database.
    Returns a list of {version, content, score, date} records.
    """
    sku = SKU_ALIASES.get(sku, sku)
    if sku == "ANKER-Q30-BLK":
        return [
            {
                "version": 1,
                "date": "2024-01-15",
                "title": "Soundcore Wireless Headphones",
                "bullet_count": 3,
                "quality_score": 42,
                "bsr_at_time": 4200,
                "notes": "Initial listing — minimal content",
            },
            {
                "version": 2,
                "date": "2024-06-01",
                "title": "Soundcore Life Q30 Wireless Noise Cancelling Headphones",
                "bullet_count": 5,
                "quality_score": 58,
                "bsr_at_time": 2800,
                "notes": "Added bullets, slight BSR improvement",
            },
            {
                "version": 3,
                "date": "2025-01-10",
                "title": "Soundcore Life Q30 Hybrid Active Noise Cancelling Headphones",
                "bullet_count": 5,
                "quality_score": 61,
                "bsr_at_time": 1843,
                "notes": "Current version — stagnant for 16 months",
            },
        ]

    return []


# ── MCP Server entrypoint ─────────────────────────────────────────────────────

TOOLS = {
    "list_products": list_products,
    "get_product_specs": get_product_specs,
    "get_brand_guidelines": get_brand_guidelines,
    "get_competitor_listings": get_competitor_listings,
    "get_historical_content": get_historical_content,
}


def call_tool(tool_name: str, **kwargs) -> dict | list:
    """Dispatch a tool call by name."""
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

    app = FastAPI(title="CatalogMCP", version="1.0")

    @app.get("/tools")
    def list_tools():
        return {"tools": list(TOOLS.keys())}

    @app.post("/call/{tool_name}")
    def call(tool_name: str, args: dict):
        return call_tool(tool_name, **args)

    uvicorn.run(app, host="0.0.0.0", port=settings.catalog_mcp_port)
