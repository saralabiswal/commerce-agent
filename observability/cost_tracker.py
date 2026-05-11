"""
Cost tracker — estimates token costs per provider/model.
Logged to SQLite for aggregate cost reporting in the dashboard.

Owner: Sarala Biswal
"""
from llm.base import LLMProvider

# Rough cost per 1M tokens — update as pricing changes
_COST_TABLE = {
    # (provider, model): (input_per_M, output_per_M)
    ("anthropic", "claude-sonnet-4-20250514"): (3.00, 15.00),
    ("anthropic", "claude-opus-4-20250514"): (15.00, 75.00),
    ("anthropic", "claude-haiku-4-5-20251001"): (0.25, 1.25),
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("ollama", "*"): (0.0, 0.0),         # Local — no cost
    ("huggingface_local", "*"): (0.0, 0.0),
    ("huggingface_api", "*"): (0.10, 0.10),
}

# Rough token estimates per pipeline stage
_STAGE_TOKEN_ESTIMATES = {
    "content_audit": 1500,
    "competitor_analysis": 2000,
    "content_generation": 2500,
}


class CostTracker:
    """Estimate token costs for configured LLM provider usage."""

    def __init__(self, provider: LLMProvider):
        """Initialize cost tracking for a provider instance."""
        self._provider = provider

    def estimate_token_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for a given token count."""
        key = (self._provider.provider_name, self._provider.model_name)
        rates = _COST_TABLE.get(key)

        if not rates:
            # Fall back to provider wildcard
            wildcard_key = (self._provider.provider_name, "*")
            rates = _COST_TABLE.get(wildcard_key, (1.0, 3.0))

        in_rate, out_rate = rates
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000

    def estimate_run_cost(self, total_tokens: int) -> float:
        """Rough run cost estimate when per-call tracking isn't available."""
        # Assume 40/60 input/output split
        input_tokens = int(total_tokens * 0.4)
        output_tokens = int(total_tokens * 0.6)
        return self.estimate_token_cost(input_tokens, output_tokens)

    def estimate_pipeline_cost(self) -> dict:
        """Estimate cost breakdown by pipeline stage."""
        result = {}
        for stage, tokens in _STAGE_TOKEN_ESTIMATES.items():
            result[stage] = {
                "estimated_tokens": tokens,
                "estimated_cost_usd": self.estimate_run_cost(tokens),
            }
        total_tokens = sum(_STAGE_TOKEN_ESTIMATES.values())
        result["total"] = {
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": self.estimate_run_cost(total_tokens),
        }
        return result
