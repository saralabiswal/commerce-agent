"""Local MCP-style tool server exports used by the agents.

Owner: Sarala Biswal
"""

from mcp_servers.catalog_mcp_server import call_tool as catalog_call
from mcp_servers.retailer_mcp_server import call_tool as retailer_call
from mcp_servers.scoring_mcp_server import call_tool as scoring_call

__all__ = ["retailer_call", "catalog_call", "scoring_call"]
