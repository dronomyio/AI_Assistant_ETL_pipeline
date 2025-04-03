from mcp.server.fastmcp import FastMCP

from connectors.source.local_dir import create_local_source, update_local_source, delete_local_source
from connectors.destination.local_dir import (
    create_local_destination,
    update_local_destination,
    delete_local_destination,
)


def register_connectors(mcp: FastMCP) -> None:
    """Register all connectors with the MCP server."""
    # Source connectors
    mcp.register_tool(create_local_source)
    mcp.register_tool(update_local_source)
    mcp.register_tool(delete_local_source)

    # Destination connectors
    mcp.register_tool(create_local_destination)
    mcp.register_tool(update_local_destination)
    mcp.register_tool(delete_local_destination)