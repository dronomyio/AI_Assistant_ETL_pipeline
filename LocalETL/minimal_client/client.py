import asyncio
import os
import subprocess
import sys
from asyncio import Event
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from mcp.client import SSEMCPClient
from mcp.client.sse import SseClientTransport
from mcp.core.internal.jsonschema_utils import (
    convert_schema_to_python_types_and_comments,
)
from mcp.core.spec.client import MCPResponse, ToolCall, ToolCallParameters, ToolResponse


class MCPClient:
    """A minimal client to interact with an MCP Server."""

    def __init__(self) -> None:
        """Initialize the MCPClient."""
        self.mcp_client: Optional[SSEMCPClient] = None
        self.process: Optional[subprocess.Popen] = None
        self.tool_metadata: Dict[str, dict] = {}  # Save tool schemas for printing to console
        self.tool_examples: Dict[str, List[tuple]] = {}  # Save example tool uses
        self.tool_calls_in_progress: Dict[str, Event] = {}
        self.client = anthropic.Anthropic()

    async def connect_to_server(self, server_path: str, args: Optional[List[str]] = None) -> None:
        """Start the MCP server and connect to it."""
        if args is None:
            args = []

        url = server_path
        if not url.startswith("http"):
            # Start local MCP server
            server_path = Path(server_path)
            project_root = server_path.parent.parent

            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(server_path.absolute()),
                ],
                cwd=project_root,
                env=os.environ.copy(),
            )
            url = "mcp"  # Local IPC connection ID

        # Create a client
        self.mcp_client = SSEMCPClient(SseClientTransport(url))
        await self.mcp_client.initialize()

        # Get the tool schemas
        for name, schema in self.mcp_client.tools.items():
            self.tool_metadata[name] = schema
            self.tool_examples[name] = []

    async def tool_call_callback(
        self,
        tool_name: str,
        parameters: ToolCallParameters,
        session_id: str,
        tool_call_id: str,
    ) -> Tuple[bool, Any]:
        """
        Called when a tool call is made by the MCP server.
        Returns (should_execute, result).
        """
        param_str = ", ".join(f"{k}={repr(v)}" for k, v in parameters.items())
        print(f"\n\033[1;36mTOOL CALL\033[0m")
        print(
            f"Accept execution of {tool_name} with args {param_str}? \033[1;35m[y/n]\033[0m \033[1;36m(y)\033[0m: ",
            end="",
        )
        sys.stdout.flush()

        # Create an event to manage this tool call
        done_event = Event()
        self.tool_calls_in_progress[tool_call_id] = done_event

        # Default to 'y'
        try:
            decision = asyncio.create_task(asyncio.to_thread(input))
            result = await asyncio.wait_for(decision, timeout=2.0)
        except asyncio.TimeoutError:
            result = "y"
            print()  # Print a newline

        if result.lower() != "n":
            # Execute the tool
            return True, None
        else:
            # Don't execute the tool
            done_event.set()
            del self.tool_calls_in_progress[tool_call_id]
            return False, "Tool execution declined by user"

    async def tool_response_callback(
        self,
        tool_response: ToolResponse,
        session_id: str,
        tool_call_id: str,
    ) -> None:
        """Called when a tool response is received from the MCP server."""
        # Print the tool result
        print("\n\033[1;36mTOOL OUTPUT\033[0m:")
        
        # Color the output to highlight things like IDs
        output = tool_response.outputs
        if isinstance(output, str):
            # Color UUIDs in gold
            import re
            uuid_pattern = r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
            output = re.sub(uuid_pattern, r"\033[93m\1\033[0m", output)
            
            # Color numbers in cyan
            number_pattern = r"(\d+)"
            output = re.sub(number_pattern, r"\033[1;36m\1\033[0m", output)
            
            # Bold parenthesis and their contents
            paren_pattern = r"(\([^)]*\))"
            output = re.sub(paren_pattern, r"\033[1m\1\033[0m", output)
            
            print(output)
        else:
            print(output)
        
        print("\n")
        
        # Mark this tool call as done
        if tool_call_id in self.tool_calls_in_progress:
            self.tool_calls_in_progress[tool_call_id].set()
            del self.tool_calls_in_progress[tool_call_id]

    async def shutdown(self) -> None:
        """Shutdown the client and server."""
        if self.mcp_client:
            await self.mcp_client.shutdown()
            self.mcp_client = None

        if self.process:
            self.process.terminate()
            self.process = None

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.shutdown()

    async def wait_for_tool_calls_completion(self) -> None:
        """Wait for all in-progress tool calls to complete."""
        if not self.tool_calls_in_progress:
            return

        # Wait for all events to be set
        await asyncio.gather(*(event.wait() for event in self.tool_calls_in_progress.values()))

    async def chat_loop(self) -> None:
        """Run a chat loop to interact with the MCP server."""
        if not self.mcp_client:
            print("Not connected to an MCP server")
            return

        print()
        query = input("\033[1;32mQuery\033[0m (q/quit to end chat): ")
        
        while query.lower() not in ("q", "quit"):
            print("\n\n")
            
            # Create a message
            response = await self.mcp_client.send_message(
                {
                    "query": query,
                },
                callback={
                    "beforeToolCall": self.tool_call_callback,
                    "afterToolCall": self.tool_response_callback,
                },
            )
            
            # Wait for all tool calls to complete
            await self.wait_for_tool_calls_completion()
            
            # Print the response
            if isinstance(response, MCPResponse):
                print("\033[1;31mASSISTANT\033[0m")
                print(response.response)
            
            # Get the next query
            print()
            query = input("\033[1;32mQuery\033[0m (q/quit to end chat): ")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_path_or_url>")
        sys.exit(1)

    server_path = sys.argv[1]
    
    client = MCPClient()
    try:
        asyncio.run(client.connect_to_server(server_path))
        asyncio.run(client.chat_loop())
    finally:
        asyncio.run(client.cleanup())