#!/usr/bin/env python3
"""
Script to launch an MCP server in the background and list its available tools.
"""

import asyncio
import yaml
import os
import shutil
import logging
import sys
from typing import List, Any
from pydantic_ai.mcp import MCPServerStdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_tools(tools: List[Any]):
    """Pretty print the tools list."""
    if not tools:
        print("\nNo tools available.")
        return

    print(f"\n{'='*60}")
    print("AVAILABLE TOOLS")
    print(f"\n{'='*60}")

    for i, tool in enumerate(tools, 1):
        print(f"\n{i}. {tool.name}")
        if tool.description:
            print(f"   Description: {tool.description}")

        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            properties = tool.inputSchema.get("properties", {})
            if properties:
                print("   Parameters:")
                required_params = tool.inputSchema.get("required", [])
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "unknown")
                    prop_desc = prop_info.get("description", "")
                    required = prop_name in required_params
                    req_str = " (required)" if required else " (optional)"
                    print(f"     - {prop_name} ({prop_type}){req_str}: {prop_desc}")

    print(f"\n{'='*60}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_tool_lister.py <server_command> [args...]")
        print("Example: python mcp_tool_lister.py uvx garth-mcp-server")
        sys.exit(1)

    server_command_args = sys.argv[1:]

    # Load config
    try:
        with open("config.yaml") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        sys.exit(1)

    garth_token = config_data.get("garth_token")
    if not garth_token:
        print("Error: garth_token not found in config.yaml")
        sys.exit(1)

    env = os.environ.copy()
    env["GARTH_TOKEN"] = garth_token

    server_command = shutil.which(server_command_args[0])
    if not server_command:
        logger.error(f"Could not find '{server_command_args[0]}' in your PATH.")
        raise FileNotFoundError(f"{server_command_args[0]} not found")

    server = MCPServerStdio(
        command=server_command,
        args=server_command_args[1:],
        env=env,
    )

    try:
        logger.info(f"Starting MCP server: {' '.join(server_command_args)}")
        async with server:
            tools = await server.list_tools()
            print_tools(tools)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
