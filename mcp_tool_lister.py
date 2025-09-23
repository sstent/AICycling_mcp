#!/usr/bin/env python3
"""
Script to launch an MCP server in the background and list its available tools.
"""

import asyncio
import json
import platform
import subprocess
import sys
import time
import yaml
import os
import shutil
import logging
from typing import Dict, List, Any, Optional

# MCP Protocol imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not available. Install with: pip install mcp")

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
        name = tool.name
        description = tool.description if hasattr(tool, 'description') else 'No description available'
        
        print(f"\n{i}. {name}")
        print(f"   Description: {description}")
        
        # Print input schema if available
        input_schema = tool.input_schema if hasattr(tool, 'input_schema') else {}
        if input_schema:
            properties = input_schema.get("properties", {})
            if properties:
                print("   Parameters:")
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "unknown")
                    prop_desc = prop_info.get("description", "")
                    required = prop_name in input_schema.get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    print(f"     - {prop_name} ({prop_type}){req_str}: {prop_desc}")
    
    print(f"\n{'='*60}")

async def main():
    if not MCP_AVAILABLE:
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python mcp_tool_lister.py <server_command> [args...]")
        print("Example: python mcp_tool_lister.py uvx garth-mcp-server")
        sys.exit(1)
    
    server_command_args = sys.argv[1:]

    # Load config
    with open("config.yaml") as f:
        config_data = yaml.safe_load(f)
    
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

    server_params = StdioServerParameters(
        command="/bin/bash",
        args=["-c", f"exec {' '.join(server_command_args)} 1>&2"],
        capture_stderr=True,
        env=env,
    )

    async def log_stderr(stderr):
        async for line in stderr:
            logger.info(f"[server-stderr] {line.decode().strip()}")

    client_context = None
    try:
        logger.info(f"Starting MCP server: {' '.join(server_command_args)}")
        client_context = stdio_client(server_params)
        streams = await client_context.__aenter__()
        if len(streams) == 3:
            read_stream, write_stream, stderr_stream = streams
            stderr_task = asyncio.create_task(log_stderr(stderr_stream))
        else:
            read_stream, write_stream = streams
            stderr_task = None

        session = ClientSession(read_stream, write_stream)
        await session.initialize()
        
        server_info = session.server_info
        if server_info:
            print(f"Server: {server_info.name} v{server_info.version}")

        tools_result = await session.list_tools() # Corrected from list__tools()
        tools = tools_result.tools if tools_result else []
        
        print_tools(tools)

        if stderr_task:
            stderr_task.cancel()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if client_context:
            await client_context.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
