#!/usr/bin/env python3
"""
Test MCP server directly
"""
import os
import yaml
import subprocess
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_direct():
    # Load token from config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    token = config['garth_token']

    # Set up environment
    env = os.environ.copy()
    env['GARTH_TOKEN'] = token

    # Set up server parameters
    server_params = StdioServerParameters(
        command="uvx",
        args=["garth-mcp-server"],
        env=env
    )

    print("Starting MCP server test...")
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            print("Initializing session...")
            result = await session.initialize()
            print("✓ Session initialized")

            print("Getting tools...")
            tools_result = await session.list_tools()
            tools = tools_result.tools if tools_result else []
            print(f"✓ Found {len(tools)} tools")

            for tool in tools[:5]:  # Show first 5 tools
                print(f"  - {tool.name}: {getattr(tool, 'description', 'No description')}")

            if len(tools) > 5:
                print(f"  ... and {len(tools) - 5} more tools")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_direct())