#!/usr/bin/env python3
"""
Test script to verify MCP tools functionality
"""
import asyncio
import yaml
from main import GarthMCPConnector, Config

async def test_mcp_tools():
    """Test the MCP tools functionality"""
    # Load config
    with open("config.yaml") as f:
        config_data = yaml.safe_load(f)

    config = Config(**config_data)
    garmin = GarthMCPConnector(config.garth_token, config.garth_mcp_server_path)

    print("Testing MCP tools retrieval...")
    try:
        tools = await garmin.get_available_tools_info()
        print(f"Successfully retrieved {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")

        # Test caching by calling again
        print("\nTesting cached tools...")
        tools2 = await garmin.get_available_tools_info()
        print(f"Cached tools: {len(tools2)} tools")

        if tools == tools2:
            print("✓ Caching works correctly!")
        else:
            print("✗ Caching failed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())