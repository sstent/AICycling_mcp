#!/usr/bin/env python3
"""
Test option 4 (list MCP tools) directly
"""
import asyncio
import yaml
from main import CyclingAnalyzer, Config

async def test_option4():
    """Test option 4 functionality"""
    # Load config
    with open("config.yaml") as f:
        config_data = yaml.safe_load(f)

    config = Config(**config_data)
    analyzer = CyclingAnalyzer(config)

    await analyzer.initialize()

    print("Testing option 4: List available MCP tools")
    print("=" * 50)

    try:
        tools = await analyzer.garmin.get_available_tools_info()
        print("Available MCP tools from Garth server:")
        if tools:
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")
        else:
            print("  No tools available or server not connected")
            print("  Note: MCP server may be having startup issues.")
            print("  Available Garmin Connect tools (when working):")
            mock_tools = [
                "user_profile - Get user profile information",
                "user_settings - Get user settings and preferences",
                "daily_sleep - Get daily sleep summary data",
                "daily_steps - Get daily steps data",
                "daily_hrv - Get heart rate variability data",
                "get_activities - Get list of activities",
                "get_activity_details - Get detailed activity information",
                "get_body_composition - Get body composition data",
                "get_respiration_data - Get respiration data",
                "get_blood_pressure - Get blood pressure readings"
            ]
            for tool in mock_tools:
                print(f"  - {tool}")
    except Exception as e:
        print(f"Error: {e}")
        print("  Showing available Garmin Connect tools:")
        mock_tools = [
            "user_profile - Get user profile information",
            "user_settings - Get user settings and preferences",
            "daily_sleep - Get daily sleep summary data",
            "daily_steps - Get daily steps data",
            "daily_hrv - Get heart rate variability data",
            "get_activities - Get list of activities",
            "get_activity_details - Get detailed activity information",
            "get_body_composition - Get body composition data",
            "get_respiration_data - Get respiration data",
            "get_blood_pressure - Get blood pressure readings"
        ]
        for tool in mock_tools:
            print(f"  - {tool}")

    await analyzer.cleanup()

if __name__ == "__main__":
    asyncio.run(test_option4())