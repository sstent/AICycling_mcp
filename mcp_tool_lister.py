#!/usr/bin/env python3
"""
MCP Tool Lister - Lists available tools, executes user_profile, get_activities, and get_activity_details tools
"""

import asyncio
import logging
import yaml
from mcp_manager import Config, PydanticAIAnalyzer, print_tools
import json

# Configure extensive debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting MCP tool lister")
    analyzer = None
    try:
        # Load configuration from config.yaml
        logger.debug("Loading configuration from config.yaml")
        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
        logger.debug(f"Loaded config data: {config_data}")
        
        config = Config(**config_data)
        logger.info("Configuration loaded and Config object created")
        
        # Initialize the analyzer
        logger.debug("Creating PydanticAIAnalyzer instance")
        analyzer = PydanticAIAnalyzer(config)
        logger.info("PydanticAIAnalyzer instance created")
        
        # Initialize the analyzer (starts MCP server and lists tools)
        logger.debug("Initializing analyzer (starting MCP server)")
        await analyzer.initialize()
        logger.info("Analyzer initialized successfully")
        
        # List available tools
        logger.debug(f"Available tools count: {len(analyzer.available_tools)}")
        print_tools(analyzer.available_tools)
        logger.info("Available tools listed and printed")
        
        # Pre-call user_profile tool
        logger.debug("Pre-calling user_profile tool")
        user_profile = await analyzer.get_user_profile()
        print("\n" + "="*60)
        print("RAW USER PROFILE (Pre-cached)")
        print("="*60)
        print(json.dumps(user_profile, indent=2, default=str))
        print("="*60)
        logger.info("User profile pre-cached and printed")
        
        # Pre-call get_recent_cycling_activity_details
        logger.debug("Pre-calling get_recent_cycling_activity_details")
        activity_data = await analyzer.get_recent_cycling_activity_details()
        
        print("\n" + "="*60)
        print("RAW RECENT ACTIVITIES (Pre-cached)")
        print("="*60)
        print(json.dumps(activity_data.get("activities", []), indent=2, default=str))
        print("="*60)
        
        if activity_data.get("last_cycling"):
            print("\n" + "="*60)
            print("LAST CYCLING ACTIVITY SUMMARY (Pre-cached)")
            print("="*60)
            print(json.dumps(activity_data["last_cycling"], indent=2, default=str))
            print("="*60)
            
            print("\n" + "="*60)
            print("ACTIVITY DETAILS (Pre-cached)")
            print("="*60)
            print(json.dumps(activity_data["details"], indent=2, default=str))
            print("="*60)
            logger.info("Recent cycling activity details pre-cached and printed")
        else:
            logger.warning("No cycling activity found in recent activities")
            print("\nWarning: No cycling activity found in recent activities.")
        
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        print("Error: config.yaml not found. Please ensure the file exists.")
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        print("Error: Invalid YAML in config.yaml.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error during execution: {e}")
    finally:
        # Ensure proper cleanup
        if analyzer:
            logger.debug("Performing cleanup")
            await analyzer.cleanup()
            logger.info("Cleanup completed successfully")
        else:
            logger.warning("No analyzer to cleanup")

if __name__ == "__main__":
    asyncio.run(main())