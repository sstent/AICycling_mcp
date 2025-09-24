#!/usr/bin/env python3
"""
MCP Manager for Pydantic AI Cycling Analyzer
"""

import os
import json
import asyncio
import shutil
import logging
from typing import List, Any
from dataclasses import dataclass
import garth

# Pydantic AI imports
try:
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None
    print("Pydantic AI not available. Install with: pip install pydantic-ai")

from templates_manager import TemplateManager

# MCP Protocol imports for direct connection
try:
    from pydantic_ai.mcp import MCPServerStdio
    from pydantic_ai import exceptions
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPServerStdio = None
    exceptions = None
    print("pydantic_ai.mcp not available. You might need to upgrade pydantic-ai.")

# Configure logging for this module
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Application configuration"""
    openrouter_api_key: str
    openrouter_model: str = "deepseek/deepseek-r1-0528:free"
    garth_token: str = ""
    garth_mcp_server_path: str = "uvx"
    rules_file: str = "rules.yaml"
    templates_dir: str = "templates"

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

class PydanticAIAnalyzer:
    """Pydantic AI powered cycling analyzer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.template_manager = TemplateManager(self.config.templates_dir)
        self.mcp_server = None
        self.available_tools = []
        self._cached_activity_details = None
        
        if not PYDANTIC_AI_AVAILABLE or not MCP_AVAILABLE:
            raise Exception("Pydantic AI or MCP not available. Please check your installation.")
        
        os.environ['OPENROUTER_API_KEY'] = config.openrouter_api_key
        os.environ['OPENAI_BASE_URL'] = "https://openrouter.ai/api/v1"
        os.environ['OPENAI_DEFAULT_HEADERS'] = json.dumps({
            "HTTP-Referer": "https://github.com/cycling-analyzer",
            "X-Title": "Cycling Workout Analyzer"
        })
        
        env = os.environ.copy()
        os.environ["GARTH_TOKEN"] = config.garth_token
        env["GARTH_TOKEN"] = config.garth_token
        
        server_executable = shutil.which(config.garth_mcp_server_path)
        if not server_executable:
            logger.error(f"'{config.garth_mcp_server_path}' not found in PATH. MCP tools will be unavailable.")
        else:
            self.mcp_server = MCPServerStdio(
                command=server_executable,
                args=["garth-mcp-server"],
                env=env,
            )

        model_name = f"openrouter:{config.openrouter_model}"
        
        main_system_prompt = self.template_manager.get_template('base/system_prompts/main_agent.txt')
        
        self.agent = Agent(
            model=model_name,
            system_prompt=main_system_prompt,
            toolsets=[self.mcp_server] if self.mcp_server else []
        )

    async def initialize(self):
        """Initialize the analyzer and connect to MCP server"""
        logger.info("Initializing Pydantic AI analyzer...")
        if self.agent and self.mcp_server:
            try:
                logger.info("Attempting to enter agent context...")
                await asyncio.wait_for(self.agent.__aenter__(), timeout=45)
                logger.info("✓ Agent context entered successfully")
                logger.info("Listing available MCP tools...")
                self.available_tools = await self.mcp_server.list_tools()
                logger.info(f"✓ Found {len(self.available_tools)} MCP tools.")
                if self.available_tools:
                    for tool in self.available_tools[:5]:  # Log first 5 tools
                        logger.info(f"  Tool: {tool.name} - {getattr(tool, 'description', 'No description')}")
                    if len(self.available_tools) > 5:
                        logger.info(f"  ... and {len(self.available_tools) - 5} more tools")
                else:
                    logger.warning("No tools returned from MCP server!")
            except asyncio.TimeoutError:
                logger.error("Agent initialization timed out. MCP tools will be unavailable.")
                self.mcp_server = None
            except Exception as e:
                logger.error(f"Agent initialization failed: {e}. MCP tools will be unavailable.")
                logger.error(f"Exception type: {type(e)}")
                import traceback
                logger.error(f"Full initialization traceback: {traceback.format_exc()}")
                self.mcp_server = None
        else:
            logger.warning("MCP server not configured. MCP tools will be unavailable.")

    async def cleanup(self):
        """Cleanup resources"""
        if self.agent and self.mcp_server:
            await self.agent.__aexit__(None, None, None)
        logger.info("Cleanup completed")

    async def get_recent_cycling_activity_details(self) -> dict:
        """Pre-call get_activities and get_activity_details to cache the last cycling activity details"""
        if self._cached_activity_details is not None:
            logger.debug("Returning cached activity details")
            return self._cached_activity_details

        if not self.mcp_server:
            logger.error("MCP server not available")
            return {}

        try:
            logger.debug("Pre-calling get_activities tool")
            activities_args = {"limit": 10}
            activities = []
            try:
                logger.debug("Bypassing direct_call_tool and using garth.connectapi directly for get_activities")
                garth.client.loads(self.config.garth_token)
                from urllib.parse import urlencode
                params = {"limit": 10}
                endpoint = "activitylist-service/activities/search/activities"
                endpoint += "?" + urlencode(params)
                activities = garth.connectapi(endpoint)
            except Exception as e:
                logger.error(f"Error calling garth.connectapi directly: {e}", exc_info=True)
                activities = []

            if not activities:
                logger.error("Failed to retrieve activities.")
                return {"error": "Failed to retrieve activities."}
            
            logger.debug(f"Retrieved {len(activities)} activities")

            # Filter for cycling activities
            cycling_activities = [
                act for act in activities
                if "cycling" in act.get("activityType", {}).get("typeKey", "").lower()
            ]

            if not cycling_activities:
                logger.warning("No cycling activities found")
                self._cached_activity_details = {"activities": activities, "last_cycling": None, "details": None}
                return self._cached_activity_details

            # Get the most recent cycling activity
            last_cycling = max(cycling_activities, key=lambda x: x.get("start_time", "1970-01-01"))
            activity_id = last_cycling["activityId"]
            logger.debug(f"Last cycling activity ID: {activity_id}")

            logger.debug("Pre-calling get_activity_details tool")
            details = garth.connectapi(f"activity-service/activity/{activity_id}")
            logger.debug("Retrieved activity details")

            self._cached_activity_details = {
                "activities": activities,
                "last_cycling": last_cycling,
                "details": details
            }
            logger.info("Cached recent cycling activity details successfully")
            return self._cached_activity_details

        except Exception as e:
            logger.error(f"Error pre-calling activity tools: {e}", exc_info=True)
            self._cached_activity_details = {"error": str(e)}
            return self._cached_activity_details

    async def get_user_profile(self) -> dict:
        """Pre-call user_profile tool to cache the response"""
        if hasattr(self, '_cached_user_profile') and self._cached_user_profile is not None:
            logger.debug("Returning cached user profile")
            return self._cached_user_profile

        if not self.mcp_server:
            logger.error("MCP server not available")
            return {}

        try:
            logger.debug("Pre-calling user_profile tool")
            profile_result = await self.mcp_server.direct_call_tool("user_profile", {})
            profile = profile_result.output if hasattr(profile_result, 'output') else profile_result
            logger.debug("Retrieved user profile")

            self._cached_user_profile = profile
            logger.info("Cached user profile successfully")
            return profile

        except Exception as e:
            logger.error(f"Error pre-calling user_profile: {e}", exc_info=True)
            self._cached_user_profile = {"error": str(e)}
            return self._cached_user_profile

    async def analyze_last_workout(self, training_rules: str) -> str:
        """Analyze the last cycling workout using Pydantic AI"""
        logger.info("Analyzing last workout with Pydantic AI...")
        
        # Get pre-cached data
        activity_data = await self.get_recent_cycling_activity_details()
        user_profile = await self.get_user_profile()
        
        if not activity_data.get("last_cycling"):
            return "No recent cycling activity found to analyze."
        
        last_activity = activity_data["last_cycling"]
        details = activity_data["details"]
        
        # Summarize key data for prompt
        activity_summary = f"""
        Last Cycling Activity:
        - Start Time: {last_activity.get('start_time', 'N/A')}
        - Duration: {last_activity.get('duration', 'N/A')} seconds
        - Distance: {last_activity.get('distance', 'N/A')} meters
        - Average Speed: {last_activity.get('averageSpeed', 'N/A')} m/s
        - Average Power: {last_activity.get('avgPower', 'N/A')} W (if available)
        - Max Power: {last_activity.get('maxPower', 'N/A')} W (if available)
        - Average Heart Rate: {last_activity.get('avgHr', 'N/A')} bpm (if available)
        
        Full Activity Details: {json.dumps(details, default=str)}
        """
        
        user_info = f"""
        User Profile:
        {json.dumps(user_profile, default=str)}
        """
        
        prompt = self.template_manager.get_template(
            'workflows/analyze_last_workout.txt',
            activity_summary=activity_summary,
            user_info=user_info,
            training_rules=training_rules
        )
 
        try:
            # Create temporary agent without tools for this analysis
            model_name = f"openrouter:{self.config.openrouter_model}"
            temp_analysis_system_prompt = self.template_manager.get_template('base/system_prompts/no_tools_analysis.txt')
            temp_agent = Agent(
                model=model_name,
                system_prompt=temp_analysis_system_prompt,
                toolsets=[]
            )
            
            # Enter context for temp agent
            await asyncio.wait_for(temp_agent.__aenter__(), timeout=30)
            
            result = await temp_agent.run(prompt)
            
            # Exit context
            await temp_agent.__aexit__(None, None, None)
            
            return str(result)
        except asyncio.TimeoutError:
            logger.error("Temp agent initialization timed out")
            return "Error: Agent initialization timed out. Please try again."
        except Exception as e:
            logger.error(f"Error in workout analysis: {e}")
            if hasattr(temp_agent, '__aexit__'):
                await temp_agent.__aexit__(None, None, None)
            return "Error analyzing workout. Please check the logs for more details."

    async def suggest_next_workout(self, training_rules: str) -> str:
        """Suggest next workout using Pydantic AI"""
        logger.info("Generating workout suggestion with Pydantic AI...")

        # Log available tools before making the call
        if self.available_tools:
            tool_names = [tool.name for tool in self.available_tools]
            logger.info(f"Available MCP tools: {tool_names}")
            if 'get_activities' not in tool_names:
                logger.warning("WARNING: 'get_activities' tool not found in available tools!")
        else:
            logger.warning("No MCP tools available!")

        prompt = self.template_manager.get_template(
            'workflows/suggest_next_workout.txt',
            training_rules=training_rules
        )

        logger.info("About to call agent.run() with workout suggestion prompt")
        try:
            result = await self.agent.run(prompt)
            logger.info("Agent run completed successfully")
            return result.text
        except Exception as e:
            logger.error(f"Error in workout suggestion: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            if "exceeded max retries" in str(e):
                return "Failed to fetch your activity data from Garmin after several attempts. Please check your connection and try again."
            return "Error suggesting workout. Please check the logs for more details."

    async def enhanced_analysis(self, analysis_type: str, training_rules: str) -> str:
        """Perform enhanced analysis using Pydantic AI with all available tools"""
        logger.info(f"Performing enhanced {analysis_type} analysis...")
        
        # Get pre-cached data
        activity_data = await self.get_recent_cycling_activity_details()
        user_profile = await self.get_user_profile()
        
        if not activity_data.get("last_cycling"):
            return f"No recent cycling activity found for {analysis_type} analysis."
        
        # Summarize recent activities
        recent_activities = activity_data.get("activities", [])
        cycling_activities_summary = "\n".join([
            f"- {act.get('start_time', 'N/A')}: {act.get('activityType', {}).get('typeKey', 'Unknown')} - Duration: {act.get('duration', 'N/A')}s"
            for act in recent_activities[-5:]  # Last 5 activities
        ])
        
        last_activity = activity_data["last_cycling"]
        details = activity_data["details"]
        
        activity_summary = f"""
        Most Recent Cycling Activity:
        - Start Time: {last_activity.get('start_time', 'N/A')}
        - Duration: {last_activity.get('duration', 'N/A')} seconds
        - Distance: {last_activity.get('distance', 'N/A')} meters
        - Average Speed: {last_activity.get('averageSpeed', 'N/A')} m/s
        - Average Power: {last_activity.get('avgPower', 'N/A')} W
        - Max Power: {last_activity.get('maxPower', 'N/A')} W
        - Average Heart Rate: {last_activity.get('avgHr', 'N/A')} bpm
        
        Full Activity Details: {json.dumps(details, default=str)}
        
        Recent Activities (last 5):
        {cycling_activities_summary}
        """
        
        user_info = f"""
        User Profile:
        {json.dumps(user_profile, default=str)}
        """
        
        prompt = self.template_manager.get_template(
            'workflows/single_workout_analysis.txt',
            analysis_type=analysis_type,
            activity_summary=activity_summary,
            user_info=user_info,
            training_rules=training_rules
        )
 
        try:
            # Create temporary agent without tools for this analysis
            model_name = f"openrouter:{self.config.openrouter_model}"
            enhanced_temp_system_prompt = self.template_manager.get_template('base/system_prompts/no_tools_analysis.txt')
            temp_agent = Agent(
                model=model_name,
                system_prompt=enhanced_temp_system_prompt,
                toolsets=[]
            )
            
            # Enter context for temp agent
            await asyncio.wait_for(temp_agent.__aenter__(), timeout=30)
            
            result = await temp_agent.run(prompt)
            
            # Exit context
            await temp_agent.__aexit__(None, None, None)
            
            return str(result)
        except asyncio.TimeoutError:
            logger.error("Temp agent initialization timed out")
            return f"Error: Agent initialization timed out for {analysis_type} analysis."
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
            if hasattr(temp_agent, '__aexit__'):
                await temp_agent.__aexit__(None, None, None)
            return f"Error in {analysis_type} analysis: {e}"