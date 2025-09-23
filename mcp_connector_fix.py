#!/usr/bin/env python3
"""
Fixed GarthMCPConnector class to resolve hanging issues
"""
import os
import json
import asyncio
import shutil
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# MCP Protocol imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

class GarthMCPConnector:
    """Fixed Connector for Garmin data via Garth MCP server"""

    def __init__(self, garth_token: str, server_path: str):
        self.garth_token = garth_token
        self.server_path = server_path
        self.server_available = False
        self.cached_tools = []  # Cache tools to avoid repeated fetches
        self._session: Optional[ClientSession] = None
        self._client_context = None
        self._read_stream = None
        self._write_stream = None
        self._connection_timeout = 30  # Timeout for operations

    async def _get_server_params(self):
        """Get server parameters for MCP connection"""
        env = os.environ.copy()
        env['GARTH_TOKEN'] = self.garth_token

        # Find the full path to the server executable
        server_command = shutil.which("garth-mcp-server")
        if not server_command:
            logger.error("Could not find 'garth-mcp-server' in your PATH.")
            logger.error("Please ensure it is installed and accessible, e.g., via 'npm install -g garth-mcp-server'.")
            raise FileNotFoundError("garth-mcp-server not found")

        return StdioServerParameters(
            command="/bin/bash",
            args=["-c", f"exec {server_command} \"$@\" 1>&2"],
            capture_stderr=True,
            env=env,
        )

    async def connect(self):
        """Start the MCP server and establish a persistent session."""
        if self._session and self.server_available:
            return True

        if not MCP_AVAILABLE:
            logger.error("MCP library not available. Install with: pip install mcp")
            return False

        try:
            logger.info("Connecting to Garth MCP server...")
            server_params = await self._get_server_params()
            
            logger.info("Starting MCP server process...")
            self._client_context = stdio_client(server_params)
            streams = await self._client_context.__aenter__()

            # Handle stderr logging in background
            if len(streams) == 3:
                self._read_stream, self._write_stream, stderr_stream = streams
                asyncio.create_task(self._log_stderr(stderr_stream))
            else:
                self._read_stream, self._write_stream = streams

            logger.info("Server process started. Waiting for it to initialize...")
            await asyncio.sleep(1.0)  # Give server more time to start

            logger.info("Initializing MCP session...")
            self._session = ClientSession(self._read_stream, self._write_stream)
            
            # Initialize with timeout
            try:
                await asyncio.wait_for(self._session.initialize(), timeout=self._connection_timeout)
            except asyncio.TimeoutError:
                logger.error("MCP session initialization timed out")
                await self.disconnect()
                return False

            logger.info("Testing connection by listing tools...")
            
            # Test tools listing with timeout
            try:
                await asyncio.wait_for(self._session.list_tools(), timeout=15)
                logger.info("✓ Successfully connected to MCP server.")
                self.server_available = True
                return True
            except asyncio.TimeoutError:
                logger.error("Tools listing timed out - server may be unresponsive")
                await self.disconnect()
                return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.disconnect()
            self.server_available = False
            return False

    async def _log_stderr(self, stderr_stream):
        """Log stderr from the server process"""
        try:
            async for line in stderr_stream:
                logger.debug(f"[garth-mcp-server] {line.decode().strip()}")
        except Exception as e:
            logger.debug(f"Error reading stderr: {e}")

    async def disconnect(self):
        """Disconnect from the MCP server and clean up resources."""
        logger.info("Disconnecting from MCP server...")
        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error during MCP client disconnection: {e}")
        
        self._session = None
        self.server_available = False
        self.cached_tools = []
        self._client_context = None
        self._read_stream = None
        self._write_stream = None
        logger.info("Disconnected.")

    async def _ensure_connected(self):
        """Ensure server is available"""
        if not self.server_available or not self._session:
            return await self.connect()
        return True

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool on the MCP server with timeout"""
        if not await self._ensure_connected():
            raise Exception("MCP server not available")

        try:
            return await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments or {}),
                timeout=self._connection_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Tool call '{tool_name}' timed out")
            raise Exception(f"Tool call '{tool_name}' timed out")
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            raise

    async def get_available_tools_info(self) -> List[Dict[str, str]]:
        """Get information about available MCP tools with proper timeout handling"""
        # Return cached tools if available
        if self.cached_tools:
            logger.debug("Returning cached tools")
            return self.cached_tools

        if not await self._ensure_connected():
            logger.warning("Could not connect to MCP server")
            return []

        try:
            logger.debug("Fetching tools from MCP server...")
            
            # Use timeout for the tools listing
            tools_result = await asyncio.wait_for(
                self._session.list_tools(), 
                timeout=15
            )
            
            if not tools_result:
                logger.warning("No tools result received from server")
                return []
                
            tools = tools_result.tools if hasattr(tools_result, 'tools') else []
            logger.info(f"Retrieved {len(tools)} tools from MCP server")

            # Cache the tools for future use
            self.cached_tools = [
                {
                    "name": tool.name,
                    "description": getattr(tool, 'description', 'No description available'),
                }
                for tool in tools
            ]

            return self.cached_tools

        except asyncio.TimeoutError:
            logger.error("Tools listing timed out after 15 seconds")
            # Don't cache empty result on timeout
            return []
        except Exception as e:
            logger.warning(f"Could not get tools info: {e}")
            return []

    async def get_activities_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get activities data via MCP or fallback to mock data"""
        if not await self._ensure_connected() or not self._session:
            logger.warning("No MCP session available, using mock data")
            return self._get_mock_activities_data(limit)
        
        try:
            # Try different possible tool names for getting activities
            possible_tools = ['get_activities', 'list_activities', 'activities', 'garmin_activities']
            available_tools = await self.get_available_tools_info()
            
            for tool_name in possible_tools:
                if any(tool['name'] == tool_name for tool in available_tools):
                    logger.info(f"Calling tool: {tool_name}")
                    result = await self.call_tool(tool_name, {"limit": limit})
                    
                    if result and hasattr(result, 'content'):
                        activities = []
                        for content in result.content:
                            if hasattr(content, 'text'):
                                try:
                                    data = json.loads(content.text)
                                    if isinstance(data, list):
                                        activities.extend(data)
                                    else:
                                        activities.append(data)
                                except json.JSONDecodeError:
                                    activities.append({"description": content.text})
                        return activities
            
            logger.warning("No suitable activity tool found, falling back to mock data")
            return self._get_mock_activities_data(limit)
            
        except Exception as e:
            logger.error(f"Failed to get activities via MCP: {e}")
            logger.warning("Falling back to mock data")
            return self._get_mock_activities_data(limit)
    
    def _get_mock_activities_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get mock activities data for testing"""
        base_activity = {
            "activityId": "12345678901",
            "activityName": "Morning Ride",
            "startTimeLocal": "2024-01-15T08:00:00",
            "activityType": {"typeKey": "cycling"},
            "distance": 25000,
            "duration": 3600,
            "averageSpeed": 6.94,
            "maxSpeed": 12.5,
            "elevationGain": 350,
            "averageHR": 145,
            "maxHR": 172,
            "averagePower": 180,
            "maxPower": 420,
            "normalizedPower": 185,
            "calories": 890,
            "averageCadence": 85,
            "maxCadence": 110
        }
        
        activities = []
        for i in range(min(limit, 10)):
            activity = base_activity.copy()
            activity["activityId"] = str(int(base_activity["activityId"]) + i)
            activity["activityName"] = f"Cycling Workout {i+1}"
            activity["distance"] = base_activity["distance"] + (i * 2000)
            activity["averagePower"] = base_activity["averagePower"] + (i * 10)
            activity["duration"] = base_activity["duration"] + (i * 300)
            activities.append(activity)
        
        return activities
    
    async def get_last_cycling_workout(self) -> Optional[Dict[str, Any]]:
        """Get the most recent cycling workout"""
        activities = await self.get_activities_data(limit=50)
        
        cycling_activities = [
            activity for activity in activities 
            if self._is_cycling_activity(activity)
        ]
        
        return cycling_activities[0] if cycling_activities else None
    
    async def get_last_n_cycling_workouts(self, n: int = 4) -> List[Dict[str, Any]]:
        """Get the last N cycling workouts"""
        activities = await self.get_activities_data(limit=50)
        
        cycling_activities = [
            activity for activity in activities 
            if self._is_cycling_activity(activity)
        ]
        
        return cycling_activities[:n]
    
    def _is_cycling_activity(self, activity: Dict[str, Any]) -> bool:
        """Check if an activity is a cycling workout"""
        activity_type = activity.get('activityType', {}).get('typeKey', '').lower()
        activity_name = activity.get('activityName', '').lower()
        
        cycling_keywords = ['cycling', 'bike', 'ride', 'bicycle']
        
        return (
            'cycling' in activity_type or 
            'bike' in activity_type or
            any(keyword in activity_name for keyword in cycling_keywords)
        )

# Test function to verify the fix
async def test_fixed_connector():
    """Test the fixed connector"""
    import yaml
    
    # Load config
    with open("config.yaml") as f:
        config_data = yaml.safe_load(f)
    
    connector = GarthMCPConnector(
        config_data['garth_token'], 
        config_data['garth_mcp_server_path']
    )
    
    print("Testing fixed MCP connector...")
    
    try:
        # Test connection
        success = await connector.connect()
        if success:
            print("✓ Connection successful")
            
            # Test tools retrieval
            tools = await connector.get_available_tools_info()
            print(f"✓ Retrieved {len(tools)} tools")
            
            for tool in tools[:5]:
                print(f"  - {tool['name']}: {tool['description']}")
                
            if len(tools) > 5:
                print(f"  ... and {len(tools) - 5} more tools")
                
        else:
            print("✗ Connection failed")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await connector.disconnect()

if __name__ == "__main__":
    asyncio.run(test_fixed_connector())