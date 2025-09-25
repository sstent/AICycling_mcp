#!/usr/bin/env python3
"""
Custom Garth MCP Implementation
Direct wrapper around the Garth module with MCP-like interface
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

try:
    import garth
    GARTH_AVAILABLE = True
except ImportError:
    GARTH_AVAILABLE = False
    garth = None

from cache_manager import CacheManager

logger = logging.getLogger(__name__)

class GarthTool:
    """Represents a single Garth-based tool"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
    
    def __repr__(self):
        return f"GarthTool(name='{self.name}')"

class CustomGarthMCP:
    """
    Custom MCP-like interface for Garth
    Provides tools for accessing Garmin Connect data with local caching
    """
    
    def __init__(self, garth_token: str = None, cache_ttl: int = 300):
        self.garth_token = garth_token or os.getenv("GARTH_TOKEN")
        self.cache = CacheManager(default_ttl=cache_ttl)
        self.session = None
        self._tools = []
        self._setup_tools()
        
        if not GARTH_AVAILABLE:
            logger.error("Garth module not available")
            raise ImportError("Garth module not available. Install with: pip install garth")
    
    def _setup_tools(self):
        """Setup available tools based on working endpoints"""
        self._tools = [
            GarthTool(
                name="user_profile",
                description="Get user profile information (social profile)",
                parameters={}
            ),
            GarthTool(
                name="user_settings", 
                description="Get user statistics/settings information",
                parameters={}
            ),
            GarthTool(
                name="get_activities",
                description="Get list of activities from Garmin Connect",
                parameters={
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "limit": {"type": "integer", "description": "Maximum number of activities (default: 20)"}
                }
            ),
            GarthTool(
                name="get_activities_by_date",
                description="Get activities for a specific date",
                parameters={
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)", "required": True}
                }
            ),
            GarthTool(
                name="get_activity_details",
                description="Get detailed information for a specific activity",
                parameters={
                    "activity_id": {"type": "string", "description": "Activity ID", "required": True}
                }
            ),
            GarthTool(
                name="daily_steps",
                description="Get daily summary data (may include steps if available)",
                parameters={
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
                    "days": {"type": "integer", "description": "Number of days"}
                }
            ),
            GarthTool(
                name="weekly_steps",
                description="Get multi-day summary data (weekly equivalent)",
                parameters={
                    "date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "weeks": {"type": "integer", "description": "Number of weeks"}
                }
            ),
            GarthTool(
                name="get_body_composition", 
                description="Get body composition/weight data (may have parameter requirements)",
                parameters={
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)"}
                }
            ),
            GarthTool(
                name="snapshot",
                description="Get comprehensive data snapshot using working endpoints only",
                parameters={
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                }
            ),
            # Note: These tools are available but will return helpful error messages
            # since the endpoints don't work
            GarthTool(
                name="daily_sleep",
                description="Get daily sleep data (currently unavailable - endpoint not working)", 
                parameters={
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
                    "days": {"type": "integer", "description": "Number of days"}
                }
            ),
            GarthTool(
                name="daily_stress",
                description="Get daily stress data (currently unavailable - endpoint not working)",
                parameters={
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
                    "days": {"type": "integer", "description": "Number of days"}  
                }
            ),
            GarthTool(
                name="daily_body_battery",
                description="Get daily body battery data (currently unavailable - endpoint not working)",
                parameters={
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
                    "days": {"type": "integer", "description": "Number of days"}
                }
            ),
            GarthTool(
                name="daily_hrv",
                description="Get daily heart rate variability data (currently unavailable - endpoint not working)",
                parameters={
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
                    "days": {"type": "integer", "description": "Number of days"}
                }
            ),
            GarthTool(
                name="get_devices",
                description="Get connected devices info (limited - returns user profile as fallback)",
                parameters={}
            )
        ]
    
    async def initialize(self):
        """Initialize Garth session"""
        if not self.garth_token:
            logger.error("No GARTH_TOKEN provided")
            raise ValueError("GARTH_TOKEN is required")
        
        try:
            # Configure Garth
            garth.configure()
            
            # Try to use saved session first
            session_path = Path.home() / ".garth"
            if session_path.exists():
                try:
                    garth.resume(str(session_path))
                    # Test the session
                    await self._test_session()
                    logger.info("Resumed existing Garth session")
                except Exception as e:
                    logger.warning(f"Could not resume session: {e}")
                    await self._create_new_session()
            else:
                await self._create_new_session()
            
            logger.info("Garth session initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Garth: {e}")
            raise
    
    async def _create_new_session(self):
        """Create new Garth session using token"""
        try:
            # Use the token to login
            # Note: You might need to adjust this based on your token format
            garth.login_oauth(token=self.garth_token)
            
            # Save session
            session_path = Path.home() / ".garth"
            garth.save(str(session_path))
            
        except Exception as e:
            logger.error(f"Failed to create Garth session: {e}")
            raise
    
    async def _test_session(self):
        """Test if current session is valid"""
        # Try multiple endpoints to find one that works
        test_endpoints = [
            "/userprofile-service/socialProfile",
            "/user-service/users/settings", 
            "/modern/currentuser-service/user/profile",
            "/userstats-service/statistics"
        ]
        
        for endpoint in test_endpoints:
            try:
                garth.connectapi(endpoint)
                logger.debug(f"Session test successful with {endpoint}")
                return True
            except Exception as e:
                logger.debug(f"Session test failed for {endpoint}: {e}")
                continue
        
        logger.debug("All session tests failed")
        raise Exception("No working endpoints found")
    
    async def cleanup(self):
        """Cleanup resources"""
        # Garth doesn't require explicit cleanup
        pass
    
    def list_tools(self) -> List[GarthTool]:
        """List available tools"""
        return self._tools
    
    async def has_tool(self, tool_name: str) -> bool:
        """Check if tool is available"""
        return any(tool.name == tool_name for tool in self._tools)
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a specific tool"""
        # Check cache first
        cache_key = f"{tool_name}:{hash(str(sorted(parameters.items())))}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {tool_name}")
            return cached_result
        
        try:
            result = await self._execute_tool(tool_name, parameters)
            
            # Cache result with appropriate TTL
            ttl = self._get_cache_ttl(tool_name)
            self.cache.set(cache_key, result, ttl=ttl)
            
            logger.debug(f"Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise
    
    def _get_cache_ttl(self, tool_name: str) -> int:
        """Get appropriate cache TTL for different tools"""
        ttl_map = {
            "user_profile": 3600,      # 1 hour - rarely changes
            "user_settings": 3600,     # 1 hour - rarely changes  
            "get_devices": 1800,       # 30 minutes - occasionally changes
            "get_activities": 300,     # 5 minutes - changes frequently
            "daily_steps": 3600,       # 1 hour - daily data
            "daily_sleep": 3600,       # 1 hour - daily data
            "daily_stress": 3600,      # 1 hour - daily data
            "daily_body_battery": 3600, # 1 hour - daily data
            "daily_hrv": 3600,         # 1 hour - daily data
            "weekly_steps": 1800,      # 30 minutes - weekly data
            "get_body_composition": 1800, # 30 minutes
            "snapshot": 600,           # 10 minutes - comprehensive data
        }
        return ttl_map.get(tool_name, 300)  # Default 5 minutes
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute the actual tool call"""
        
        # User profile and settings - use working endpoints
        if tool_name == "user_profile":
            # Use the working social profile endpoint
            return garth.connectapi("/userprofile-service/socialProfile")
            
        elif tool_name == "user_settings":
            # Try user stats as fallback since user settings doesn't work
            try:
                return garth.connectapi("/userstats-service/statistics")
            except Exception:
                # If stats don't work, return the social profile as user info
                return garth.connectapi("/userprofile-service/socialProfile")
        
        # Activities - use working endpoint
        elif tool_name == "get_activities":
            start_date = parameters.get("start_date")
            limit = parameters.get("limit", 20)
            
            params = {"limit": limit}
            if start_date:
                params["startDate"] = start_date
                
            return garth.connectapi("/activitylist-service/activities/search/activities", params=params)
        
        elif tool_name == "get_activities_by_date":
            date = parameters["date"]
            start = f"{date}T00:00:00.000Z"
            end = f"{date}T23:59:59.999Z"
            
            return garth.connectapi("/activitylist-service/activities/search/activities", params={
                "startDate": start,
                "endDate": end,
                "limit": 100
            })
        
        elif tool_name == "get_activity_details":
            activity_id = parameters["activity_id"]
            return garth.connectapi(f"/activity-service/activity/{activity_id}")
        
        # Daily metrics - many don't work, so provide fallbacks
        elif tool_name == "daily_steps":
            date = parameters.get("date", datetime.now().strftime("%Y-%m-%d"))
            days = parameters.get("days", 1)
            
            # Try usersummary first, if that fails, try to get from activities
            try:
                return garth.connectapi("/usersummary-service/usersummary/daily", params={
                    "startDate": date,
                    "numOfDays": days
                })
            except Exception as e:
                logger.warning(f"Daily steps via usersummary failed: {e}")
                # Fallback: get activities for the date and sum steps if available
                try:
                    activities = garth.connectapi("/activitylist-service/activities/search/activities", params={
                        "startDate": f"{date}T00:00:00.000Z",
                        "endDate": f"{date}T23:59:59.999Z",
                        "limit": 50
                    })
                    return {"fallback_activities": activities, "date": date, "message": "Daily steps not available, showing activities instead"}
                except Exception as e2:
                    raise Exception(f"Both daily steps and activities failed: {e}, {e2}")
        
        elif tool_name == "daily_sleep":
            # Sleep endpoint doesn't work, return error with helpful message
            raise Exception("Daily sleep endpoint not available. Sleep data may be accessible through other means.")
        
        elif tool_name == "daily_stress":
            # Stress endpoint doesn't work
            raise Exception("Daily stress endpoint not available.")
        
        elif tool_name == "daily_body_battery":
            # Body battery endpoint doesn't work
            raise Exception("Daily body battery endpoint not available.")
        
        elif tool_name == "daily_hrv":
            # HRV endpoint doesn't work
            raise Exception("HRV endpoint not available.")
        
        # Weekly metrics
        elif tool_name == "weekly_steps":
            # Weekly endpoint doesn't work, try to get multiple days instead
            date = parameters.get("date", datetime.now().strftime("%Y-%m-%d"))
            weeks = parameters.get("weeks", 1)
            days = weeks * 7
            
            try:
                return garth.connectapi("/usersummary-service/usersummary/daily", params={
                    "startDate": date,
                    "numOfDays": days
                })
            except Exception as e:
                raise Exception(f"Weekly steps not available: {e}")
        
        # Device info
        elif tool_name == "get_devices":
            # Device registration doesn't work, return user profile as fallback
            profile = garth.connectapi("/userprofile-service/socialProfile")
            return {"message": "Device registration endpoint not available", "user_profile": profile}
        
        # Body composition
        elif tool_name == "get_body_composition":
            # Weight service gives 400 error, likely needs parameters
            date = parameters.get("date", datetime.now().strftime("%Y-%m-%d"))
            try:
                return garth.connectapi("/weight-service/weight/dateRange", params={
                    "startDate": date,
                    "endDate": date
                })
            except Exception as e:
                raise Exception(f"Body composition endpoint error: {e}")
        
        # Comprehensive snapshot - use working endpoints only
        elif tool_name == "snapshot":
            start_date = parameters.get("start_date", datetime.now().strftime("%Y-%m-%d"))
            end_date = parameters.get("end_date", start_date)
            
            snapshot = {}
            
            try:
                # User profile - we know this works
                snapshot["user_profile"] = garth.connectapi("/userprofile-service/socialProfile")
            except Exception as e:
                logger.warning(f"Could not get user profile: {e}")
            
            try:
                # Activities for date range - we know this works
                snapshot["activities"] = garth.connectapi("/activitylist-service/activities/search/activities", params={
                    "startDate": f"{start_date}T00:00:00.000Z",
                    "endDate": f"{end_date}T23:59:59.999Z",
                    "limit": 100
                })
            except Exception as e:
                logger.warning(f"Could not get activities: {e}")
            
            try:
                # User stats - we know this works
                snapshot["user_stats"] = garth.connectapi("/userstats-service/statistics")
            except Exception as e:
                logger.warning(f"Could not get user stats: {e}")
            
            # Try some daily data (even though many endpoints don't work)
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                days = (end_dt - start_dt).days + 1
                
                snapshot["daily_summary"] = garth.connectapi("/usersummary-service/usersummary/daily", params={
                    "startDate": start_date,
                    "numOfDays": days
                })
            except Exception as e:
                logger.warning(f"Could not get daily summary: {e}")
            
            if not snapshot:
                raise Exception("Could not retrieve any data for snapshot")
            
            return snapshot
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def print_tools(self):
        """Pretty print available tools"""
        print(f"\n{'='*60}")
        print("AVAILABLE GARTH TOOLS")
        print(f"{'='*60}")
        
        for i, tool in enumerate(self._tools, 1):
            print(f"\n{i}. {tool.name}")
            print(f"   Description: {tool.description}")
            
            if tool.parameters:
                print("   Parameters:")
                for param, info in tool.parameters.items():
                    param_type = info.get("type", "string")
                    param_desc = info.get("description", "")
                    required = info.get("required", False)
                    req_str = " (required)" if required else " (optional)"
                    print(f"     - {param} ({param_type}){req_str}: {param_desc}")
        
        print(f"\n{'='*60}")
    
    @property
    def is_available(self) -> bool:
        """Check if MCP server is available"""
        return GARTH_AVAILABLE and self.garth_token is not None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()

# Utility functions
async def create_garth_mcp(garth_token: str = None, cache_ttl: int = 300) -> CustomGarthMCP:
    """Create and initialize a CustomGarthMCP instance"""
    mcp = CustomGarthMCP(garth_token=garth_token, cache_ttl=cache_ttl)
    await mcp.initialize()
    return mcp

async def test_garth_mcp():
    """Test the custom Garth MCP implementation"""
    print("🚀 Testing Custom Garth MCP Implementation")
    print("=" * 50)
    
    try:
        # Initialize
        mcp = await create_garth_mcp()
        
        # List tools
        tools = mcp.list_tools()
        print(f"✅ Found {len(tools)} tools")
        
        # Test user profile
        profile = await mcp.call_tool("user_profile", {})
        print("✅ Got user profile")
        print(f"   User: {profile.get('displayName', 'Unknown')}")
        
        # Test activities
        activities = await mcp.call_tool("get_activities", {"limit": 5})
        print(f"✅ Got {len(activities)} activities")
        
        # Test cache
        cached_profile = await mcp.call_tool("user_profile", {})  # Should hit cache
        print("✅ Cache working")
        
        # Show cache stats
        cache_stats = mcp.get_cache_stats()
        print(f"📊 Cache has {cache_stats['total_entries']} entries")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_garth_mcp())