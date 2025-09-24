#!/usr/bin/env python3
"""
Simple MCP Test App - Test MCP connection and user profile loading
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Import our modules
from config import Config, load_config, create_sample_config
from mcp_client import MCPClient
from cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPTestApp:
    """Simple test application for MCP functionality"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mcp_client = MCPClient(config)
        self.cache_manager = CacheManager(default_ttl=300)
        
    async def initialize(self):
        """Initialize MCP client"""
        logger.info("Initializing MCP test app...")
        await self.mcp_client.initialize()
        
    async def cleanup(self):
        """Cleanup resources"""
        await self.mcp_client.cleanup()
        
    async def test_mcp_connection(self):
        """Test basic MCP connection and list tools"""
        print("\n" + "="*60)
        print("MCP CONNECTION TEST")
        print("="*60)
        
        if not self.mcp_client.is_available:
            print("❌ MCP server not available")
            return False
            
        print("✅ MCP server connected")
        
        # List available tools
        tools = await self.mcp_client.list_tools()
        print(f"📋 Found {len(tools)} tools:")
        
        if tools:
            for i, tool in enumerate(tools, 1):
                print(f"  {i}. {tool.name}")
                if hasattr(tool, 'description') and tool.description:
                    print(f"     {tool.description}")
        else:
            print("  No tools available")
            
        return len(tools) > 0
        
    async def test_user_profile(self):
        """Test user profile loading and caching"""
        print("\n" + "="*60)
        print("USER PROFILE TEST")
        print("="*60)
        
        # Check if user_profile tool is available
        if not await self.mcp_client.has_tool("user_profile"):
            print("❌ user_profile tool not available")
            return None
            
        print("✅ user_profile tool found")
        
        try:
            # Call user_profile tool
            print("📞 Calling user_profile tool...")
            profile_data = await self.mcp_client.call_tool("user_profile", {})
            
            # Cache the profile
            self.cache_manager.set("user_profile", profile_data, ttl=3600)
            print("💾 User profile cached (TTL: 1 hour)")
            
            # Pretty print the profile
            print("\n" + "-"*40)
            print("USER PROFILE DATA:")
            print("-"*40)
            print(json.dumps(profile_data, indent=2, default=str))
            print("-"*40)
            
            return profile_data
            
        except Exception as e:
            print(f"❌ Error getting user profile: {e}")
            logger.error(f"User profile error: {e}", exc_info=True)
            return None
            
    async def test_cached_retrieval(self):
        """Test retrieving cached user profile"""
        print("\n" + "="*60)
        print("CACHE RETRIEVAL TEST")
        print("="*60)
        
        # Try to get cached profile
        cached_profile = self.cache_manager.get("user_profile")
        
        if cached_profile:
            print("✅ User profile retrieved from cache")
            print(f"📊 Cache stats: {self.cache_manager.get_stats()}")
            return cached_profile
        else:
            print("❌ No cached user profile found")
            return None
            
    async def test_activities_preview(self):
        """Test getting recent activities if available"""
        print("\n" + "="*60)
        print("ACTIVITIES PREVIEW TEST")
        print("="*60)
        
        if not await self.mcp_client.has_tool("get_activities"):
            print("❌ get_activities tool not available")
            return None
            
        print("✅ get_activities tool found")
        
        try:
            print("📞 Calling get_activities (limit=5)...")
            activities = await self.mcp_client.call_tool("get_activities", {"limit": 5})
            
            if activities:
                print(f"📋 Retrieved {len(activities)} activities")
                
                # Show basic info for each activity
                print("\nRecent Activities:")
                for i, activity in enumerate(activities[:3], 1):  # Show first 3
                    activity_type = activity.get('activityType', {}).get('typeKey', 'Unknown')
                    start_time = activity.get('startTimeLocal', 'Unknown time')
                    duration = activity.get('duration', 0)
                    
                    print(f"  {i}. {activity_type} - {start_time}")
                    print(f"     Duration: {duration // 60}m {duration % 60}s")
                
                # Cache activities
                self.cache_manager.set("recent_activities", activities, ttl=900)
                print("💾 Activities cached (TTL: 15 minutes)")
                
                return activities
            else:
                print("📋 No activities found")
                return []
                
        except Exception as e:
            print(f"❌ Error getting activities: {e}")
            logger.error(f"Activities error: {e}", exc_info=True)
            return None
            
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting MCP Test Suite")
        
        results = {}
        
        # Test 1: MCP Connection
        results['mcp_connection'] = await self.test_mcp_connection()
        
        if not results['mcp_connection']:
            print("\n❌ MCP connection failed - skipping remaining tests")
            return results
            
        # Test 2: User Profile
        results['user_profile'] = await self.test_user_profile()
        
        # Test 3: Cache Retrieval
        results['cache_retrieval'] = await self.test_cached_retrieval()
        
        # Test 4: Activities Preview (optional)
        results['activities'] = await self.test_activities_preview()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            
        # Cache summary
        cache_stats = self.cache_manager.get_stats()
        print(f"\nCache Status: {cache_stats['total_entries']} entries")
        for key in cache_stats['keys']:
            print(f"  - {key}")
            
        return results

def validate_config(config: Config) -> bool:
    """Validate configuration for MCP testing"""
    issues = []
    
    if not config.garth_token:
        issues.append("GARTH_TOKEN not set")
        
    if not config.garth_mcp_server_path:
        issues.append("garth_mcp_server_path not set")
        
    if issues:
        print("❌ Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nTo fix:")
        print("1. Run 'uvx garth login' to get GARTH_TOKEN")
        print("2. Install garth-mcp-server: 'npm install -g garth-mcp-server'")
        print("3. Update config.yaml with your tokens")
        return False
        
    return True

async def main():
    """Main entry point"""
    print("MCP Test App - Simple MCP and User Profile Test")
    print("=" * 50)
    
    try:
        # Setup config
        create_sample_config()
        config = load_config()
        
        if not validate_config(config):
            return
        
        # Create and run test app
        app = MCPTestApp(config)
        
        try:
            await app.initialize()
            results = await app.run_all_tests()
            
            # Exit with appropriate code
            if all(results.values()):
                print("\n🎉 All tests passed!")
                sys.exit(0)
            else:
                print("\n⚠️  Some tests failed")
                sys.exit(1)
                
        finally:
            await app.cleanup()
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        logger.error(f"Main error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())