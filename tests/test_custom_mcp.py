#!/usr/bin/env python3
"""
Test Custom MCP Implementation
Test our custom Garth MCP wrapper
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Import our modules
from config import Config, load_config, create_sample_config
from mcp_client import MCPClient  # Updated MCP client
from cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomMCPTestApp:
    """Test application for custom MCP implementation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mcp_client = MCPClient(config)  # Will use custom implementation
        
    async def initialize(self):
        """Initialize MCP client"""
        logger.info("Initializing Custom MCP test app...")
        await self.mcp_client.initialize()
        
    async def cleanup(self):
        """Cleanup resources"""
        await self.mcp_client.cleanup()
        
    async def test_connection_and_tools(self):
        """Test basic connection and list tools"""
        print("\n" + "="*60)
        print("CUSTOM MCP CONNECTION TEST")
        print("="*60)
        
        if not self.mcp_client.is_available:
            print("❌ Custom MCP not available")
            return False
            
        print("✅ Custom MCP connected")
        print(f"📋 Implementation: {self.mcp_client.implementation_type}")
        
        # List available tools
        tools = await self.mcp_client.list_tools()
        print(f"📋 Found {len(tools)} tools")
        
        if tools:
            print("\nAvailable tools:")
            for i, tool in enumerate(tools[:10], 1):  # Show first 10
                print(f"  {i:2d}. {tool.name}")
                if hasattr(tool, 'description'):
                    print(f"      {tool.description}")
        else:
            print("  No tools available")
            
        return len(tools) > 0
        
    async def test_user_profile(self):
        """Test user profile retrieval"""
        print("\n" + "="*60)
        print("USER PROFILE TEST")
        print("="*60)
        
        # Check if tool exists
        if not await self.mcp_client.has_tool("user_profile"):
            print("❌ user_profile tool not available")
            return None
            
        print("✅ user_profile tool found")
        
        try:
            # Call user_profile tool
            print("📞 Calling user_profile tool...")
            profile_data = await self.mcp_client.call_tool("user_profile", {})
            
            print("✅ User profile retrieved")
            
            # Show profile summary
            if isinstance(profile_data, dict):
                display_name = profile_data.get('displayName', 'Unknown')
                full_name = profile_data.get('fullName', 'Unknown')
                user_name = profile_data.get('userName', 'Unknown')
                
                print(f"   Display Name: {display_name}")
                print(f"   Full Name: {full_name}")
                print(f"   Username: {user_name}")
                print(f"   Profile contains {len(profile_data)} fields")
            
            return profile_data
            
        except Exception as e:
            print(f"❌ Error getting user profile: {e}")
            logger.error(f"User profile error: {e}", exc_info=True)
            return None
    
    async def test_activities(self):
        """Test activities retrieval"""
        print("\n" + "="*60)
        print("ACTIVITIES TEST")
        print("="*60)
        
        if not await self.mcp_client.has_tool("get_activities"):
            print("❌ get_activities tool not available")
            return None
            
        print("✅ get_activities tool found")
        
        try:
            print("📞 Calling get_activities (limit=5)...")
            activities = await self.mcp_client.call_tool("get_activities", {"limit": 5})
            
            if activities and len(activities) > 0:
                print(f"✅ Retrieved {len(activities)} activities")
                
                # Show activity summaries
                print("\nRecent Activities:")
                for i, activity in enumerate(activities[:3], 1):
                    activity_id = activity.get('activityId', 'Unknown')
                    activity_type = activity.get('activityType', {})
                    type_key = activity_type.get('typeKey', 'Unknown') if isinstance(activity_type, dict) else str(activity_type)
                    start_time = activity.get('startTimeLocal', 'Unknown time')
                    
                    print(f"  {i}. {type_key} (ID: {activity_id})")
                    print(f"     Start: {start_time}")
                    
                    # Try to get duration
                    duration = activity.get('duration')
                    if duration:
                        minutes = duration // 60
                        seconds = duration % 60
                        print(f"     Duration: {minutes}m {seconds}s")
                
                return activities
            else:
                print("📋 No activities found")
                return []
                
        except Exception as e:
            print(f"❌ Error getting activities: {e}")
            logger.error(f"Activities error: {e}", exc_info=True)
            return None
    
    async def test_activity_details(self, activities):
        """Test activity details retrieval"""
        print("\n" + "="*60)
        print("ACTIVITY DETAILS TEST")
        print("="*60)
        
        if not activities or len(activities) == 0:
            print("❌ No activities available for details test")
            return None
        
        if not await self.mcp_client.has_tool("get_activity_details"):
            print("❌ get_activity_details tool not available")
            return None
        
        # Get details for first activity
        first_activity = activities[0]
        activity_id = str(first_activity.get('activityId'))
        
        print(f"📞 Getting details for activity {activity_id}...")
        
        try:
            details = await self.mcp_client.call_tool("get_activity_details", {
                "activity_id": activity_id
            })
            
            print("✅ Activity details retrieved")
            
            if isinstance(details, dict):
                print(f"   Details contain {len(details)} fields")
                
                # Show some key details
                activity_name = details.get('activityName', 'Unnamed')
                sport = details.get('sportTypeId', 'Unknown sport')
                distance = details.get('distance')
                elapsed_duration = details.get('elapsedDuration')
                
                print(f"   Activity: {activity_name}")
                print(f"   Sport: {sport}")
                if distance:
                    print(f"   Distance: {distance/1000:.2f} km")
                if elapsed_duration:
                    minutes = elapsed_duration // 60
                    seconds = elapsed_duration % 60
                    print(f"   Duration: {minutes}m {seconds}s")
            
            return details
            
        except Exception as e:
            print(f"❌ Error getting activity details: {e}")
            logger.error(f"Activity details error: {e}", exc_info=True)
            return None
    
    async def test_daily_metrics(self):
        """Test daily metrics retrieval"""
        print("\n" + "="*60)
        print("DAILY METRICS TEST")
        print("="*60)
        
        metrics_to_test = ["daily_steps", "daily_sleep", "daily_stress"]
        results = {}
        
        for metric in metrics_to_test:
            if await self.mcp_client.has_tool(metric):
                try:
                    print(f"📞 Testing {metric}...")
                    data = await self.mcp_client.call_tool(metric, {"days": 1})
                    
                    if data:
                        print(f"✅ {metric} data retrieved")
                        if isinstance(data, list) and len(data) > 0:
                            print(f"   Contains {len(data)} day(s) of data")
                        results[metric] = data
                    else:
                        print(f"⚠️  {metric} returned no data")
                        
                except Exception as e:
                    print(f"❌ Error with {metric}: {e}")
                    logger.debug(f"{metric} error: {e}")
            else:
                print(f"❌ {metric} tool not available")
        
        return results
    
    async def test_cache_functionality(self):
        """Test cache functionality"""
        print("\n" + "="*60)
        print("CACHE FUNCTIONALITY TEST")
        print("="*60)
        
        if self.mcp_client.implementation_type != "custom_garth":
            print("❌ Cache testing only available with custom implementation")
            return False
        
        try:
            # Call user_profile twice to test caching
            print("📞 First user_profile call (should hit API)...")
            profile1 = await self.mcp_client.call_tool("user_profile", {})
            
            print("📞 Second user_profile call (should hit cache)...")
            profile2 = await self.mcp_client.call_tool("user_profile", {})
            
            # Verify same data
            if profile1 == profile2:
                print("✅ Cache working correctly")
            else:
                print("⚠️  Cache data differs from API data")
            
            # Show cache stats
            cache_stats = self.mcp_client.get_cache_stats()
            print(f"📊 Cache stats: {cache_stats}")
            
            return True
            
        except Exception as e:
            print(f"❌ Cache test failed: {e}")
            logger.error(f"Cache test error: {e}", exc_info=True)
            return False
    
    async def test_comprehensive_snapshot(self):
        """Test comprehensive data snapshot"""
        print("\n" + "="*60)
        print("COMPREHENSIVE SNAPSHOT TEST")
        print("="*60)
        
        if not await self.mcp_client.has_tool("snapshot"):
            print("❌ snapshot tool not available")
            return None
        
        try:
            print("📞 Getting comprehensive data snapshot...")
            snapshot = await self.mcp_client.call_tool("snapshot", {
                "start_date": "2024-09-20",
                "end_date": "2024-09-24"
            })
            
            if isinstance(snapshot, dict):
                print("✅ Snapshot retrieved successfully")
                print(f"   Snapshot contains {len(snapshot)} data categories:")
                
                for category, data in snapshot.items():
                    if isinstance(data, list):
                        print(f"     - {category}: {len(data)} items")
                    elif isinstance(data, dict):
                        print(f"     - {category}: {len(data)} fields")
                    else:
                        print(f"     - {category}: {type(data).__name__}")
            
            return snapshot
            
        except Exception as e:
            print(f"❌ Snapshot test failed: {e}")
            logger.error(f"Snapshot error: {e}", exc_info=True)
            return None
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting Custom MCP Test Suite")
        
        results = {}
        
        # Test 1: Connection and tools
        results['connection'] = await self.test_connection_and_tools()
        
        if not results['connection']:
            print("\n❌ Connection failed - skipping remaining tests")
            return results
        
        # Test 2: User profile
        profile = await self.test_user_profile()
        results['user_profile'] = profile is not None
        
        # Test 3: Activities
        activities = await self.test_activities()
        results['activities'] = activities is not None
        
        # Test 4: Activity details (if we have activities)
        if activities and len(activities) > 0:
            details = await self.test_activity_details(activities)
            results['activity_details'] = details is not None
        
        # Test 5: Daily metrics
        metrics = await self.test_daily_metrics()
        results['daily_metrics'] = len(metrics) > 0
        
        # Test 6: Cache functionality
        results['cache'] = await self.test_cache_functionality()
        
        # Test 7: Comprehensive snapshot
        snapshot = await self.test_comprehensive_snapshot()
        results['snapshot'] = snapshot is not None
        
        # Summary
        self._print_test_summary(results)
        
        return results
    
    def _print_test_summary(self, results):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = 0
        total = 0
        
        for test_name, result in results.items():
            total += 1
            if result:
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            print(f"{test_name.replace('_', ' ').title():.<40} {status}")
        
        print("-" * 60)
        print(f"Total: {passed}/{total} tests passed")
        
        if self.mcp_client.implementation_type == "custom_garth":
            try:
                cache_stats = self.mcp_client.get_cache_stats()
                print(f"\nCache Status: {cache_stats.get('total_entries', 0)} entries")
                for key in cache_stats.get('keys', []):
                    print(f"  - {key}")
            except Exception as e:
                logger.debug(f"Could not get cache stats: {e}")

def validate_config(config: Config) -> bool:
    """Validate configuration"""
    issues = []
    
    if not config.garth_token:
        issues.append("GARTH_TOKEN not set")
        
    if issues:
        print("❌ Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nTo fix:")
        print("1. Run 'pip install garth' to install Garth module")
        print("2. Run authentication to get token")
        print("3. Update config.yaml with your token")
        return False
        
    return True

async def main():
    """Main entry point"""
    print("Custom MCP Test App - Test Custom Garth Implementation")
    print("=" * 60)
    
    try:
        # Setup config
        create_sample_config()
        config = load_config()
        
        if not validate_config(config):
            return
        
        # Create and run test app
        app = CustomMCPTestApp(config)
        
        try:
            await app.initialize()
            results = await app.run_all_tests()
            
            # Exit with appropriate code
            passed_tests = sum(1 for result in results.values() if result)
            total_tests = len(results)
            
            if passed_tests == total_tests:
                print(f"\n🎉 All {total_tests} tests passed!")
                sys.exit(0)
            elif passed_tests > 0:
                print(f"\n⚠️  {passed_tests}/{total_tests} tests passed")
                sys.exit(1)
            else:
                print(f"\n❌ All {total_tests} tests failed")
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