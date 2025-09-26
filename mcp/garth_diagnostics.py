#!/usr/bin/env python3
"""
Garth API Diagnostics
Test various Garmin Connect API endpoints to see what's available
"""

import json
from pathlib import Path

try:
    import garth
    print("✅ Garth module available")
except ImportError:
    print("❌ Garth module not found")
    exit(1)

def test_endpoints():
    """Test various Garmin Connect API endpoints"""
    print("🔍 Testing Garmin Connect API Endpoints")
    print("=" * 50)
    
    # Load session
    session_path = Path.home() / ".garth"
    if not session_path.exists():
        print("❌ No Garth session found. Run setup_garth.py first.")
        return
    
    try:
        garth.resume(str(session_path))
        print("✅ Session loaded")
    except Exception as e:
        print(f"❌ Could not load session: {e}")
        return
    
    # List of endpoints to test
    endpoints_to_test = [
        # User/Profile endpoints
        ("/userprofile-service/socialProfile", "Social Profile"),
        ("/user-service/user", "User Service"),
        ("/user-service/users/settings", "User Settings"),
        ("/modern/currentuser-service/user/profile", "Modern User Profile"),
        ("/userprofile-service/userprofile", "User Profile Alt"),
        
        # Activity endpoints  
        ("/activitylist-service/activities/search/activities?limit=1", "Recent Activities"),
        ("/activity-service/activity", "Activity Service"),
        
        # Wellness endpoints
        ("/wellness-service/wellness", "Wellness Service"),
        ("/wellness-service/wellness/dailySleep", "Daily Sleep"),
        ("/wellness-service/wellness/dailyStress", "Daily Stress"),
        ("/wellness-service/wellness/dailyBodyBattery", "Daily Body Battery"),
        
        # Stats endpoints
        ("/userstats-service/statistics", "User Statistics"),
        ("/usersummary-service/usersummary/daily", "Daily Summary"),
        ("/usersummary-service/usersummary/weekly", "Weekly Summary"),
        
        # Device endpoints
        ("/device-service/deviceRegistration", "Device Registration"),
        ("/device-service/deviceService/app-info", "Device App Info"),
        
        # HRV and health
        ("/hrv-service/hrv", "HRV Service"),
        ("/weight-service/weight/dateRange", "Weight Service"),
        
        # Other services
        ("/badge-service/badge", "Badge Service"),
        ("/golf-service/golf", "Golf Service"),
        ("/content-service/content", "Content Service"),
    ]
    
    working_endpoints = []
    failed_endpoints = []
    
    for endpoint, name in endpoints_to_test:
        try:
            print(f"\n📡 Testing: {name}")
            print(f"   Endpoint: {endpoint}")
            
            # Extract base endpoint and parameters
            if "?" in endpoint:
                base_endpoint, params_str = endpoint.split("?", 1)
                # Parse simple parameters
                params = {}
                if params_str:
                    for param in params_str.split("&"):
                        if "=" in param:
                            key, value = param.split("=", 1)
                            params[key] = value
                result = garth.connectapi(base_endpoint, params=params)
            else:
                result = garth.connectapi(endpoint)
            
            if result is not None:
                if isinstance(result, dict):
                    keys_info = f"Dict with {len(result)} keys: {list(result.keys())[:5]}"
                elif isinstance(result, list):
                    keys_info = f"List with {len(result)} items"
                else:
                    keys_info = f"{type(result).__name__}: {str(result)[:50]}"
                
                print(f"   ✅ SUCCESS: {keys_info}")
                working_endpoints.append((endpoint, name, result))
            else:
                print(f"   ⚠️  SUCCESS but empty response")
                working_endpoints.append((endpoint, name, None))
                
        except Exception as e:
            error_str = str(e)
            if "404" in error_str:
                print(f"   ❌ NOT FOUND (404)")
            elif "403" in error_str:
                print(f"   🔒 FORBIDDEN (403) - May need different auth")
            elif "401" in error_str:
                print(f"   🚫 UNAUTHORIZED (401) - Auth expired?")
            elif "500" in error_str:
                print(f"   💥 SERVER ERROR (500)")
            else:
                print(f"   ❌ ERROR: {error_str[:100]}")
            
            failed_endpoints.append((endpoint, name, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Working Endpoints ({len(working_endpoints)}):")
    for endpoint, name, _ in working_endpoints:
        print(f"   {name}: {endpoint}")
    
    print(f"\n❌ Failed Endpoints ({len(failed_endpoints)}):")
    for endpoint, name, error in failed_endpoints:
        short_error = error.split('\n')[0][:50]
        print(f"   {name}: {short_error}")
    
    # Show sample data from best working endpoint
    if working_endpoints:
        print(f"\n📋 SAMPLE DATA from first working endpoint:")
        endpoint, name, data = working_endpoints[0]
        print(f"Endpoint: {name} ({endpoint})")
        if data:
            if isinstance(data, dict):
                # Show first few key-value pairs
                for i, (key, value) in enumerate(data.items()):
                    if i >= 10:  # Limit to first 10 items
                        print(f"   ... and {len(data) - 10} more fields")
                        break
                    value_str = str(value)[:100]
                    print(f"   {key}: {value_str}")
            elif isinstance(data, list) and len(data) > 0:
                print(f"   List with {len(data)} items")
                if isinstance(data[0], dict):
                    print(f"   First item keys: {list(data[0].keys())}")
                else:
                    print(f"   First item: {str(data[0])[:100]}")
            else:
                print(f"   Data: {str(data)[:200]}")
    
    print(f"\n💡 Recommendation:")
    if working_endpoints:
        best_endpoint = working_endpoints[0]
        print(f"Use '{best_endpoint[0]}' for user profile data")
    else:
        print("No working endpoints found. Check authentication.")

if __name__ == "__main__":
    test_endpoints()