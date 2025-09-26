#!/usr/bin/env python3
"""
Garth Setup Script
Helper script to set up Garth authentication and test connection
"""

import os
import json
import sys
from pathlib import Path

try:
    import garth
    print("✅ Garth module available")
except ImportError:
    print("❌ Garth module not found")
    print("Install with: pip install garth")
    sys.exit(1)

def setup_garth_auth():
    """Setup Garth authentication"""
    print("🔐 Setting up Garth authentication...")
    print("This will guide you through authenticating with Garmin Connect")
    print()
    
    # Check if session already exists
    session_path = Path.home() / ".garth"
    if session_path.exists():
        print("📁 Existing Garth session found")
        choice = input("Do you want to use existing session? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                garth.resume(str(session_path))
                # Test session
                test_result = test_garth_connection()
                if test_result:
                    print("✅ Existing session is valid")
                    return True
                else:
                    print("❌ Existing session is invalid, creating new one...")
            except Exception as e:
                print(f"⚠️  Could not resume session: {e}")
                print("Creating new session...")
    
    # Create new session
    print("\n🆕 Creating new Garth session...")
    print("You'll need your Garmin Connect credentials")
    print()
    
    try:
        # Configure Garth
        garth.configure()
        
        # Get credentials
        username = input("Garmin Connect username/email: ").strip()
        password = input("Garmin Connect password: ").strip()
        
        if not username or not password:
            print("❌ Username and password are required")
            return False
        
        print("\n🔄 Logging into Garmin Connect...")
        
        # Login
        garth.login(username, password)
        
        # Save session
        garth.save(str(session_path))
        print(f"💾 Session saved to {session_path}")
        
        # Test connection
        if test_garth_connection():
            print("✅ Authentication successful!")
            return True
        else:
            print("❌ Authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False

def test_garth_connection():
    """Test Garth connection"""
    print("🧪 Testing Garth connection...")
    
    # Use the endpoint we know works
    try:
        print("  Trying social profile endpoint...")
        user_info = garth.connectapi("/userprofile-service/socialProfile")
        
        if user_info and isinstance(user_info, dict):
            display_name = user_info.get('displayName', 'Unknown')
            full_name = user_info.get('fullName', 'Unknown')
            print(f"✅ Connected as: {display_name} ({full_name})")
            
            # Test activities too
            try:
                activities = garth.connectapi("/activitylist-service/activities/search/activities", params={"limit": 1})
                if activities and len(activities) > 0:
                    print(f"✅ Activities accessible: Found {len(activities)} recent activity")
                else:
                    print(f"⚠️  Activities accessible but none found")
            except Exception as e:
                print(f"⚠️  Activities test failed: {str(e)[:50]}")
            
            return True
        else:
            print("❌ Invalid response from social profile endpoint")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def extract_token_info():
    """Extract token information for config"""
    print("\n🎫 Extracting token information...")
    
    session_path = Path.home() / ".garth"
    if not session_path.exists():
        print("❌ No Garth session found. Run authentication first.")
        return None
    
    try:
        # Load session data
        with open(session_path, 'r') as f:
            session_data = json.load(f)
        
        # Extract token (this might need adjustment based on Garth's session format)
        token = session_data.get('oauth_token') or session_data.get('token')
        
        if token:
            print(f"✅ Token extracted: {token[:20]}...")
            print("\n📝 Add this to your config.yaml:")
            print(f"garth_token: \"{token}\"")
            return token
        else:
            print("❌ Could not extract token from session")
            print("Available session keys:", list(session_data.keys()))
            return None
            
    except Exception as e:
        print(f"❌ Error extracting token: {e}")
        return None

def show_user_info():
    """Show current user information"""
    print("\n👤 User Information:")
    
    # Try multiple endpoints to get user info
    endpoints_to_try = [
        ("/userprofile-service/socialProfile", "Social Profile"),
        ("/user-service/users/settings", "User Settings"),
        ("/modern/currentuser-service/user/profile", "User Profile"),
        ("/userprofile-service/userprofile", "User Profile Alt"),
    ]
    
    user_info = None
    working_endpoint = None
    
    for endpoint, name in endpoints_to_try:
        try:
            print(f"  Trying {name}...")
            data = garth.connectapi(endpoint)
            if data and isinstance(data, dict):
                user_info = data
                working_endpoint = endpoint
                print(f"  ✅ {name} successful")
                break
        except Exception as e:
            print(f"  ❌ {name} failed: {str(e)[:100]}")
            continue
    
    if user_info:
        print(f"\n  Working Endpoint: {working_endpoint}")
        print(f"  Data keys available: {list(user_info.keys())}")
        
        # Extract common fields
        display_name = (
            user_info.get('displayName') or 
            user_info.get('fullName') or 
            user_info.get('userName') or 
            user_info.get('profileDisplayName') or
            'Unknown'
        )
        
        user_id = (
            user_info.get('id') or 
            user_info.get('userId') or 
            user_info.get('profileId') or
            'Unknown'
        )
        
        print(f"  Display Name: {display_name}")
        print(f"  User ID: {user_id}")
        
        # Show first few fields for debugging
        print(f"  Sample data:")
        for key, value in list(user_info.items())[:5]:
            print(f"    {key}: {str(value)[:50]}")
        
        # Try to get activities
        try:
            print(f"\n  Testing activities access...")
            activities = garth.connectapi("/activitylist-service/activities/search/activities", 
                                        params={"limit": 1})
            if activities and len(activities) > 0:
                print(f"  Recent Activities: ✅ Available ({len(activities)} found)")
                activity = activities[0]
                activity_type = activity.get('activityType', {})
                if isinstance(activity_type, dict):
                    type_name = activity_type.get('typeKey', 'Unknown')
                else:
                    type_name = str(activity_type)
                print(f"  Last Activity: {type_name}")
            else:
                print(f"  Recent Activities: ⚠️  None found")
        except Exception as e:
            print(f"  Recent Activities: ❌ Error ({str(e)[:50]})")
    else:
        print("  ❌ Could not retrieve any user information")
        
        # Show what endpoints are available
        print("\n  Available endpoints check:")
        test_endpoints = [
            "/userstats-service/statistics",
            "/wellness-service/wellness",
            "/device-service/deviceRegistration",
        ]
        
        for endpoint in test_endpoints:
            try:
                garth.connectapi(endpoint)
                print(f"  ✅ {endpoint} - Available")
            except Exception as e:
                if "404" in str(e):
                    print(f"  ❌ {endpoint} - Not Found")
                elif "403" in str(e):
                    print(f"  ⚠️  {endpoint} - Forbidden (auth issue)")
                else:
                    print(f"  ❌ {endpoint} - Error: {str(e)[:30]}")

def main():
    """Main setup function"""
    print("Garth Setup and Authentication")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Setup/Login to Garmin Connect")
        print("2. Test existing connection")
        print("3. Show user information")
        print("4. Extract token for config")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            success = setup_garth_auth()
            if success:
                show_user_info()
                
        elif choice == "2":
            test_garth_connection()
            
        elif choice == "3":
            show_user_info()
            
        elif choice == "4":
            token = extract_token_info()
            if token:
                # Also show how to set environment variable
                print(f"\nOr set as environment variable:")
                print(f"export GARTH_TOKEN=\"{token}\"")
                
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()