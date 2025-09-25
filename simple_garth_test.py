#!/usr/bin/env python3
"""
Simple Garth Test - Minimal test to verify Garth is working
"""

import garth
from pathlib import Path

def simple_test():
    print("🔧 Simple Garth Connection Test")
    print("=" * 40)
    
    # Check if session exists
    session_path = Path.home() / ".garth"
    if session_path.exists():
        print("✅ Session file exists")
        try:
            garth.resume(str(session_path))
            print("✅ Session loaded")
        except Exception as e:
            print(f"❌ Session load failed: {e}")
            return False
    else:
        print("❌ No session file found")
        return False
    
    # Try the simplest possible API call
    simple_endpoints = [
        "/",
        "/ping", 
        "/health",
        "/status"
    ]
    
    print("\n🧪 Testing simple endpoints...")
    for endpoint in simple_endpoints:
        try:
            result = garth.connectapi(endpoint)
            print(f"✅ {endpoint} worked: {type(result)}")
            return True
        except Exception as e:
            print(f"❌ {endpoint} failed: {str(e)[:50]}")
    
    # Try to access the raw client
    print("\n🔍 Checking Garth client details...")
    try:
        client = garth.client
        print(f"Client type: {type(client)}")
        print(f"Client attributes: {[attr for attr in dir(client) if not attr.startswith('_')][:5]}")
        
        # Try to see if we can get base URL
        if hasattr(client, 'domain'):
            print(f"Domain: {client.domain}")
        if hasattr(client, 'base_url'):
            print(f"Base URL: {client.base_url}")
            
    except Exception as e:
        print(f"❌ Client inspection failed: {e}")
    
    return False

if __name__ == "__main__":
    simple_test()