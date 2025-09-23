#!/usr/bin/env python3
"""
Test script to verify GARTH_TOKEN validity
"""
import os
import yaml

try:
    import garth
    print("✓ Garth library imported successfully")
except ImportError:
    print("✗ Garth library not installed")
    exit(1)

# Load token from config
try:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    token = config.get('garth_token')
    if not token:
        print("✗ No garth_token found in config.yaml")
        exit(1)
    print("✓ Token loaded from config.yaml")
except Exception as e:
    print(f"✗ Error loading config: {e}")
    exit(1)

# Test token
try:
    print("Testing token validity...")
    garth.client.loads(token)
    print("✓ Token loaded successfully")

    # Try to get user profile
    print("Testing API access...")
    user_profile = garth.UserProfile.get()
    print("✓ API access successful")
    print(f"User Profile: {user_profile}")
    print(f"Display Name: {getattr(user_profile, 'display_name', 'N/A')}")
    print(f"Full Name: {getattr(user_profile, 'full_name', 'N/A')}")

except Exception as e:
    print(f"✗ Token validation failed: {e}")
    import traceback
    traceback.print_exc()