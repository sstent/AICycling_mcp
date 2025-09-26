#!/usr/bin/env python3
"""
Standalone MCP Test - Single file test for MCP connection and user profile
No external dependencies on the modular architecture - just tests MCP directly
"""

import asyncio
import json
import os
import shutil
import yaml
from pathlib import Path

# Check dependencies
try:
    from pydantic_ai.mcp import MCPServerStdio
    print("✅ pydantic-ai MCP available")
except ImportError:
    print("❌ pydantic-ai MCP not available")
    print("Install with: pip install pydantic-ai")
    exit(1)

def load_config_from_yaml():
    """Load configuration from config.yaml file"""
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ config.yaml not found")
        print("Please create config.yaml with your settings:")
        print("""
garth_token: "your_garth_token_here"
openrouter_api_key: "your_openrouter_api_key_here" 
openrouter_model: "deepseek/deepseek-chat-v3.1"
garth_mcp_server_path: "uvx"
""")
        return None
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        print(f"✅ Loaded config from {config_file}")
        return config_data
        
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config.yaml: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading config.yaml: {e}")
        return None

async def test_mcp_user_profile():
    """Simple test to connect to MCP and get user profile"""
    
    print("🚀 MCP User Profile Test")
    print("=" * 40)
    
    # 1. Load configuration from config.yaml
    config = load_config_from_yaml()
    if not config:
        return False
    
    # 2. Get garth_token from config
    garth_token = config.get("garth_token")
    if not garth_token or garth_token == "your_garth_token_here":
        print("❌ garth_token not properly set in config.yaml")
        print("Please run: uvx garth login")
        print("Then update config.yaml with your token")
        return False
    
    print("✅ GARTH_TOKEN loaded from config.yaml")
    
    # 3. Get server path from config
    server_path = config.get("garth_mcp_server_path", "uvx")
    server_executable = shutil.which(server_path)
    if not server_executable:
        print(f"❌ {server_path} not found")
        print("Please install uvx and garth-mcp-server")
        return False
        
    print(f"✅ {server_path} found")
    
    # 4. Setup MCP server
    print("🔧 Setting up MCP server...")
    
    env = os.environ.copy()
    env["GARTH_TOKEN"] = garth_token
    
    mcp_server = MCPServerStdio(
        command=server_executable,
        args=["garth-mcp-server"],
        env=env,
    )
    
    try:
        # 5. List available tools
        print("📋 Listing MCP tools...")
        tools = await mcp_server.list_tools()
        
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  • {tool.name}")
        
        # 6. Check for user_profile tool
        user_profile_tool = next((t for t in tools if t.name == "user_profile"), None)
        if not user_profile_tool:
            print("❌ user_profile tool not available")
            return False
            
        print("✅ user_profile tool found")
        
        # 7. Call user_profile tool
        print("📞 Getting user profile...")
        result = await mcp_server.direct_call_tool("user_profile", {})
        
        # Extract data
        profile_data = result.output if hasattr(result, 'output') else result
        
        # 8. Display results
        print("\n" + "=" * 50)
        print("USER PROFILE RETRIEVED")
        print("=" * 50)
        print(json.dumps(profile_data, indent=2, default=str))
        print("=" * 50)
        
        # 9. Quick analysis
        if isinstance(profile_data, dict):
            print(f"\n📊 Profile contains {len(profile_data)} fields:")
            for key in list(profile_data.keys())[:5]:  # Show first 5 keys
                print(f"  • {key}")
            if len(profile_data) > 5:
                print(f"  ... and {len(profile_data) - 5} more")
        
        print("\n🎉 Test completed successfully!")
        
        # 10. Show config info used
        print(f"\n📝 Configuration used:")
        print(f"  • Model: {config.get('openrouter_model', 'Not set')}")
        print(f"  • OpenRouter API Key: {'Set' if config.get('openrouter_api_key') else 'Not set'}")
        print(f"  • Server Path: {server_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

async def main():
    """Run the test"""
    try:
        success = await test_mcp_user_profile()
        if success:
            print("\n✅ MCP user profile test PASSED")
        else:
            print("\n❌ MCP user profile test FAILED")
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    print("Standalone MCP User Profile Test")
    print("This will test MCP connection and retrieve your Garmin user profile")
    print()
    
    # Check prerequisites
    print("Prerequisites check:")
    
    # Check if config.yaml exists
    config_file = Path("config.yaml")
    if config_file.exists():
        print("✅ config.yaml found")
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check garth_token in config
            if config.get("garth_token") and config.get("garth_token") != "your_garth_token_here":
                print("✅ garth_token set in config.yaml")
            else:
                print("❌ garth_token not properly set in config.yaml")
                
            # Check openrouter_api_key
            if config.get("openrouter_api_key") and config.get("openrouter_api_key") != "your_openrouter_api_key_here":
                print("✅ openrouter_api_key set in config.yaml")
            else:
                print("❌ openrouter_api_key not set in config.yaml")
                
        except Exception as e:
            print(f"❌ Error reading config.yaml: {e}")
    else:
        print("❌ config.yaml not found")
        
    if shutil.which("uvx"):
        print("✅ uvx command available")
    else:
        print("❌ uvx not found - install it first")
        
    print()
    
    # Run the test
    asyncio.run(main())