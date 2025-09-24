#!/usr/bin/env python3
"""
Minimal MCP Test - Just test MCP connection and user profile
"""

import asyncio
import json
import logging
import os
import yaml
from pathlib import Path

# Minimal imports - just what we need
try:
    from pydantic_ai.mcp import MCPServerStdio
    import shutil
    MCP_AVAILABLE = True
except ImportError:
    print("❌ pydantic-ai MCP not available")
    print("Install with: pip install pydantic-ai")
    exit(1)

# Simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalMCPTest:
    """Minimal MCP test class"""
    
    def __init__(self, garth_token: str, server_path: str = "uvx"):
        self.garth_token = garth_token
        self.server_path = server_path
        self.mcp_server = None
        self.cached_profile = None
        
    def setup_mcp_server(self):
        """Setup MCP server connection"""
        # Set environment
        os.environ["GARTH_TOKEN"] = self.garth_token
        env = os.environ.copy()
        
        # Find server executable
        server_executable = shutil.which(self.server_path)
        if not server_executable:
            raise FileNotFoundError(f"'{self.server_path}' not found in PATH")
            
        self.mcp_server = MCPServerStdio(
            command=server_executable,
            args=["garth-mcp-server"],
            env=env,
        )
        
        print("✅ MCP server configured")
        
    async def test_connection(self):
        """Test basic MCP connection"""
        if not self.mcp_server:
            raise RuntimeError("MCP server not configured")
            
        try:
            # Try to list tools
            tools = await self.mcp_server.list_tools()
            print(f"✅ MCP connected - found {len(tools)} tools")
            
            # Show tools
            for tool in tools:
                print(f"   📋 {tool.name}: {getattr(tool, 'description', 'No description')}")
                
            return tools
            
        except Exception as e:
            print(f"❌ MCP connection failed: {e}")
            raise
            
    async def get_user_profile(self):
        """Get and cache user profile"""
        try:
            print("📞 Calling user_profile tool...")
            
            # Direct tool call
            result = await self.mcp_server.direct_call_tool("user_profile", {})
            profile_data = result.output if hasattr(result, 'output') else result
            
            # Cache it
            self.cached_profile = profile_data
            
            print("✅ User profile retrieved and cached")
            return profile_data
            
        except Exception as e:
            print(f"❌ Failed to get user profile: {e}")
            raise
            
    def print_profile(self):
        """Print cached profile"""
        if not self.cached_profile:
            print("❌ No cached profile")
            return
            
        print("\n" + "="*50)
        print("USER PROFILE")
        print("="*50)
        print(json.dumps(self.cached_profile, indent=2, default=str))
        print("="*50)
        
    async def run_test(self):
        """Run the complete test"""
        print("🚀 Starting Minimal MCP Test\n")
        
        # Setup
        self.setup_mcp_server()
        
        # Test connection
        tools = await self.test_connection()
        
        # Check if user_profile tool exists
        user_profile_tool = next((t for t in tools if t.name == "user_profile"), None)
        if not user_profile_tool:
            print("❌ user_profile tool not found")
            return False
            
        # Get user profile
        await self.get_user_profile()
        
        # Show results
        self.print_profile()
        
        print("\n🎉 Test completed successfully!")
        return True

def get_config():
    """Get configuration from config.yaml or environment"""
    config_file = Path("config.yaml")
    
    # Try to load from config.yaml first
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            garth_token = config_data.get("garth_token")
            server_path = config_data.get("garth_mcp_server_path", "uvx")
            
            if garth_token and garth_token != "your_garth_token_here":
                print("✅ Configuration loaded from config.yaml")
                return garth_token, server_path
            else:
                print("⚠️  garth_token not properly set in config.yaml")
                
        except yaml.YAMLError as e:
            print(f"❌ Error parsing config.yaml: {e}")
        except Exception as e:
            print(f"❌ Error reading config.yaml: {e}")
    else:
        print("ℹ️  config.yaml not found")
    
    # Fallback to environment variables
    print("Trying environment variables...")
    garth_token = os.getenv("GARTH_TOKEN")
    
    if not garth_token:
        print("❌ GARTH_TOKEN not found in config.yaml or environment")
        print("Please either:")
        print("1. Create config.yaml with garth_token")
        print("2. Run 'uvx garth login' and set GARTH_TOKEN environment variable")
        raise ValueError("GARTH_TOKEN is required")
        
    server_path = os.getenv("GARTH_MCP_SERVER_PATH", "uvx")
    print("✅ Configuration loaded from environment variables")
    
    return garth_token, server_path

async def main():
    """Main entry point"""
    try:
        # Get config
        garth_token, server_path = get_config()
        
        # Run test
        test = MinimalMCPTest(garth_token, server_path)
        success = await test.run_test()
        
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Tests failed!")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        logger.error("Test error", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())