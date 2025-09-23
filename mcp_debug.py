#!/usr/bin/env python3
"""
Debug script to identify MCP tools hanging issue
"""
import asyncio
import yaml
import logging
import signal
from main import GarthMCPConnector, Config

# Set up more detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

async def debug_mcp_connection():
    """Debug the MCP connection step by step"""
    # Load config
    with open("config.yaml") as f:
        config_data = yaml.safe_load(f)

    config = Config(**config_data)
    garmin = GarthMCPConnector(config.garth_token, config.garth_mcp_server_path)

    print("=== MCP CONNECTION DEBUG ===")
    
    # Step 1: Test connection
    print("\n1. Testing MCP connection...")
    try:
        success = await garmin.connect()
        if success:
            print("✓ MCP connection successful")
        else:
            print("✗ MCP connection failed")
            return
    except Exception as e:
        print(f"✗ MCP connection error: {e}")
        return

    # Step 2: Test session availability
    print("\n2. Testing session availability...")
    if garmin._session:
        print("✓ Session is available")
    else:
        print("✗ No session available")
        return

    # Step 3: Test tools listing with timeout
    print("\n3. Testing tools listing (with 10s timeout)...")
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        tools_result = await garmin._session.list_tools()
        signal.alarm(0)  # Cancel timeout
        
        if tools_result:
            print(f"✓ Tools result received: {type(tools_result)}")
            if hasattr(tools_result, 'tools'):
                tools = tools_result.tools
                print(f"✓ Found {len(tools)} tools")
                
                # Show first few tools
                for i, tool in enumerate(tools[:3]):
                    print(f"  Tool {i+1}: {tool.name} - {getattr(tool, 'description', 'No desc')}")
                
                if len(tools) > 3:
                    print(f"  ... and {len(tools) - 3} more tools")
            else:
                print(f"✗ tools_result has no 'tools' attribute: {dir(tools_result)}")
        else:
            print("✗ No tools result received")
            
    except TimeoutError:
        print("✗ Tools listing timed out after 10 seconds")
        print("This suggests the MCP server is hanging on list_tools()")
    except Exception as e:
        print(f"✗ Tools listing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal.alarm(0)  # Make sure to cancel any pending alarm

    # Step 4: Test our wrapper method
    print("\n4. Testing our get_available_tools_info() method...")
    try:
        # Clear cache first
        garmin.cached_tools = []
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # 15 second timeout
        
        tools = await garmin.get_available_tools_info()
        signal.alarm(0)
        
        print(f"✓ get_available_tools_info() returned {len(tools)} tools")
        for tool in tools[:3]:
            print(f"  - {tool['name']}: {tool['description']}")
            
    except TimeoutError:
        print("✗ get_available_tools_info() timed out")
    except Exception as e:
        print(f"✗ get_available_tools_info() error: {e}")
    finally:
        signal.alarm(0)

    # Cleanup
    print("\n5. Cleaning up...")
    await garmin.disconnect()
    print("✓ Cleanup complete")

async def test_alternative_approach():
    """Test an alternative approach to getting tools info"""
    print("\n=== TESTING ALTERNATIVE APPROACH ===")
    
    # Load config
    with open("config.yaml") as f:
        config_data = yaml.safe_load(f)

    config = Config(**config_data)
    
    # Create a simpler MCP connector for testing
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import os
    import shutil
    
    try:
        # Set up environment
        env = os.environ.copy()
        env['GARTH_TOKEN'] = config.garth_token
        
        # Find server command
        server_command = shutil.which("garth-mcp-server")
        if not server_command:
            print("✗ garth-mcp-server not found")
            return
            
        print(f"✓ Found server at: {server_command}")
        
        # Create server parameters
        server_params = StdioServerParameters(
            command="/bin/bash",
            args=["-c", f"exec {server_command} \"$@\" 1>&2"],
            env=env,
        )
        
        print("Starting direct MCP test...")
        async with stdio_client(server_params) as streams:
            if len(streams) == 3:
                read_stream, write_stream, stderr_stream = streams
            else:
                read_stream, write_stream = streams
                
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            print("✓ Direct session initialized")
            
            # Try to list tools with timeout
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                
                print("Calling list_tools()...")
                tools_result = await session.list_tools()
                signal.alarm(0)
                
                print(f"✓ Direct list_tools() successful: {len(tools_result.tools) if tools_result and hasattr(tools_result, 'tools') else 0} tools")
                
                if tools_result and hasattr(tools_result, 'tools'):
                    for tool in tools_result.tools[:3]:
                        print(f"  - {tool.name}")
                        
            except TimeoutError:
                print("✗ Direct list_tools() timed out")
            except Exception as e:
                print(f"✗ Direct list_tools() error: {e}")
            finally:
                signal.alarm(0)
                
    except Exception as e:
        print(f"✗ Alternative approach failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    async def main():
        await debug_mcp_connection()
        await test_alternative_approach()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Debug script failed: {e}")
        import traceback
        traceback.print_exc()