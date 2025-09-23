#!/usr/bin/env python3
"""
Script to launch an MCP server in the background and list its available tools.
"""

import asyncio
import json
import platform
import subprocess
import sys
import time
from typing import Dict, List, Any, Optional

class MCPClient:
    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process = None
        self.request_id = 1
        
    async def start_server(self):
        """Start the MCP server process."""
        print(f"Starting MCP server: {' '.join(self.server_command)}")
        print(f"Python version: {platform.python_version()}")
        
        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Give the server a moment to start
        await asyncio.sleep(0.5)
        
        # Debug: show process object type
        print(f"Process object type: {type(self.process)}")
        
        # Check if process has terminated
        if self.process.returncode is not None:
            stderr = await self.process.stderr.read()
            raise Exception(f"Server failed to start. Error: {stderr.decode()}")
            
        print("Server started successfully")
        
    async def send_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP server."""
        if not self.process:
            raise Exception("Server not started")
            
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        
        if params is not None:
            request["params"] = params
            
        self.request_id += 1
        
        # Send request
        request_json = json.dumps(request) + "\n"
        print(f"Sending request: {request_json.strip()}")
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise Exception("No response from server")
            
        try:
            response_str = response_line.decode().strip()
            print(f"Received response: {response_str}")
            response = json.loads(response_str)
            return response
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {e}. Response: {response_str}")
    
    async def initialize(self):
        """Initialize the MCP server."""
        print("Initializing server...")
        
        try:
            response = await self.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "mcp-tool-lister",
                    "version": "1.0.0"
                }
            })
            
            if "error" in response:
                raise Exception(f"Initialization failed: {response['error']}")
                
            print("Server initialized successfully")
            return response.get("result", {})
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        print("Requesting tools list...")
        
        try:
            # Pass empty parameters object to satisfy server requirements
            response = await self.send_request("tools/list", {})
            
            if "error" in response:
                raise Exception(f"Failed to list tools: {response['error']}")
                
            tools = response.get("result", {}).get("tools", [])
            print(f"Found {len(tools)} tools")
            return tools
        except Exception as e:
            print(f"Tool listing error: {str(e)}")
            raise
        
    async def stop_server(self):
        """Stop the MCP server process."""
        if self.process:
            print("Stopping server...")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                print("Server didn't stop gracefully, killing...")
                self.process.kill()
                await self.process.wait()
            print("Server stopped")

def print_tools(tools: List[Dict[str, Any]]):
    """Pretty print the tools list."""
    if not tools:
        print("\nNo tools available.")
        return
        
    print(f"\n{'='*60}")
    print("AVAILABLE TOOLS")
    print(f"{'='*60}")
    
    for i, tool in enumerate(tools, 1):
        name = tool.get("name", "Unknown")
        description = tool.get("description", "No description available")
        
        print(f"\n{i}. {name}")
        print(f"   Description: {description}")
        
        # Print input schema if available
        input_schema = tool.get("inputSchema", {})
        if input_schema:
            properties = input_schema.get("properties", {})
            if properties:
                print("   Parameters:")
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "unknown")
                    prop_desc = prop_info.get("description", "")
                    required = prop_name in input_schema.get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    print(f"     - {prop_name} ({prop_type}){req_str}: {prop_desc}")
    
    print(f"\n{'='*60}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_tool_lister.py <server_command> [args...]")
        print("Example: python mcp_tool_lister.py uvx my-mcp-server")
        sys.exit(1)
    
    server_command = sys.argv[1:]
    client = MCPClient(server_command)
    
    try:
        # Start and initialize the server
        await client.start_server()
        init_result = await client.initialize()
        
        # Print server info
        server_info = init_result.get("serverInfo", {})
        if server_info:
            print(f"Server: {server_info.get('name', 'Unknown')} v{server_info.get('version', 'Unknown')}")
        
        capabilities = init_result.get("capabilities", {})
        if capabilities:
            print(f"Server capabilities: {', '.join(capabilities.keys())}")
        
        # List and display tools
        tools = await client.list_tools()
        print_tools(tools)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await client.stop_server()

if __name__ == "__main__":
    asyncio.run(main())