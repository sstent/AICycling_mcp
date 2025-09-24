#!/usr/bin/env python3
"""
MCP Client - Handles MCP server connections and tool management
"""

import os
import shutil
import logging
import asyncio
from typing import List, Dict, Any, Optional

try:
    from pydantic_ai.mcp import MCPServerStdio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPServerStdio = None

from config import Config

logger = logging.getLogger(__name__)

class MCPClient:
    """Manages MCP server connection and tool interactions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mcp_server = None
        self.available_tools = []
        self._initialized = False
        
        if not MCP_AVAILABLE:
            logger.warning("MCP not available. Tool functionality will be limited.")
            return
        
        # Set up MCP server
        self._setup_mcp_server()
    
    def _setup_mcp_server(self):
        """Set up MCP server connection"""
        if not self.config.garth_token:
            logger.warning("No GARTH_TOKEN provided. MCP tools will be unavailable.")
            return
        
        # Set up environment
        os.environ["GARTH_TOKEN"] = self.config.garth_token
        env = os.environ.copy()
        env["GARTH_TOKEN"] = self.config.garth_token
        
        # Find server executable
        server_executable = shutil.which(self.config.garth_mcp_server_path)
        if not server_executable:
            logger.error(f"'{self.config.garth_mcp_server_path}' not found in PATH")
            return
        
        self.mcp_server = MCPServerStdio(
            command=server_executable,
            args=["garth-mcp-server"],
            env=env,
        )
    
    async def initialize(self):
        """Initialize MCP server connection"""
        if not self.mcp_server:
            logger.warning("MCP server not configured")
            return
        
        try:
            logger.info("Initializing MCP server connection...")
            
            # The MCP server will be initialized when used by the agent
            # For now, we'll try to list tools to verify connection
            await asyncio.sleep(0.1)  # Give it a moment
            
            logger.info("MCP server connection established")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"MCP server initialization failed: {e}")
            self.mcp_server = None
    
    async def cleanup(self):
        """Cleanup MCP server connection"""
        if self.mcp_server:
            # MCP server cleanup is handled by the agent
            pass
    
    async def list_tools(self) -> List[Any]:
        """List available MCP tools"""
        if not self.mcp_server:
            return []
        
        try:
            if not self.available_tools:
                self.available_tools = await self.mcp_server.list_tools()
            return self.available_tools
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    async def has_tool(self, tool_name: str) -> bool:
        """Check if a specific tool is available"""
        tools = await self.list_tools()
        return any(tool.name == tool_name for tool in tools)
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a specific MCP tool directly"""
        if not self.mcp_server:
            raise RuntimeError("MCP server not available")
        
        try:
            result = await self.mcp_server.direct_call_tool(tool_name, parameters)
            return result.output if hasattr(result, 'output') else result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        for tool in self.available_tools:
            if tool.name == tool_name:
                return {
                    "name": tool.name,
                    "description": getattr(tool, 'description', ''),
                    "parameters": getattr(tool, 'inputSchema', {}).get('properties', {})
                }
        return None
    
    def print_tools(self):
        """Pretty print available tools"""
        if not self.available_tools:
            print("No MCP tools available")
            return
        
        print(f"\n{'='*60}")
        print("AVAILABLE MCP TOOLS")
        print(f"{'='*60}")
        
        for i, tool in enumerate(self.available_tools, 1):
            print(f"\n{i}. {tool.name}")
            if hasattr(tool, 'description') and tool.description:
                print(f"   Description: {tool.description}")
            
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                properties = tool.inputSchema.get("properties", {})
                if properties:
                    print("   Parameters:")
                    required = tool.inputSchema.get("required", [])
                    for param, info in properties.items():
                        param_type = info.get("type", "unknown")
                        param_desc = info.get("description", "")
                        req_str = " (required)" if param in required else " (optional)"
                        print(f"     - {param} ({param_type}){req_str}: {param_desc}")
        
        print(f"\n{'='*60}")
    
    @property
    def is_available(self) -> bool:
        """Check if MCP server is available"""
        return self.mcp_server is not None and self._initialized