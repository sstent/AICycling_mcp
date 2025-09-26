#!/usr/bin/env python3
"""
Updated MCP Client - Uses our custom Garth implementation
Fallback to standard MCP if needed, but prefer custom implementation
"""

import logging
from typing import List, Dict, Any, Optional

from ..config import Config

# Try to import both implementations
try:
    from .custom_garth_mcp import CustomGarthMCP, GarthTool
    CUSTOM_GARTH_AVAILABLE = True
except ImportError:
    CUSTOM_GARTH_AVAILABLE = False
    CustomGarthMCP = None
    GarthTool = None

try:
    from pydantic_ai.mcp import MCPServerStdio
    import shutil
    import os
    STANDARD_MCP_AVAILABLE = True
except ImportError:
    STANDARD_MCP_AVAILABLE = False
    MCPServerStdio = None

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Enhanced MCP Client that prefers custom Garth implementation
    Falls back to standard MCP if needed
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.garth_mcp = None
        self.standard_mcp = None
        self.available_tools = []
        self._initialized = False
        self._use_custom = True
        
        # Decide which implementation to use
        if CUSTOM_GARTH_AVAILABLE and config.garth_token:
            logger.info("Using custom Garth MCP implementation")
            self._use_custom = True
        elif STANDARD_MCP_AVAILABLE and config.garth_token:
            logger.info("Falling back to standard MCP implementation")
            self._use_custom = False
        else:
            logger.warning("No MCP implementation available")
    
    async def initialize(self):
        """Initialize the preferred MCP implementation"""
        if not self.config.garth_token:
            logger.warning("No GARTH_TOKEN provided. MCP tools will be unavailable.")
            return
        
        try:
            if self._use_custom and CUSTOM_GARTH_AVAILABLE:
                await self._initialize_custom_garth()
            elif STANDARD_MCP_AVAILABLE:
                await self._initialize_standard_mcp()
            else:
                logger.error("No MCP implementation available")
                return
            
            self._initialized = True
            logger.info("MCP client initialized successfully")
            
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            # Try fallback if custom failed
            if self._use_custom and STANDARD_MCP_AVAILABLE:
                logger.info("Trying fallback to standard MCP")
                try:
                    self._use_custom = False
                    await self._initialize_standard_mcp()
                    self._initialized = True
                    logger.info("Fallback MCP initialization successful")
                except Exception as fallback_error:
                    logger.error(f"Fallback MCP initialization also failed: {fallback_error}")
    
    async def _initialize_custom_garth(self):
        """Initialize custom Garth MCP"""
        self.garth_mcp = CustomGarthMCP(
            garth_token=self.config.garth_token,
            cache_ttl=self.config.cache_ttl
        )
        await self.garth_mcp.initialize()
        logger.info("Custom Garth MCP initialized")
    
    async def _initialize_standard_mcp(self):
        """Initialize standard MCP (fallback)"""
        if not self.config.garth_token:
            raise ValueError("GARTH_TOKEN required for standard MCP")
        
        # Set up environment
        os.environ["GARTH_TOKEN"] = self.config.garth_token
        env = os.environ.copy()
        env["GARTH_TOKEN"] = self.config.garth_token
        
        # Find server executable
        server_executable = shutil.which(self.config.garth_mcp_server_path)
        if not server_executable:
            raise FileNotFoundError(f"'{self.config.garth_mcp_server_path}' not found in PATH")
        
        self.standard_mcp = MCPServerStdio(
            command=server_executable,
            args=["garth-mcp-server"],
            env=env,
        )
        logger.info("Standard MCP initialized")
    
    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.garth_mcp:
            await self.garth_mcp.cleanup()
        # Standard MCP cleanup is handled by the agent
    
    async def list_tools(self) -> List[Any]:
        """List available MCP tools"""
        if not self._initialized:
            return []
        
        try:
            if self._use_custom and self.garth_mcp:
                if not self.available_tools:
                    self.available_tools = self.garth_mcp.list_tools()
                return self.available_tools
            elif self.standard_mcp:
                if not self.available_tools:
                    self.available_tools = await self.standard_mcp.list_tools()
                return self.available_tools
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
        
        return []
    
    def list_tools_sync(self) -> List[Any]:
        """Synchronous version for compatibility"""
        if self._use_custom and self.garth_mcp:
            return self.garth_mcp.list_tools()
        return []
    
    async def has_tool(self, tool_name: str) -> bool:
        """Check if a specific tool is available"""
        if not self._initialized:
            return False
        
        try:
            if self._use_custom and self.garth_mcp:
                return await self.garth_mcp.has_tool(tool_name)
            elif self.standard_mcp:
                tools = await self.list_tools()
                return any(tool.name == tool_name for tool in tools)
        except Exception as e:
            logger.error(f"Error checking tool {tool_name}: {e}")
            return False
        
        return False
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a specific MCP tool"""
        if not self._initialized:
            raise RuntimeError("MCP client not initialized")
        
        try:
            if self._use_custom and self.garth_mcp:
                result = await self.garth_mcp.call_tool(tool_name, parameters)
                logger.debug(f"Custom MCP tool {tool_name} called successfully")
                return result
            elif self.standard_mcp:
                result = await self.standard_mcp.direct_call_tool(tool_name, parameters)
                # Handle different result formats
                if hasattr(result, 'output'):
                    return result.output
                elif hasattr(result, 'content'):
                    return result.content
                else:
                    return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        tools = self.list_tools_sync() if self._use_custom else []
        
        for tool in tools:
            if tool.name == tool_name:
                if self._use_custom and isinstance(tool, GarthTool):
                    return {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                else:
                    # Standard MCP tool
                    return {
                        "name": tool.name,
                        "description": getattr(tool, 'description', ''),
                        "parameters": getattr(tool, 'inputSchema', {}).get('properties', {})
                    }
        return None
    
    def print_tools(self):
        """Pretty print available tools"""
        if self._use_custom and self.garth_mcp:
            self.garth_mcp.print_tools()
            return
        
        tools = self.list_tools_sync()
        if not tools:
            print("No MCP tools available")
            return
        
        print(f"\n{'='*60}")
        print("AVAILABLE MCP TOOLS")
        print(f"{'='*60}")
        
        for i, tool in enumerate(tools, 1):
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
        """Check if MCP is available and initialized"""
        return self._initialized and (
            (self._use_custom and self.garth_mcp is not None) or
            (not self._use_custom and self.standard_mcp is not None)
        )
    
    @property
    def implementation_type(self) -> str:
        """Get the type of MCP implementation being used"""
        if self._use_custom:
            return "custom_garth"
        else:
            return "standard_mcp"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (only available with custom implementation)"""
        if self._use_custom and self.garth_mcp:
            return self.garth_mcp.get_cache_stats()
        else:
            return {"message": "Cache stats only available with custom implementation"}
    
    def clear_cache(self):
        """Clear cache (only available with custom implementation)"""
        if self._use_custom and self.garth_mcp:
            self.garth_mcp.clear_cache()
            logger.info("Cache cleared")
        else:
            logger.warning("Cache clearing only available with custom implementation")
    
    # Compatibility methods for existing code
    @property
    def mcp_server(self):
        """Compatibility property for existing code using agent integration"""
        if self._use_custom:
            # For custom implementation, we can't provide direct agent integration
            # Return None to indicate tools should be called directly
            return None
        else:
            return self.standard_mcp