#!/usr/bin/env python3
"""
LLM Client - Handles all LLM interactions
"""

import os
import json
import logging
import asyncio
from typing import Optional, Any

try:
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None

from ..config import Config

logger = logging.getLogger(__name__)

class LLMClient:
    """Handles LLM interactions with and without MCP tools"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agent_with_tools = None
        self.agent_no_tools = None
        
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError("Pydantic AI not available. Install with: pip install pydantic-ai")
        
        # Set up OpenRouter environment
        self._setup_openrouter_env()
    
    def _setup_openrouter_env(self):
        """Configure OpenRouter environment variables"""
        os.environ['OPENROUTER_API_KEY'] = self.config.openrouter_api_key
        os.environ['OPENAI_BASE_URL'] = "https://openrouter.ai/api/v1"
        os.environ['OPENAI_DEFAULT_HEADERS'] = json.dumps({
            "HTTP-Referer": "https://github.com/cycling-analyzer",
            "X-Title": "Cycling Workout Analyzer"
        })
    
    async def initialize(self):
        """Initialize LLM clients"""
        logger.info("Initializing LLM clients...")
        
        model_name = f"openrouter:{self.config.openrouter_model}"
        
        # Agent without tools for analysis
        self.agent_no_tools = Agent(
            model=model_name,
            system_prompt="You are an expert cycling coach. Analyze the provided data comprehensively.",
            toolsets=[]
        )
        
        logger.info("LLM clients initialized")
    
    async def cleanup(self):
        """Cleanup LLM clients"""
        if self.agent_with_tools:
            await self.agent_with_tools.__aexit__(None, None, None)
        if self.agent_no_tools:
            await self.agent_no_tools.__aexit__(None, None, None)
    
    async def generate(self, prompt: str, use_tools: bool = False) -> str:
        """Generate response using LLM without tools"""
        if not self.agent_no_tools:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Initialize agent context if not already done
            if not hasattr(self.agent_no_tools, '_initialized'):
                await asyncio.wait_for(self.agent_no_tools.__aenter__(), timeout=30)
                self.agent_no_tools._initialized = True
            
            result = await self.agent_no_tools.run(prompt)
            return str(result)
        
        except asyncio.TimeoutError:
            logger.error("LLM request timed out")
            return "Error: LLM request timed out"
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {e}"
    
    async def generate_with_tools(self, prompt: str, mcp_client) -> str:
        """Generate response using LLM with MCP tools"""
        # Create a temporary agent with tools for this request
        try:
            model_name = f"openrouter:{self.config.openrouter_model}"
            
            temp_agent = Agent(
                model=model_name,
                system_prompt="You are an expert cycling coach with access to comprehensive Garmin Connect data. Use available tools to gather data and provide detailed analysis.",
                toolsets=[mcp_client.mcp_server] if mcp_client.mcp_server else []
            )
            
            # Initialize temporary agent
            await asyncio.wait_for(temp_agent.__aenter__(), timeout=30)
            
            # Generate response
            result = await temp_agent.run(prompt)
            
            # Cleanup
            await temp_agent.__aexit__(None, None, None)
            
            return result.text if hasattr(result, 'text') else str(result)
        
        except asyncio.TimeoutError:
            logger.error("LLM with tools request timed out")
            return "Error: LLM request with tools timed out"
        except Exception as e:
            logger.error(f"LLM generation with tools error: {e}")
            return f"Error generating response with tools: {e}"
    
    async def chat(self, messages: list, use_tools: bool = False) -> str:
        """Chat-style interaction (for future extension)"""
        # Convert messages to prompt for now
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return await self.generate(prompt, use_tools)