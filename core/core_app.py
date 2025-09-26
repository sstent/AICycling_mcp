#!/usr/bin/env python3
"""
Core Application - Clean skeleton with separated concerns
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import Config, load_config
from ..llm.llm_client import LLMClient
from ..mcp.mcp_client import MCPClient
from .cache_manager import CacheManager
from .template_engine import TemplateEngine

logger = logging.getLogger(__name__)

class CyclingAnalyzerApp:
    """Main application class - orchestrates all components"""
    
    def __init__(self, config: Config, test_mode: bool = False):
            self.config = config
            self.test_mode = test_mode
            self.llm_client = LLMClient(config)
            self.mcp_client = MCPClient(config)
            self.cache_manager = CacheManager()
            self.template_engine = TemplateEngine(config.templates_dir)
            
            logger.info("DEBUG: Cache contents after init:")
            for key in ["user_profile", "last_cycling_details"]:
                data = self.cache_manager.get(key, {})
                logger.info(f"  {key}: keys={list(data.keys()) if data else 'EMPTY'}, length={len(data) if data else 0}")
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing application components...")
        
        await self.llm_client.initialize()
        await self.mcp_client.initialize()
        await self._preload_cache()
        
        logger.info("Application initialization complete")
    
    async def cleanup(self):
        """Cleanup all components"""
        await self.mcp_client.cleanup()
        await self.llm_client.cleanup()
    
    async def _preload_cache(self):
        """Pre-load and cache common MCP responses"""
        logger.info("Pre-loading cache...")
        
        # Cache user profile
        if await self.mcp_client.has_tool("user_profile"):
            profile = await self.mcp_client.call_tool("user_profile", {})
            self.cache_manager.set("user_profile", profile)
        
        # Cache recent activities
        if await self.mcp_client.has_tool("get_activities"):
            activities = await self.mcp_client.call_tool("get_activities", {"limit": 10})
            self.cache_manager.set("recent_activities", activities)
            
            # Find and cache last cycling activity details
            cycling_activity = self._find_last_cycling_activity(activities)
            if cycling_activity and await self.mcp_client.has_tool("get_activity_details"):
                details = await self.mcp_client.call_tool(
                    "get_activity_details", 
                    {"activity_id": cycling_activity["activityId"]}
                )
                self.cache_manager.set("last_cycling_details", details)
    
    def _find_last_cycling_activity(self, activities: list) -> Optional[Dict[str, Any]]:
        """Find the most recent cycling activity from activities list"""
        cycling_activities = [
            act for act in activities
            if "cycling" in act.get("activityType", {}).get("typeKey", "").lower()
        ]
        return max(cycling_activities, key=lambda x: x.get("startTimeGmt", 0)) if cycling_activities else None
    
    # Core functionality methods
    
    async def analyze_workout(self, analysis_type: str = "last_workout", **kwargs) -> str:
        """Analyze workout using LLM with cached data"""
        template_name = f"workflows/{analysis_type}.txt"
        
        # Prepare enhanced context with data quality assessment
        context = self._prepare_analysis_context(**kwargs)
        
        # Load and render template
        logger.info(f"Rendering template {template_name} with context keys: {list(context.keys())}")
        prompt = self.template_engine.render(template_name, **context)
        
        if self.test_mode:
            logger.info("Test mode: Printing rendered prompt instead of calling LLM")
            print("\n" + "="*60)
            print("RENDERED PROMPT FOR LLM:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")
            return f"TEST MODE: Prompt rendered (length: {len(prompt)} characters)"
        
        # Call LLM
        return await self.llm_client.generate(prompt)
    
    def _prepare_analysis_context(self, **kwargs) -> Dict[str, Any]:
            """Prepare analysis context with data quality assessment"""
            user_info = self.cache_manager.get("user_profile", {})
            activity_summary = self.cache_manager.get("last_cycling_details", {})
            
            logger.info(f"DEBUG: user_info keys: {list(user_info.keys()) if user_info else 'EMPTY'}, length: {len(user_info) if user_info else 0}")
            logger.info(f"DEBUG: activity_summary keys: {list(activity_summary.keys()) if activity_summary else 'EMPTY'}, length: {len(activity_summary) if activity_summary else 0}")
            
            # Assess data quality
            data_quality = self._assess_data_quality(activity_summary)
            logger.info(f"DEBUG: data_quality: {data_quality}")
            
            context = {
                "user_info": user_info,
                "activity_summary": activity_summary,
                "data_quality": data_quality,
                "missing_metrics": data_quality.get("missing", []),
                **kwargs
            }
            
            logger.debug(f"Prepared context with data quality: {data_quality.get('overall', 'N/A')}")
            return context
    
    def _assess_data_quality(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality and completeness of activity data"""
        summary_dto = activity_data.get('summaryDTO', {})
        is_indoor = activity_data.get('is_indoor', False)
        
        missing = []
        overall = "complete"
        
        # Key metrics for outdoor cycling
        outdoor_metrics = ['averageSpeed', 'maxSpeed', 'elevationGain', 'elevationLoss']
        # Key metrics for indoor cycling
        indoor_metrics = ['averagePower', 'maxPower', 'averageHR', 'maxHR']
        
        if is_indoor:
            expected = indoor_metrics
            note = "Indoor activity - focus on power and heart rate metrics"
        else:
            expected = outdoor_metrics
            note = "Outdoor activity - full metrics expected"
        
        for metric in expected:
            if summary_dto.get(metric) is None:
                missing.append(metric)
        
        if missing:
            overall = "incomplete"
            note += f" | Missing: {', '.join(missing)}"
        
        return {
            "overall": overall,
            "is_indoor": is_indoor,
            "missing": missing,
            "note": note,
            "available_metrics": [k for k, v in summary_dto.items() if v is not None]
        }
    
    async def suggest_next_workout(self, **kwargs) -> str:
        """Generate workout suggestion using MCP tools and LLM"""
        # Use MCP-enabled agent for dynamic tool usage
        template_name = "workflows/suggest_next_workout.txt"
        
        # Prepare enhanced context
        context = self._prepare_analysis_context(**kwargs)
        context["training_rules"] = kwargs.get("training_rules", "")
        
        prompt = self.template_engine.render(template_name, **context)
        
        if self.test_mode:
            logger.info("Test mode: Printing rendered prompt instead of calling LLM with tools")
            print("\n" + "="*60)
            print("RENDERED PROMPT FOR LLM WITH TOOLS:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")
            return f"TEST MODE: Prompt rendered (length: {len(prompt)} characters)"
        
        # Use MCP-enabled LLM client for this
        return await self.llm_client.generate_with_tools(prompt, self.mcp_client)
    
    async def enhanced_analysis(self, analysis_type: str, **kwargs) -> str:
        """Perform enhanced analysis with full MCP tool access"""
        template_name = "workflows/enhanced_analysis.txt"
        
        # Prepare enhanced context
        context = self._prepare_analysis_context(**kwargs)
        context.update({
            "analysis_type": analysis_type,
            "cached_data": self.cache_manager.get_all(),
        })
        
        prompt = self.template_engine.render(template_name, **context)
        
        if self.test_mode:
            logger.info("Test mode: Printing rendered prompt instead of calling LLM with tools")
            print("\n" + "="*60)
            print("RENDERED PROMPT FOR ENHANCED ANALYSIS:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")
            return f"TEST MODE: Prompt rendered (length: {len(prompt)} characters)"
        
        return await self.llm_client.generate_with_tools(prompt, self.mcp_client)
    
    # Utility methods
    
    async def list_available_tools(self) -> list:
        """Get list of available MCP tools"""
        return await self.mcp_client.list_tools()
    
    def list_templates(self) -> list:
        """Get list of available templates"""
        return self.template_engine.list_templates()
    
    def get_cached_data(self, key: str = None) -> Any:
        """Get cached data by key, or all if no key provided"""
        return self.cache_manager.get(key) if key else self.cache_manager.get_all()

async def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        config = load_config()
        app = CyclingAnalyzerApp(config)
        
        await app.initialize()
        
        # Example usage
        print("Available tools:", len(await app.list_available_tools()))
        print("Available templates:", len(app.list_templates()))
        
        # Run analysis
        # analysis = await app.analyze_workout("analyze_last_workout",
        #                                     training_rules="Sample rules")
        # print("Analysis:", analysis[:200] + "...")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if 'app' in locals():
            await app.cleanup()

if __name__ == "__main__":
    asyncio.run(main())