#!/usr/bin/env python3
"""
Cycling Workout Analyzer with Pydantic AI and MCP Server Integration
A Python app that uses Pydantic AI with MCP tools to analyze cycling workouts
"""

import os
import json
import asyncio
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml
from dataclasses import dataclass

# Pydantic AI imports
try:
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    print("Pydantic AI not available. Install with: pip install pydantic-ai")

# MCP Protocol imports for direct connection
try:
    from pydantic_ai.mcp import MCPServerStdio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("pydantic_ai.mcp not available. You might need to upgrade pydantic-ai.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Application configuration"""
    openrouter_api_key: str
    openrouter_model: str = "deepseek/deepseek-r1-0528:free"
    garth_token: str = ""
    garth_mcp_server_path: str = "uvx"
    rules_file: str = "rules.yaml"
    templates_dir: str = "templates"

def print_tools(tools: List[Any]):
    """Pretty print the tools list."""
    if not tools:
        print("\nNo tools available.")
        return

    print(f"\n{'='*60}")
    print("AVAILABLE TOOLS")
    print(f"\n{'='*60}")

    for i, tool in enumerate(tools, 1):
        print(f"\n{i}. {tool.name}")
        if tool.description:
            print(f"   Description: {tool.description}")

        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            properties = tool.inputSchema.get("properties", {})
            if properties:
                print("   Parameters:")
                required_params = tool.inputSchema.get("required", [])
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "unknown")
                    prop_desc = prop_info.get("description", "")
                    required = prop_name in required_params
                    req_str = " (required)" if required else " (optional)"
                    print(f"     - {prop_name} ({prop_type}){req_str}: {prop_desc}")

    print(f"\n{'='*60}")

class PydanticAIAnalyzer:
    """Pydantic AI powered cycling analyzer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mcp_server = None
        self.available_tools = []
        
        if not PYDANTIC_AI_AVAILABLE or not MCP_AVAILABLE:
            raise Exception("Pydantic AI or MCP not available. Please check your installation.")
        
        os.environ['OPENROUTER_API_KEY'] = config.openrouter_api_key
        os.environ['OPENAI_BASE_URL'] = "https://openrouter.ai/api/v1"
        os.environ['OPENAI_DEFAULT_HEADERS'] = json.dumps({
            "HTTP-Referer": "https://github.com/cycling-analyzer",
            "X-Title": "Cycling Workout Analyzer"
        })
        
        env = os.environ.copy()
        env["GARTH_TOKEN"] = config.garth_token
        
        server_executable = shutil.which(config.garth_mcp_server_path)
        if not server_executable:
            logger.error(f"'{config.garth_mcp_server_path}' not found in PATH. MCP tools will be unavailable.")
        else:
            self.mcp_server = MCPServerStdio(
                command=server_executable,
                args=["garth-mcp-server"],
                env=env,
            )

        model_name = f"openrouter:{config.openrouter_model}"
        
        self.agent = Agent(
            model=model_name,
            system_prompt="""You are an expert cycling coach with access to comprehensive Garmin Connect data.
            You analyze cycling workouts, provide performance insights, and give actionable training recommendations.
            Use the available tools to gather detailed workout data and provide comprehensive analysis.""",
            toolsets=[self.mcp_server] if self.mcp_server else []
        )

    async def initialize(self):
        """Initialize the analyzer and connect to MCP server"""
        logger.info("Initializing Pydantic AI analyzer...")
        if self.agent and self.mcp_server:
            try:
                await asyncio.wait_for(self.agent.__aenter__(), timeout=45)
                logger.info("✓ Agent context entered successfully")
                self.available_tools = await self.mcp_server.list_tools()
                logger.info(f"✓ Found {len(self.available_tools)} MCP tools.")
            except asyncio.TimeoutError:
                logger.error("Agent initialization timed out. MCP tools will be unavailable.")
                self.mcp_server = None
            except Exception as e:
                logger.error(f"Agent initialization failed: {e}. MCP tools will be unavailable.")
                self.mcp_server = None
        else:
            logger.warning("MCP server not configured. MCP tools will be unavailable.")

    async def cleanup(self):
        """Cleanup resources"""
        if self.agent and self.mcp_server:
            await self.agent.__aexit__(None, None, None)
        logger.info("Cleanup completed")

    async def analyze_last_workout(self, training_rules: str) -> str:
        """Analyze the last cycling workout using Pydantic AI"""
        logger.info("Analyzing last workout with Pydantic AI...")
        
        prompt = f"""
        Please analyze my most recent cycling workout. Use the get_activities tool to fetch my recent activities, 
        then focus on the latest cycling workout.

        My training rules and goals:
        {training_rules}

        Please provide:
        1. Overall assessment of the workout
        2. How well it aligns with my rules and goals
        3. Areas for improvement
        4. Specific feedback on power, heart rate, duration, and intensity
        5. Recovery recommendations
        6. Comparison with typical performance metrics
        
        Use additional Garmin tools (like hrv_data or nightly_sleep) if they would provide relevant context.
        """

        try:
            result = await self.agent.run(prompt)
            return result.text
        except Exception as e:
            logger.error(f"Error in workout analysis: {e}")
            return f"Error analyzing workout: {e}"

    async def suggest_next_workout(self, training_rules: str) -> str:
        """Suggest next workout using Pydantic AI"""
        logger.info("Generating workout suggestion with Pydantic AI...")
        
        prompt = f"""
        Please suggest my next cycling workout based on my recent training history. Use the get_activities tool 
        to get my recent activities and analyze the training pattern.

        My training rules and goals:
        {training_rules}

        Please provide:
        1. Analysis of my recent training pattern
        2. Identified gaps or imbalances in my training
        3. Specific workout recommendation for my next session
        4. Target zones (power, heart rate, duration)
        5. Rationale for the recommendation based on recent performance
        6. Alternative options if weather/time constraints exist
        7. How this fits into my overall training progression

        Use additional tools like hrv_data or nightly_sleep to inform recovery status and workout readiness.
        """

        try:
            result = await self.agent.run(prompt)
            return result.text
        except Exception as e:
            logger.error(f"Error in workout suggestion: {e}")
            return f"Error suggesting workout: {e}"

    async def enhanced_analysis(self, analysis_type: str, training_rules: str) -> str:
        """Perform enhanced analysis using Pydantic AI with all available tools"""
        logger.info(f"Performing enhanced {analysis_type} analysis...")
        
        prompt = f"""
        Please perform a comprehensive {analysis_type} analysis of my cycling training data. 
        Use all available Garmin tools to gather relevant data including:
        - Recent activities and workout details
        - User profile information
        - Heart rate variability data
        - Sleep quality data
        - Any other relevant metrics

        My training rules and goals:
        {training_rules}

        Focus your {analysis_type} analysis on:
        1. **Data Gathering**: Use multiple tools to get comprehensive data
        2. **Performance Analysis**: Analyze power, heart rate, training load, and recovery metrics  
        3. **Training Periodization**: Consider my training phase and progression
        4. **Actionable Recommendations**: Provide specific, measurable guidance
        5. **Risk Assessment**: Identify any signs of overtraining or injury risk

        Be thorough and use multiple data points to support your recommendations.
        """

        try:
            result = await self.agent.run(prompt)
            return result.text
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
            return f"Error in {analysis_type} analysis: {e}"

class TemplateManager:
    """Manages prompt templates (kept for compatibility)"""
    
    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)

    def list_templates(self) -> List[str]:
        """List available templates"""
        return [f.name for f in self.templates_dir.glob("*.txt")]

class RulesManager:
    """Manages training rules and goals"""
    
    def __init__(self, rules_file: str):
        self.rules_file = Path(rules_file)
        self._create_default_rules()
    
    def _create_default_rules(self):
        """Create default rules file if it doesn't exist"""
        if not self.rules_file.exists():
            default_rules = {
                "training_goals": [
                    "Improve FTP (Functional Threshold Power)",
                    "Build endurance for 100km rides",
                    "Maintain consistent training 4-5x per week"
                ],
                "power_zones": {
                    "zone_1_active_recovery": "< 142W",
                    "zone_2_endurance": "142-162W", 
                    "zone_3_tempo": "163-180W",
                    "zone_4_lactate_threshold": "181-196W",
                    "zone_5_vo2_max": "197-224W",
                    "zone_6_anaerobic": "> 224W"
                },
                "heart_rate_zones": {
                    "zone_1": "< 129 bpm",
                    "zone_2": "129-146 bpm",
                    "zone_3": "147-163 bpm", 
                    "zone_4": "164-181 bpm",
                    "zone_5": "> 181 bpm"
                },
                "weekly_structure": {
                    "easy_rides": "60-70% of weekly volume",
                    "moderate_rides": "20-30% of weekly volume", 
                    "hard_rides": "5-15% of weekly volume"
                },
                "recovery_rules": [
                    "At least 1 full rest day per week",
                    "Easy spin after hard workouts",
                    "Listen to body - skip workout if overly fatigued"
                ],
                "workout_preferences": [
                    "Prefer morning rides when possible",
                    "Include variety - not just steady state",
                    "Focus on consistency over peak performance"
                ]
            }
            
            with open(self.rules_file, 'w') as f:
                yaml.dump(default_rules, f, default_flow_style=False)
            logger.info(f"Created default rules file: {self.rules_file}")
    
    def get_rules(self) -> str:
        """Get rules as formatted string"""
        with open(self.rules_file, 'r') as f:
            rules = yaml.safe_load(f)
        
        return yaml.dump(rules, default_flow_style=False)

class CyclingAnalyzer:
    """Main application class using Pydantic AI"""
    
    def __init__(self, config: Config):
        self.config = config
        self.analyzer = PydanticAIAnalyzer(config)
        self.templates = TemplateManager(config.templates_dir)
        self.rules = RulesManager(config.rules_file)
    
    async def initialize(self):
        """Initialize the application"""
        logger.info("Initializing Pydantic AI Cycling Analyzer...")
        await self.analyzer.initialize()
        logger.info("Application initialization complete")

    async def cleanup(self):
        """Cleanup resources"""
        await self.analyzer.cleanup()
        logger.info("Application cleanup completed")
    
    async def analyze_last_workout(self):
        """Analyze the last cycling workout"""
        rules_text = self.rules.get_rules()
        return await self.analyzer.analyze_last_workout(rules_text)
    
    async def suggest_next_workout(self):
        """Suggest next workout based on recent activities"""
        rules_text = self.rules.get_rules()
        return await self.analyzer.suggest_next_workout(rules_text)
    
    async def enhanced_analysis(self, analysis_type: str):
        """Perform enhanced analysis using all available tools"""
        rules_text = self.rules.get_rules()
        return await self.analyzer.enhanced_analysis(analysis_type, rules_text)
    
    async def list_available_tools(self):
        """List available Garmin tools"""
        return self.analyzer.available_tools
    
    async def run(self):
        """Main application loop"""
        logger.info("Starting Cycling Workout Analyzer with Pydantic AI...")
        
        logger.info("Calling initialize()...")
        await self.initialize()
        logger.info("Initialize() completed, starting main loop...")
        
        try:
            while True:
                print("\n" + "="*60)
                print("CYCLING WORKOUT ANALYZER (Pydantic AI + MCP)")
                print("="*60)
                print("1. Analyze last cycling workout")
                print("2. Get next workout suggestion")
                print("3. Enhanced analysis using all MCP tools")
                print("4. List available MCP tools")
                print("5. List available templates")
                print("6. View current rules")
                print("7. Exit")
                print("-"*60)
                
                choice = input("Enter your choice (1-7): ").strip()
                logger.info(f"User selected option: {choice}")
                
                try:
                    if choice == "1":
                        print("\nAnalyzing your last workout with Pydantic AI...")
                        analysis = await self.analyze_last_workout()
                        print("\n" + "="*50)
                        print("WORKOUT ANALYSIS (Pydantic AI)")
                        print("="*50)
                        print(analysis)
                    
                    elif choice == "2":
                        print("\nGenerating workout suggestion with Pydantic AI...")
                        suggestion = await self.suggest_next_workout()
                        print("\n" + "="*50)
                        print("NEXT WORKOUT SUGGESTION (Pydantic AI)")
                        print("="*50)
                        print(suggestion)
                    
                    elif choice == "3":
                        print("\nSelect analysis type:")
                        print("a) Performance trends")
                        print("b) Training load analysis")
                        print("c) Recovery assessment")
                        analysis_choice = input("Enter choice (a-c): ").strip().lower()
                        
                        analysis_types = {
                            'a': 'performance trends',
                            'b': 'training load',
                            'c': 'recovery assessment'
                        }
                        
                        if analysis_choice in analysis_types:
                            print(f"\nPerforming {analysis_types[analysis_choice]} analysis...")
                            analysis = await self.enhanced_analysis(
                                analysis_types[analysis_choice]
                            )
                            print(f"\n{ '='*50}")
                            print(f"ENHANCED {analysis_types[analysis_choice].upper()} ANALYSIS")
                            print("="*50)
                            print(analysis)
                        else:
                            print("Invalid choice.")
                    
                    elif choice == "4":
                        tools = await self.list_available_tools()
                        print_tools(tools)
                    
                    elif choice == "5":
                        templates = self.templates.list_templates()
                        print(f"\nAvailable templates in {self.config.templates_dir}:")
                        for template in templates:
                            print(f"  - {template}")
                    
                    elif choice == "6":
                        rules = self.rules.get_rules()
                        print(f"\nCurrent rules from {self.config.rules_file}:")
                        print("-"*30)
                        print(rules)
                    
                    elif choice == "7":
                        print("Goodbye!")
                        break
                    
                    else:
                        print("Invalid choice. Please try again.")
                
                except Exception as e:
                    logger.error(f"Error: {e}")
                    print(f"An error occurred: {e}")
                
                input("\nPress Enter to continue...")
        
        finally:
            await self.cleanup()

def load_config() -> Config:
    """Load configuration from environment and config files"""
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenRouter API key: ").strip()

    return Config(
        openrouter_api_key=api_key,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free"),
        garth_token=os.getenv("GARTH_TOKEN", ""),
        garth_mcp_server_path=os.getenv("GARTH_MCP_SERVER_PATH", "uvx"),
    )

def create_sample_config():
    """Create a sample config file"""
    config_file = Path("config.yaml")
    if not config_file.exists():
        sample_config = {
            "openrouter_api_key": "your_openrouter_api_key_here",
            "openrouter_model": "google/gemini-flash-1.5",
            "garth_token": "your_garth_token_here",
            "garth_mcp_server_path": "uvx",
            "rules_file": "rules.yaml",
            "templates_dir": "templates"
        }

        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False)
        print(f"Created sample config file: {config_file}")
        print("Please edit it with your actual OpenRouter API key and GARTH_TOKEN.")
        print("Get your GARTH_TOKEN by running: uvx garth login")

async def main():
    """Main entry point"""
    create_sample_config()
    
    try:
        config = load_config()
        analyzer = CyclingAnalyzer(config)
        await analyzer.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    asyncio.run(main())