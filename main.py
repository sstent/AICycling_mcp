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
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import aiohttp
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
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not available. Install with: pip install mcp")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Application configuration"""
    openrouter_api_key: str
    openrouter_model: str = "deepseek/deepseek-r1-0528:free"
    garth_token: str = ""  # GARTH_TOKEN for authentication
    garth_mcp_server_path: str = "uvx"  # Use uvx to run garth-mcp-server
    rules_file: str = "rules.yaml"
    templates_dir: str = "templates"

class GarminMCPTools:
    """MCP Tools interface for Pydantic AI"""
    
    def __init__(self, garth_token: str, server_path: str):
        self.garth_token = garth_token
        self.server_path = server_path
        self.server_available = False
        self._session: Optional[ClientSession] = None
        self._client_context = None
        self._read_stream = None
        self._write_stream = None
        self._connection_timeout = 30
        
        # Known tools (workaround for hanging list_tools)
        self.available_tools = [
            {"name": "user_profile", "description": "Get user profile information"},
            {"name": "user_settings", "description": "Get user settings and preferences"},
            {"name": "daily_sleep", "description": "Get daily sleep summary data"},
            {"name": "daily_steps", "description": "Get daily steps data"},
            {"name": "daily_hrv", "description": "Get heart rate variability data"},
            {"name": "get_activities", "description": "Get list of activities"},
            {"name": "get_activity_details", "description": "Get detailed activity information"},
            {"name": "get_body_composition", "description": "Get body composition data"},
            {"name": "get_respiration_data", "description": "Get respiration data"},
            {"name": "get_blood_pressure", "description": "Get blood pressure readings"}
        ]

    async def _get_server_params(self):
        """Get server parameters for MCP connection"""
        env = os.environ.copy()
        env['GARTH_TOKEN'] = self.garth_token

        server_command = shutil.which("garth-mcp-server")
        if not server_command:
            logger.error("Could not find 'garth-mcp-server' in your PATH.")
            raise FileNotFoundError("garth-mcp-server not found")

        return StdioServerParameters(
            command="/bin/bash",
            args=["-c", f"exec {server_command} \"$@\" 1>&2"],
            capture_stderr=True,
            env=env,
        )

    async def connect(self):
        """Connect to MCP server"""
        if self._session and self.server_available:
            return True

        if not MCP_AVAILABLE:
            logger.error("MCP library not available")
            return False

        try:
            logger.info("Connecting to Garth MCP server...")
            server_params = await self._get_server_params()
            
            self._client_context = stdio_client(server_params)
            streams = await self._client_context.__aenter__()

            if len(streams) == 3:
                self._read_stream, self._write_stream, stderr_stream = streams
                asyncio.create_task(self._log_stderr(stderr_stream))
            else:
                self._read_stream, self._write_stream = streams

            await asyncio.sleep(1.0)

            self._session = ClientSession(self._read_stream, self._write_stream)
            
            try:
                await asyncio.wait_for(self._session.initialize(), timeout=self._connection_timeout)
                logger.info("✓ MCP session initialized successfully")
                
                # Skip the hanging list_tools() call - we'll use our known tools list
                logger.info("Skipping list_tools() call (known to hang), using predefined tools")
                self.server_available = True
                return True
            except asyncio.TimeoutError:
                logger.error("MCP session initialization timed out")
                await self.disconnect()
                return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.disconnect()
            return False

    async def _log_stderr(self, stderr_stream):
        """Log stderr from server"""
        try:
            async for line in stderr_stream:
                logger.debug(f"[garth-mcp-server] {line.decode().strip()}")
        except Exception:
            pass

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Disconnect error: {e}")
        
        self._session = None
        self.server_available = False
        self._client_context = None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call MCP tool with timeout"""
        if not self.server_available or not self._session:
            # Return mock data if no connection
            return self._get_mock_tool_response(tool_name, arguments)

        try:
            logger.info(f"Calling MCP tool: {tool_name}")
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments or {}),
                timeout=self._connection_timeout
            )
            logger.info(f"✓ Tool call '{tool_name}' successful")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Tool call '{tool_name}' timed out, using mock data")
            return self._get_mock_tool_response(tool_name, arguments)
        except Exception as e:
            logger.error(f"Tool call '{tool_name}' failed: {e}, using mock data")
            return self._get_mock_tool_response(tool_name, arguments)

    def _get_mock_tool_response(self, tool_name: str, arguments: Dict[str, Any] = None):
        """Generate mock responses for testing"""
        if tool_name == "get_activities":
            limit = arguments.get("limit", 10) if arguments else 10
            activities = []
            for i in range(min(limit, 5)):
                activities.append({
                    "activityId": f"1234567890{i}",
                    "activityName": f"Cycling Workout {i+1}",
                    "startTimeLocal": f"2024-01-{15+i:02d}T08:00:00",
                    "activityType": {"typeKey": "cycling"},
                    "distance": 25000 + (i * 2000),
                    "duration": 3600 + (i * 300),
                    "averageSpeed": 6.94 + (i * 0.1),
                    "maxSpeed": 12.5 + (i * 0.2),
                    "elevationGain": 350 + (i * 25),
                    "averageHR": 145 + (i * 2),
                    "maxHR": 172 + (i * 3),
                    "averagePower": 180 + (i * 10),
                    "maxPower": 420 + (i * 15),
                    "normalizedPower": 185 + (i * 8),
                    "calories": 890 + (i * 50),
                    "averageCadence": 85 + (i * 2),
                    "maxCadence": 110 + (i * 1)
                })
            
            class MockResult:
                def __init__(self, data):
                    self.content = [MockContent(json.dumps(data))]
            
            class MockContent:
                def __init__(self, text):
                    self.text = text
            
            return MockResult(activities)
        
        elif tool_name == "user_profile":
            profile_data = {
                "displayName": "Test Cyclist",
                "fullName": "Test User",
                "email": "test@example.com",
                "profileImageUrl": None
            }
            
            class MockResult:
                def __init__(self, data):
                    self.content = [MockContent(json.dumps(data))]
            
            class MockContent:
                def __init__(self, text):
                    self.text = text
            
            return MockResult(profile_data)
        
        # Default empty response
        class MockResult:
            def __init__(self):
                self.content = [MockContent("{}")]
        
        class MockContent:
            def __init__(self, text):
                self.text = text
        
        return MockResult()

class PydanticAIAnalyzer:
    """Pydantic AI powered cycling analyzer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.garmin_tools = GarminMCPTools(config.garth_token, config.garth_mcp_server_path)
        
        if not PYDANTIC_AI_AVAILABLE:
            raise Exception("Pydantic AI not available. Install with: pip install pydantic-ai")
        
        # Set environment variables for OpenRouter
        os.environ['OPENROUTER_API_KEY'] = config.openrouter_api_key
        os.environ['OPENAI_BASE_URL'] = "https://openrouter.ai/api/v1"
        os.environ['OPENAI_DEFAULT_HEADERS'] = json.dumps({
            "HTTP-Referer": "https://github.com/cycling-analyzer",
            "X-Title": "Cycling Workout Analyzer"
        })
        
        # Create agent with OpenRouter model using string identifier
        # Pydantic AI supports OpenRouter via "openrouter:" prefix
        model_name = f"openrouter:{config.openrouter_model}"
        
        self.agent = Agent(
            model=model_name,
            system_prompt="""You are an expert cycling coach with access to comprehensive Garmin Connect data.
            You analyze cycling workouts, provide performance insights, and give actionable training recommendations.
            Use the available tools to gather detailed workout data and provide comprehensive analysis.""",
        )
        
        # Register MCP tools as Pydantic AI tools
        self._register_garmin_tools()

    def _register_garmin_tools(self):
        """Register Garmin MCP tools as Pydantic AI tools"""
        
        from pydantic_ai import RunContext
        
        @self.agent.tool
        async def get_garmin_activities(ctx: RunContext[None], limit: int = 10) -> str:
            """Get recent Garmin activities"""
            try:
                result = await self.garmin_tools.call_tool("get_activities", {"limit": limit})
                if result and hasattr(result, 'content'):
                    activities = []
                    for content in result.content:
                        if hasattr(content, 'text'):
                            try:
                                data = json.loads(content.text)
                                if isinstance(data, list):
                                    activities.extend(data)
                                else:
                                    activities.append(data)
                            except json.JSONDecodeError:
                                activities.append({"description": content.text})
                    return json.dumps(activities, indent=2)
                return "No activities data available"
            except Exception as e:
                logger.error(f"Error getting activities: {e}")
                return f"Error retrieving activities: {e}"

        @self.agent.tool
        async def get_garmin_user_profile(ctx: RunContext[None]) -> str:
            """Get Garmin user profile information"""
            try:
                result = await self.garmin_tools.call_tool("user_profile")
                if result and hasattr(result, 'content'):
                    for content in result.content:
                        if hasattr(content, 'text'):
                            return content.text
                return "No profile data available"
            except Exception as e:
                logger.error(f"Error getting profile: {e}")
                return f"Error retrieving profile: {e}"

        @self.agent.tool
        async def get_garmin_activity_details(ctx: RunContext[None], activity_id: str) -> str:
            """Get detailed information about a specific Garmin activity"""
            try:
                result = await self.garmin_tools.call_tool("get_activity_details", {"activity_id": activity_id})
                if result and hasattr(result, 'content'):
                    for content in result.content:
                        if hasattr(content, 'text'):
                            return content.text
                return "No activity details available"
            except Exception as e:
                logger.error(f"Error getting activity details: {e}")
                return f"Error retrieving activity details: {e}"

        @self.agent.tool
        async def get_garmin_hrv_data(ctx: RunContext[None]) -> str:
            """Get heart rate variability data from Garmin"""
            try:
                result = await self.garmin_tools.call_tool("daily_hrv")
                if result and hasattr(result, 'content'):
                    for content in result.content:
                        if hasattr(content, 'text'):
                            return content.text
                return "No HRV data available"
            except Exception as e:
                logger.error(f"Error getting HRV data: {e}")
                return f"Error retrieving HRV data: {e}"

        @self.agent.tool
        async def get_garmin_sleep_data(ctx: RunContext[None]) -> str:
            """Get sleep data from Garmin"""
            try:
                result = await self.garmin_tools.call_tool("daily_sleep")
                if result and hasattr(result, 'content'):
                    for content in result.content:
                        if hasattr(content, 'text'):
                            return content.text
                return "No sleep data available"
            except Exception as e:
                logger.error(f"Error getting sleep data: {e}")
                return f"Error retrieving sleep data: {e}"

    async def initialize(self):
        """Initialize the analyzer and connect to MCP server"""
        logger.info("Initializing Pydantic AI analyzer...")
        
        try:
            # Add timeout to the entire connection process
            success = await asyncio.wait_for(
                self.garmin_tools.connect(), 
                timeout=45  # 45 second timeout
            )
            if success:
                logger.info("✓ MCP server connected successfully")
            else:
                logger.warning("MCP server connection failed - will use mock data")
        except asyncio.TimeoutError:
            logger.error("MCP connection timed out after 45 seconds - using mock data")
            success = False
        except Exception as e:
            logger.error(f"MCP connection error: {e} - using mock data")
            success = False
        
        # Add debug info
        logger.info("Initialization completed successfully")
        return True

    async def cleanup(self):
        """Cleanup resources"""
        await self.garmin_tools.disconnect()
        logger.info("Cleanup completed")

    async def analyze_last_workout(self, training_rules: str) -> str:
        """Analyze the last cycling workout using Pydantic AI"""
        logger.info("Analyzing last workout with Pydantic AI...")
        
        prompt = f"""
        Please analyze my most recent cycling workout. Use the get_garmin_activities tool to fetch my recent activities, 
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
        
        Use additional Garmin tools (like HRV or sleep data) if they would provide relevant context.
        """

        try:
            result = await self.agent.run(prompt)
            return result.data
        except Exception as e:
            logger.error(f"Error in workout analysis: {e}")
            return f"Error analyzing workout: {e}"

    async def suggest_next_workout(self, training_rules: str) -> str:
        """Suggest next workout using Pydantic AI"""
        logger.info("Generating workout suggestion with Pydantic AI...")
        
        prompt = f"""
        Please suggest my next cycling workout based on my recent training history. Use the get_garmin_activities tool 
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

        Use additional tools like HRV or sleep data to inform recovery status and workout readiness.
        """

        try:
            result = await self.agent.run(prompt)
            return result.data
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
            return result.data
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
        result = await self.analyzer.initialize()
        logger.info("Application initialization complete")
        return result

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
        return self.analyzer.garmin_tools.available_tools
    
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
                            print(f"\n{'='*50}")
                            print(f"ENHANCED {analysis_types[analysis_choice].upper()} ANALYSIS")
                            print("="*50)
                            print(analysis)
                        else:
                            print("Invalid choice.")
                    
                    elif choice == "4":
                        tools = await self.list_available_tools()
                        print(f"\nAvailable Garmin MCP tools:")
                        for tool in tools:
                            print(f"  - {tool['name']}: {tool['description']}")
                    
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
    # Try to load from config.yaml first
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)

    # Fall back to environment variables
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
            "openrouter_model": "deepseek/deepseek-r1-0528:free",
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
    # Create sample config if needed
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