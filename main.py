#!/usr/bin/env python3
"""
Cycling Workout Analyzer with Garth MCP Server Integration
A Python app that uses OpenRouter AI and Garmin data via MCP to analyze cycling workouts
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

# MCP Protocol imports
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
    
class OpenRouterClient:
    """Client for OpenRouter AI API"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
    
    async def generate_response(self, prompt: str, available_tools: List[Dict] = None) -> str:
        """Generate AI response from prompt, optionally with MCP tools available"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-username/cycling-analyzer",
            "X-Title": "Cycling Workout Analyzer"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        # Add tool information if available
        if available_tools:
            tool_info = "\n\nAvailable Garmin data tools:\n"
            for tool in available_tools:
                tool_info += f"- {tool['name']}: {tool.get('description', 'No description')}\n"
            messages[0]["content"] += tool_info
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API error: {response.status} - {error_text}")

class GarthMCPConnector:
    """Connector for Garmin data via Garth MCP server"""

    def __init__(self, garth_token: str, server_path: str):
        self.garth_token = garth_token
        self.server_path = server_path
        self.server_available = False
        self.cached_tools = []  # Cache tools to avoid repeated fetches
        self._session: Optional[ClientSession] = None
        self._client_context = None  # To hold the stdio_client context
        self._read_stream = None
        self._write_stream = None

    async def _get_server_params(self):
        """Get server parameters for MCP connection"""
        env = os.environ.copy()
        env['GARTH_TOKEN'] = self.garth_token

        # Find the full path to the server executable to avoid issues with intermediate tools like uvx
        server_command = shutil.which("garth-mcp-server")
        if not server_command:
            logger.error("Could not find 'garth-mcp-server' in your PATH.")
            logger.error("Please ensure it is installed and accessible, e.g., via 'npm install -g garth-mcp-server'.")
            raise FileNotFoundError("garth-mcp-server not found")

        # The garth-mcp-server logs to stdout during startup, which interferes
        # with the MCP JSON-RPC communication. To redirect its stdout to stderr,
        # we must run it via a shell command that performs the redirection.
        # StdioServerParameters does not have a 'shell' argument, so we make
        # the 'command' itself a shell interpreter, and pass the actual command
        # with redirection as an argument to the shell.
        return StdioServerParameters(
            command="/bin/bash", # Use bash to execute the command with redirection
            # The -c flag tells bash to read commands from the string.
            # "exec ..." replaces the bash process with garth-mcp-server.
            # "1>&2" redirects stdout (file descriptor 1) to stderr (file descriptor 2).
            # "$@" passes any additional arguments from StdioServerParameters.args (which is currently empty).
            args=["-c", f"exec {server_command} \"$@\" 1>&2"],
            capture_stderr=True,  # Capture the stderr stream for debugging
            env=env,
        )

    async def connect(self):
        """Start the MCP server and establish a persistent session."""
        if self._session and self.server_available:
            return True

        if not MCP_AVAILABLE:
            logger.error("MCP library not available. Install with: pip install mcp")
            return False

        try:
            # Create a background task to log stderr from the server process
            async def log_stderr(stderr_stream):
                async for line in stderr_stream:
                    logger.error(f"[garth-mcp-server-stderr] {line.decode().strip()}")

            logger.info("Connecting to Garth MCP server...")
            server_params = await self._get_server_params()
            
            # The stdio_client is an async context manager, we need to enter it.
            # We'll store the process and streams to manage them manually.
            logger.info("Starting MCP server process...")
            self._client_context = stdio_client(server_params)  # type: ignore
            streams = await self._client_context.__aenter__()

            # Handle both cases: with and without stderr capture
            if len(streams) == 3:
                self._read_stream, self._write_stream, stderr_stream = streams
                # Start the stderr logging task
                stderr_task = asyncio.create_task(log_stderr(stderr_stream))
            else:
                self._read_stream, self._write_stream = streams
                stderr_task = None

            logger.info("Server process started. Waiting for it to initialize...")
            
            # A short wait for the shell and server process to start.
            await asyncio.sleep(0.5)

            logger.info("Initializing MCP session...")
            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.initialize()
            
            logger.info("Testing connection by listing tools...")
            await self._session.list_tools()

            self.server_available = True
            logger.info("✓ Successfully connected to MCP server.")
            if stderr_task:
                stderr_task.cancel()  # Stop logging stderr once connected
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.disconnect()  # Clean up on failure
            self.server_available = False
            # Use the variable from the outer scope
            if 'stderr_task' in locals() and stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return False

    async def disconnect(self):
        """Disconnect from the MCP server and clean up resources."""
        logger.info("Disconnecting from MCP server...")
        if self._client_context:
            try:
                # Properly exit the context manager to clean up the subprocess
                await self._client_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error during MCP client disconnection: {e}")
        
        self._session = None
        self.server_available = False
        self.cached_tools = []  # Clear cache
        self._client_context = None
        self._read_stream = None
        self._write_stream = None
        logger.info("Disconnected.")

    async def _ensure_connected(self):
        """Ensure server is available"""
        if not self.server_available or not self._session:
            return await self.connect()
        return True

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool on the MCP server"""
        if not await self._ensure_connected():
            raise Exception("MCP server not available")

        try:
            return await self._session.call_tool(tool_name, arguments or {})
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            raise
    
    async def get_activities_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get activities data via MCP or fallback to mock data"""
        if not await self._ensure_connected() or not self._session:
            logger.warning("No MCP session available, using mock data")
            return self._get_mock_activities_data(limit)
        
        try:
            # Try different possible tool names for getting activities
            possible_tools = ['get_activities', 'list_activities', 'activities', 'garmin_activities']
            available_tools = await self.get_available_tools_info()
            for tool_name in possible_tools:
                if any(tool['name'] == tool_name for tool in available_tools):
                    result = await self.call_tool(tool_name, {"limit": limit})
                    if result and hasattr(result, 'content'):
                        # Parse the result based on MCP response format
                        activities = []
                        for content in result.content:
                            if hasattr(content, 'text'):
                                # Try to parse as JSON
                                try:
                                    data = json.loads(content.text)
                                    if isinstance(data, list):
                                        activities.extend(data)
                                    else:
                                        activities.append(data)
                                except json.JSONDecodeError:
                                    # If not JSON, treat as text description
                                    activities.append({"description": content.text})
                        return activities
            
            logger.warning("No suitable activity tool found, falling back to mock data")
            return self._get_mock_activities_data(limit)
            
        except Exception as e:
            logger.error(f"Failed to get activities via MCP: {e}")
            logger.warning("Falling back to mock data")
            return self._get_mock_activities_data(limit)
    
    def _get_mock_activities_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get mock activities data for testing"""
        base_activity = {
            "activityId": "12345678901",
            "activityName": "Morning Ride",
            "startTimeLocal": "2024-01-15T08:00:00",
            "activityType": {"typeKey": "cycling"},
            "distance": 25000,  # meters
            "duration": 3600,   # seconds
            "averageSpeed": 6.94,  # m/s
            "maxSpeed": 12.5,      # m/s
            "elevationGain": 350,  # meters
            "averageHR": 145,
            "maxHR": 172,
            "averagePower": 180,
            "maxPower": 420,
            "normalizedPower": 185,
            "calories": 890,
            "averageCadence": 85,
            "maxCadence": 110
        }
        
        activities = []
        for i in range(min(limit, 10)):
            activity = base_activity.copy()
            activity["activityId"] = str(int(base_activity["activityId"]) + i)
            activity["activityName"] = f"Cycling Workout {i+1}"
            # Vary the data slightly
            activity["distance"] = base_activity["distance"] + (i * 2000)
            activity["averagePower"] = base_activity["averagePower"] + (i * 10)
            activity["duration"] = base_activity["duration"] + (i * 300)
            activities.append(activity)
        
        return activities
    
    async def get_last_cycling_workout(self) -> Optional[Dict[str, Any]]:
        """Get the most recent cycling workout"""
        activities = await self.get_activities_data(limit=50)
        
        # Filter for cycling activities
        cycling_activities = [
            activity for activity in activities 
            if self._is_cycling_activity(activity)
        ]
        
        return cycling_activities[0] if cycling_activities else None
    
    async def get_last_n_cycling_workouts(self, n: int = 4) -> List[Dict[str, Any]]:
        """Get the last N cycling workouts"""
        activities = await self.get_activities_data(limit=50)
        
        # Filter for cycling activities
        cycling_activities = [
            activity for activity in activities 
            if self._is_cycling_activity(activity)
        ]
        
        return cycling_activities[:n]
    
    def _is_cycling_activity(self, activity: Dict[str, Any]) -> bool:
        """Check if an activity is a cycling workout"""
        activity_type = activity.get('activityType', {}).get('typeKey', '').lower()
        activity_name = activity.get('activityName', '').lower()
        
        cycling_keywords = ['cycling', 'bike', 'ride', 'bicycle']
        
        return (
            'cycling' in activity_type or 
            'bike' in activity_type or
            any(keyword in activity_name for keyword in cycling_keywords)
        )
    
    async def get_available_tools_info(self) -> List[Dict[str, str]]:
        """Get information about available MCP tools"""
        # Return cached tools if available
        if self.cached_tools:
            return self.cached_tools

        if not await self._ensure_connected():
            return []

        try:
            tools_result = await self._session.list_tools()
            tools = tools_result.tools if tools_result else []

            # Cache the tools for future use
            self.cached_tools = [
                {
                    "name": tool.name,
                    "description": getattr(tool, 'description', 'No description available'),
                }
                for tool in tools
            ]

            return self.cached_tools
        except Exception as e:
            logger.warning(f"Could not get tools info: {e}")
            return []

class TemplateManager:
    """Manages prompt templates"""
    
    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default template files if they don't exist"""
        templates = {
            "single_workout_analysis.txt": """
Analyze my cycling workout against my training rules and goals.

WORKOUT DATA:
{workout_data}

MY TRAINING RULES:
{rules}

You have access to additional Garmin data through MCP tools if needed.

Please provide:
1. Overall assessment of the workout
2. How well it aligns with my rules and goals
3. Areas for improvement
4. Specific feedback on power, heart rate, duration, and intensity
5. Recovery recommendations
6. Comparison with my typical performance metrics
            """.strip(),
            
            "workout_recommendation.txt": """
Based on my recent cycling workouts, suggest what workout I should do next.

RECENT WORKOUTS:
{workouts_data}

MY TRAINING RULES:
{rules}

You have access to additional Garmin data and tools to analyze my fitness trends.

Please provide:
1. Analysis of my recent training pattern
2. Identified gaps or imbalances in my training
3. Specific workout recommendation for my next session
4. Target zones (power, heart rate, duration)
5. Rationale for the recommendation based on my recent performance
6. Alternative options if weather/time constraints exist
7. How this fits into my overall training progression
            """.strip(),
            
            "mcp_enhanced_analysis.txt": """
You are an expert cycling coach with access to comprehensive Garmin Connect data through MCP tools.

CONTEXT:
- User's Training Rules: {rules}
- Analysis Type: {analysis_type}
- Recent Data: {recent_data}

AVAILABLE MCP TOOLS:
{available_tools}

Please use the available MCP tools to gather additional relevant data and provide a comprehensive analysis. Focus on:

1. **Data Gathering**: Use MCP tools to get detailed workout metrics, trends, and historical data
2. **Performance Analysis**: Analyze power, heart rate, training load, and recovery metrics  
3. **Training Periodization**: Consider the user's training phase and progression
4. **Actionable Recommendations**: Provide specific, measurable guidance for future workouts
5. **Risk Assessment**: Identify any signs of overtraining or injury risk

Be thorough in your analysis and use multiple data points to support your recommendations.
            """.strip()
        }
        
        for filename, content in templates.items():
            template_path = self.templates_dir / filename
            if not template_path.exists():
                template_path.write_text(content)
                logger.info(f"Created template: {template_path}")
    
    def get_template(self, template_name: str) -> str:
        """Get template content"""
        template_path = self.templates_dir / template_name
        if template_path.exists():
            return template_path.read_text()
        else:
            raise FileNotFoundError(f"Template not found: {template_path}")
    
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
    """Main application class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.openrouter = OpenRouterClient(config.openrouter_api_key, config.openrouter_model)
        self.garmin = GarthMCPConnector(
            config.garth_token,
            config.garth_mcp_server_path
        )
        self.templates = TemplateManager(config.templates_dir)
        self.rules = RulesManager(config.rules_file)
    
    async def initialize(self):
        """Initialize the application and connect to MCP server"""
        logger.info("Initializing application and connecting to MCP server...")
        success = await self.garmin.connect()
        if success:
            logger.info("Application initialized successfully")
        else:
            logger.warning("Application initialized but MCP server connection failed - will retry on demand")
        return True  # Always return True to allow the app to start

    async def cleanup(self):
        """Cleanup resources"""
        await self.garmin.disconnect()
        logger.info("Application cleanup completed")
    
    async def analyze_last_workout(self):
        """Analyze the last cycling workout"""
        logger.info("Analyzing last cycling workout...")
        
        try:
            # Get workout data via MCP
            workout = await self.garmin.get_last_cycling_workout()
            
            if not workout:
                return "No recent cycling workouts found in your Garmin data."
            
            # Get rules
            rules_text = self.rules.get_rules()
            
            # Format workout data
            workout_text = json.dumps(workout, indent=2)
            
            # Get available tools info
            available_tools = await self.garmin.get_available_tools_info()
            
            # Get template and format prompt
            template = self.templates.get_template("single_workout_analysis.txt")
            prompt = template.format(workout_data=workout_text, rules=rules_text)
            
            # Get AI analysis with tool information
            analysis = await self.openrouter.generate_response(prompt, available_tools)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing workout: {e}")
            return f"Error analyzing workout: {e}"
    
    async def suggest_next_workout(self):
        """Suggest next workout based on recent activities"""
        logger.info("Analyzing recent workouts and suggesting next workout...")
        
        try:
            # Get last 4 workouts via MCP
            workouts = await self.garmin.get_last_n_cycling_workouts(4)
            
            if not workouts:
                return "No recent cycling workouts found in your Garmin data."
            
            # Get rules
            rules_text = self.rules.get_rules()
            
            # Format workouts data
            workouts_text = json.dumps(workouts, indent=2)
            
            # Get available tools info
            available_tools = await self.garmin.get_available_tools_info()
            
            # Get template and format prompt
            template = self.templates.get_template("workout_recommendation.txt")
            prompt = template.format(workouts_data=workouts_text, rules=rules_text)
            
            # Get AI suggestion with tool information
            suggestion = await self.openrouter.generate_response(prompt, available_tools)
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error suggesting workout: {e}")
            return f"Error suggesting next workout: {e}"
    
    async def mcp_enhanced_analysis(self, analysis_type: str):
        """Perform enhanced analysis using MCP tools directly"""
        logger.info(f"Performing MCP-enhanced {analysis_type} analysis...")
        
        try:
            # Get rules
            rules_text = self.rules.get_rules()
            
            # Get recent data
            recent_workouts = await self.garmin.get_last_n_cycling_workouts(7)
            recent_data = json.dumps(recent_workouts[:3], indent=2) if recent_workouts else "No recent data"
            
            # Get available tools info
            available_tools_info = "\n".join([
                f"- {tool['name']}: {tool['description']}"
                for tool in await self.garmin.get_available_tools_info()
            ])
            
            # Get enhanced template
            template = self.templates.get_template("mcp_enhanced_analysis.txt")
            prompt = template.format(
                rules=rules_text,
                analysis_type=analysis_type,
                recent_data=recent_data,
                available_tools=available_tools_info
            )
            
            # Get AI analysis with full tool context
            analysis = await self.openrouter.generate_response(
                prompt,
                await self.garmin.get_available_tools_info()
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in MCP enhanced analysis: {e}")
            return f"Error in enhanced analysis: {e}"
    
    async def run(self):
        """Main application loop"""
        logger.info("Starting Cycling Workout Analyzer with Garth MCP Server...")
        
        # Initialize MCP connection (with fallback mode)
        await self.initialize()
        
        try:
            while True:
                print("\n" + "="*60)
                print("CYCLING WORKOUT ANALYZER (with Garth MCP Integration)")
                print("="*60)
                print("1. Analyze last cycling workout")
                print("2. Get next workout suggestion")
                print("3. Enhanced analysis using MCP tools")
                print("4. List available MCP tools")
                print("5. List available templates")
                print("6. View current rules")
                print("7. Exit")
                print("-"*60)
                
                choice = input("Enter your choice (1-7): ").strip()
                
                try:
                    if choice == "1":
                        print("\nAnalyzing your last workout...")
                        analysis = await self.analyze_last_workout()
                        print("\n" + "="*50)
                        print("WORKOUT ANALYSIS")
                        print("="*50)
                        print(analysis)
                    
                    elif choice == "2":
                        print("\nAnalyzing recent workouts and generating suggestion...")
                        suggestion = await self.suggest_next_workout()
                        print("\n" + "="*50)
                        print("NEXT WORKOUT SUGGESTION")
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
                            analysis = await self.mcp_enhanced_analysis(
                                analysis_types[analysis_choice]
                            )
                            print(f"\n{'='*50}")
                            print(f"ENHANCED {analysis_types[analysis_choice].upper()} ANALYSIS")
                            print("="*50)
                            print(analysis)
                        else:
                            print("Invalid choice.")
                    
                    elif choice == "4":
                        try:
                            tools = await self.garmin.get_available_tools_info()
                            print(f"\nAvailable MCP tools from Garth server:")
                            if tools:
                                for tool in tools:
                                    print(f"  - {tool['name']}: {tool['description']}")
                            else:
                                print("  No tools available or server not connected")
                                print("  Note: MCP server may be having startup issues.")
                                print("  Available Garmin Connect tools (when working):")
                                mock_tools = [
                                    "user_profile - Get user profile information",
                                    "user_settings - Get user settings and preferences",
                                    "daily_sleep - Get daily sleep summary data",
                                    "daily_steps - Get daily steps data",
                                    "daily_hrv - Get heart rate variability data",
                                    "get_activities - Get list of activities",
                                    "get_activity_details - Get detailed activity information",
                                    "get_body_composition - Get body composition data",
                                    "get_respiration_data - Get respiration data",
                                    "get_blood_pressure - Get blood pressure readings"
                                ]
                                for tool in mock_tools:
                                    print(f"  - {tool}")
                        except Exception as e:
                            logger.error(f"Error listing tools: {e}")
                            print(f"Error: {e}")
                            print("  Showing available Garmin Connect tools:")
                            mock_tools = [
                                "user_profile - Get user profile information",
                                "user_settings - Get user settings and preferences",
                                "daily_sleep - Get daily sleep summary data",
                                "daily_steps - Get daily steps data",
                                "daily_hrv - Get heart rate variability data",
                                "get_activities - Get list of activities",
                                "get_activity_details - Get detailed activity information",
                                "get_body_composition - Get body composition data",
                                "get_respiration_data - Get respiration data",
                                "get_blood_pressure - Get blood pressure readings"
                            ]
                            for tool in mock_tools:
                                print(f"  - {tool}")
                        # Add small delay to keep output visible
                        time.sleep(3)
                    
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
            "garth_token": "your_garth_token_here",  # Get this with: uvx garth login
            "garth_mcp_server_path": "uvx",  # Use uvx to run garth-mcp-server
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