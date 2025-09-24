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

from mcp_manager import Config, print_tools, PydanticAIAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




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
        
        # Pre-call user_profile tool
        logger.info("Pre-caching user profile...")
        user_profile = await self.analyzer.get_user_profile()
        print("\n" + "="*60)
        print("RAW USER PROFILE (Pre-cached)")
        print("="*60)
        print(json.dumps(user_profile, indent=2, default=str))
        print("="*60)
        logger.info("User profile pre-cached")
        
        # Pre-call get_recent_cycling_activity_details
        logger.info("Pre-caching recent cycling activity details...")
        activity_data = await self.analyzer.get_recent_cycling_activity_details()
        
        print("\n" + "="*60)
        print("RAW RECENT ACTIVITIES (Pre-cached)")
        print("="*60)
        print(json.dumps(activity_data.get("activities", []), indent=2, default=str))
        print("="*60)
        
        if activity_data.get("last_cycling"):
            print("\n" + "="*60)
            print("LAST CYCLING ACTIVITY SUMMARY (Pre-cached)")
            print("="*60)
            print(json.dumps(activity_data["last_cycling"], indent=2, default=str))
            print("="*60)
            
            print("\n" + "="*60)
            print("ACTIVITY DETAILS (Pre-cached)")
            print("="*60)
            print(json.dumps(activity_data["details"], indent=2, default=str))
            print("="*60)
            logger.info("Recent cycling activity details pre-cached")
        else:
            logger.warning("No cycling activity found in recent activities")
            print("\nWarning: No cycling activity found in recent activities.")
        
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