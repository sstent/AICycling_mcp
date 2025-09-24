#!/usr/bin/env python3
"""
CLI Interface - Simple command line interface for the cycling analyzer
"""

import asyncio
import logging
from pathlib import Path

from config import Config, load_config, create_sample_config
from core_app import CyclingAnalyzerApp
from template_engine import create_default_templates

class CLI:
    """Command line interface"""
    
    def __init__(self):
        self.app = None
    
    async def run(self):
        """Main CLI loop"""
        print("Cycling Workout Analyzer")
        print("=" * 40)
        
        # Setup configuration
        try:
            config = self._setup_config()
            self.app = CyclingAnalyzerApp(config)
            
            # Setup logging
            logging.basicConfig(level=getattr(logging, config.log_level.upper()))
            
            # Initialize app
            await self.app.initialize()
            
            # Show initial status
            self._show_status()
            
            # Main loop
            await self._main_loop()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"CLI error: {e}", exc_info=True)
        finally:
            if self.app:
                await self.app.cleanup()
    
    def _setup_config(self) -> Config:
        """Setup configuration and default files"""
        # Create sample config if needed
        create_sample_config()
        
        # Load config
        config = load_config()
        
        # Validate required settings
        if not config.openrouter_api_key or config.openrouter_api_key == "your_openrouter_api_key_here":
            print("Please edit config.yaml with your OpenRouter API key")
            print("Get your key from: https://openrouter.ai")
            raise ValueError("OpenRouter API key not configured")
        
        # Create default templates
        create_default_templates(config.templates_dir)
        
        return config
    
    def _show_status(self):
        """Show application status"""
        print(f"\nStatus:")
        print(f"- Available tools: {len(self.app.list_available_tools())}")
        print(f"- Available templates: {len(self.app.list_templates())}")
        print(f"- Cached data keys: {list(self.app.get_cached_data().keys())}")
    
    async def _main_loop(self):
        """Main interaction loop"""
        while True:
            print(f"\n{'='*60}")
            print("CYCLING WORKOUT ANALYZER")
            print(f"{'='*60}")
            print("1. Analyze last cycling workout")
            print("2. Get next workout suggestion")
            print("3. Enhanced analysis")
            print("4. List available MCP tools")
            print("5. List available templates")
            print("6. Show cached data")
            print("7. Clear cache")
            print("8. Exit")
            print("-" * 60)
            
            choice = input("Enter your choice (1-8): ").strip()
            
            try:
                if choice == "1":
                    await self._analyze_last_workout()
                elif choice == "2":
                    await self._suggest_next_workout()
                elif choice == "3":
                    await self._enhanced_analysis()
                elif choice == "4":
                    self._list_tools()
                elif choice == "5":
                    self._list_templates()
                elif choice == "6":
                    self._show_cached_data()
                elif choice == "7":
                    self._clear_cache()
                elif choice == "8":
                    break
                else:
                    print("Invalid choice. Please try again.")
            
            except Exception as e:
                print(f"Error: {e}")
                logging.error(f"Menu action error: {e}")
            
            input("\nPress Enter to continue...")
    
    async def _analyze_last_workout(self):
        """Analyze last workout"""
        print("\nAnalyzing your last workout...")
        
        # Load training rules
        rules = self._load_training_rules()
        
        result = await self.app.analyze_workout(
            analysis_type="analyze_last_workout",
            training_rules=rules
        )
        
        print("\n" + "="*50)
        print("WORKOUT ANALYSIS")
        print("="*50)
        print(result)
    
    async def _suggest_next_workout(self):
        """Suggest next workout"""
        print("\nGenerating workout suggestion...")
        
        # Load training rules
        rules = self._load_training_rules()
        
        result = await self.app.suggest_next_workout(training_rules=rules)
        
        print("\n" + "="*50)
        print("NEXT WORKOUT SUGGESTION")
        print("="*50)
        print(result)
    
    async def _enhanced_analysis(self):
        """Enhanced analysis menu"""
        print("\nSelect analysis type:")
        print("a) Performance trends")
        print("b) Training load analysis")
        print("c) Recovery assessment")
        
        choice = input("Enter choice (a-c): ").strip().lower()
        
        analysis_types = {
            'a': 'performance trends',
            'b': 'training load',
            'c': 'recovery assessment'
        }
        
        if choice not in analysis_types:
            print("Invalid choice.")
            return
        
        analysis_type = analysis_types[choice]
        print(f"\nPerforming {analysis_type} analysis...")
        
        # Load training rules
        rules = self._load_training_rules()
        
        result = await self.app.enhanced_analysis(
            analysis_type,
            training_rules=rules
        )
        
        print(f"\n{'='*50}")
        print(f"ENHANCED {analysis_type.upper()} ANALYSIS")
        print("="*50)
        print(result)
    
    def _list_tools(self):
        """List available tools"""
        tools = self.app.list_available_tools()
        if tools:
            self.app.mcp_client.print_tools()
        else:
            print("No MCP tools available")
    
    def _list_templates(self):
        """List available templates"""
        templates = self.app.list_templates()
        print(f"\nAvailable templates ({len(templates)}):")
        for template in templates:
            print(f"  - {template}")
    
    def _show_cached_data(self):
        """Show cached data"""
        cached_data = self.app.get_cached_data()
        print(f"\nCached data ({len(cached_data)} items):")
        for key, value in cached_data.items():
            data_type = type(value).__name__
            if isinstance(value, (dict, list)):
                size = len(value)
                print(f"  - {key}: {data_type} (size: {size})")
            else:
                print(f"  - {key}: {data_type}")
    
    def _clear_cache(self):
        """Clear cache"""
        self.app.cache_manager.clear()
        print("Cache cleared")
    
    def _load_training_rules(self) -> str:
        """Load training rules from file"""
        rules_file = Path(self.app.config.rules_file)
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                return f.read()
        else:
            # Create default rules
            default_rules = """
Training Goals:
- Improve FTP (Functional Threshold Power)  
- Build endurance for long rides
- Maintain consistent training

Power Zones (adjust based on your FTP):
- Zone 1 (Active Recovery): < 55% FTP
- Zone 2 (Endurance): 56-75% FTP  
- Zone 3 (Tempo): 76-90% FTP
- Zone 4 (Lactate Threshold): 91-105% FTP
- Zone 5 (VO2 Max): 106-120% FTP

Weekly Structure:
- 70-80% easy/moderate intensity
- 20-30% high intensity
- At least 1 rest day per week
"""
            rules_file.parent.mkdir(exist_ok=True)
            with open(rules_file, 'w') as f:
                f.write(default_rules)
            return default_rules

async def main():
    """CLI entry point"""
    cli = CLI()
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main())