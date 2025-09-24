# Cycling Workout Analyzer - Clean Architecture

A modular, extensible cycling workout analyzer built with a clean architecture that separates core concerns into focused modules.

## Architecture Overview

The application is structured into distinct, focused modules:

```
├── core_app.py          # Main orchestrator
├── config.py            # Configuration management  
├── llm_client.py        # LLM interactions
├── mcp_client.py        # MCP server management
├── cache_manager.py     # Data caching with TTL
├── template_engine.py   # Template loading/rendering
├── cli_interface.py     # Command line interface
└── requirements.txt     # Dependencies
```

## Core Features

### 🤖 LLM Integration
- OpenRouter API support with multiple models
- Both tool-enabled and tool-free analysis modes
- Async request handling with timeouts

### 🔧 MCP Tool Management  
- Automatic MCP server discovery and connection
- Tool listing and direct tool calling
- Garth MCP server integration for Garmin data

### 💾 Smart Caching
- TTL-based caching system
- Pre-loading of common data (user profile, activities)
- Specialized cycling data cache helpers

### 📝 Template System
- Modular template structure
- Section includes and variable substitution
- Auto-creation of default templates

### ⚙️ Configuration
- YAML config files with environment variable fallback
- Automatic sample config generation
- Extensible configuration structure

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# Install MCP server for Garmin data
npm install -g garth-mcp-server
```

### 2. Configure
```bash
# Run once to create config.yaml
python cli_interface.py

# Edit config.yaml with your API keys
```

### 3. Run
```bash
python cli_interface.py
```

## Usage Examples

### Basic Analysis
```python
from config import load_config
from core_app import CyclingAnalyzerApp

config = load_config()
app = CyclingAnalyzerApp(config)

await app.initialize()

# Analyze last workout
analysis = await app.analyze_workout("analyze_last_workout")
print(analysis)

# Get workout suggestion  
suggestion = await app.suggest_next_workout()
print(suggestion)

await app.cleanup()
```

### Custom Analysis
```python
# Enhanced analysis with tools
analysis = await app.enhanced_analysis(
    "performance_trends", 
    training_rules="Custom rules here"
)

# Check what tools are available
tools = app.list_available_tools()
for tool in tools:
    print(f"- {tool.name}: {tool.description}")
```

### Cache Management
```python
# Check cached data
cached = app.get_cached_data()
print("Cached keys:", list(cached.keys()))

# Cache custom data
app.cache_manager.set("custom_key", {"data": "value"}, ttl=600)
```

## Configuration

### config.yaml
```yaml
# LLM Settings
openrouter_api_key: "your_api_key_here"
openrouter_model: "deepseek/deepseek-r1-0528:free"

# MCP Settings  
garth_token: "your_garth_token_here"
garth_mcp_server_path: "uvx"

# Application Settings
templates_dir: "templates"
rules_file: "rules.yaml"
cache_ttl: 300
log_level: "INFO"
```

### Environment Variables
```bash
export OPENROUTER_API_KEY="your_key"
export GARTH_TOKEN="your_token"
export LOG_LEVEL="DEBUG"
```

## Extension Points

### Custom Analysis Types
```python
# Add new analysis in core_app.py
async def custom_analysis(self, **kwargs) -> str:
    template = "workflows/custom_analysis.txt" 
    context = {"custom_data": kwargs}
    prompt = self.template_engine.render(template, **context)
    return await self.llm_client.generate(prompt)
```

### Custom MCP Tools
```python
# Add new tool support in mcp_client.py  
async def call_custom_tool(self, parameters: dict) -> dict:
    return await self.call_tool("custom_tool", parameters)
```

### Custom Templates
Create new templates in `templates/workflows/`:
```
templates/
├── workflows/
│   ├── my_analysis.txt
│   └── custom_report.txt
├── base/
│   ├── data_sections/
│   └── analysis_frameworks/
```

### Custom Cache Strategies
```python
from cache_manager import CacheManager

class CustomCache(CacheManager):
    def cache_performance_data(self, data, athlete_id):
        self.set(f"performance_{athlete_id}", data, ttl=1800)
```

## Architecture Benefits

### Separation of Concerns
- **Config**: Handles all configuration logic
- **LLM Client**: Pure LLM interactions
- **MCP Client**: Tool management only
- **Cache**: Data persistence with TTL
- **Templates**: Prompt composition
- **CLI**: User interface

### Extensibility
- Easy to add new LLM providers
- Plugin-style MCP tool additions
- Template-based prompt customization
- Configurable caching strategies

### Testability
- Each module has single responsibility
- Clear interfaces between components
- Mock-friendly async design
- Dependency injection ready

### Maintainability
- Small, focused files
- Clear naming conventions
- Comprehensive logging
- Error handling at boundaries

## Advanced Features

### Template Inheritance
Templates can include sections and inherit from base templates:
```
{activity_summary_section}  # Includes base/data_sections/activity_summary.txt
{assessment_points}         # Includes base/analysis_frameworks/assessment_points.txt
```

### Dynamic Tool Selection
The app automatically detects available tools and adjusts functionality:
```python
if await self.mcp_client.has_tool("hrv_data"):
    hrv_data = await self.mcp_client.call_tool("hrv_data", {})
```

### Cache Warming
Common data is pre-loaded during initialization:
- User profile (1 hour TTL)
- Recent activities (15 min TTL)  
- Last cycling activity details (1 hour TTL)

## Troubleshooting

### MCP Connection Issues
```bash
# Check if garth-mcp-server is installed
which garth-mcp-server

# Test Garth token  
uvx garth login
```

### Template Errors
```bash
# List available templates
python -c "from template_engine import TemplateEngine; print(TemplateEngine('templates').list_templates())"

# Check template variables
python -c "from template_engine import TemplateEngine; print(TemplateEngine('templates').get_template_info('workflows/analyze_last_workout.txt'))"
```

### Cache Issues  
```bash
# Clear cache
python -c "from cache_manager import CacheManager; CacheManager().clear()"
```

## Contributing

The modular architecture makes contributions straightforward:

1. **New LLM Provider**: Extend `LLMClient`
2. **New Data Source**: Create new MCP client
3. **New Analysis**: Add templates and methods  
4. **New Interface**: Create alternative to CLI
5. **New Cache Strategy**: Extend `CacheManager`

Each module is independently testable and can be developed in isolation.