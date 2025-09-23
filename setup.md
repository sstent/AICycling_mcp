# Cycling Workout Analyzer Setup Guide

## Prerequisites

1. **Python 3.8+** installed on your system
2. **OpenRouter API account** - Get your API key from [OpenRouter.ai](https://openrouter.ai)
3. **Garmin Connect account** with workout data

## Installation Steps

### 1. Install the Garth MCP Server

First, install the Garth MCP server that will connect to your Garmin data:

```bash
# Install the Garth MCP server
npm install -g garth-mcp-server

# Or if using pip/uv (check the repo for latest instructions)
# pip install garth-mcp-server
```

### 2. Set Up the Python Application

```bash
# Clone or download the cycling analyzer files
# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configure the Application

Run the application once to generate the configuration file:

```bash
python main.py
```

This will create a `config.yaml` file. Edit it with your credentials:

```yaml
openrouter_api_key: "your_openrouter_api_key_here"
openrouter_model: "deepseek/deepseek-r1-0528:free"
garmin_email: "your_garmin_email@example.com"
garmin_password: "your_garmin_password"
garth_mcp_server_path: "garth-mcp-server"  # or full path if needed
rules_file: "rules.yaml"
templates_dir: "templates"
```

### 4. Set Up Environment Variables (Alternative)

Instead of using the config file, you can set environment variables:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
export GARMIN_EMAIL="your_email@example.com"
export GARMIN_PASSWORD="your_password"
export GARTH_MCP_SERVER_PATH="garth-mcp-server"
```

### 5. Customize Your Training Rules

Edit the generated `rules.yaml` file with your specific:
- Training goals
- Power zones (based on your FTP)
- Heart rate zones
- Weekly training structure preferences
- Recovery rules

### 6. Customize Prompt Templates

Edit the template files in the `templates/` directory:
- `single_workout_analysis.txt` - For analyzing individual workouts
- `workout_recommendation.txt` - For getting next workout suggestions
- `mcp_enhanced_analysis.txt` - For enhanced analysis using MCP tools

## Running the Application

```bash
python main.py
```

## Features

### 1. Basic Analysis
- Analyze your last cycling workout against your rules
- Get suggestions for your next workout based on recent training

### 2. MCP-Enhanced Analysis
- Uses the Garth MCP server to access comprehensive Garmin data
- Provides detailed performance trends, training load analysis, and recovery assessment
- The LLM has direct access to your Garmin tools and can fetch additional data as needed

### 3. Customizable
- Edit your training rules and goals
- Modify prompt templates to get the analysis style you want
- Configure different AI models through OpenRouter

## Troubleshooting

### MCP Connection Issues
- Ensure `garth-mcp-server` is properly installed and accessible
- Check that your Garmin credentials are correct
- Verify the server path in your configuration

### API Issues
- Confirm your OpenRouter API key is valid and has credits
- Check your internet connection
- Try a different model if the default one is unavailable

### No Workout Data
- Ensure you have recent cycling activities in Garmin Connect
- Check that the MCP server can authenticate with Garmin
- Verify your Garmin credentials

## File Structure

```
cycling-analyzer/
├── main.py                           # Main application
├── config.yaml                       # Configuration file
├── rules.yaml                        # Your training rules and zones
├── requirements.txt                  # Python dependencies
└── templates/                        # Prompt templates
    ├── single_workout_analysis.txt
    ├── workout_recommendation.txt
    └── mcp_enhanced_analysis.txt
```

## Advanced Usage

### Custom Templates
You can create additional templates for specific analysis types. The application will automatically detect `.txt` files in the templates directory. Template variables available:
- `{workout_data}` - Individual workout data
- `{workouts_data}` - Multiple workouts data
- `{rules}` - Your training rules
- `{available_tools}` - MCP tools information

### Custom Analysis Types
Add new analysis options by:
1. Creating a new template file
2. Adding the analysis logic to the `CyclingAnalyzer` class
3. Adding menu options in the main loop

### Multiple AI Models
You can experiment with different AI models through OpenRouter:
- `deepseek/deepseek-r1-0528:free` (default, free)
- `anthropic/claude-3-sonnet`
- `openai/gpt-4-turbo`
- `google/gemini-pro`

### Integration with Other Tools
The MCP architecture allows easy integration with other fitness tools and data sources. You can extend the application to work with:
- Training Peaks
- Strava (via MCP server)
- Wahoo, Polar, or other device manufacturers
- Custom training databases

### Automated Analysis
You can run the analyzer in automated mode by modifying the `run()` method to:
- Analyze workouts automatically after each session
- Generate weekly training reports
- Send recommendations via email or notifications

## Example Workflow

1. **After a workout**: Run option 1 to get immediate feedback on your session
2. **Planning next session**: Use option 2 to get AI-powered recommendations
3. **Weekly review**: Use option 3 for enhanced analysis of trends and patterns
4. **Adjust training**: Modify your `rules.yaml` based on insights and goals changes

## Security Notes

- Store your credentials securely
- Consider using environment variables instead of config files for sensitive data
- The MCP server runs locally and connects directly to Garmin - no data is sent to third parties except the AI provider (OpenRouter)

## Support and Contributions

- Check the Garth MCP server repository for Garmin-specific issues
- Refer to OpenRouter documentation for API-related questions
- Customize templates and rules to match your specific training methodology

## What Makes This Unique

This application bridges three powerful technologies:
1. **Garth MCP Server** - Direct access to comprehensive Garmin data
2. **Model Context Protocol (MCP)** - Standardized way for AI to access tools and data
3. **OpenRouter** - Access to multiple state-of-the-art AI models

The AI doesn't just analyze static workout data - it can actively query your Garmin account for additional context, trends, and historical data to provide much more comprehensive and personalized recommendations.