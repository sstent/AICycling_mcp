#!/usr/bin/env python3
"""
Template Engine - Simplified template loading and rendering
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class TemplateEngine:
    """Simple template engine for prompt management"""
    
    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Create basic directory structure
        self._ensure_structure()
    
    def _ensure_structure(self):
        """Ensure basic template directory structure exists"""
        dirs = [
            "workflows",
            "base/system_prompts", 
            "base/data_sections",
            "base/analysis_frameworks"
        ]
        
        for dir_path in dirs:
            (self.templates_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def list_templates(self) -> List[str]:
        """List all available templates"""
        templates = []
        
        # Get all .txt files in templates directory and subdirectories
        for template_file in self.templates_dir.rglob("*.txt"):
            rel_path = template_file.relative_to(self.templates_dir)
            templates.append(str(rel_path))
        
        return sorted(templates)
    
    def template_exists(self, template_name: str) -> bool:
        """Check if template exists"""
        template_path = self._resolve_template_path(template_name)
        return template_path.exists() if template_path else False
    
    def _resolve_template_path(self, template_name: str) -> Path:
        """Resolve template name to full path"""
        # Handle different template name formats
        if template_name.endswith('.txt'):
            template_path = self.templates_dir / template_name
        else:
            template_path = self.templates_dir / f"{template_name}.txt"
        
        return template_path
    
    def load_template(self, template_name: str) -> str:
        """Load raw template content"""
        template_path = self._resolve_template_path(template_name)
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_name}")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.debug(f"Loaded template: {template_name}")
            return content
        
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            raise
    
    def render(self, template_name: str, **kwargs) -> str:
        """Load and render template with variables"""
        content = self.load_template(template_name)
        
        # Handle section includes (simple replacement)
        content = self._process_includes(content, **kwargs)
        
        try:
            rendered = content.format(**kwargs)
            logger.debug(f"Rendered template: {template_name}")
            return rendered
        
        except KeyError as e:
            logger.error(f"Missing variable in template {template_name}: {e}")
            logger.debug(f"Available variables: {list(kwargs.keys())}")
            raise ValueError(f"Missing variable in template {template_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise
    
    def _process_includes(self, content: str, **kwargs) -> str:
        """Process section includes like {activity_summary_section}"""
        import re
        
        # Define section mappings
        section_mappings = {
            'activity_summary_section': 'base/data_sections/activity_summary.txt',
            'user_info_section': 'base/data_sections/user_info.txt',
            'training_rules_section': 'base/data_sections/training_rules.txt',
            'workout_data_section': 'base/data_sections/workout_data.txt',
            'assessment_points': 'base/analysis_frameworks/assessment_points.txt',
            'performance_analysis': 'base/analysis_frameworks/performance_analysis.txt',
        }
        
        # Find and replace section placeholders
        section_pattern = re.compile(r'\{(\w+_section|\w+_points|\w+_analysis)\}')
        
        for match in section_pattern.finditer(content):
            placeholder = match.group(0)
            section_name = match.group(1)
            
            if section_name in section_mappings:
                section_file = section_mappings[section_name]
                try:
                    section_content = self.load_template(section_file)
                    # Render section with same kwargs
                    # Recursively render the section content
                    section_rendered = self.render(section_file, **kwargs)
                    content = content.replace(placeholder, section_rendered)
                except (FileNotFoundError, KeyError, ValueError) as e:
                    logger.warning(f"Could not process section {section_name}: {e}")
                    # Replace with empty string if section fails
                    content = content.replace(placeholder, "")
        
        return content
    
    def create_template(self, template_name: str, content: str) -> None:
        """Create a new template file"""
        template_path = self._resolve_template_path(template_name)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created template: {template_name}")
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template"""
        if not self.template_exists(template_name):
            return {"exists": False}
        
        template_path = self._resolve_template_path(template_name)
        content = self.load_template(template_name)
        
        # Extract variables used in template
        import re
        variables = set(re.findall(r'\{(\w+)\}', content))
        
        return {
            "exists": True,
            "path": str(template_path),
            "size": len(content),
            "variables": sorted(list(variables)),
            "line_count": len(content.splitlines())
        }

# Utility functions for template management
def create_default_templates(templates_dir: str) -> None:
    """Create default template files if they don't exist"""
    engine = TemplateEngine(templates_dir)
    
    # Default system prompts
    default_templates = {
        "base/system_prompts/main_agent.txt": 
            "You are an expert cycling coach with access to comprehensive Garmin Connect data.\n"
            "You analyze cycling workouts, provide performance insights, and give actionable training recommendations.\n"
            "Use the available tools to gather detailed workout data and provide comprehensive analysis.",
        
        "base/system_prompts/no_tools_analysis.txt":
            "You are an expert cycling coach. Perform comprehensive analysis using the provided data.\n"
            "Do not use any tools - all relevant data is included in the prompt.",
        
        "base/data_sections/activity_summary.txt":
            "ACTIVITY SUMMARY:\n{activity_summary}",
        
        "base/data_sections/user_info.txt":
            "USER INFO:\n{user_info}",
        
        "base/data_sections/training_rules.txt":
            "My training rules and goals:\n{training_rules}",
        
        "base/analysis_frameworks/assessment_points.txt":
            "Please provide:\n"
            "1. Overall assessment of the workout\n"
            "2. How well it aligns with my rules and goals\n"
            "3. Areas for improvement\n"
            "4. Specific feedback on power, heart rate, duration, and intensity\n"
            "5. Recovery recommendations\n"
            "6. Comparison with typical performance metrics",
        
        "workflows/analyze_last_workout.txt":
            "Analyze my most recent cycling workout using the provided data.\n\n"
            "{activity_summary_section}\n\n"
            "{user_info_section}\n\n"
            "{training_rules_section}\n\n"
            "{assessment_points}\n\n"
            "Focus on the provided activity details for your analysis.",
        
        "workflows/suggest_next_workout.txt":
            "Please suggest my next cycling workout based on my recent training history.\n\n"
            "{training_rules_section}\n\n"
            "Please provide:\n"
            "1. Analysis of my recent training pattern\n"
            "2. Identified gaps or imbalances in my training\n"
            "3. Specific workout recommendation for my next session\n"
            "4. Target zones (power, heart rate, duration)\n"
            "5. Rationale for the recommendation based on recent performance",
        
        "workflows/enhanced_analysis.txt":
            "Perform enhanced {analysis_type} analysis using all available data and tools.\n\n"
            "Available cached data: {cached_data}\n\n"
            "Use MCP tools as needed to gather additional data for comprehensive analysis."
    }
    
    for template_name, content in default_templates.items():
        if not engine.template_exists(template_name):
            engine.create_template(template_name, content)
            logger.info(f"Created default template: {template_name}")