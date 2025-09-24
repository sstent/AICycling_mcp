import os
import logging
from pathlib import Path

class TemplateManager:
    """Manages prompt templates for the cycling analyzer"""
    
    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
    
    def list_templates(self) -> list[str]:
        """List available template files"""
        return [f.name for f in self.templates_dir.glob("*.txt")]
    
    def get_template(self, template_name: str, **kwargs) -> str:
        """Load and format a template with provided variables"""
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template '{template_name}' not found in {self.templates_dir}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # Debug logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Loading template: {template_name}")
        logger.debug(f"Template content length: {len(template_content)}")
        logger.debug(f"Available kwargs: {list(kwargs.keys())}")

        try:
            formatted_template = template_content.format(**kwargs)
            return formatted_template
        except KeyError as e:
            logger.error(f"Template content preview: {template_content[:200]}...")
            logger.error(f"Missing variable in template '{template_name}': {e}")
            logger.error(f"Available kwargs: {list(kwargs.keys())}")
            raise ValueError(f"Missing variable in template '{template_name}': {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template '{template_name}': {e}")