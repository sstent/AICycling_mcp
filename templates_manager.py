import os
import logging
import yaml
from pathlib import Path
from template_validator import TemplateValidator

logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages prompt templates for the cycling analyzer with inheritance, versioning, and validation"""
    
    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.validator = TemplateValidator(str(self.templates_dir))
        self.components_dir = self.templates_dir / "base"
        self.versions_dir = self.templates_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
    
    def list_templates(self) -> list[str]:
        """List available template files, including versioned ones"""
        templates = []
        # Base templates
        for f in self.templates_dir.glob("*.txt"):
            templates.append(f.name)
        # Workflows and subdirs
        for f in self.templates_dir.rglob("*.txt"):
            if f.parent.name in ["workflows", "base"]:
                rel_path = f.relative_to(self.templates_dir)
                templates.append(str(rel_path))
        # Versions
        for ver_dir in self.versions_dir.iterdir():
            if ver_dir.is_dir():
                for f in ver_dir.glob("*.txt"):
                    templates.append(f"{ver_dir.name}@{f.stem}")
        return sorted(set(templates))
    
    def _resolve_path(self, template_name: str) -> Path:
        """Resolve template path, handling versions and subdirs"""
        # Handle versioned
        if '@' in template_name:
            name, version = template_name.rsplit('@', 1)
            ver_path = self.versions_dir / name / f"{version}.txt"
            if ver_path.exists():
                return ver_path
        
        # Handle subdir paths like workflows/xxx.txt or base/yyy.txt
        if '/' in template_name:
            path = self.templates_dir / template_name
            if path.exists():
                return path
        
        # Plain name
        path = self.templates_dir / f"{template_name}.txt"
        if path.exists():
            return path
        
        raise FileNotFoundError(f"Template '{template_name}' not found")
    
    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter."""
        frontmatter = {}
        body = content
        if content.startswith("---\n"):
            end = content.find("\n---\n")
            if end != -1:
                try:
                    frontmatter = yaml.safe_load(content[4:end]) or {}
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML frontmatter in {template_name}: {e}")
                body = content[end + 5:]
        return frontmatter, body
    
    def _load_and_compose(self, template_name: str, visited: set = None, **kwargs) -> str:
        """Recursively load and compose template with inheritance and includes."""
        if visited is None:
            visited = set()
        if template_name in visited:
            raise ValueError(f"Inheritance cycle detected for {template_name}")
        visited.add(template_name)
        
        path = self._resolve_path(template_name)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        frontmatter, body = self._parse_frontmatter(content)
        
        # Handle extends
        extends = frontmatter.get('extends')
        if extends:
            base_content = self._load_and_compose(extends, visited, **kwargs)
            # Simple override: child body replaces base, but could be more sophisticated
            body = base_content.replace("{body}", body, 1) if "{body}" in base_content else base_content + "\n\n" + body
        
        # Handle includes
        includes = frontmatter.get('includes', [])
        for inc in includes:
            inc_path = self.components_dir / inc
            if inc_path.exists():
                with open(inc_path, 'r') as f:
                    inc_content = f.read().format(**kwargs)
                body += f"\n\n{inc_content}"
            else:
                logger.warning(f"Include '{inc}' not found")
        
        # Dynamic selection based on kwargs
        dynamic_includes = []
        if 'workout_data' in kwargs:
            dynamic_includes.append('data_sections/workout_data.txt')
        if 'user_info' in kwargs:
            dynamic_includes.append('data_sections/user_info.txt')
        if 'training_rules' in kwargs:
            dynamic_includes.append('data_sections/training_rules.txt')
        # Add more as needed
        
        for dinc in dynamic_includes:
            if dinc not in includes:  # Avoid duplicates
                dinc_path = self.components_dir / dinc
                if dinc_path.exists():
                    with open(dinc_path, 'r') as f:
                        dinc_content = f.read().format(**kwargs)
                    body += f"\n\n{dinc_content}"
        
        # Replace section placeholders
        import re
        section_pattern = re.compile(r'\{(\w+_section|\w+)\}')
        sections_map = {
            'activity_summary_section': 'data_sections/activity_summary.txt',
            'user_info_section': 'data_sections/user_info.txt',
            'training_rules_section': 'data_sections/training_rules.txt',
            'workout_data_section': 'data_sections/workout_data.txt',
            'workouts_data': 'data_sections/workouts_data.txt',
            'available_tools_section': 'data_sections/available_tools.txt',
            'recent_data_section': 'data_sections/recent_data.txt',
            'assessment_points': 'analysis_frameworks/assessment_points.txt',
            'performance_analysis': 'analysis_frameworks/performance_analysis.txt',
            'data_gathering': 'analysis_frameworks/data_gathering.txt',
        }

        for match in section_pattern.finditer(body):
            placeholder = match.group(0)
            section_name = match.group(1)
            if section_name in sections_map:
                section_file = sections_map[section_name]
                section_path = self.components_dir / section_file
                if section_path.exists():
                    with open(section_path, 'r', encoding='utf-8') as f:
                        section_content = f.read().format(**kwargs)
                    body = body.replace(placeholder, section_content)

        return body
    
    def get_template(self, template_name: str, **kwargs) -> str:
        """Load, compose, validate, and format a template."""
        # Validate first
        validation = self.validator.full_validate(template_name)
        if not validation["valid"]:
            raise ValueError(f"Template validation failed: {validation['errors']}")
        
        # Compose
        composed_content = self._load_and_compose(template_name, **kwargs)
        
        # Debug logging
        logger.debug(f"Loading template: {template_name}")
        logger.debug(f"Composed content length: {len(composed_content)}")
        logger.debug(f"Available kwargs: {list(kwargs.keys())}")
        
        # Format
        try:
            formatted_template = composed_content.format(**kwargs)
            return formatted_template
        except KeyError as e:
            logger.error(f"Missing variable in template '{template_name}': {e}")
            logger.error(f"Available kwargs: {list(kwargs.keys())}")
            raise ValueError(f"Missing variable in template '{template_name}': {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template '{template_name}': {e}")