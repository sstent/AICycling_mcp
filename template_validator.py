import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TemplateValidator:
    """Validates template syntax, inheritance, and versioning."""

    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)
        self.components_dir = self.templates_dir / "base"
        self.versions_dir = self.templates_dir / "versions"
        self.legacy_mappings = {'main_agent_system_prompt.txt': 'base/system_prompts/main_agent.txt'}

    def parse_frontmatter(self, content: str) -> dict:
        """Parse YAML frontmatter from template content."""
        frontmatter = {}
        if content.startswith("---\n"):
            end = content.find("\n---\n")
            if end != -1:
                try:
                    frontmatter = yaml.safe_load(content[4:end])
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML frontmatter: {e}")
                content = content[end + 5:]
        return frontmatter, content

    def validate_syntax(self, template_path: Path) -> bool:
        """Validate extends/includes syntax in frontmatter."""
        with open(template_path, 'r') as f:
            frontmatter, _ = self.parse_frontmatter(f.read())
        
        extends = frontmatter.get('extends')
        includes = frontmatter.get('includes', [])
        
        if extends and not isinstance(extends, str):
            raise ValueError("extends must be a string")
        
        if not isinstance(includes, list):
            raise ValueError("includes must be a list")
        
        for inc in includes:
            if not isinstance(inc, str):
                raise ValueError("Each include must be a string")
        
        return True

    def detect_inheritance_cycle(self, template_name: str, visited: set = None) -> bool:
        """Detect cycles in inheritance chain."""
        if visited is None:
            visited = set()
        
        if template_name in visited:
            return True  # Cycle detected
        
        visited.add(template_name)
        template_path = self._find_template(template_name)
        if not template_path:
            return False
        
        with open(template_path, 'r') as f:
            frontmatter, _ = self.parse_frontmatter(f.read())
        
        extends = frontmatter.get('extends')
        if extends:
            if self.detect_inheritance_cycle(extends, visited):
                return True
        
        return False

    def validate_components_exist(self, template_path: Path) -> bool:
        """Check if all included components exist."""
        with open(template_path, 'r') as f:
            frontmatter, _ = self.parse_frontmatter(f.read())
        
        includes = frontmatter.get('includes', [])
        for inc in includes:
            comp_path = self.components_dir / inc
            if not comp_path.exists():
                raise FileNotFoundError(f"Component '{inc}' not found")
        
        return True

    def validate_version(self, version_str: str) -> bool:
        """Validate semantic version format."""
        import re
        if re.match(r'^v?\d+\.\d+\.\d+$', version_str):
            return True
        raise ValueError(f"Invalid version format: {version_str}")

    def validate_backward_compatibility(self, template_path: Path) -> bool:
        """Ensure template can be loaded as plain if no frontmatter."""
        with open(template_path, 'r') as f:
            content = f.read()
        try:
            frontmatter, body = self.parse_frontmatter(content)
            # If no frontmatter, it's compatible
            return True
        except ValueError:
            # No frontmatter, plain template
            return True

    def _find_template(self, name: str) -> Path | None:
        """Find template path, handling versions."""
        # Check legacy mappings first
        if name in self.legacy_mappings:
            name = self.legacy_mappings[name]
        # Handle versioned
        if '@' in name:
            name_part, version = name.rsplit('@', 1)
            ver_path = self.versions_dir / name_part / f"{version}.txt"
            if ver_path.exists():
                return ver_path
        # Handle subdir paths like workflows/xxx.txt or base/yyy.txt
        if '/' in name:
            path = self.templates_dir / name
            if path.exists():
                return path
        # Plain name
        path = self.templates_dir / f"{name}.txt"
        if path.exists():
            return path
        return None

    def full_validate(self, template_name: str) -> dict:
        """Perform full validation and return report."""
        template_path = self._find_template(template_name)
        if not template_path:
            raise FileNotFoundError(f"Template '{template_name}' not found")
        
        errors = []
        try:
            self.validate_syntax(template_path)
        except ValueError as e:
            errors.append(str(e))
        
        try:
            self.validate_components_exist(template_path)
        except FileNotFoundError as e:
            errors.append(str(e))
        
        if self.detect_inheritance_cycle(template_name):
            errors.append("Inheritance cycle detected")
        
        version_str = template_name.split('@')[-1] if '@' in template_name else None
        if version_str:
            try:
                self.validate_version(version_str)
            except ValueError as e:
                errors.append(str(e))
        
        self.validate_backward_compatibility(template_path)
        
        return {"valid": len(errors) == 0, "errors": errors}