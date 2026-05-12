"""
Template manager for L2P CLI.

Handles loading templates from package resources or custom paths.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from l2p.utils.temp.pddl_parser import load_file
from l2p.cli.utils.errors import TemplateError


class TemplateManager:
    """Manages template loading and discovery."""
    
    # template categories and their default files
    TEMPLATE_CATEGORIES = {
        "domain": {
            "description": "Domain generation templates",
            "files": [
                "formalize_type.txt",
                "formalize_type_hierarchy.txt", 
                "formalize_constants.txt",
                "formalize_predicates.txt",
                "formalize_functions.txt",
                "formalize_parameters.txt",
                "formalize_preconditions.txt",
                "formalize_effects.txt",
                "formalize_pddl_action.txt",
                "formalize_pddl_actions.txt",
                "extract_nl_actions.txt",
                "formalize_domain_spec.txt",
            ]
        },
        "task": {
            "description": "Task/problem generation templates",
            "files": [
                "formalize_task.txt",
                "formalize_objects.txt",
                "formalize_initial.txt", 
                "formalize_goal.txt",
            ]
        },
        "feedback": {
            "description": "Feedback generation templates",
            "files": [
                "feedback.txt",
            ]
        }
    }
    
    def __init__(self, config_manager=None):
        """Initialize template manager."""
        self.config_manager = config_manager
        self._package_templates = None
    
    def get_template(self, template_name: str, category: str = "domain") -> str:
        """Get template content by name."""
        # ensure .txt extension
        if not template_name.endswith(".txt"):
            template_name = f"{template_name}.txt"
        
        # first try custom template path
        custom_content = self._try_custom_template(template_name, category)
        if custom_content is not None:
            return custom_content
        
        # then try package templates
        package_content = self._try_package_template(template_name, category)
        if package_content is not None:
            return package_content
        
        # template not found
        available = self.list_templates(category)
        raise TemplateError(
            f"[ERROR] Template not found: {template_name} (category: {category})",
            [
                f"Available templates in '{category}' category:",
                *[f"  - {t}" for t in available],
                "Use 'l2p templates list' to see all available templates",
                "Specify custom template with --template-file option"
            ]
        )
    
    def _try_custom_template(self, template_name: str, category: str) -> Optional[str]:
        """Try to load template from custom path."""
        
        if not self.config_manager:
            return None
        
        templates_config = self.config_manager.get_templates_config()
        custom_path = templates_config.get("custom_path")
        
        if not custom_path:
            return None

        custom_dir = Path(custom_path).expanduser().resolve()
        
        # try category subdirectory first
        category_dir = custom_dir / category
        if category_dir.exists():
            template_path = category_dir / template_name
            if template_path.exists():
                try:
                    return template_path.read_text()
                except Exception as e:
                    raise TemplateError(
                        f"[ERROR] Failed to read custom template: {template_path}",
                        [f"Error: {e}", "Check file permissions and encoding"]
                    )
        
        # try direct in custom directory
        template_path = custom_dir / template_name
        if template_path.exists():
            try:
                return template_path.read_text()
            except Exception as e:
                raise TemplateError(
                    f"[ERROR] Failed to read custom template: {template_path}",
                    [f"Error: {e}", "Check file permissions and encoding"]
                )
        
        return None
    
    def _try_package_template(self, template_name: str, category: str) -> Optional[str]:
        """Try to load template from package resources."""
        try:
            # try to load from package resources
            template_path = f"templates/{category}_templates/{template_name}"
            content = load_file(template_path)
            return content
        except (ImportError, FileNotFoundError, ModuleNotFoundError):
            # fall back to file system for development
            try:
                # try relative to package root
                package_root = Path(__file__).parent.parent.parent.parent
                template_path = package_root / "templates" / f"{category}_templates" / template_name
                if template_path.exists():
                    return template_path.read_text()
            except Exception:
                pass
        
        return None
    
    def list_templates(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List available templates."""
        if category:
            if category not in self.TEMPLATE_CATEGORIES:
                raise TemplateError(
                    f"[ERROR] Invalid template category: {category}",
                    [f"Valid categories: {', '.join(self.TEMPLATE_CATEGORIES.keys())}"]
                )
            return {category: self.TEMPLATE_CATEGORIES[category]["files"]}
        
        # return all categories
        return {
            cat: info["files"]
            for cat, info in self.TEMPLATE_CATEGORIES.items()
        }
    
    def get_template_path(self, template_name: str, category: str = "domain") -> Optional[Path]:
        """Get filesystem path to a template if it exists."""
        
        # try custom path first
        if self.config_manager:
            templates_config = self.config_manager.get_templates_config()
            custom_path = templates_config.get("custom_path")
            if custom_path:
                custom_dir = Path(custom_path).expanduser().resolve()
                
                # try category subdirectory
                category_dir = custom_dir / category
                if category_dir.exists():
                    template_path = category_dir / template_name
                    if template_path.exists():
                        return template_path
                
                # try direct
                template_path = custom_dir / template_name
                if template_path.exists():
                    return template_path
        
        # try package templates on filesystem (for development)
        try:
            package_root = Path(__file__).parent.parent.parent.parent
            template_path = package_root / "templates" / f"{category}_templates" / template_name
            if template_path.exists():
                return template_path
        except Exception:
            pass
        
        return None
    
    def validate_template(self, template_content: str, required_vars: List[str]) -> Tuple[bool, List[str]]:
        """Validate template content for required variables."""
        missing = []
        for var in required_vars:
            placeholder = f"{{{var}}}"
            if placeholder not in template_content:
                missing.append(var)
        
        return len(missing) == 0, missing


# convenience function for getting template manager instance
_template_manager: Optional[TemplateManager] = None

def get_template_manager(config_manager=None) -> TemplateManager:
    """Get or create template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager(config_manager)
    elif config_manager is not None:
        _template_manager.config_manager = config_manager
    
    return _template_manager