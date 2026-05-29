"""
Template manager for L2P CLI.

All templates are loaded from the package ``l2p/templates/`` directory.
Custom/user-defined templates are not supported — the CLI uses the built-in
prompts that ship with L2P.
"""

from typing import Dict, List, Optional

from l2p.utils.pddl_prompt import load_default_template
from l2p.cli.utils.errors import TemplateError


class TemplateManager:
    """Manages template loading from the L2P package templates directory."""

    # ------------------------------------------------------------------
    # Mapping:  friendly name  →  folder / filename  (without extension)
    # ------------------------------------------------------------------
    DOMAIN_TEMPLATES: Dict[str, str] = {
        "domain": "domain/prompt_domain",
        "requirements": "domain/prompt_requirements",
        "types": "domain/prompt_types",
        "constants": "domain/prompt_constants",
        "predicates": "domain/prompt_predicates",
        "functions": "domain/prompt_functions",
        "constraints": "domain/prompt_constraints",
        "derived_predicates": "domain/prompt_derived_predicates",
        "actions": "domain/prompt_actions",
        "durative_actions": "domain/prompt_durative_actions",
        "nl_actions": "domain/prompt_nl_actions",
        "nl_durative_actions": "domain/prompt_nl_durative_actions",
        "preconditions": "domain/prompt_preconditions",
        "effects": "domain/prompt_effects",
        "durative_conditions": "domain/prompt_durative_conditions",
        "durative_effects": "domain/prompt_durative_effects",
        "parameters": "domain/prompt_parameters",
        "events": "domain/prompt_events",
        "processes": "domain/prompt_processes",
    }

    PROBLEM_TEMPLATES: Dict[str, str] = {
        "problem": "problem/prompt_problem",
        "objects": "problem/prompt_objects",
        "initial_states": "problem/prompt_initial_states",
        "goal_states": "problem/prompt_goal_states",
        "constraints": "problem/prompt_constraints",
        "metric": "problem/prompt_metric",
    }

    CATEGORY_MAP: Dict[str, Dict[str, str]] = {
        "domain": DOMAIN_TEMPLATES,
        "problem": PROBLEM_TEMPLATES,
    }

    def __init__(self, config_manager=None):
        self.config_manager = config_manager

    def get_template(self, template_name: str, category: str = "domain") -> str:
        """
        Load a template by its friendly name and category.

        Args:
            template_name: e.g. ``"types"``, ``"objects"``, ``"actions"``.
            category: ``"domain"`` or ``"problem"``.

        Returns:
            The template content as a string.

        Raises:
            TemplateError: If the template cannot be found.
        """
        # strip .md / .txt suffix if the caller passed a full filename
        template_name = template_name.removesuffix(".md").removesuffix(".txt")

        # resolve the resource path
        cat_templates = self.CATEGORY_MAP.get(category)
        if cat_templates is None:
            raise TemplateError(
                f"[ERROR] Unknown template category: '{category}'. "
                f"Valid categories: {', '.join(self.CATEGORY_MAP)}"
            )

        resource_path = cat_templates.get(template_name)
        if resource_path is None:
            available = self.list_templates(category)
            raise TemplateError(
                f"[ERROR] Template not found: '{template_name}' (category: {category}).",
                [
                    f"Available templates in '{category}' category:",
                    *[f"  - {t}" for t in available],
                ],
            )

        folder, filename = resource_path.split("/", 1)
        content = load_default_template(folder, f"{filename}.md")
        if content.startswith("[ERROR]"):
            raise TemplateError(content)

        return content

    def list_templates(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List available template names."""
        if category:
            cat_templates = self.CATEGORY_MAP.get(category)
            if cat_templates is None:
                raise TemplateError(
                    f"[ERROR] Unknown template category: '{category}'. "
                    f"Valid categories: {', '.join(self.CATEGORY_MAP)}"
                )
            return {category: list(cat_templates.keys())}

        return {cat: list(t.keys()) for cat, t in self.CATEGORY_MAP.items()}


# ------------------------------------------------------------------
# Singleton-ish convenience
# ------------------------------------------------------------------
_template_manager: Optional[TemplateManager] = None


def get_template_manager(config_manager=None) -> TemplateManager:
    """Get or create the template manager singleton."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager(config_manager)
    elif config_manager is not None:
        _template_manager.config_manager = config_manager
    return _template_manager
