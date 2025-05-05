"""
This file contains collection of functions for formatting Python PDDL components into PDDL format.
"""

from collections import defaultdict
from .pddl_types import Predicate, Action

def format_action_params(action: Action) -> str:
    """Helper function to format action parameters"""
    grouped_params = defaultdict(list)
    for name, type_ in action["params"].items():
        # ensure name starts with '?'
        name = name if name.startswith("?") else f"?{name}"
        clean_type = type_.strip() if type_ else None
        grouped_params[clean_type].append(name)

    param_parts = []
    for type_, names in grouped_params.items():
        if type_ is None:
            param_parts.append(" ".join(names))  # untyped parameters
        else:
            param_parts.append(f"{' '.join(names)} - {type_}")

    return " ".join(param_parts)


def format_types(types: dict[str,str] | list[dict[str,str]]) -> dict[str, str]:
    """Formats both flat and nested Python type hierarchies into a PDDL-style dictionary."""
    result = {}

    def is_nested_format(typ) -> bool:
        if isinstance(typ, list):
            return True
        return any(isinstance(v, list) and k == "children" for k, v in typ.items())

    def process_node(node, parent_type=None):
        if not isinstance(node, dict):
            return

        type_name = next((k for k in node if k != "children"), None)
        if type_name is None:
            return

        description = node[type_name]
        parent = parent_type if parent_type else type_name

        name = f"{type_name} - {parent}" if type_name != parent else type_name
        result[name] = f"; {description}" if description else ""

        for child in node.get("children", []):
            process_node(child, type_name)

    if is_nested_format(types):
        if isinstance(types, list):
            for node in types:
                process_node(node)
        else:
            process_node(types)
    else:
        for type_name, description in types.items():
            result[type_name] = f"; {description}" if description else ""

    return result
    
    
def format_types_to_string(types: dict[str, str] | list[dict[str, str]]) -> str:
    """
    Formats a type hierarchy (flat or nested) into a PDDL-style string.

    Args:
        types (dict[str, str] | list[dict[str, str]]): Type hierarchy in flat or nested format.

    Returns:
        str: A string where each line represents a type with optional description as a comment.
    """
    formatted = format_types(types)
    lines = [f"{type_name} {desc}" if desc else f"{type_name}" for type_name, desc in formatted.items()]
    return "\n".join(lines)


def format_predicates(predicates: list[Predicate]) -> str:
    """Helper function that formats predicate list into string"""
    return "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])