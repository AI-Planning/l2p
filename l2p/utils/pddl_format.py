"""
This file contains collection of functions for formatting Python PDDL components into PDDL format strings.
"""

import json, re
from collections import defaultdict, OrderedDict
from .pddl_types import Predicate, Action

def indent(string: str, level: int = 2):
    """Indent string helper function to format PDDL domain/task"""
    return "   " * level + string.replace("\n", f"\n{'   ' * level}")


def pretty_print_dict(data):
    """Formats dictionary or list of dictionaries in JSON format for readability."""
    if isinstance(data, (dict, list)):
        return json.dumps(data, indent=4)
    else:
        raise TypeError("Input must be a dictionary or a list of dictionaries")


def pretty_print_predicates(predicates: list[Predicate]) -> str:
    """Formats list of predicates easier for readability"""
    if not predicates:
        return ""
    return "\n".join(
        f"{i + 1}. {pred['name']}: {pred.get('desc', 'No description provided') or 'No description provided'}"
        for i, pred in enumerate(predicates)
    )


def action_desc(action: Action) -> str:
    """Helper function to format individual action descriptions"""
    param_str = format_action_params(action)
    
    preconditions = "\n".join(
        line for line in action['preconditions'].splitlines() if line.strip()
    )
    
    effects = "\n".join(
        line for line in action['effects'].splitlines() if line.strip()
    )
    
    desc = f"(:action {action['name']}\n"
    desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
    desc += f"   :precondition\n{indent(string=preconditions, level=2)}\n"
    desc += f"   :effect\n{indent(string=effects, level=2)}\n"
    desc += ")"
    return desc

def format_actions(actions: list[Action]) -> str:
    """Formats a list of Actions into a PDDL-style string."""
    desc = ""
    for action in actions:
        desc += "\n\n" + indent(action_desc(action), level=1)
    return desc


def format_action_params(action: Action) -> str:
    """Helper function to format action parameters into a PDDL-style string."""
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


def format_params(parameters: OrderedDict) -> str:
    """Helper function to format parameters (as its type) into a PDDL-style string."""
    grouped_params = defaultdict(list)
    for name, type_ in parameters.items():
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


def format_types(
        types: dict[str,str] | list[dict[str,str]] | None = None
        ) -> dict[str, str]:
    """
    Formats nested Python type hierarchies into flat dictionaries. Flat type dictionaries 
    (no hierarchies) are returned as default.
    """

    if not types:
        return None

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
    
    
def format_types_to_string(
    types: dict[str, str] | list[dict[str, str]],
    append_obj_type_to_parent: bool = False # flag for appending `object` type to parent type
    ) -> str:
    """Formats a type hierarchy (flat or nested) into a PDDL-style string."""
    formatted = format_types(types)

    # appends `object` to parent type (required for some PDDL parsers)
    type_groups = {}
    for type_name, desc in formatted.items():
        if " - " in type_name:
            type_part, parent_part = type_name.split(" - ")
            parent = parent_part.strip()
            type_part = type_part.strip()
        else:
            parent = "object" if append_obj_type_to_parent else None
            type_part = type_name.strip()
        
        if parent:
            if parent not in type_groups:
                type_groups[parent] = []
            type_groups[parent].append(type_part)
    
    # Build the output lines
    lines = []
    
    # Handle top-level objects first
    if "object" in type_groups:
        top_level_types = sorted(t for t in type_groups["object"] if t != "object")
        if top_level_types:
            if append_obj_type_to_parent:
                lines.append(f"{' '.join(top_level_types)} - object")
            else:
                lines.append(f"{' '.join(top_level_types)}")
        del type_groups["object"]
    
    # Handle other groups
    for parent, child_types in sorted(type_groups.items()):
        child_line = " ".join(sorted(child_types))
        lines.append(f"{child_line} - {parent}")
    
    return "\n".join(lines)


def format_predicates(predicates: list[Predicate]) -> str:
    """Formats predicate list into a PDDL-style string, removing exact duplicates."""
    unique = dict()  # key = (name.lower(), tuple(params)), value = clean string
    for pred in predicates:
        key = (pred["name"].lower(), tuple(pred["params"]))
        if key not in unique:
            unique[key] = pred["clean"].replace(":", " ; ")

    return "\n".join(unique.values())


def format_objects(objects: dict[str, str]) -> str:
    """Formats task objects into a PDDL-style string."""
    objects = "\n".join([f"{obj} - {type}" if type else f"{obj}" for obj, type in objects.items()])
    return objects


def format_initial(initial_states: list[dict[str, str]]) -> str:
    """Formats task initial states into a PDDL-style string."""

    full_str = []

    for state in initial_states:
        # if function statement
        if state.get("func_name"):
            full_str.append(
                f"({state["op"]} ({state['func_name']} {' '.join(state['params'])}) {state['value']})"
            )
        # if predicate statement
        elif state.get("pred_name"):

            inner_str = f"({state['pred_name']} {' '.join(state['params'])})"

            full_str.append(
                f"(not {inner_str})" if state["neg"] else inner_str
              )

    initial_states_str = "\n".join(
        full_str
    )  # combine the states into a single string

    return initial_states_str


def format_goal(goal_states: list[dict[str, str]]) -> str:
    """Formats task goal states into a PDDL-style string."""
    full_str = []

    for state in goal_states:
         # if function statement
        if state.get("func_name"):
            full_str.append(
                f"({state["op"]} ({state['func_name']} {' '.join(state['params'])}) {state['value']})"
            )
        # if predicate statement
        elif state.get("pred_name"):
            inner_str = f"({state['pred_name']} {' '.join(state['params'])})"
            full_str.append(
                f"(not {inner_str})" if state["neg"] else inner_str
              )
        
        goal_states_str = "\n".join(
        full_str
    )  # combine the states into a single string

    goal_states_str = f"(and \n{indent(goal_states_str, 1)}\n)"

    return goal_states_str

def remove_comments(pddl_str):
    # Remove all semicolon-style comments (from ';' to the end of the line)
    while True:
        match = re.search(r";[^\n]*\n", pddl_str)
        if match is None:
            break
        start, end = match.start(), match.end()
        pddl_str = pddl_str[:start] + pddl_str[end:]

    # Remove lines that contain only whitespace
    pddl_str = re.sub(r'^[ \t]*\n', '', pddl_str, flags=re.MULTILINE)

    # Replace multiple consecutive newlines with just two
    pddl_str = re.sub(r'\n{2,}', '\n\n', pddl_str)

    return pddl_str


def format_pddl_expr(expr):
    if isinstance(expr, list):
        return "(" + " ".join(format_pddl_expr(e) for e in expr) + ")"
    else:
        return str(expr)