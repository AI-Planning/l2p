"""
This file contains collection of functions for formatting Python PDDL components into PDDL format strings.
"""

import json
from collections import defaultdict
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
    
    desc = f"(:action {action['name']}\n"
    desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
    desc += f"   :precondition\n{indent(string=action['preconditions'], level=2)}\n"
    desc += f"   :effect\n{indent(string=action['effects'], level=2)}\n"
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
    
    
def format_types_to_string(types: dict[str, str] | list[dict[str, str]]) -> str:
    """Formats a type hierarchy (flat or nested) into a PDDL-style string."""
    formatted = format_types(types)
    lines = [f"{type_name} {desc}" if desc else f"{type_name}" for type_name, desc in formatted.items()]
    return "\n".join(lines)


def format_predicates(predicates: list[Predicate]) -> str:
    """Formats predicate list into a PDDL-style string, removing exact duplicates."""
    unique = dict()  # key = (name.lower(), tuple(params)), value = clean string
    for pred in predicates:
        key = (pred["name"].lower(), tuple(pred["params"]))
        if key not in unique:
            unique[key] = pred["clean"].replace(":", " ; ")

    return "\n".join(unique.values())


def format_action(self, actions: list[Action]) -> str:
    desc = ""
    for action in actions:
        param_str = "\n".join(
            [f"{name} - {type}" for name, type in action["params"].items()]
        )  # name includes ?
        desc += f"(:action {action['name']}\n"
        desc += f"   :parameters (\n{indent(param_str,2)}\n   )\n"
        desc += f"   :precondition\n{indent(action['preconditions'],2)}\n"
        desc += f"   :effect\n{indent(action['effects'],2)}\n"
        desc += ")\n"
    return desc


def format_objects(objects: dict[str, str]) -> str:
    """Formats task objects into a PDDL-style string."""
    objects = "\n".join([f"{obj} - {type}" if type else f"{obj}" for obj, type in objects.items()])
    return objects


def format_initial(initial_states: list[dict[str, str]]) -> str:
    """Formats task initial states into a PDDL-style string."""
    inner_str = [
        f"({state['name']} {' '.join(state['params'])})" for state in initial_states
    ]  # The main part of each predicate
    full_str = [
        f"(not {inner})" if state["neg"] else inner
        for state, inner in zip(initial_states, inner_str)
    ]  # add `not` if needed
    initial_states_str = "\n".join(
        full_str
    )  # combine the states into a single string

    return initial_states_str


def format_goal(goal_states: list[dict[str, str]]) -> str:
    """Formats task goal states into a PDDL-style string."""
    goal_states_str = "(AND \n"

    # loop through each dictionary in the list
    for item in goal_states:
        # extract the name and parameters from the dictionary
        name = item["name"]
        params = " ".join(item["params"])
        goal_states_str += (
            f"   ({name} {params}) \n"  # append the predicate in the desired format
        )

    goal_states_str += ")"

    return goal_states_str