"""
This file contains collection of functions for extracting/parsing information from LLM output
"""

from pddl.formatter import domain_to_string, problem_to_string
from pddl import parse_domain, parse_problem
from .pddl_types import Action, Predicate
from collections import OrderedDict
from copy import deepcopy
from typing import Optional
import re
import ast
import json
import sys
import os


def load_file(file_path: str):
    _, ext = os.path.splitext(file_path)
    with open(file_path, "r") as file:
        if ext == ".json":
            return json.load(file)
        else:
            return file.read().strip()


def load_files(folder_path: str):
    file_contents = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                file_contents.append(file.read())
    return file_contents


def parse_params(llm_output):
    """
    Parses parameters from LLM into Python format (refer to example templates to see
    how these parameters should be formatted in LLM response).

    LLM output header should contain '### Parameters' along with structured content.
    """
    params_info = OrderedDict()
    params_heading = re.split(
        r"\n#+\s", llm_output.split("Parameters")[1].strip(), maxsplit=1
    )[0]
    params_str = combine_blocks(params_heading)
    params_raw = []
    for line in params_str.split("\n"):
        if line.strip() == "" or ("." not in line and not line.strip().startswith("-")):
            print(
                f"[WARNING] checking param object types - empty line or not a valid line: '{line}'"
            )
            continue
        if not (line.split(".")[0].strip().isdigit() or line.startswith("-")):
            print(f"[WARNING] checking param object types - not a valid line: '{line}'")
            continue
        try:
            params_raw.append(line.strip())
            p_info = [e for e in line.split(":")[0].split(" ") if e != ""]
            param_name, param_type = p_info[1].strip(" `"), p_info[3].strip(" `")
            params_info[param_name] = param_type
        except Exception:
            print(f"[WARNING] checking param object types - fail to parse: {line}")
            break
    return params_info, params_raw


def parse_new_predicates(llm_output) -> list[Predicate]:
    """
    Parses new predicates from LLM into Python format (refer to example templates to see
    how these predicates should be formatted in LLM response).

    LLM output header should contain '### New Predicates' along with structured content.
    """
    new_predicates = list()
    try:
        predicate_heading = (
            llm_output.split("New Predicates\n")[1].strip().split("###")[0]
        )
    except:
        raise Exception(
            "Could not find the 'New Predicates' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
    predicate_output = combine_blocks(predicate_heading)

    for p_line in predicate_output.split("\n"):
        if ("." not in p_line or not p_line.split(".")[0].strip().isdigit()) and not (
            p_line.startswith("-") or p_line.startswith("(")
        ):
            if len(p_line.strip()) > 0:
                print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue
        predicate_info = p_line.split(": ")[0].strip(" 1234567890.(-)`").split(" ")
        predicate_name = predicate_info[0]
        predicate_desc = p_line.split(": ")[1].strip() if ": " in p_line else ""

        # get the predicate type info
        if len(predicate_info) > 1:
            predicate_type_info = predicate_info[1:]
            predicate_type_info = [
                l.strip(" ()`") for l in predicate_type_info if l.strip(" ()`")
            ]
        else:
            predicate_type_info = []
        params = OrderedDict()
        next_is_type = False
        upcoming_params = []

        for p in predicate_type_info:
            if next_is_type:
                if p.startswith("?"):
                    print(
                        f"[WARNING] `{p}` is not a valid type for a variable, but it is being treated as one. Should be checked by syntax check later."
                    )
                for up in upcoming_params:
                    params[up] = p
                next_is_type = False
                upcoming_params = []
            elif p == "-":
                next_is_type = True
            elif p.startswith("?"):
                upcoming_params.append(p)  # the next type will be for this variable
            else:
                print(
                    f"[WARNING] `{p}` is not corrrectly formatted. Assuming it's a variable name."
                )
                upcoming_params.append(f"?{p}")
        if next_is_type:
            print(
                f"[WARNING] The last type is not specified for `{p_line}`. Undefined are discarded."
            )
        if len(upcoming_params) > 0:
            print(
                f"[WARNING] The last {len(upcoming_params)} is not followed by a type name for {upcoming_params}. These are discarded"
            )

        # generate a clean version of the predicate
        clean = f"({predicate_name} {' '.join([f'{k} - {v}' for k, v in params.items()])}): {predicate_desc}"

        # drop the index/dot
        p_line = p_line.strip(" 1234567890.-`")
        new_predicates.append(
            {
                "name": predicate_name,
                "desc": predicate_desc,
                "raw": p_line,
                "params": params,
                "clean": clean,
            }
        )

    return new_predicates


def parse_predicates(all_predicates):
    """
    This function assumes the predicate definitions adhere to PDDL grammar.
    Assigns `params` to the predicate arguments properly. This should be run
    after retrieving a predicate list to ensure predicates are set correctly.
    """
    all_predicates = deepcopy(all_predicates)
    for i, pred in enumerate(all_predicates):
        if "params" in pred:
            continue
        pred_def = pred["raw"].split(": ")[0]
        pred_def = pred_def.strip(" ()`")  # drop any leading/strange formatting
        split_predicate = pred_def.split(" ")[1:]  # discard the predicate name
        split_predicate = [e for e in split_predicate if e != ""]

        pred["params"] = OrderedDict()
        for j, p in enumerate(split_predicate):
            if j % 3 == 0:
                assert "?" in p, f"invalid predicate definition: {pred_def}"
                assert (
                    split_predicate[j + 1] == "-"
                ), f"invalid predicate definition: {pred_def}"
                param_name, param_obj_type = p, split_predicate[j + 2]
                pred["params"][param_name] = param_obj_type
    return all_predicates


def parse_action(llm_response: str, action_name: str) -> Action:
    """
    Parse an action from a given LLM output.

    Args:
        llm_response (str): The LLM output.
        action_name (str): The name of the action.

    Returns:
        Action: The parsed action.
    """
    parameters, _ = parse_params(llm_response)
    try:
        preconditions = (
            llm_response.split("Preconditions\n")[1]
            .split("###")[0]
            .split("```")[1]
            .strip(" `\n")
        )
    except:
        raise Exception(
            "Could not find the 'Preconditions' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
    try:
        effects = (
            llm_response.split("Effects\n")[1]
            .split("###")[0]
            .split("```")[1]
            .strip(" `\n")
        )
    except:
        raise Exception(
            "Could not find the 'Effects' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
    return {
        "name": action_name,
        "params": parameters,
        "preconditions": preconditions,
        "effects": effects,
    }


def parse_objects(llm_response: str) -> dict[str, str]:
    """
    Extract objects from LLM response and returns dictionary string pairs object(name, type)
    Args:
        - llm_response (str):
        - types (dict[str,str]): WILL BE USED FOR CHECK ERROR RAISES
        - predicates (list[Predicate]): WILL BE USED FOR CHECK ERROR RAISES
    Returns:
        - dict[str,str]: objects
    """

    objects_head = extract_heading(llm_response, "OBJECTS")
    objects_raw = combine_blocks(objects_head)

    objects_clean = clear_comments(
        text=objects_raw, comments=[":", "//", "#", ";", "(", ")"]
    )  # Remove comments

    objects = {
        obj.split(" - ")[0].strip(" `"): obj.split(" - ")[1].strip(" `").lower()
        for obj in objects_clean.split("\n")
        if obj.strip()
    }

    return objects


def parse_initial(llm_response: str) -> list[dict[str, str]]:
    """
    Extracts state (PDDL-init) from LLM response and returns it as a list of dict strings

    Args:
        llm_response (str): The LLM output.

    Returns:
        states (list[dict[str,str]]): list of initial states in dictionaries
    """
    state_head = extract_heading(llm_response, "INITIAL")
    state_raw = combine_blocks(state_head)
    state_clean = clear_comments(state_raw)

    states = []
    for line in state_clean.split("\n"):
        line = line.strip("- `()")
        if not line:  # Skip empty lines
            continue
        name = line.split(" ")[0]
        if name == "not":
            neg = True
            name = line.split(" ")[1].strip(
                "()"
            )  # Remove the `not` and the parentheses
            params = line.split(" ")[2:]
        else:
            neg = False
            params = line.split(" ")[1:] if len(line.split(" ")) > 1 else []
        states.append({"name": name, "params": params, "neg": neg})

    return states


def parse_goal(llm_response: str) -> list[dict[str, str]]:
    """
    Extracts goal (PDDL-goal) from LLM response and returns it as a string

    Args:
        llm_response (str): The LLM output.

    Returns:
        states (list[dict[str,str]]): list of goal states in dictionaries
    """
    goal_head = extract_heading(llm_response, "GOAL")

    if goal_head.count("```") != 2:
        raise ValueError(
            "Could not find exactly one block in the goal section of the LLM output. The goal has to be specified in a single block and as valid PDDL using the `and` and `not` operators. Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task."
        )
    goal_raw = goal_head.split("```")[1].strip()  # Only a single block in the goal
    goal_clean = clear_comments(goal_raw)

    goal_pure = (
        goal_clean.replace("and", "")
        .replace("AND", "")
        .replace("not", "")
        .replace("NOT", "")
    )
    goal = []
    for line in goal_pure.split("\n"):
        line = line.strip(" ()")
        if not line:  # Skip empty lines
            continue
        name = line.split(" ")[0]
        params = line.split(" ")[1:] if len(line.split(" ")) > 1 else []
        goal.append({"name": name, "params": params})

    return goal  # Since the goal uses `and` and `not` recombining it is difficult


def prune_types(
    types: dict[str, str] | list[dict[str, str]],
    predicates: list[Predicate],
    actions: list[Action]
) -> dict[str, str]:
    """
    Prune types that are not used in any predicate or action.

    Args:
        types (dict or list): Either a flat dict of {type: description} or a nested list of type hierarchies.
        predicates (list[Predicate]): A list of predicates.
        actions (list[Action]): A list of actions.

    Returns:
        dict[str, str]: A dictionary of used types.
    """
    used_types = {}
    all_type_names = {}

    def collect_all_types(obj):
        """Recursively extract all type names and their descriptions from the input."""
        
        # if setup is normal types dictionary
        if isinstance(obj, dict):
            for type_name, desc in obj.items():
                all_type_names[type_name] = desc
                
        # if set up is type hierarchy list dictionary
        elif isinstance(obj, list):
            for node in obj:
                type_name = next((k for k in node if k != "children"), None)
                if type_name:
                    all_type_names[type_name] = node[type_name]
                    collect_all_types(node.get("children", []))

    # flatten all types into all_type_names
    collect_all_types(types)

    # identify used types
    for type_name in all_type_names:
        base_name = type_name.split(" ")[0]
        found = False

        for pred in predicates:
            if base_name in pred["params"].values():
                used_types[type_name] = all_type_names[type_name]
                found = True
                break
        if found:
            continue

        for action in actions:
            if base_name in action["params"].values():
                used_types[type_name] = all_type_names[type_name]
                break
            if (
                base_name in action["preconditions"]
                or base_name in action["effects"]
            ):
                used_types[type_name] = all_type_names[type_name]
                break

    return used_types


def prune_predicates(
    predicates: list[Predicate], actions: list[Action]
) -> list[Predicate]:
    """
    Remove predicates that are not used in any action.

    Args:
        predicates (list[Predicate]): A list of predicates.
        actions (list[Action]): A list of actions.

    Returns:
        list[Predicate]: The pruned list of predicates.
    """
    used_predicates = []
    seen_predicate_names = set()

    for pred in predicates:
        for action in actions:
            # Add a space or a ")" to avoid partial matches
            names = [f"{pred['name']} ", f"{pred['name']})"]
            for name in names:
                if name in action["preconditions"] or name in action["effects"]:
                    if pred["name"] not in seen_predicate_names:
                        used_predicates.append(pred)
                        seen_predicate_names.add(pred["name"])
                    break

    return used_predicates


def extract_heading(llm_output: str, heading: str):
    """Extract the text between the heading and the next second level heading in the LLM output."""
    if heading not in llm_output:
        print("#" * 10, "LLM Output", "#" * 10)
        print(llm_output)
        print("#" * 30)
        raise ValueError(
            f"Could not find heading {heading} in the LLM output. Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task."
        )
    heading_str = (
        llm_output.split(heading)[1].split("\n### ")[0].strip()
    )  # Get the text between the heading and the next heading
    return heading_str


def convert_to_dict(llm_response: str) -> Optional[dict[str, str]]:
    """
    Safely extracts and evaluates a dictionary structure from a string (LLM response).

    Args:
        llm_response (str): Raw string from the LLM expected to contain a flat dictionary.

    Returns:
        dict[str, str] | None: Parsed dictionary if valid, else None.
    """
    try:
        # Regex to extract the first dictionary-like structure
        dict_pattern = re.compile(r"{[^{}]*}", re.DOTALL)
        match = dict_pattern.search(llm_response)

        if not match:
            print("No dictionary found in the LLM response.")
            return None

        dict_str = match.group(0)
        parsed = ast.literal_eval(dict_str)

        # Validate it is a flat dictionary with string keys and values
        if isinstance(parsed, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
        ):
            return parsed

        print("Parsed object is not a flat dictionary of string keys and values.")
        return None

    except Exception as e:
        print(f"Error parsing dictionary: {e}")
        return None
    
    
def convert_to_list_of_dict(llm_response: str) -> Optional[list[dict[str,str]]]:
    """
    Safely parses LLM response into a list of nested dictionaries representing the type hierarchy.

    Args:
        llm_response (str): raw LLM output expected to contain a Python list of dictionaries.

    Returns:
        list[dict[str,str]] | None: Parsed type hierarchy if valid, else None.
    """
    try:
        parsed = ast.literal_eval(llm_response)

        # Ensure it's a list of dicts with proper structure
        if not isinstance(parsed, list):
            return None

        def validate_type_node(node):
            if not isinstance(node, dict):
                return False
            if "children" not in node:
                return False
            if not isinstance(node["children"], list):
                return False
            for child in node["children"]:
                if not validate_type_node(child):
                    return False
            return True

        if all(validate_type_node(item) for item in parsed):
            return parsed

        return None

    except Exception as e:
        print(f"Failed to convert response to dict: {e}")
        return None


def clear_comments(text: str, comments=[":", "//", "#", ";"]) -> str:
    """Remove comments from the text."""
    for comment in comments:
        text = "\n".join([line.split(comment)[0] for line in text.split("\n")])
    return text


def combine_blocks(heading_str: str):
    """Combine the inside of blocks from the heading string into a single string."""

    possible_blocks = heading_str.split("```")
    blocks = [
        possible_blocks[i] for i in range(1, len(possible_blocks), 2)
    ]  # obtain string between ```

    combined = "\n".join(blocks)

    return combined.replace(
        "\n\n", "\n"
    ).strip()  # remove leading/trailing whitespace and internal empty lines


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


def indent(string: str, level: int = 2):
    """Indent string helper function to format PDDL domain/task"""
    return "   " * level + string.replace("\n", f"\n{'   ' * level}")


def check_parse_domain(file_path: str):
    """Run PDDL library to check if file is syntactically correct"""
    try:
        domain = parse_domain(file_path)
        pddl_domain = domain_to_string(domain)
        return pddl_domain
    except Exception as e:
        print("------------------")
        print(f"Error parsing domain: {e}", file=sys.stderr)
        print("------------------")
        sys.exit(1)


def check_parse_problem(file_path: str):
    """Run PDDL library to check if file is syntactically correct"""
    try:
        problem = parse_problem(file_path)
        pddl_problem = problem_to_string(problem)
        return pddl_problem
    except Exception as e:
        print("------------------")
        print(f"Error parsing domain: {e}", file=sys.stderr)
        print("------------------")
        sys.exit(1)
