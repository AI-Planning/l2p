"""
This module contains a collection of helper functions that parses information from LLM output.
"""

import ast
import json
import os
import re
import sys

from collections import OrderedDict
from copy import deepcopy
from typing import Optional

from pddl import parse_domain, parse_problem
from pddl.formatter import domain_to_string, problem_to_string

from .pddl_format import remove_comments
from .pddl_types import Action, Function, Predicate


# ---- PDDL DOMAIN PARSERS ----


def parse_types(llm_output: str, heading: str = "TYPES") -> Optional[dict[str, str]]:
    """
    Safely extracts and evaluates a dictionary structure from a string (LLM response).

    Args:
        llm_output (str): raw string from the LLM expected to contain a flat dictionary

    Returns:
        types_parsed (dict[str, str]) | None: parsed dictionary of :types if valid, else None.
    """
    try:

        types_head = parse_heading(llm_output, heading)
        if types_head.count("```") != 2:
            raise ValueError(
                "Could not find exactly one block in the types section enclosed by [```, ```] of the LLM output."
            )

        types_raw = combine_blocks(types_head)

        # Regex to extract the first dictionary-like structure
        dict_pattern = re.compile(r"{[^{}]*}", re.DOTALL)
        match = dict_pattern.search(types_raw)

        if not match:
            print("No dictionary found in the LLM response.")
            return None

        dict_str = match.group(0)
        types_parsed = ast.literal_eval(dict_str)

        # Validate it is a flat dictionary with string keys and values
        if isinstance(types_parsed, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in types_parsed.items()
        ):
            return types_parsed

        print("Parsed object is not a flat dictionary of string keys and values.")
        return None

    except Exception as e:
        print(f"Error parsing dictionary: {e}")
        return None


def parse_type_hierarchy(llm_output: str) -> Optional[list[dict[str, str]]]:
    """
    Safely parses LLM response into a list of nested dictionaries representing the type hierarchy.

    Args:
        llm_output (str): raw LLM output expected to contain a Python list of dictionaries.

    Returns:
        types_parsed (list[dict[str,str]]) | None: parsed type hierarchy if valid, else None.
    """
    try:

        types_head = parse_heading(llm_output, "TYPES")
        if types_head.count("```") != 2:
            raise ValueError(
                "Could not find exactly one block in the types section enclosed by [```, ```] of the LLM output."
            )

        types_raw = combine_blocks(types_head)

        # catch if types is empty
        if not types_raw:
            return None

        types_parsed = ast.literal_eval(types_raw)

        # ensure it's a list of dicts with proper structure
        if not isinstance(types_parsed, list):
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

        if all(validate_type_node(item) for item in types_parsed):
            return types_parsed

        return None

    except Exception as e:
        print(f"Failed to convert response to dict: {e}")
        return None


def parse_constants(llm_output: str) -> Optional[dict[str, str]]:
    """
    Safely extracts and evaluates a dictionary structure from a string (LLM response).

    Args:
        llm_output (str): Raw string from the LLM expected to contain a flat dictionary.

    Returns:
        constants_parsed (dict[str, str]) | None: parsed dictionary if valid, else None.
    """
    try:

        constant_head = parse_heading(llm_output, "CONSTANTS")
        if constant_head.count("```") != 2:
            raise ValueError(
                "Could not find exactly one block in the constants section enclosed by [```, ```] of the LLM output."
            )

        constants_raw = combine_blocks(constant_head)

        # regex to extract the first dictionary-like structure
        dict_pattern = re.compile(r"{[^{}]*}", re.DOTALL)
        match = dict_pattern.search(constants_raw)

        if not match:
            print("No dictionary found in the LLM response.")
            return None

        dict_str = match.group(0)
        constants_parsed = ast.literal_eval(dict_str)

        # validate it is a flat dictionary with string keys and values
        if isinstance(constants_parsed, dict) and all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in constants_parsed.items()
        ):
            return constants_parsed

        print("Parsed object is not a flat dictionary of string keys and values.")
        return None

    except Exception as e:
        print(f"Error parsing dictionary: {e}")
        return None


def parse_predicates(all_predicates: list[Predicate]) -> list[Predicate]:
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
        p_line = p_line.strip()
        if not p_line or p_line.startswith("```"):
            continue  # skip empty lines and code block markers

        # skip lines that do not look like predicate definitions
        if not (p_line.startswith("-") or p_line.startswith("(")):
            if len(p_line) > 0:
                print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue

        # extract predicate signature and description
        if ":" in p_line:
            pred_part, desc = p_line.split(":", 1)
            predicate_desc = desc.strip().strip("'\"")
        elif ";" in p_line:
            pred_part, desc = p_line.split(";", 1)
            predicate_desc = desc.strip().strip("'\"")
        else:
            pred_part = p_line
            predicate_desc = ""

        # clean the predicate signature
        pred_part = pred_part.strip("- ()").strip()

        # split into name and parameters
        parts = pred_part.split()
        if not parts:
            continue

        predicate_name = parts[0]
        param_parts = parts[1:]

        params = OrderedDict()
        current_param = None

        i = 0
        while i < len(param_parts):
            part = param_parts[i]
            if part.startswith("?"):
                # found a parameter
                current_param = part
                params[current_param] = ""  # Default to untyped
                i += 1
            elif part == "-":
                # found type indicator
                if current_param is None:
                    print(
                        f"[WARNING] Found type indicator without parameter in: {p_line}"
                    )
                    i += 1
                    continue
                if i + 1 < len(param_parts):
                    params[current_param] = param_parts[i + 1]
                    i += 2
                else:
                    print(f"[WARNING] Missing type after '-' in: {p_line}")
                    i += 1
            else:
                # might be a typeless parameter (accept it with warning)
                print(
                    f"[WARNING] Assuming '{part}' is an untyped parameter in: {p_line}"
                )
                current_param = f"?{part.lstrip('?')}"
                params[current_param] = ""
                i += 1

        # generate clean PDDL representation
        clean_params = []
        for param, type_ in params.items():
            if type_:
                clean_params.append(f"{param} - {type_}")
            else:
                clean_params.append(param)
        clean = f"({predicate_name} {' '.join(clean_params)})"

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


def parse_functions(llm_output: str) -> list[Function]:
    """
    Parses function from LLM into Python format (refer to example templates to see
    how these functions should be formatted in LLM response).

    LLM output header should contain '### FUNCTIONS' along with structured content.
    """
    functions = list()
    try:
        function_heading = llm_output.split("FUNCTIONS\n")[1].strip().split("###")[0]
    except:
        raise Exception(
            "Could not find the 'FUNCTIONS' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )
    function_output = combine_blocks(function_heading)

    for f_line in function_output.split("\n"):
        f_line = f_line.strip()
        if not f_line or f_line.startswith("```"):
            continue  # skip empty lines and code block markers

        # skip lines that do not look like function definitions
        if not (f_line.startswith("-") or f_line.startswith("(")):
            if len(f_line) > 0:
                print(f'[WARNING] unable to parse the line: "{f_line}"')
            continue

        # extract function signature and description
        if ":" in f_line:
            func_part, desc = f_line.split(":", 1)
            function_desc = desc.strip().strip("'\"")
        elif ";" in f_line:
            func_part, desc = f_line.split(";", 1)
            function_desc = desc.strip().strip("'\"")
        else:
            func_part = f_line
            function_desc = ""

        # clean the function signature
        func_part = func_part.strip("- ()").strip()

        # split into name and parameters
        parts = func_part.split()
        if not parts:
            continue

        function_name = parts[0]
        param_parts = parts[1:]

        params = OrderedDict()
        current_param = None

        i = 0
        while i < len(param_parts):
            part = param_parts[i]
            if part.startswith("?"):
                # found a parameter
                current_param = part
                params[current_param] = ""  # default to untyped
                i += 1
            elif part == "-":
                # found type indicator
                if current_param is None:
                    print(
                        f"[WARNING] Found type indicator without parameter in: {f_line}"
                    )
                    i += 1
                    continue
                if i + 1 < len(param_parts):
                    params[current_param] = param_parts[i + 1]
                    i += 2
                else:
                    print(f"[WARNING] Missing type after '-' in: {f_line}")
                    i += 1
            else:
                # might be a typeless parameter (accept it with warning)
                print(
                    f"[WARNING] Assuming '{part}' is an untyped parameter in: {f_line}"
                )
                current_param = f"?{part.lstrip('?')}"
                params[current_param] = ""
                i += 1

        # generate clean PDDL representation
        clean_params = []
        for param, type_ in params.items():
            if type_:
                clean_params.append(f"{param} - {type_}")
            else:
                clean_params.append(param)
        clean = f"({function_name} {' '.join(clean_params)})"

        functions.append(
            {
                "name": function_name,
                "desc": function_desc,
                "raw": f_line,
                "params": params,
                "clean": clean,
            }
        )

    return functions


def parse_action(llm_output: str, action_name: str) -> Action:
    """
    Parse an action from a given LLM output.

    Args:
        llm_output (str): raw LLM output
        action_name (str): the name of the action

    Returns:
        Action: the parsed PDDL action
    """
    parameters, _ = parse_params(llm_output)
    try:
        preconditions = (
            llm_output.split("Preconditions\n")[1]
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
            llm_output.split("Effects\n")[1]
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


def parse_params(llm_output: str) -> tuple[OrderedDict, list]:
    """
    Parses parameters from LLM into Python format (refer to example templates to see
    how these parameters should be formatted in LLM response).

    LLM output header should contain 'Parameters' along with structured content.
    """
    params_info = OrderedDict()
    params_heading = re.split(
        r"\n#+\s", llm_output.split("Parameters")[1].strip(), maxsplit=1
    )[0]
    params_str = combine_blocks(params_heading)
    params_raw = []

    for line in params_str.split("\n"):
        line = line.strip()
        if not line:  # skip empty lines
            continue

        if line.startswith("-"):
            line = line[1:].strip()  # remove the dash and clean up

        if not line.startswith("?"):
            print(f"[WARNING] Invalid parameter line - must start with '?': '{line}'")

        try:
            params_raw.append(line)
            # split into param name and type
            parts = line.split(":")

            left_side = parts[0].strip()
            param_parts = [p.strip() for p in left_side.split("-")]

            param_name = param_parts[0].strip(" `")
            if len(param_parts) == 2 and param_parts[1]:
                param_type = param_parts[1].strip(" `")
            else:
                print(
                    f"[WARNING] Invalid parameter format: '{line}'. Defaulting to no type."
                )
                param_type = ""  # no type provided

            params_info[param_name] = param_type

        except Exception as e:
            print(f"[WARNING] Failed to parse parameter line: '{line}' - {str(e)}")
            continue

    return params_info, params_raw


def parse_preconditions(llm_output: str) -> str:
    """Parses precondition string from LLM output"""
    try:
        preconditions = (
            llm_output.split("Preconditions\n")[1]
            .split("###")[0]
            .split("```")[1]
            .strip(" `\n")
        )

        return preconditions
    except:
        raise Exception(
            "Could not find the 'Preconditions' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )


def parse_effects(llm_output: str) -> str:
    """Parses effect string from LLM output"""
    try:
        effects = (
            llm_output.split("Effects\n")[1]
            .split("###")[0]
            .split("```")[1]
            .strip(" `\n")
        )

        return effects
    except:
        raise Exception(
            "Could not find the 'Preconditions' section in the output. Provide the entire response, including all headings even if some are unchanged."
        )


def prune_types(
    types: dict[str, str] | list[dict[str, str]],
    predicates: list[Predicate],
    actions: list[Action],
) -> dict[str, str]:
    """
    Prune types that are not used in any predicate or action.

    Parameters:
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
            if base_name in action["preconditions"] or base_name in action["effects"]:
                used_types[type_name] = all_type_names[type_name]
                break

    return used_types


def prune_predicates(
    predicates: list[Predicate], actions: list[Action]
) -> list[Predicate]:
    """
    Remove predicates that are not used in any action.

    Args:
        predicates (list[Predicate]): a list of predicates
        actions (list[Action]): a list of actions

    Returns:
        list[Predicate]: the pruned list of predicates
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


# ---- PDDL PROBLEM PARSERS ----


def parse_objects(llm_output: str) -> dict[str, str]:
    """
    Extract objects from LLM response and returns dictionary string pairs object(name, type)

    Args:
        llm_output (str): raw LLM output
        types (dict[str,str]): WILL BE USED FOR CHECK ERROR RAISES
        predicates (list[Predicate]): WILL BE USED FOR CHECK ERROR RAISES

    Returns:
        objects_parsed (dict[str,str]): PDDL task objects
    """

    objects_head = parse_heading(llm_output, "OBJECTS")
    objects_raw = combine_blocks(objects_head)
    objects_clean = remove_comments(objects_raw)

    objects_parsed = {
        obj.split(" - ")[0].strip(" `"): obj.split(" - ")[1].strip(" `")
        for obj in objects_clean.split("\n")
        if obj.strip()
    }

    return objects_parsed


def parse_initial(llm_output: str) -> list[dict[str, str]]:
    """
    Extracts state (PDDL-init) from LLM response and returns it as a list of dict strings

    Parameters:
        llm_output (str): raw LLM output

    Returns:
        states (list[dict[str,str]]): list of initial states in dictionaries
    """
    initial_head = parse_heading(llm_output, "INITIAL")
    if initial_head.count("```") != 2:
        raise ValueError(
            "Could not find exactly one block in the initial section enclosed by [```, ```] of the LLM output. The initial states have to be specified in a single block Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task."
        )

    initial_raw = combine_blocks(initial_head)
    initial_clean = remove_comments(initial_raw)
    initial_parsed = parse_pddl(f"({initial_clean})")

    return parse_task_states(initial_parsed)


def parse_goal(llm_output: str) -> list[dict[str, str]]:
    """
    Extracts goal (PDDL-goal) from LLM response and returns it as a string

    Parameters:
        llm_output (str): raw LLM output

    Returns:
        states (list[dict[str,str]]): list of goal states in dictionaries
    """
    goal_head = parse_heading(llm_output, "GOAL")
    if goal_head.count("```") != 2:
        raise ValueError(
            "Could not find exactly one block in the goal section enclosed by [```, ```] of the LLM output. The goal states have to be specified in a single block Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task."
        )

    goal_raw = combine_blocks(goal_head)
    goal_clean = remove_comments(goal_raw)
    goal_parsed = parse_pddl(f"({goal_clean})")

    return parse_task_states(goal_parsed)


def parse_task_states(parsed_states: list) -> list[dict]:
    states = []
    for line in parsed_states:

        if isinstance(line, str):
            line = [line]

        # only parse lines
        if isinstance(line, list):
            name = line[0].split(" ")[0]

            if name == "and":
                continue
            # if comparsion operator
            if name in ["=", ">", "<", ">=", "<="]:
                op = name
                name = line[1][0].split(" ")[0]  # retrieve function name
                params = line[1][0].split(" ")[1:]
                value = line[2]
                states.append(
                    {"func_name": name, "params": params, "value": value, "op": op}
                )
                continue
            if name == "not":
                name = line[1][0].split(" ")[0]  # retrieve function name
                params = line[1][0].split(" ")[1:]
                neg = True
            else:
                neg = False
                params = line[0].split(" ")[1:]
            states.append({"pred_name": name, "params": params, "neg": neg})

    return states


# ---- SUPPLEMENTARY PARSING FUNCTIONS ----


def parse_heading(llm_output: str, heading: str) -> str:
    """Extract the text between the heading and the next second level heading in the LLM output."""
    if heading not in llm_output:
        raise ValueError(
            f"Could not find heading {heading} in the LLM output:\n{llm_output}\n. Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task"
        )
    heading_str = (
        llm_output.split(heading)[1].split("\n### ")[0].strip()
    )  # Get the text between the heading and the next heading
    return heading_str


def parse_pddl(pddl_str: str) -> list:
    """
    Simplified PDDL parser that converts the string into a nested list structure.
    """

    # tokenize string
    tokens = re.sub(r"([()])", r" \1 ", pddl_str).split()
    stack = []
    current = []

    # iterate each token, if it is a bracket, we append to new list
    for token in tokens:
        if token == "(":
            stack.append(current)
            current = []
        elif token == ")":
            if stack:
                parent = stack.pop()
                parent.append(current)
                current = parent
        else:
            current.append(token)

    if len(current) != 1:
        raise ValueError("Malformed PDDL expression")

    nested_pddl = concatenate_strings(current[0])
    return nested_pddl


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


def concatenate_strings(nested_list):
    """Helper function that concatenates strings within a list together."""
    if not isinstance(nested_list, list):
        return nested_list

    new_list = []
    i = 0
    while i < len(nested_list):
        current = nested_list[i]
        if isinstance(current, str):
            # start collecting consecutive strings
            concatenated = current
            j = i + 1
            while j < len(nested_list) and isinstance(nested_list[j], str):
                concatenated += " " + nested_list[j]
                j += 1
            new_list.append(concatenated)
            i = j
        else:
            # recurse for sublists
            new_list.append(concatenate_strings(current))
            i += 1
    return new_list

    # ---- SUPPLEMENTARY FUNCTIONS ----


# ---- HELPER FUNCTIONS ----


def load_file(file_path: str):
    """Helper function that loads a single file into a string"""
    _, ext = os.path.splitext(file_path)
    with open(file_path, "r") as file:
        if ext == ".json":
            return json.load(file)
        else:
            return file.read().strip()


def load_files(folder_path: str):
    """Helper function that loads multiple files into a string list"""
    file_contents = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                file_contents.append(file.read())
    return file_contents


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
