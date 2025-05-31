"""
This module contains a collection of PDDL syntax validation functions. Users MUST specify what validation checker
is being used in `error_types` when using an extraction function found in DomainBuilder/TaskBuilder class.

For instance:
    syntax_validator = SyntaxValidator()

    self.syntax_validator.error_types = [
                "validate_header",
                "validate_duplicate_headers",
                "validate_unsupported_keywords",
                "validate_params",
                "validate_types_predicates",
                "validate_format_predicates",
                "validate_usage_action",
            ]

Is supported in: DomainBuilder.extract_pddl_action(**kwargs, syntax_validator)
"""

import re
from collections import OrderedDict
from .pddl_format import *
from .pddl_parser import *
from .pddl_types import Predicate, Function


ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}

# constant declarations for PDDL types
LOGICAL_CONNECTIVES = {"and", "not", "or", "imply"}
QUANTIFIERS = {"forall", "exists"}
CONDITIONAL_EFFECTS = {"when"}

NUMERIC_OPERATORS = {"+", "-", "/", "*"}
COMPARISON_OPERATORS = {"=", ">", "<", ">=", "<="}
ASSIGNMENT_OPERATORS = {"assign", "increase", "decrease", "scale-up", "scale-down"}

# TODO: implement preferences and temporal features
TEMPORAL_KEYWORDS = {"at", "over", "start", "end"}
PREFERENCE_KEYWORDS = {
    "sometime-after",
    "sometime-before",
    "always-within",
    "hold-during",
    "hold-after",
    "at end",
}


class SyntaxValidator:
    def __init__(
        self,
        headers: list[str] | None = None,
        error_types: list[str] | None = None,
        unsupported_keywords: list[str] | None = None,
    ) -> None:
        """
        Initializes an L2P custom syntax validator checker object.

        Args:
            headers (list[str]): headers to extract from LLM output
            error_types (list[str]): error types to execute formalization functions
            unsupported_keywords (list[str]): keywords to check against LLM output
        """

        # assign default unsupported keywords
        default_unsupported = ["pddl", "lisp", "python"]

        self.headers = headers if headers else []
        self.error_types = error_types if error_types else []
        self.unsupported_keywords = (
            default_unsupported
            if unsupported_keywords is None
            else unsupported_keywords
        )

    # ---- PDDL TYPE CHECKS ----

    def validate_type(
        self,
        target_type: str,
        claimed_type: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Check if the claimed_type is valid for the target_type according to the type hierarchy.

        Args:
            target_type (str): type that is expected for the parameter (from :predicates)
            claimed_type (str): type that is provided in the LLM output PDDL.
            types (dict[str,str] | list[dict[str,str]]): current types in domain

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        # check if the claimed type matches the target type
        if claimed_type == target_type:
            feedback_msg = "[PASS]: claimed type matches target type definition."
            return True, feedback_msg

        types = format_types(types)  # flatten hierarchy

        # extract all types from the keys in the types dictionary
        all_types = set()
        for key in types.keys():
            main_type, *subtype = key.split(" - ")
            all_types.add(main_type.strip())
            if subtype:
                all_types.add(subtype[0].strip())

        # check if target type is not found in all types
        if target_type not in all_types:
            feedback_msg = f"[ERROR]: target type `{target_type}` is not found in :types definition: {all_types}."
            return False, feedback_msg

        # iterate through the types hierarchy to check if claimed_type is a subtype of target_type
        current_type = claimed_type
        while current_type in all_types:
            # find the key that starts with the current type

            parent_type_entry = next(
                (k for k in types.keys() if k.startswith(f"{current_type} - ")), None
            )

            if parent_type_entry:
                # extract the parent type from the key
                super_type = parent_type_entry.split(" - ")[1].strip()

                if super_type == target_type:
                    feedback_msg = (
                        "[PASS]: claimed type matches target type definition."
                    )
                    return True, feedback_msg
                current_type = super_type
            else:
                break

        feedback_msg = f"[ERROR]: claimed type `{claimed_type}` does not match target `{target_type}` or any of its possible sub-types."
        return False, feedback_msg

    def validate_format_types(
        self, types: dict[str, str] | list[dict[str, str]] | None = None
    ) -> tuple[bool, str]:
        """
        Checks if type variables contain `?` characters.

        Args:
            types (dict[str,str] | list[dict[str,str]]): types given from LLM output

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        if not types:
            return True, "[PASS]: no types provided."

        # flatten types if it is in a hierarchy
        types = format_types(types)

        invalid_types = []
        for t_name, _ in types.items():
            if t_name.startswith("?"):
                invalid_types.append(f" - {t_name}")

        if invalid_types:
            invalid_types_str = "\n".join(invalid_types)
            feedback_msg = (
                f"[ERROR]: There are type(s) with name(s) that start with character `?`. This is not allowed in PDDL."
                f"\n\nRemove `?` from the following type(s):\n"
                f"{invalid_types_str}"
                f"\n\nMake sure each entry defines a type using the format:"
                f"\n - <type_name>: <description of type>"
                f"\n\nFor example:"
                f"\n - car: a car that can drive"
            )
            return False, feedback_msg

        feedback_msg = "[PASS]: all types are formatted correctly."
        return True, feedback_msg

    def validate_cyclic_types(
        self,
        types: dict[str, str] | list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Checks if the types given contain an invalid cyclic hierarchy.

        Args:
            types (dict[str,str] | list[dict[str,str]]): current types in domain

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        def visit_type(node, visited, path, all_types):
            """Traverses the type hierarchy and check for cycles."""

            node_key = next(iter(node))  # get first type

            # if already visited this type, it indicates a cycle
            if node_key in visited:
                cycle_path = " -> ".join(path[path.index(node_key) :] + [node_key])
                violated_type = node_key  # type that caused the cycle
                return False, cycle_path, violated_type

            # mark the current type as visited in this traversal path
            visited.add(node_key)
            path.append(node_key)

            # visit all children of the current node
            if "children" in node:
                for child in node["children"]:
                    result, cycle, violated_type = visit_type(
                        child, visited, path, all_types
                    )
                    if not result:
                        return result, cycle, violated_type

            # also check if type appears elsewhere in the hierarchy with children
            for other_type in all_types:
                other_key = next(iter(other_type))
                if (
                    other_key == node_key
                    and "children" in other_type
                    and other_type["children"]
                ):
                    for child in other_type["children"]:
                        result, cycle, violated_type = visit_type(
                            child, visited, path, all_types
                        )
                        if not result:
                            return result, cycle, violated_type

            # remove the type from the visited once its children are processed
            visited.remove(node_key)
            path.pop()
            return True, "", ""

        # if no types are provided, return invalid
        if not types:
            return True, "[PASS]: no types provided."

        if isinstance(types, dict):
            types = [{k: v} for k, v in types.items()]

        # iterate over all the top-level types
        for type_node in types:
            visited = set()
            path = []  # This will keep track of the current path
            result, cycle, violated_type = visit_type(type_node, visited, path, types)
            if not result:
                feedback_msg = (
                    f"[ERROR]: Circular dependency detected in the type hierarchy: {cycle}"
                    f"\n\nThis means the type '{violated_type}' indirectly inherits from itself through the chain:"
                    f"\n - Starts with: '{path[0]}'"
                    f"\n - Leads back to: '{violated_type}' via: {cycle}"
                    f"\n\nThis creates an infinite loop in the type system where '{violated_type}' cannot be properly defined because its parent eventually depends on itself"
                    f"\n\nPossible Solutions:"
                    f"\n(1) Remove or modify one of the relationships in the cycle"
                    f"\n(2) Consider flattening your hierarchy if circular references are needed"
                )
                return False, feedback_msg

        feedback_msg = "[PASS]: Type hierarchy is valid."
        return True, feedback_msg

    def validate_constant_types(
        self,
        constants: dict[str, str] | None = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Checks if constant types are found in current :types.

        Args:
            constants (dict[str,str]): current :constants in domain
            types (dict[str,str] | list[dict[str,str]]): current types in domain

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        if not constants:
            return True, "[PASS]: no constants provided."

        types = format_types(types)  # flatten type hierarchy

        if types:
            for const_name, const_type in constants.items():
                if const_type not in types.keys():
                    return (
                        False,
                        f"[ERROR]: constant `{const_name}` contains type `{const_type}` "
                        f"that is not found in list of available types:\n"
                        f"{pretty_print_dict(types)}\n\n"
                        f"Make sure that constants only point to types that exist.",
                    )

        return True, "[PASS]: all constants are valid."

    # ---- PDDL FUNCTION CHECKS ----

    def validate_format_functions(
        self,
        functions: list[Function] | None = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Check for any PDDL syntax found within functions, allowing untyped variables.

        Args:
            functions (list[Function]): list of functions in domain
            types (dict[str,str] | list[dict[str,str]]): current types in domain

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        if not functions:
            return True, "[PASS]: no functions provided."

        # retrieve types
        valid_types = list()
        if types:
            types = format_types(types)
            valid_types = [
                type_key.split(" - ")[0].strip() for type_key in types.keys()
            ]
        else:
            valid_types = []

        all_invalid_params = []

        for func in functions:
            func_def = func["clean"]
            func_def = func_def.strip(" ()`")  # discard parentheses and similar

            # check if function name declared
            if func_def.startswith("?") or func["name"].startswith("?"):
                return (
                    False,
                    f"[ERROR]: Line `({func_def})` does not contain a function name. Function names must not start with `?`. Revise function to include name. For example: `(battery-level ?c - car)` where `battery-level` is the function name.",
                )

            split_function = func_def.split(" ")[1:]  # discard function name
            split_function = [e for e in split_function if e != ""]

            i = 0
            while i < len(split_function):
                f = split_function[i]
                # variable name must start with `?`
                if "?" not in f:
                    # catches random character declarations
                    if re.match(r"^[^\w\s]+$", f) or re.match(
                        r"^[^\w]", f
                    ):  # all non-word or starts with symbol
                        return (
                            False,
                            f"[ERROR]: For PDDL, function `({func_def})` appears to contain invalid or unexpected symbol `{f}`. This might be a parsing error or stray character. Make sure each parameter follows the format `?name - type`.",
                        )

                    raw_func = func["raw"]

                    feedback_msg = (
                        f"[ERROR]: For PDDL, there is a syntax issue in the function definition."
                        f"\n`{f}` appears where a variable is expected in function `{raw_func}`."
                        f"\n\nPossible causes:"
                        f"\n(1) `{f}` is intended to be a variable but is missing the `?` prefix. All variables must start with `?`, like `?block`."
                        f"\n(2) `{f}` is actually a type, in which case it should appear after a `-` in a declaration like `?x - {f}`."
                    )

                    return False, feedback_msg

                # check if variable is followed by `- type` or nothing (untyped)
                if i + 1 < len(split_function) and split_function[i + 1] == "-":
                    if i + 2 >= len(split_function):
                        return (
                            False,
                            f"[ERROR]: For PDDL, there is a missing type after the `-` for parameter `{f}` in function `{func_def}`. Make sure each parameter follows the format `?name - type`.",
                        )

                    param_obj_type = split_function[i + 2]

                    if param_obj_type not in valid_types:
                        all_invalid_params.append((param_obj_type, f, func_def))

                    i += 3  # skip ?var - type
                else:
                    # untyped variable
                    i += 1  # move to next token

        if all_invalid_params:
            feedback_msg = (
                "[ERROR]: For PDDL, there are invalid object types in the functions:"
            )
            for param_obj_type, f, func_def in all_invalid_params:
                feedback_msg += (
                    f"\n - `{param_obj_type}` for the parameter `{f}` in the definition of the function `{func_def}` "
                    + (
                        f"not found in types: {valid_types}."
                        if valid_types
                        else "contain types when no types are available."
                    )
                )
            feedback_msg += "\n\nRevise function parameters such that their types are assigned correctly. Otherwise leave variable untyped."
            return False, feedback_msg

        feedback_msg = "[PASS]: All functions are formatted correctly."
        return True, feedback_msg

    # ---- PDDL PREDICATE CHECKS ----

    def validate_types_predicates(
        self,
        predicates: list[Predicate] | None = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Check if predicate name is found within any type definitions.

        Args:
            predicates (list[Predicate]): current predicates in domain / generated from LLM
            types (dict[str,str] | list[dict[str,str]]): current types in domain

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        if not predicates:
            return True, "[PASS]: no predicates provided."

        # if types is None or empty, return true
        if not types:
            feedback_msg = "[PASS]: No types declared, all predicate names are unique."
            return True, feedback_msg

        types = format_types(types)

        invalid_predicates = list()
        for pred in predicates:

            for type_key in types.keys():
                # extract the actual type name, disregarding hierarchical or descriptive parts
                type_name = type_key.split(" - ")[0].strip()

                # check if the predicate name is exactly the same as the type name
                if pred["name"] == type_name:
                    invalid_predicates.append(pred)

        if invalid_predicates:
            feedback_msg = "[ERROR]: The following predicate(s) have the same name(s) as existing object types:"
            for pred_i, pred in enumerate(invalid_predicates):
                feedback_msg += f"\n{pred_i + 1}. `{pred['name']}` from {pred['clean']}"
            feedback_msg += f"\nRename these predicates that are unique from types: {list(types.keys())}"
            return False, feedback_msg

        feedback_msg = "[PASS]: All predicate names are unique to object type names"
        return True, feedback_msg

    def validate_duplicate_predicates(
        self,
        curr_predicates: list[Predicate] | None = None,
        new_predicates: list[Predicate] | None = None,
    ) -> tuple[bool, str]:
        """
        Checks if predicates have the same name but different parameters.

        Args:
            curr_predicates (list[predicate]): current predicates in domain
            new_predicates (list[Predicate]): new predicates generated from LLM

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        if not new_predicates:
            return True, "[PASS]: no new predicates were created."

        if curr_predicates:
            curr_pred_dict = {pred["name"]: pred for pred in curr_predicates}

            duplicated_predicates = list()
            for new_pred in new_predicates:
                name_lower = new_pred["name"]
                if name_lower in curr_pred_dict:
                    curr = curr_pred_dict[name_lower]

                    if len(curr["params"]) != len(new_pred["params"]) or any(
                        t1 != t2 for t1, t2 in zip(curr["params"], new_pred["params"])
                    ):
                        duplicated_predicates.append((new_pred["raw"], curr["raw"]))

            if duplicated_predicates:
                feedback_msg = "[ERROR]: Duplicate predicate name(s) found with mismatched parameters.\n"
                feedback_msg += "You have defined predicates with the same name as existing ones but with different parameters, which is not allowed.\n\n"
                feedback_msg += "Conflicting predicate definitions:\n"

                for i, (new_pred, existing_pred) in enumerate(duplicated_predicates, 1):
                    feedback_msg += (
                        f"{i}. New: {new_pred.replace(':', ';')}\n"
                        f"   Conflicts with existing: {existing_pred.replace(':', ';')}\n"
                    )

                feedback_msg += "\n\nIf you're trying to use the same concept, ensure the parameters match the existing definition exactly.\n"
                feedback_msg += "If this is a new concept, use a different predicate name to avoid confusion.\n"

                return False, feedback_msg

        return (
            True,
            "[PASS]: All predicates are uniquely named and consistently defined.",
        )

    def validate_overflow_predicates(
        self, llm_response: str, limit: int = 30
    ) -> tuple[bool, str]:
        """
        Checks if LLM output contains too many predicates in precondition/effects (based on users assigned limit).
        This error is very rare, but can occur. Thus, it is omitted in core functions but is still available.

        Args:
            llm_response (str): raw LLM output
            limit (int): max number of states declared, default to 30

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        spacer = "=" * 50

        assert "Preconditions" in llm_response, llm_response
        precond_str = llm_response.split("Preconditions")[1].split("```\n")[1].strip()
        precond_str = remove_comments(precond_str)
        num_prec_pred = len(precond_str.split("\n")) - 2  # for outer brackets
        if num_prec_pred > limit:
            feedback_msg = (
                f"[ERROR]: You seem to have generated an action model with an unusually long list of precondition predicates.\n\n"
                f"{spacer}\n"
                f"{precond_str}\n"
                f"{spacer}\n"
                f"Extracted predicates: {num_prec_pred}\n\n"
                f"Please include only the relevant preconditions and keep the action model concise or raise limit of predicates."
            )
            return False, feedback_msg

        eff_str = llm_response.split("Effects")[1].split("```\n")[1].strip()
        num_eff_pred = len(eff_str.split("\n")) - 2  # for outer brackets
        if num_eff_pred > limit:
            feedback_msg = (
                f"[ERROR]: You seem to have generated an action model with an unusually long list of effect predicates.\n\n"
                f"{spacer}\n"
                f"{eff_str}\n"
                f"{spacer}\n"
                f"Extracted predicates: {num_eff_pred}\n\n"
                f"Please include only the relevant effects and keep the action model concise or raise limit of predicates."
            )
            return False, feedback_msg

        feedback_msg = (
            f"[PASS]: predicate count satisfies limit of {limit}.\n"
            f"Approximate predicates in preconditions: {num_prec_pred}\n"
            f"Approximate Predicates in effects: {num_eff_pred}"
        )
        return True, feedback_msg

    def validate_format_predicates(
        self,
        predicates: list[dict] | None = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Checks for any PDDL syntactic errors found within predicates, allowing untyped variables.

        Args:
            predicates (list[Predicate]): current predicates in domain / generated from LLM
            types (dict[str,str] | list[dict[str,str]]): current types in domain

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        if not predicates:
            return True, "[PASS: no predicates provided."

        valid_types = list()
        # flatten type hierarchy if exists
        if types:
            types = format_types(types)
            valid_types = [
                type_key.split(" - ")[0].strip() for type_key in types.keys()
            ]
        else:
            valid_types = []

        all_invalid_params = []

        for pred in predicates:
            pred_def = pred["clean"]
            pred_def = pred_def.strip(" ()`")  # discard parentheses and similar

            # check if predicate name declared
            if pred_def.startswith("?") or pred["name"].startswith("?"):
                feedback_msg = f"[ERROR]: Line `({pred_def})` does not contain a predicate name. Predicate names must not start with `?`. Revise predicate to include name. For example: `(stack ?b - block ?t - table)` where `stack` is the predicate name."
                return False, feedback_msg

            split_predicate = pred_def.split(" ")[1:]  # discard the predicate name
            split_predicate = [e for e in split_predicate if e != ""]

            i = 0
            while i < len(split_predicate):
                p = split_predicate[i]
                # variable name must start with `?`
                if "?" not in p:
                    # catches random character declarations
                    if re.match(r"^[^\w\s]+$", p) or re.match(
                        r"^[^\w]", p
                    ):  # all non-word or starts with symbol
                        feedback_msg = f"[ERROR]: For PDDL, predicate `({pred_def})` appears to contain invalid or unexpected symbol `{p}`. This might be a parsing error or stray character. Make sure each parameter follows the format `?name - type`."
                        return False, feedback_msg

                    raw_pred = pred["raw"]

                    feedback_msg = (
                        f"[ERROR]: For PDDL, there is a syntax issue in the predicate definition."
                        f"\n`{p}` appears where a variable is expected in predicate `{raw_pred}`."
                        f"\n\nPossible causes:"
                        f"\n(1) `{p}` is intended to be a variable but is missing the `?` prefix. All variables must start with `?`, like `?block`."
                        f"\n(2) `{p}` is actually a type, in which case it should appear after a `-` in a declaration like `?x - {p}`."
                    )

                    return False, feedback_msg

                # check if variable is followed by `- type` or nothing (untyped)
                if i + 1 < len(split_predicate) and split_predicate[i + 1] == "-":
                    if i + 2 >= len(split_predicate):
                        feedback_msg = f"[ERROR]: For PDDL, there is a missing type after the `-` for parameter `{p}` in new predicate `{pred_def}`. Make sure each parameter follows the format `?name - type`."
                        return False, feedback_msg

                    param_obj_type = split_predicate[i + 2]

                    if param_obj_type not in valid_types:
                        all_invalid_params.append((param_obj_type, p, pred_def))

                    i += 3  # skip ?var - type
                else:
                    # untyped variable (just ?var)
                    i += 1  # move to next token

        if all_invalid_params:
            feedback_msg = (
                "[ERROR]: For PDDL, there are invalid object types in the predicates:"
            )
            for param_obj_type, p, pred_def in all_invalid_params:
                feedback_msg += (
                    f"\n - `{param_obj_type}` for the parameter `{p}` in the definition of the predicate `{pred_def}` "
                    + (
                        f"not found in types: {valid_types}."
                        if valid_types
                        else "contain types when no types are available."
                    )
                )
            feedback_msg += "\n\nRevise predicate parameters such that their types are assigned correctly. Otherwise leave variable untyped."
            return False, feedback_msg

        feedback_msg = "[PASS]: All predicates are formatted correctly."
        return True, feedback_msg

    # ---- PDDL ACTION CHECKS ----

    def validate_pddl_action(
        self,
        pddl: str,
        predicates: list[Predicate],
        action_params: OrderedDict,
        functions: list[Function] | None = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        part="preconditions",
    ) -> tuple[bool, str]:
        """
        Validates predicates and fluent expression in nested PDDL list format. There are many specific
        checks in this validation function. This is where LLMs encounter the most syntax errors.

        Performs three main kinds of checks:
            (i) if predicate / functions statements are misused or does not exist
            (ii) if state parameters align with original definition parameters
            (iii) if arguments are being passed correctly (i.e. conditional-effects)

        Args:
            pddl (str): part of PDDL section from LLM
            predicates (list[Predicate]): current predicates in domain / generated from LLM
            action_params (OrderedDict): PDDL parameters of current action
            functions (list[Function]): list of current functions in domain
            types (dict[str,str] | list[dict[str,str]]): current types in domain
            part (str): section of the PDDL to focus on

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        pddl = parse_pddl(pddl)  # parse into nested list

        # retrieve dict comprehension for each predicate/function
        pred_index = {pred["name"]: pred for pred in predicates}
        func_index = {func["name"]: func for func in functions} if functions else {}

        def get_ordinal_suffix(num):
            """Helper function that appends parameter index to string"""
            return (
                ORDINAL_SUFFIXES.get(num % 10, "th")
                if num % 100 not in (11, 12, 13)
                else "th"
            )

        def is_value(s):
            """Helper function that checks if a string is numeric"""
            try:
                float(s)
                return True
            except ValueError:
                return False

        def split_var_type_pairs(raw_list):
            """Helper function that for parsing typed variable declarations in PDDL-like syntax."""
            if not raw_list:
                return []
            tokens = raw_list[0].split()
            result, current_vars = [], []
            for i, token in enumerate(tokens):
                if token == "-":
                    continue
                if current_vars and i > 0 and tokens[i - 1] == "-":
                    var_type = token
                    result.extend(f"{v} - {var_type}" for v in current_vars)
                    current_vars = []
                else:
                    current_vars.append(token)
            result.extend(current_vars)
            return result

        def validate_term(node, term, scoped_params):
            """Recursive function that validates a term which may be a function or value."""

            # if nested expression
            if isinstance(term, list):
                # (1) if term is a numeric operator
                head_parts = term[0].strip().split()
                head = head_parts[0]

                if head in NUMERIC_OPERATORS:
                    terms_ = term[0].split(" ") + term[1:]

                    if len(terms_) != 3:
                        return (
                            False,
                            f"[ERROR]: Numeric operator `{head}` requires two arguments: {format_pddl_expr(terms_)}\n\n"
                            f"Parsed line: {format_pddl_expr(node)}\n\n"
                            f"Make sure that only fluents are used and not variables.",
                        )

                    for arg in term[1:]:
                        valid, msg = validate_term(node, arg, scoped_params)
                        if not valid:
                            return False, msg
                    return True, "[PASS]"

                # (2) if term is a function or nested expression
                func_ = term[0].split(" ")
                func_name = func_[0]
                func_args = func_[1:]

                # checks if function name in pddl not found in :function list
                if func_name not in func_index:

                    # extra catch - could be predicate but used as a function!
                    if func_name in pred_index:
                        return (
                            False,
                            f"[ERROR]: `{func_name}` is a predicate, not a function. Predicates cannot be used to return numeric values but instead as standalone conditions.\n\n"
                            f"Parsed line: {format_pddl_expr(node)}\n\n"
                            f"Predicates can only be used within a boolean context—that is, they must appear inside logical expressions like `and`, `or`, `not`, `when`, or standalone conditions in PDDL actions.",
                        )

                    available_funcs = (
                        (
                            " - "
                            + "\n - ".join(
                                [f"{f['name']}: {f['desc']}" for f in functions]
                            )
                        )
                        if functions
                        else "No functions available."
                    )

                    available_funcs = (
                        "List of available function(s) are:\n - "
                        + "\n - ".join([f["raw"] for f in functions])
                        if functions
                        else "Numeric fluents cannot be used because no functions are declared in the :functions section of the domain."
                    )
                    return False, (
                        f"[ERROR]: Undeclared function `({func_name})` found in {part}.\n\n"
                        f"{available_funcs}\n\n"
                        f"Parsed line: {format_pddl_expr(node)}"
                    )

                # retrieve target function arguments
                target_func = func_index[func_name]
                expected_args = list(target_func["params"].keys())
                expected_types = list(target_func["params"].values())

                # checks if function arguments align with :function list
                if len(func_args) != len(expected_args):
                    return (
                        False,
                        f"[ERROR]: Function `{target_func['clean']}` expects {len(expected_args)} variable parameter(s), "
                        f"but found {len(term[1:])} in {part}.\n\nParsed line: ({format_pddl_expr(node)})",
                    )

                # recursively checks if function argument is nested
                for i, arg in enumerate(func_args):
                    if isinstance(arg, list):
                        valid, msg = validate_term(node, arg, scoped_params)
                        if not valid:
                            return False, msg
                    else:
                        # leaf of function call
                        expected_type = expected_types[i]
                        actual_type = scoped_params.get(arg)
                        if expected_type and actual_type:

                            # checks if :types exists
                            if not types:
                                return (
                                    False,
                                    f"[ERROR]: Types declared but type dictionary is empty.",
                                )

                            # validates if variable type aligns with target :type
                            flag, _ = self.validate_type(
                                expected_type, actual_type, types
                            )
                            if not flag:
                                param_number = i + 1
                                suffix = get_ordinal_suffix(param_number)
                                return (
                                    False,
                                    f"[ERROR]: The {param_number}{suffix} parameter of `{target_func['clean']}` "
                                    f"should be of type `{expected_type}`, but `{arg}` is `{actual_type}`\n\n"
                                    f"Parsed line: {format_pddl_expr(term)}\n"
                                    f"Found in: {format_pddl_expr(node)}",
                                )

                        # checks if variable is in the scope of available variables
                        if arg not in scoped_params and not is_value(arg):
                            return (
                                False,
                                f"[ERROR]: Argument `{arg}` not found in scope for function `{func_name}`.\nScope: {list(scoped_params.keys())}\n\n"
                                f"Parsed line: {format_pddl_expr(term)}\n"
                                f"Found in: {format_pddl_expr(node)}\n\n"
                                f"Make sure the function uses the correct variables from its scope. Otherwise, you may need to add a new variable to the parameters section.",
                            )
                return True, "[PASS]"

            # if not nested expression, it is a variable or constant
            else:
                if term not in scoped_params and not is_value(term):
                    return (
                        False,
                        f"[ERROR]: Variable `{term}` not in scope: {list(scoped_params.keys())}",
                    )
                return True, "[PASS]"

        def traverse(node, scoped_params):
            """Recursive function that goes through nested list."""

            # if reached end node with no errors, return true
            if isinstance(node, str) or not isinstance(node, list) or len(node) == 0:
                return True, "[PASS]"

            keyword = node[0].split(" ")[0]  # extract keyword from node

            # (1) if keyword is a logical connective (and, not, or)
            if keyword in LOGICAL_CONNECTIVES:

                # recursively branch child nodes
                children = node[1:] if keyword != "not" else [node[1]]
                for child in children:
                    valid, msg = traverse(child, scoped_params)
                    if not valid:
                        return False, msg
                return True, "[PASS]"

            # (2) if keyword is a quantifier (forall, exists)
            if keyword in QUANTIFIERS:

                # validates correct arguments provided into quantifier
                if len(node) != 3:
                    return (
                        False,
                        f"[ERROR]: malformed usage of `{keyword}` statement. There should be 3 main arguments, but {len(node)} was given."
                        f"\nMake sure to adhere to valid PDDL syntax. For example: `({keyword} (<variable_list>) (<logical_expression(s)>))`\n\n"
                        f"Parsed line: {format_pddl_expr(node)}\n\n"
                        f"Possible solutions:\n"
                        f"  (1) Always wrap the list of variables with their types in parentheses, even for a single variable.\n"
                        f"  (2) The second argument of {keyword} should be a valid condition, or a use of `and/or` expression to wrap multiple conditions.",
                    )

                # parse quantified variable declarations
                param_spec = split_var_type_pairs(node[1])
                body = node[2]
                new_scope = scoped_params.copy()
                for spec in param_spec:
                    parts = spec.split(" ")
                    var = parts[0]
                    var_type = parts[2] if len(parts) >= 3 and parts[1] == "-" else ""

                    # validate if type exists in :types
                    if not var_type or var_type not in types:
                        return False, (
                            f"[ERROR]: Unknown type declared `{var_type}` for `{var}` in quantifier `{keyword}`\n\n"
                            f"Parsed line: {format_pddl_expr(node)}\n\n"
                            f"Make sure that the PDDL actions do not contain any type declarations. For example: `(drive ?c)` is correct, but `(drive ?c - car)` is invalid"
                        )
                    new_scope[var] = var_type  # update variable scope environment
                return traverse(body, new_scope)

            # (3) if keyword is a conditional effect (when)
            if keyword in CONDITIONAL_EFFECTS:
                # ensures conditional effects only found in :effects
                if part != "effects":
                    return (
                        False,
                        f"[ERROR]: `{keyword}` is only allowed in the :effects section, but found in {part}.\n\n"
                        f"Parsed line: {format_pddl_expr(node)}",
                    )

                if len(node) != 3:
                    return (
                        False,
                        f"[ERROR]: malformed usage of `{keyword}` statement. There should be 3 main arguments, but {len(node)} was given."
                        f"\nMake sure to adhere to valid PDDL syntax. For example: `({keyword} (<condition(s)>) (<effect(s)>))`\n\n"
                        f"Parsed line: {format_pddl_expr(node)}\n\n"
                        f"Possible solutions:\n"
                        f"  (1) `{keyword}` must have two arguments: `(when <condition> <effect>)` where both condition and effect must be wrapped in parentheses.\n"
                        f"  (2) If there are multiple conditions or effects, they must use a logical connective like `and` expression to wrap arguments.",
                    )
                valid, msg = traverse(node[1], scoped_params)
                if not valid:
                    return False, f"Invalid condition in 'when': {msg}"
                valid, msg = traverse(node[2], scoped_params)
                if not valid:
                    return False, f"Invalid effect in 'when': {msg}"
                return True, "[PASS]"

            # (4) if keyword is a numeric-fluent operator
            if (
                keyword
                in COMPARISON_OPERATORS | NUMERIC_OPERATORS | ASSIGNMENT_OPERATORS
            ):
                # ensures assignment operators only found in :effects
                if part != "effects" and keyword in ASSIGNMENT_OPERATORS:
                    return (
                        False,
                        f"[ERROR]: `{keyword}` is only allowed in the :effects section, but found in {part}.\n\n"
                        f"Parsed line: {format_pddl_expr(node)}",
                    )

                # check if equality is doing object comparison
                if keyword == "=":
                    parts = node[0].split(" ")
                    if len(parts) == 3:
                        arg_1, arg_2 = parts[1], parts[2]

                        if arg_1.startswith("?") and arg_2.startswith("?"):
                            for arg in (arg_1, arg_2):
                                if arg not in scoped_params:
                                    return (
                                        False,
                                        f"[ERROR]: variable `{arg}` not found in scope of the {part} section. "
                                        f"Available variables in scope: {list(scoped_params.keys())}\n\n"
                                        f"Parsed line: {format_pddl_expr(node)}\n\n"
                                        f"Possible solutions:\n"
                                        f"  (1) Revise line to only use variables in scope.\n"
                                        f"  (2) If necessary, create new variables in the parameters or consider using quantifiers.",
                                    )

                            # check if the arguments are the same type
                            if scoped_params[arg_1] != scoped_params[arg_2]:
                                return (
                                    False,
                                    f"[ERROR]: invalid object equality usage in the {part} section.\nVariable `{arg_1}` points to type `{scoped_params[arg_1]}`. However, variable `{arg_2}` points to type `{scoped_params[arg_2]}`\n\n"
                                    f"Parsed line: {format_pddl_expr(node)}\n\n"
                                    f"Make sure that when using object equality, the variables share the same types.",
                                )

                            return True, "[PASS]"

                if len(node) != 3:
                    return (
                        False,
                        f"[ERROR]: `{keyword}` operator requires exactly two arguments.\n\n"
                        f"Parsed line: {format_pddl_expr(node)}\n\n"
                        f"Verify that the expression has exactly two arguments in the form: ({keyword} (<arg1>) (<arg2>)),\n"
                        f"where both <arg1> and <arg2> can be numeric constants (e.g., 5, 3.14, -2)\n"
                        f"or numeric function expressions (e.g., (total-cost), (fuel-level ?v)).",
                    )
                for term in node[1:]:
                    # traverse terms of operator statement
                    valid, msg = validate_term(node, term, scoped_params)
                    if not valid:
                        return False, msg

                return True, "[PASS]"

            # (5) if keyword is a predicate
            pred_name = keyword
            pred_args = node[0].split(" ")[1:]

            # validate if predicate is found in :predicates
            if pred_name not in pred_index:

                # extra catch - could be function but used as a predicate!
                if pred_name in func_index:
                    return (
                        False,
                        f"[ERROR]: `{pred_name}` is a function, not a predicate. Functions return numeric values and cannot be used as standalone conditions.\n\n"
                        f"Parsed line: {format_pddl_expr(node)}\n\n"
                        f"Functions can only be used within a numeric context such as comparison (e.g., `(= (<function_name> <?var>) <numeric_expression>)`)",
                    )

                available_preds = (
                    (
                        " - "
                        + "\n - ".join(
                            [f"{p['name']}: {p['desc']}" for p in predicates]
                        )
                    )
                    if predicates
                    else "No predicates available."
                )

                return (
                    False,
                    f"[ERROR]: Undeclared predicate `({pred_name})` found in {part}.\n\n"
                    f"Available predicates:\n{available_preds}\n\n"
                    f"Parsed line: {format_pddl_expr(node)}",
                )

            # retrieve target predicate arguments
            target_pred = pred_index[pred_name]
            expected_args = list(target_pred["params"].keys())
            expected_types = list(target_pred["params"].values())

            # checks if predicate arguments align with :predicates list
            if len(pred_args) != len(expected_args):

                return (
                    False,
                    f"[ERROR]: Predicate `{target_pred['clean']}` expects {len(expected_args)} parameters, "
                    f"but found {len(pred_args)} in {part}.\n\nParsed line: {format_pddl_expr(node)}\n\n"
                    f"Make sure that only variables (i.e. `?var`) is being passed through a predicate argument and not its type. For example: `(pred_name ?var)` is correct; `(pred_name ?var - object)` is incorrect.",
                )

            for i, arg in enumerate(pred_args):
                # recursively checks if predicate argument is nested
                if isinstance(arg, list):
                    valid, msg = validate_term(node, arg, scoped_params)
                    if not valid:
                        return False, msg
                else:
                    # checks if parameter variables are found in current scope
                    if arg not in scoped_params:
                        return (
                            False,
                            f"[ERROR]: Variable `{arg}` not found in scope for predicate `{pred_name}`.\n\nAvailable variables in scope: {list(scoped_params.keys())}\n\n"
                            f"Parsed line: {format_pddl_expr(node)}\n\n"
                            f"Make sure the predicate uses the correct variables from its scope. Otherwise, you may need to add a new variable to the parameters section.",
                        )

                    # check if type of variable matches predicate type
                    expected_type = expected_types[i]
                    actual_type = scoped_params[arg]
                    if expected_type and actual_type:

                        # if types is empty and it calls a type
                        if not types:
                            return (
                                False,
                                f"[ERROR]: Types declared in predicate, but types list is empty.",
                            )
                        flag, _ = self.validate_type(expected_type, actual_type, types)
                        if not flag:
                            suffix = get_ordinal_suffix(i + 1)
                            return (
                                False,
                                f"[ERROR]: The {i+1}{suffix} parameter of `{target_pred['clean']}` "
                                f"should be of type `{expected_type}`, but `{arg}` is `{actual_type}`\n\n"
                                f"Parsed line: {format_pddl_expr(node)}",
                            )

            return True, "[PASS]"

        # resurively invoked function to branch nested list
        return traverse(pddl, scoped_params=action_params.copy())

    def validate_params(
        self,
        parameters: OrderedDict,
        types: dict[str, str] | list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Checks whether a PDDL action parameter is correctly
        formatted and type declaration assigned correctly.

        Args:
            parameters (OrderedDict): parameters of an action
            types (dict[str,str] | list[dict[str,str]]): current types in domain

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        # check if parameter names (i.e. ?a) contains '?'
        invalid_param_names = list()
        for param_name, param_type in parameters.items():
            if not param_name.startswith("?"):
                invalid_param_names.append(f"{param_name} - {param_type}")

        if invalid_param_names:
            feedback_msg = (
                f"[ERROR]: Character `?` is not found in parameter(s) `{invalid_param_names}` "
                "Please insert `?` in front of the parameter names (i.e. ?boat - vehicle)"
            )
            return False, feedback_msg

        # catch if dash is used but type is missing
        missing_type_after_dash = list()
        for param_name, param_type in parameters.items():
            if "-" in param_name:
                parts = param_name.split("-")
                if len(parts) < 2 or parts[1].strip() == "":
                    missing_type_after_dash.append(param_name)

        if missing_type_after_dash:
            feedback_msg = (
                f"[ERROR]: One or more parameters use `-` but do not specify a type: {missing_type_after_dash}. "
                "Each parameter using `-` must be followed by a valid type (e.g., `?c - car`)."
            )
            return False, feedback_msg

        types = format_types(types)
        # if no types are defined, check if parameters contain types
        if not types:
            for param_name, param_type in parameters.items():
                if param_type is not None and param_type != "":
                    feedback_msg = (
                        f"[ERROR]: The parameter `{param_name}` has an object type `{param_type}` "
                        "while no types are defined. Please remove the object type from this parameter."
                    )
                    return False, feedback_msg

            # if all parameter names do not contain a type
            return True, "[PASS]: All parameters are valid."

        # otherwise check that parameter types are valid in the given types
        else:
            for param_name, param_type in parameters.items():

                if not any(param_type in t for t in types.keys()):
                    feedback_msg = (
                        f"[ERROR]: There is an invalid object type `{param_type}` for the parameter `{param_name}` not found in the types {list(types.keys())} found in the action parameters section. Parameter types should align with the provided types, otherwise just leave parameter untyped.\n\n"
                        f"Make sure each line defines a parameter using the format: "
                        f"`?<parameter_name> - <type_name>: <description of parameter>`\n\n"
                        f"For example:\n"
                        f"`?c - car: a car that can drive`\n"
                        f"or `?c: a car that can drive` if not using a type"
                    )
                    return False, feedback_msg

            feedback_msg = "[PASS]: All parameter types found in object types."
            return True, feedback_msg

    def validate_usage_action(
        self,
        llm_response: str,
        curr_predicates: list[Predicate] | None = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        functions: list[Function] | None = None,
        extract_new_preds: bool = False,
    ) -> tuple[bool, str]:
        """
        Higher level function that performs checks over whether the predicates/functions are used in a
        valid way in the PDDL action. Invokes `validate_pddl_action` to perform deep syntax checks.

        Args:
            llm_response (str): raw LLM output
            curr_predicates (list[predicate]): current predicates in domain
            types (dict[str,str] | list[dict[str,str]]): current types in domain
            functions (list[Function]): list of current functions in domain
            extract_new_preds (bool): flag for if new predicates are being extracted, defaults to False

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        # parse predicates
        if extract_new_preds:
            new_predicates = parse_new_predicates(llm_response)
        else:
            new_predicates = []

        if curr_predicates is None:
            curr_predicates = new_predicates
        else:
            curr_predicates.extend(new_predicates)
        curr_predicates = parse_predicates(curr_predicates)

        # get action params
        params_info = parse_params(llm_response)

        # check preconditions
        precond_str = parse_preconditions(llm_response)
        precond_str = remove_comments(precond_str)
        precond_str = (
            precond_str.replace("\n", " ").replace("(", " ( ").replace(")", " ) ")
        )
        validation_info = self.validate_pddl_action(
            pddl=precond_str,
            predicates=curr_predicates,
            action_params=params_info[0],
            functions=functions,
            types=types,
            part="preconditions",
        )

        if not validation_info[0]:
            return validation_info

        # check effects
        eff_str = parse_effects(llm_response)
        eff_str = remove_comments(eff_str)
        eff_str = eff_str.replace("\n", " ").replace("(", " ( ").replace(")", " ) ")
        return self.validate_pddl_action(
            pddl=eff_str,
            predicates=curr_predicates,
            action_params=params_info[0],
            functions=functions,
            types=types,
            part="effects",
        )

    # ---- PDDL TASK CHECKS ----

    def validate_task_objects(
        self,
        objects: dict[str, str],
        types: dict[str, str] | list[dict[str, str]] | None = None,
    ) -> tuple[bool, str]:
        """
        Checks if task objects are declared correctly. Performs the following cases:
            (i) if object type is found within list of available types
            (ii) if object name is the same as a type (invalid)

        Args:
            objects (dict[str,str]): task objects generated from LLM
            types (dict[str,str] | list[dict[str,str]]): current types in domain

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        # if type hierarchy format
        if isinstance(types, list):
            types = format_types(types)

        type_keys = types.keys() if types else []

        for obj_name, obj_type in objects.items():

            obj_type_found = False

            for type_key in type_keys:
                # Try to split into current_type and parent_type if possible
                if " - " in type_key:
                    current_type, parent_type = type_key.split(" - ")
                else:
                    current_type = type_key
                    parent_type = None

                # Match object type to current or parent
                if obj_type == current_type or (
                    parent_type and obj_type == parent_type
                ):
                    obj_type_found = True

                # Check for name conflict with types
                if obj_name == current_type:
                    parsed_line = f"{obj_name} - {obj_type}"
                    return (
                        False,
                        f"[ERROR]: Object variable '{obj_name}' matches the type name '{current_type}', change it to be unique from types: {list(type_keys)}\n"
                        f"Violated object declaration: ({parsed_line if obj_type else obj_name})\n",
                    )

                if parent_type and obj_name == parent_type:
                    parsed_line = f"{obj_name} - {obj_type}"
                    return (
                        False,
                        f"[ERROR]: Object variable '{obj_name}' matches the type name '{parent_type}', change it to be unique from types: {list(type_keys)}\n"
                        f"Violated object declaration: ({parsed_line if obj_type else obj_name})\n",
                    )

            # if object does not contain a type
            if not obj_type:
                continue

            if not obj_type_found:
                message = (
                    f"not found in types: {list(type_keys)}"
                    if type_keys
                    else "but there are no types declared."
                )
                return (
                    False,
                    f"[ERROR]: Object variable '{obj_name}' has an invalid type '{obj_type}' {message}\n"
                    f"If an object variable requires a type, it must be assigned to the given types. If there are no types, leave object variable untyped.\n\n"
                    f"Parsed line: ({obj_name} - {obj_type})\n",
                )

        feedback_msg = "[PASS]: All objects are valid."
        return True, feedback_msg

    def validate_task_states(
        self,
        states: list[dict[str, str]],
        objects: dict[str, str],
        predicates: list[Predicate],
        functions: list[Function] | None = None,
        state_type: str = "initial",
    ) -> tuple[bool, str]:
        """
        Checks if task states are declared correcly. Performs following checks:
            (i) if predicates/functions in states exist in the domain
            (ii) if all object variables in states are declared in task objects
            (iii) if types of object variables match predicate parameter types

        Args:
            states (list[dict[str,str]]): a list of dictionaries of the states
            objects (dict[str,str]): a dictionary of the task objects and their types
            predicates (list[Predicate]): current predicates in domain
            functions (list[Function]): list of current functions in domain
            state_type (str): optional; 'initial' or 'goal' to label messages

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        for state in states:

            # new check: function states
            state_name = state.get("func_name", None)
            if state_name:
                state_params = state["params"]
                state_val = state["value"]
                state_op = state["op"]

                # (i) check if function name exists in domain functions
                if not functions:
                    functions = []
                function_names = [f["name"] for f in functions]
                if state_name not in function_names:
                    function_names_str = (
                        function_names if functions else "No functions provided"
                    )
                    return (
                        False,
                        f"[ERROR]: In the {state_type} state, '({state_op} ({state_name} {' '.join(state_params)}) {state_val})' "
                        f"uses function '{state_name}', which is not defined in the domain: {function_names_str}.\n\n"
                        f"If there are no functions, do not include this state.",
                    )

                # (ii) Check if all parameters exist in the task objects
                missing_params = [p for p in state_params if p not in objects]
                if missing_params:
                    return (
                        False,
                        f"[ERROR]: In the {state_type} state, '({state_name} {' '.join(state_params)})' "
                        f"contains parameter(s) {missing_params} not found in task objects {list(objects.keys())}.",
                    )

                # (iii) Check if object types match expected predicate types
                target_func = next(f for f in functions if f["name"] == state_name)
                expected_types = list(target_func["params"].values())
                actual_types = [objects[param] for param in state_params]
                actual_name_type_pairs = [
                    f"'{param}' is type: '{objects[param]}'" for param in state_params
                ]

                if expected_types != actual_types:
                    return (
                        False,
                        f"[ERROR]: In the {state_type} state, '({state_op} ({state_name} {' '.join(state_params)}) {state_val})' has mismatched types.\n\n"
                        f"Declared objects: {[f'{obj_name} - {obj_type}' for obj_name, obj_type in objects.items()]}\n\n"
                        f"Function `{target_pred['clean']}` expects type(s): {expected_types}\n"
                        f"Parsed line: ({state_op} ({state_name} {' '.join(state_params)}) {state_val}) contains type(s): {actual_types}, where [{', '.join(actual_name_type_pairs)}]\n\n"
                        f"Revise the state such that their parameter types align with the original function definition types.",
                    )

            else:
                state_name = state["pred_name"]
                state_params = state["params"]

                # (i) Check if predicate name exists in domain predicates
                predicate_names = [p["name"] for p in predicates]
                if state_name not in predicate_names:
                    return (
                        False,
                        f"[ERROR]: In the {state_type} state, '({state_name} {' '.join(state_params)})' "
                        f"uses predicate '{state_name}', which is not defined in the domain ({predicate_names}).",
                    )

                # (ii) Check if all parameters exist in the task objects
                missing_params = [p for p in state_params if p not in objects]
                if missing_params:
                    return (
                        False,
                        f"[ERROR]: In the {state_type} state, '({state_name} {' '.join(state_params)})' "
                        f"contains parameter(s) {missing_params} not found in task objects {list(objects.keys())}.",
                    )

                # (iii) Check if object types match expected predicate types
                target_pred = next(p for p in predicates if p["name"] == state_name)
                expected_types = list(target_pred["params"].values())
                actual_types = [objects[param] for param in state_params]
                actual_name_type_pairs = [
                    f"'{param}' is type: '{objects[param]}'" for param in state_params
                ]

                if expected_types != actual_types:
                    return (
                        False,
                        f"[ERROR]: In the {state_type} state, '({state_name} {' '.join(state_params)})' has mismatched types.\n\n"
                        f"Declared objects: {[f'{obj_name} - {obj_type}' for obj_name, obj_type in objects.items()]}\n\n"
                        f"Predicate `{target_pred['clean']}` expects type(s): {expected_types}\n"
                        f"Parsed line: ({state_name} {' '.join(state_params)}) contains type(s): {actual_types}, where [{', '.join(actual_name_type_pairs)}]\n\n"
                        f"Revise the state predicates to match the original predicate parameter types. Do not include type annotations in the predicate—e.g., use (drive ?c), not (drive ?c - car).",
                    )

        feedback_msg = "[PASS]: All task states are valid."
        return True, feedback_msg

    # ---- COMMON CHECKS ----

    def validate_header(
        self,
        llm_response: str,
    ) -> tuple[bool, str]:
        """
        Checks if domain headers and formatted code block syntax are found in LLM output.
        Headers to check must be declared as `self.headers = ['Action Preconditions']`

        Args:
            llm_response (str): raw LLM output

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        # catches if a header is not present
        for header in self.headers:
            if header not in llm_response:
                feedback_msg = (
                    f"[ERROR]: The header `{header}` is missing in the PDDL model. Please include the `### {header}` section following by its content enclosed by [```] like:\n\n"
                    f"### {header}\n"
                    f"```\n"
                    f"[CONTENT]\n"
                    f"```"
                )
                return False, feedback_msg

            # catches if the section does not contain correct code block format
            if llm_response.split(f"{header}")[1].split("###")[0].count("```") < 2:
                feedback_msg = (
                    f"[ERROR]: The header `{header}` contains an incorrect formalised code block. Please include the `### {header}` section following by its content enclosed by [```] like:\n\n"
                    f"### {header}\n"
                    f"```\n"
                    f"[CONTENT]\n"
                    f"```"
                    f"\n\n Source of error may be that keyword `{header}` was used somewhere else. Make sure the header is only stated once in your response."
                )
                return False, feedback_msg

        feedback_msg = "[PASS]: headers are identified properly in LLM output."
        return True, feedback_msg

    def validate_duplicate_headers(
        self,
        llm_response: str,
    ) -> tuple[bool, str]:
        """
        Checks if the LLM attempts to create a new action (so two or more actions defined in the same response).
        Headers to check must be declared as `self.headers = ['Action Preconditions']`

        Args:
            llm_response (str): raw LLM output

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        invalid = False
        duplicate_headers = []

        for header in self.headers:
            count = llm_response.count(f"### {header}")
            if count > 1:
                invalid = True
                duplicate_headers.append({"header": f"`### {header}`", "count": count})

        if invalid:
            feedback_msg = (
                "[ERROR]: Detected multiple definitions of the following header(s):\n"
                + "\n".join(
                    f"- {item['header']} found {item['count']} times"
                    for item in duplicate_headers
                )
                + "\n\nEach header section should only be declared once in your response for: "
                + ", ".join(item["header"] for item in duplicate_headers)
                + "."
            )
            return False, feedback_msg

        return True, "[PASS]: no duplicate sections created."

    def validate_unsupported_keywords(self, llm_response: str) -> tuple[bool, str]:
        """
        Checks whether PDDL model uses unsupported logic keywords
        Unsupported keywords to check must be declared as `self.unsupported_keywords = ['lisp']`

        Args:
            llm_response (str): raw LLM output

        Returns:
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        if not self.unsupported_keywords:
            feedback_msg = "[PASS]: No unsupported keywords declared."
            return True, feedback_msg

        for key in self.unsupported_keywords:
            if f"{key}" in llm_response:
                feedback_msg = f"[ERROR]: The PDDL model contains the keyword `{key}`. Revise the model so that it does not use this keyword."
                return False, feedback_msg

        feedback_msg = "[PASS]: Unsupported keywords not found in PDDL model."
        return True, feedback_msg
