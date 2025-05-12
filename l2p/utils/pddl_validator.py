"""
This file contains collection of functions PDDL syntax validations. Users MUST specify what validation checker
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
                "validate_usage_predicates",
            ]

Is supported in:
    DomainBuilder.extract_pddl_action(**kwargs, syntax_validator)
"""

from collections import OrderedDict
from .pddl_parser import parse_params, parse_new_predicates, parse_predicates, parse_pddl
from .pddl_types import Predicate
from .pddl_format import format_types, remove_comments
import re

from typing import List, OrderedDict, Tuple, Union, Optional, Dict

class SyntaxValidator:
    def __init__(
            self, 
            headers: list[str] | None = None,
            error_types: list[str] | None = None,
            unsupported_keywords: list[str] | None = None
            ) -> None:

        default_unsupported = [
            "forall", "when",
            "exists", "implies",
            "pddl", "lisp", "python"
        ]
        
        default_headers = [
            'Action Parameters', 
            'Action Preconditions', 
            'Action Effects', 
            'New Predicates',
        ]
        
        self.PDDL_KEYWORDS = {
            "and", "or", "not", "when", "imply",
            "exists", "forall", "either",
            # "increase", "decrease", "assign", "scale-up", "scale-down",
            # "=", "<", ">", "<=", ">=",
            # "number", "object", "total-cost",
        }
        
        self.error_types = error_types
        self.unsupported_keywords = (
            default_unsupported
            if unsupported_keywords is None
            else unsupported_keywords
        )
        self.headers = default_headers if headers is None else headers

    # ---- PDDL PARAMETER CHECKS ----

    def validate_params(
        self, 
        parameters: OrderedDict, 
        types: dict[str, str] | list[dict[str,str]] | None = None
    ) -> tuple[bool, str]:
        """Checks whether a PDDL action parameter is correctly formatted and type declaration assigned correctly."""

        # check if parameter names (i.e. ?a) contains '?'
        invalid_param_names = list()
        for param_name, param_type in parameters.items():
            if not param_name.startswith("?"):
                invalid_param_names.append(f"{param_name} - {param_type}")
        
        if invalid_param_names:
            feedback_msg = (
                f'[ERROR]: Character `?` is not found in parameter(s) `{invalid_param_names}` '
                'Please insert `?` in front of the parameter names (i.e. ?boat - vehicle)'
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
                        f'[ERROR]: The parameter `{param_name}` has an object type `{param_type}` '
                        'while no types are defined. Please remove the object type from this parameter.'
                    )
                    return False, feedback_msg
            
            # if all parameter names do not contain a type
            return True, "[PASS]: All parameters are valid."

        # otherwise check that parameter types are valid in the given types
        else:
            for param_name, param_type in parameters.items():

                if not any(param_type in t for t in types.keys()):
                    feedback_msg = (
                        f"[ERROR]: There is an invalid object type `{param_type}` for the parameter `{param_name}` not found in the types {list(types.keys())}. Parameter types should align with the provided types, otherwise just leave parameter untyped."
                        f"\n\nMake sure each line defines a parameter using the format:"
                        f"\n?<parameter_name> - <type_name>: <description of parameter>"
                        f"\n\nFor example:"
                        f"\n`?c - car: a car that can drive`"
                        f"\nor `?c: a car that can drive` if not using a type"
                    )
                    return False, feedback_msg

            feedback_msg = "[PASS]: All parameter types found in object types."
            return True, feedback_msg

    
    # ---- PDDL TYPE CHECKS ----

    def validate_type(
            self, 
            target_type: str, 
            claimed_type: str, 
            types: dict[str,str] | list[dict[str,str]] | None = None
            ) -> tuple[bool, str]:
        """
        Check if the claimed_type is valid for the target_type according to the type hierarchy.

        Parameters:
            - target_type (str): The type that is expected for the parameter (from predicate).
            - claimed_type (str): The type that is provided in the PDDL.
            - types (dict[str, str]): A dictionary mapping subtypes to their supertypes.

        Returns:
            - bool: True if claimed_type is valid, False otherwise.
        """

        # check if the claimed type matches the target type
        if claimed_type == target_type:
            feedback_msg = "[PASS]: claimed type matches target type definition."
            return True, feedback_msg

        types = format_types(types) # flatten hierarchy

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
                    feedback_msg = "[PASS]: claimed type matches target type definition."
                    return True, feedback_msg
                current_type = super_type
            else:
                break

        feedback_msg = f"[ERROR]: claimed type `{claimed_type}` does not match target `{target_type}` or any of its possible sub-types."
        return False, feedback_msg
    
    
    def validate_format_types(
            self, 
            types: dict[str,str] | list[dict[str,str]]
            ) -> tuple[bool, str]: 
        """Checks if type variables contain `?` characters."""

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
        types: dict[str,str] | list[dict[str,str]] | None = None,
    ) -> tuple[bool, str]:
        """Checks if the types given contain an invalid cyclic hierarchy."""
        
        def visit_type(node, visited, path, all_types):
            """Traverses the type hierarchy and check for cycles."""

            node_key = next(iter(node))  # get first type
            
            # if already visited this type, it indicates a cycle
            if node_key in visited:
                cycle_path = ' -> '.join(path[path.index(node_key):] + [node_key])
                violated_type = node_key  # type that caused the cycle
                return False, cycle_path, violated_type
            
            # mark the current type as visited in this traversal path
            visited.add(node_key)
            path.append(node_key)
            
            # visit all children of the current node
            if "children" in node:
                for child in node["children"]:
                    result, cycle, violated_type = visit_type(child, visited, path, all_types)
                    if not result:
                        return result, cycle, violated_type
            
            # also check if type appears elsewhere in the hierarchy with children
            for other_type in all_types:
                other_key = next(iter(other_type))
                if other_key == node_key and "children" in other_type and other_type["children"]:
                    for child in other_type["children"]:
                        result, cycle, violated_type = visit_type(child, visited, path, all_types)
                        if not result:
                            return result, cycle, violated_type
            
            # remove the type from the visited once its children are processed
            visited.remove(node_key)
            path.pop()
            return True, "", ""
        
        # if no types are provided, return invalid
        if types is None:
            return False, "No types provided."

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
    

    # ---- PDDL PREDICATE CHECKS ----

    def validate_types_predicates(
        self, 
        predicates: list[Predicate], 
        types: dict[str, str] | list[dict[str,str]] | None = None,
    ) -> tuple[bool, str]:
        """Check if predicate name is found within any type definitions"""
        
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
        new_predicates: list[Predicate] | None = None
    ) -> tuple[bool, str]:
        """Checks if predicates have the same name but different parameters."""
        
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
                        duplicated_predicates.append(
                            (new_pred["raw"], curr["raw"])
                        )

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

        return True, "[PASS]: All predicates are uniquely named and consistently defined."


    def validate_format_predicates(
        self,
        predicates: list[dict],
        types: dict[str, str] | list[dict[str, str]] | None = None
    ) -> tuple[bool, str]:
        """Checks for any PDDL syntax found within predicates, allowing untyped variables."""

        valid_types = list()
        # flatten type hierarchy if exists
        if types:
            types = format_types(types)
            valid_types = [
                            type_key.split(" - ")[0].strip()
                            for type_key in types.keys()
                        ]
        else:
            valid_types = []

        all_invalid_params = []

        for pred in predicates:
            pred_def = pred["clean"]
            pred_def = pred_def.strip(" ()`")  # discard parentheses and similar
            
            # check if predicate name declared
            if pred_def.startswith("?") or pred['name'].startswith("?"):
                feedback_msg = f"[ERROR]: Predicate `({pred_def})` does not contain a predicate name. Predicate names must not start with `?`. Revise predicate to include name like `(stack ?b - block ?t - table)` where `stack` is the predicate name."
                return False, feedback_msg

            split_predicate = pred_def.split(" ")[1:]  # discard the predicate name
            split_predicate = [e for e in split_predicate if e != ""]

            i = 0
            while i < len(split_predicate):
                p = split_predicate[i]
                # variable name must start with `?`
                if "?" not in p:
                    
                    # catches random character declarations
                    if re.match(r"^[^\w\s]+$", p) or re.match(r"^[^\w]", p):  # all non-word or starts with symbol
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
            feedback_msg = "[ERROR]: For PDDL, there are invalid object types in the predicates:"
            for param_obj_type, p, pred_def in all_invalid_params:
                feedback_msg += (
                f"\n - `{param_obj_type}` for the parameter `{p}` in the definition of the predicate `{pred_def}` "
                + (f"not found in types: {valid_types}." if valid_types else "contain types when no types are available.")
                )
            feedback_msg += "\n\nRevise predicate parameters such that their types are assigned correctly. Otherwise leave variable untyped."
            return False, feedback_msg

        feedback_msg = "[PASS]: All predicates are formatted correctly."
        return True, feedback_msg


    def validate_pddl_usage_predicates(
        self,
        pddl: str,
        predicates: list[Predicate],
        action_params: OrderedDict,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        part="preconditions",
    ) -> tuple[bool, str]:
        """
        Validates predicates in nested PDDL list format.
        """
        
        pddl = remove_comments(pddl)
        pddl = parse_pddl(pddl)
        pred_index = {pred["name"]: pred for pred in predicates}

        def get_ordinal_suffix(_num):
            return (
                {1: "st", 2: "nd", 3: "rd"}.get(_num % 10, "th")
                if _num not in (11, 12, 13)
                else "th"
            )

        def traverse(node, scoped_params):
            if isinstance(node, str):
                return True, "[PASS]"

            if not isinstance(node, list) or len(node) == 0:
                return True, "[PASS]"

            keyword = node[0].lower()

            # Logical connectives
            if keyword in {"and", "or", "not", "imply"}:
                children = node[1:] if keyword != "not" else [node[1]]
                for child in children:
                    valid, msg = traverse(child, scoped_params)
                    if not valid:
                        return False, msg
                return True, "[PASS]"

            # Quantifiers
            if keyword in {"forall", "exists"}:
                if len(node) != 3:
                    return False, f"[ERROR]: Malformed quantifier in {part}: {node}"
                
                param_spec = node[1]  # e.g., ['?b - box']
                body = node[2]

                new_scope = scoped_params.copy()

                if isinstance(param_spec[0], list):  # e.g., [['?x - block'], ['?y - block']]
                    for spec in param_spec:                        
                        var = spec[0]
                        _type = spec[2] if len(spec) >= 3 and spec[1] == "-" else ""
                        new_scope[var] = _type
                else:  # e.g., ['?x - block']
                    param_spec = param_spec[0].split(" ")
                    var = param_spec[0]
                    _type = param_spec[2] if len(param_spec) >= 3 and param_spec[1] == "-" else ""
                    
                    new_scope[var] = _type

                return traverse(body, new_scope)
            
            # retrieve predicate
            pred_ = keyword
            pred_name = pred_.split(" ")[0]
            pred_args = node[0].split(" ")[1:]
            
            if pred_name == "=":
                if len(pred_args) != 2:
                    return (
                        False,
                        f"[ERROR]: Equality statement `=` in {part} must have exactly two arguments.\nParsed line: {node}"
                    )
                for var in node[0].split(" ")[1:]:
                    if var not in scoped_params:
                        return (
                            False,
                            f"[ERROR]: Variable `{var}` in for equality statement not found in scope:\n"
                            f"Scoped parameters: {list(scoped_params.keys())}\n"
                            f"\nParsed line: ({node[0]})\n\n"
                            f"Make sure variables are being used are found in parameter scope."
                        )
                return True, "[PASS]"

            if pred_name not in pred_index:
                available_preds = "\n - " + "\n - ".join([p["raw"] for p in predicates])
                return (
                    False,
                    f"[ERROR]: Undeclared predicate `({pred_})` found in {part}.\n"
                    f"List of available predicates are:{available_preds}"
                )

            target_pred = pred_index[pred_name]
            expected_args = list(target_pred["params"].keys())
            expected_types = list(target_pred["params"].values())

            # (i) Arity check
            if len(pred_args) != len(expected_args):
                return (
                    False,
                    f"[ERROR]: Predicate `{target_pred['clean']}` expects {len(expected_args)} parameter(s), "
                    f"but found {len(pred_args)} in {part}.\nParsed line: ({node[0]})\n\n"
                    f"Possible solutions:\n"
                    f"(1) Make sure that a predicate used in the action does not include the parameter's variable type. For instance: (drive ?c) is correct, but (drive ?c - car) is not."
                )

            for i, param in enumerate(pred_args):
                if param not in scoped_params:
                    return (
                        False,
                        f"[ERROR]: Parameter `{param}` in predicate `{pred_name}` not found in scope.\n"
                        f"Available variables: {list(scoped_params.keys())}\nParsed line: ({node[0]})\n\n"
                        f"Possible solutions:\n"
                        f"(1) Make sure the parameter variable being used is found in the available parameter variables list.\n"
                        f"(2) Make sure that variable is assigned the correct type for use in predicate `{pred_name}`."
                        
                    )

                # (iii) Type check
                expected_type = expected_types[i]
                actual_type = scoped_params[param]

                # type mismatch only if both are defined
                if expected_type and actual_type:
                    if (types is None or len(types) == 0):
                        return (
                            False,
                            f"[ERROR]: Types are declared in the {part} predicates, but list of available types are empty.\n"
                            f"Remove all types from predicates."
                        )
                    flag, _ = self.validate_type(expected_type, actual_type, types)
                    if not flag:
                        param_number = i + 1
                        ordinal_suffix = get_ordinal_suffix(param_number)
                        violating_var = node[0].split(" ")[param_number]
                        return (
                            False,
                            f"[ERROR]: The {param_number}{ordinal_suffix} parameter of `{target_pred['clean']}` "
                            f"should be of type `{expected_type}`, but type `{actual_type}` was assigned to `{violating_var}` "
                            f"in predicate: ({node[0]})\n\n"
                            f"Possible solutions:\n"
                            f"(1) Make sure variable `{violating_var}` is assigned the correct type for use in predicate `{pred_name}`\n"
                            f"(2) If `{violating_var}` was recently introduced (i.e. forall quantifier), make sure the variable in that declaration line is assigned the correct type"
                        )

            return True, "[PASS]"

        # start traversal with base action parameters
        return traverse(pddl, scoped_params=action_params.copy())


    def validate_usage_predicates(
        self, 
        llm_response: str, 
        curr_predicates: list[Predicate] | None = None, 
        types: dict[str, str] | list[dict[str,str]] | None = None
    ):
        """
        This function performs very basic check over whether the predicates are used in a valid way.
            This check should be performed at the end.
        """

        # parse predicates
        new_predicates = parse_new_predicates(llm_response)
        if curr_predicates is None:
            curr_predicates = new_predicates
        else:
            curr_predicates.extend(new_predicates)
        curr_predicates = parse_predicates(curr_predicates)

        # get action params
        params_info = parse_params(llm_response)

        # check preconditions
        precond_str = llm_response.split("Preconditions")[1].split("```\n")[1].strip()
        precond_str = (
            precond_str.replace("\n", " ").replace("(", " ( ").replace(")", " ) ")
        )

        validation_info = self.validate_pddl_usage_predicates(
            precond_str, curr_predicates, params_info[0], types, part="preconditions"
        )
        if not validation_info[0]:
            return validation_info

        # check effects
        if llm_response.split("Effects")[1].count("```\n") < 2:
            return True, "invalid_predicate_usage"
        eff_str = llm_response.split("Effects")[1].split("```\n")[1].strip()
        eff_str = eff_str.replace("\n", " ").replace("(", " ( ").replace(")", " ) ")
        return self.validate_pddl_usage_predicates(
            eff_str, curr_predicates, params_info[0], types, part="effects"
        )


    def validate_overflow_predicates(
        self, llm_response: str, limit: int = 30
    ) -> tuple[bool, str]:
        """
        Checks if LLM output contains too many predicates in precondition/effects (based on users assigned limit)
        """

        spacer = "="*50

        assert "Preconditions" in llm_response, llm_response
        precond_str = (llm_response.split("Preconditions")[1].split("```\n")[1].strip())
        num_prec_pred = len(precond_str.split("\n")) - 2 # for outer brackets
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
        num_eff_pred = len(eff_str.split("\n")) - 2 # for outer brackets
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


    # ---- PDDL TASK CHECKS ----

    def validate_task_objects(
        self, 
        objects: dict[str, str], 
        types: dict[str, str] | list[dict[str,str]] | None = None
    ) -> tuple[bool, str]:
        """
        Parameters:
            - objects (dict[str,str]): a dictionary of the task objects.
            - types (dict[str,str]): a dictionary of the domain types.

        Returns:
            - check, feedback_msg (bool, str)

        Checks following cases:
            (i) if object type is the same as type
            (ii) if object name is the same as type
        """

        # if type hierarchy format
        if isinstance(types, list):
            types = format_types(types)

        valid = True
        feedback_msgs = []

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
                if obj_type == current_type or (parent_type and obj_type == parent_type):
                    obj_type_found = True

                # Check for name conflict with types
                if obj_name == current_type:
                    parsed_line = f"{obj_name} - {obj_type}"
                    feedback_msgs.append(
                        f"[ERROR]: Object variable '{obj_name}' matches the type name '{current_type}', change it to be unique from types: {list(type_keys)}\n"
                        f"Violated object declaration: ({parsed_line if obj_type else obj_name})\n"
                    )
                    valid = False
                    break
                if parent_type and obj_name == parent_type:
                    parsed_line = f"{obj_name} - {obj_type}"
                    feedback_msgs.append(
                        f"[ERROR]: Object variable '{obj_name}' matches the type name '{parent_type}', change it to be unique from types: {list(type_keys)}\n"
                        f"Violated object declaration: ({parsed_line if obj_type else obj_name})\n"
                    )
                    valid = False
                    break

            # if object does not contain a type
            if not obj_type:
                continue

            if not obj_type_found:
                message = f"not found in types: {list(type_keys)}" if type_keys else "but there are no types declared."
                feedback_msgs.append(
                    f"[ERROR]: Object variable '{obj_name}' has an invalid type '{obj_type}' {message}\n"
                    f"If an object variable requires a type, it must be assigned to the given types. If there are no types, leave object variable untyped.\n\n"
                    f"Parsed line: ({obj_name} - {obj_type})\n"
                )
                valid = False

        feedback_msg = (
            "\n".join(feedback_msgs) if not valid else "[PASS]: all objects are valid."
        )

        return valid, feedback_msg


    def validate_task_states(
        self,
        states: list[dict[str, str]],
        objects: dict[str, str],
        predicates: list[Predicate],
        state_type: str = "initial",
    ) -> tuple[bool, str]:
        """
        Parameters:
            - states (list[dict[str,str]]): a list of dictionaries of the state states.
            - objects (dict[str,str]): a dictionary of the task objects and their types.
            - predicates (list[Predicate]): a list of predicate definitions.
            - state_type (str): optional; 'initial' or 'goal' to label messages.

        Returns:
            - valid (bool): True if all checks pass, False otherwise.
            - feedback_msg (str): summary of validation results.

        Checks:
            (i) if predicates in states exist in the domain
            (ii) if all object variables in states are declared in task objects
            (iii) if types of object variables match predicate parameter types
        """
        valid = True
        feedback_msgs = []

        for state in states:
            state_name = state["name"]
            state_params = state["params"]

            # (i) Check if predicate name exists in domain predicates
            predicate_names = [p["name"] for p in predicates]
            if state_name not in predicate_names:
                feedback_msgs.append(
                    f"[ERROR]: In the {state_type} state, '({state_name} {' '.join(state_params)})' "
                    f"uses predicate '{state_name}', which is not defined in the domain ({predicate_names})."
                )
                valid = False
                continue  # Skip further checks for this state

            # (ii) Check if all parameters exist in the task objects
            missing_params = [p for p in state_params if p not in objects]
            if missing_params:
                for mp in missing_params:
                    feedback_msgs.append(
                        f"[ERROR]: In the {state_type} state, '({state_name} {' '.join(state_params)})' "
                        f"contains parameter '{mp}' not found in task objects {list(objects.keys())}."
                    )
                valid = False
                continue  # Skip type check if parameters are invalid

            # (iii) Check if object types match expected predicate types
            target_pred = next(p for p in predicates if p["name"] == state_name)
            expected_types = list(target_pred["params"].values())
            actual_types = [objects[param] for param in state_params]
            actual_name_type_pairs = [f"'{param}': '{objects[param]}'" for param in state_params]

            if expected_types != actual_types:
                feedback_msgs.append(
                    f"[ERROR]: In the {state_type} state, '({state_name} {' '.join(state_params)})' has mismatched types.\n\n"
                    f"Declared objects: {[f'{obj_name} - {obj_type}' for obj_name, obj_type in objects.items()]}\n\n"
                    f"Predicate `{target_pred['clean']}` expects type(s): {expected_types}\n"
                    f"Parsed line: ({state_name} {' '.join(state_params)}) contains type(s): {actual_types}, where [{', '.join(actual_name_type_pairs)}]\n\n"
                    f"Revise the state predicates such that their parameter types align with the original predicate definition types."
                )
                valid = False

        feedback_msg = (
            "\n".join(feedback_msgs) if not valid else "[PASS]: All task states are valid."
        )
        return valid, feedback_msg

    # ---- TEXT PARSE CHECKS ----

    def validate_header(
            self, 
            llm_response: str,
            ):
        """Checks if domain headers and formatted code block syntax are found in LLM output"""

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
                    f"\n\n Make sure the header is only stated once in your response."          
                )
                return False, feedback_msg

        feedback_msg = "[PASS]: headers are identified properly in LLM output."
        return True, feedback_msg
    
    
    def validate_duplicate_headers(
        self, 
        llm_response: str,
    ) -> tuple[bool, str]:
        """Checks if the LLM attempts to create a new action (so two or more actions defined in the same response)."""

        invalid = False
        duplicate_headers = []

        for header in self.headers:
            count = llm_response.count(f"### {header}")
            if count > 1:
                invalid = True
                duplicate_headers.append({'header': f'`### {header}`', 'count': count})

        if invalid:
            feedback_msg = (
                "[ERROR]: Detected multiple definitions of the following header(s):\n"
                + "\n".join(f"- {item['header']} found {item['count']} times" for item in duplicate_headers)
                + "\n\nEach header section should only be declared once in your response for: "
                + ", ".join(item['header'] for item in duplicate_headers) + "."
            )
            return False, feedback_msg

        return True, "[PASS]: no duplicate sections created."


    def validate_unsupported_keywords(
        self, 
        llm_response: str
    ) -> tuple[bool, str]:
        """Checks whether PDDL model uses unsupported logic keywords"""

        if not self.unsupported_keywords:
            feedback_msg = "[PASS]: No unsupported keywords declared."
            return True, feedback_msg

        for key in self.unsupported_keywords:
            if f"{key}" in llm_response:
                feedback_msg = (
                    f"[ERROR]: The PDDL model contains the keyword `{key}`. Revise the model so that it does not use this keyword."
                )
                return False, feedback_msg

        feedback_msg = "[PASS]: Unsupported keywords not found in PDDL model."
        return True, feedback_msg