"""
This file contains collection of functions PDDL syntax validations
"""

from collections import OrderedDict
from .pddl_parser import parse_params, parse_new_predicates, parse_predicates
from .pddl_types import Predicate
from .pddl_format import format_types
import re


class SyntaxValidator:
    def __init__(
            self, 
            error_types: list[str] | None = None,
            unsupported_keywords: list[str] | None = None
            ) -> None:

        # current [ERROR] types available
        default_error_types = [
            "invalid_header",
            "invalid_keyword_usage",
            "unsupported_keywords",
            "invalid_params",
            "invalid_predicate_name",
            "invalid_predicate_format",
            "invalid_predicate_usage",
        ]

        default_unsupported = [
            "forall",
            "when",
            "exists",
            "implies",
        ]
        
        self.PDDL_KEYWORDS = {
            # Logical keywords
            "and", "or", "not", "when", "imply",
            "exists", "forall", "either",

            # Assignment operators
            "increase", "decrease", "assign", "scale-up", "scale-down",

            # Relational operators
            "=", "<", ">", "<=", ">=",

            # Type and metric-related
            "number", "object", "total-cost",
        }
        
        self.error_types = default_error_types if error_types is None else error_types
        self.unsupported_keywords = (
            default_unsupported
            if unsupported_keywords is None
            else unsupported_keywords
        )

    # PARAMETER CHECKS

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
                    feedback_msg = (f'[ERROR]: There is an invalid object type `{param_type}` for the parameter `{param_name}` '
                                    f'not found in the types {list(types.keys())}. Make sure parameter types align with '
                                    'provided types, otherwise just leave parameter untyped.'
                    )
                    return False, feedback_msg

            feedback_msg = "[PASS]: All parameter types found in object types."
            return True, feedback_msg

    # PREDICATE CHECKS

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
                type_name = type_key.split(" - ")[0].strip().lower()

                # check if the predicate name is exactly the same as the type name
                if pred["name"].lower() == type_name:
                    invalid_predicates.append(pred)

        if invalid_predicates:
            feedback_msg = "[ERROR]: The following predicate(s) have the same name(s) as existing object types:"
            for pred_i, pred in enumerate(invalid_predicates):
                feedback_msg += f"\n{pred_i + 1}. `{pred['name'].lower()}` from {pred['clean']}"
            feedback_msg += f"\nRename these predicates that are unique from types: {list(types.keys())}"
            return False, feedback_msg

        feedback_msg = "[PASS]: All predicate names are unique to object type names"
        return True, feedback_msg


    def validate_duplicate_predicates(
        self, curr_predicates: list[Predicate], new_predicates: list[Predicate]
    ) -> tuple[bool, str]:
        """Checks if predicates have the same name but different parameters."""

        curr_pred_dict = {pred["name"].lower(): pred for pred in curr_predicates}

        duplicated_predicates = list()
        for new_pred in new_predicates:
            name_lower = new_pred["name"].lower()
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
                            type_key.split(" - ")[0].strip().lower()
                            for type_key in types.keys()
                        ]
        else:
            valid_types = []

        all_invalid_params = []

        for pred in predicates:
            pred_def = pred["raw"].split(": ")[0]
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

                    param_obj_type = split_predicate[i + 2].lower()

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
        action_params: list[str],
        types: dict[str, str] | list[dict[str,str]] | None = None,
        part="preconditions",
    ) -> tuple[bool, str]:
        """
        This function checks three types of errors:
            - (i) check if the num of params given matches the num of params in predicate definition
            - (ii) check if there is any param that is not listed under `Action Parameters`
            - (iii) check if the param type matches that in the predicate definition
        """

        def get_ordinal_suffix(_num):
            return (
                {1: "st", 2: "nd", 3: "rd"}.get(_num % 10, "th")
                if _num not in (11, 12, 13)
                else "th"
            )

        pred_names = {predicates[i]["name"]: i for i in range(len(predicates))}
        pddl_elems = [e for e in pddl.replace("(", " ( ").replace(")", " ) ").split() if e != ""]
        idx = 0
                
        action_predicates = dict()

        while idx < len(pddl_elems):
            if pddl_elems[idx] == "(" and idx + 1 < len(pddl_elems):
                pred_candidate = pddl_elems[idx + 1]
                
                # put predicate name into list
                if pred_candidate not in self.PDDL_KEYWORDS and pred_candidate not in {"(", ")"}:
                    
                    orig_idx = idx
                    curr_pred_tokens = []
                    idx += 2
                    
                    while idx < len(pddl_elems):
                        token = pddl_elems[idx]
                        if token == ")":
                            break
                        curr_pred_tokens.append(token)
                        idx += 1
                        
                    action_predicates[pred_candidate] = f"{' '.join(curr_pred_tokens)}"
                    idx = orig_idx
                
                if pred_candidate in pred_names:
                    curr_pred_name = pred_candidate
                    target_pred_info = predicates[pred_names[curr_pred_name]]
                    
                    curr_pred_tokens = []
                    idx += 2

                    # Collect predicate parameters until closing ')'
                    while idx < len(pddl_elems):
                        token = pddl_elems[idx]
                        if token == ")":
                            break
                        curr_pred_tokens.append(token)
                        idx += 1

                    curr_pred_line = f"({curr_pred_name} {' '.join(curr_pred_tokens)})"
                    

                    # Clean out comments if any (denoted by ';')
                    raw_param_str = " ".join(curr_pred_tokens).split(";")[0].strip()
                    curr_pred_params = raw_param_str.split()

                    # --- (i) Parameter count check ---
                    n_expected_param = len(target_pred_info["params"])
                    n_actual_param = len(curr_pred_params)

                    if n_expected_param != n_actual_param:
                        feedback_msg = (
                            f"[ERROR]: There is a syntax mistake in the {part}.\n"
                            f"Predicate `{target_pred_info['clean']}` expects {n_expected_param} parameter variable(s), "
                            f"but found {n_actual_param}.\n\n"
                            f"Parsed line: {curr_pred_line}\n"
                            f"Extracted variables: {curr_pred_params}"
                            f"\n\nPossible causes:"
                            f"\n(1) Missing variable(s). Example: `(drive ?c)` has only 1 variable, but should be `(drive ?c ?from)` "
                            f"to match the definition `(drive ?c - car ?from - location)`."
                            f"\n(2) Object types included incorrectly. Example: `{part}` predicate `(drive ?c ?from)` is correct, "
                            f"but `(drive ?c - car ?from - location)` is incorrect."
                        )

                        return False, feedback_msg

                    # --- (ii) Unknown parameter check ---
                    for curr_param in curr_pred_params:
                        if curr_param not in action_params[0]:
                            feedback_msg = (
                                f"[ERROR]: A predicate in {part} contains parameter variable(s) not found in the action parameter list: "
                                f"{list(action_params[0].keys())}\n\n"
                                f"Parsed line: {curr_pred_line}\n"
                                f"Unknown variable: {curr_param}"
                                f"\n\nPossible solutions:"
                                f"\n(1) Ensure that the variables used in the predicate match those defined in the action's parameter list."
                                f"\n(2) If needed, update the action's parameter list and their types to reflect the correct requirements for the action."
                            )

                            return False, feedback_msg

                    # --- (iii) Type mismatch check ---
                    if (types is None or len(types) == 0) and (
                    any(target_pred_info["params"].values()) or any(action_params[0].values())
                    ):
                        source_of_types = []
                        if any(target_pred_info["params"].values()):
                            source_of_types.append(f"the predicate(s)")
                        if any(action_params[0].values()):
                            source_of_types.append(f"the action parameter list")

                        type_sources = " and ".join(source_of_types)

                        feedback_msg = f"[ERROR]: Type information is declared in {type_sources}, but the `types` dictionary is empty or undefined.\n\n"

                        # collect all predicates that contain types
                        predicates_with_types = [
                            pred["clean"] for pred in predicates if any(pred["params"].values())
                        ]
                        
                        if predicates_with_types:
                            feedback_msg += f"Predicates declared with types: {', '.join(predicates_with_types)}\n"

                        violated_params = [f"{k} - {v}" for k, v in action_params[0].items() if v]
                        if violated_params:
                            for i in violated_params:
                                feedback_msg += f"Action parameter with types: {i}\n"
                                
                        feedback_msg += (
                            f"\nTo resolve this, either:\n"
                            f"1. Remove type annotations from {type_sources} if type checking is not required.\n"
                            f"2. Provide a valid `types` dictionary that defines the available types.\n"
                        )

                        return False, feedback_msg
                    
                    target_param_types = list(target_pred_info["params"].values())

                    for param_idx, target_type in enumerate(target_param_types):
                        curr_param = curr_pred_params[param_idx]
                        claimed_type = action_params[0][curr_param]
                        flag, _ = self.validate_type(target_type, claimed_type, types)

                        if not flag:
                            param_number = param_idx + 1
                            ordinal_suffix = get_ordinal_suffix(param_number)
                            parsed_vars = [
                                f"{param} - {action_params[0][param]}"
                                for param in curr_pred_params
                            ]
                            pred_name = target_pred_info["name"]
                            clean_pred = target_pred_info["clean"]

                            feedback_msg = (
                                f"[ERROR]: There is a syntax mistake in the {part}.\n"
                                f"- The {param_number}{ordinal_suffix} parameter of predicate `{clean_pred}` should be of type `{target_type}`,\n"
                                f"  but `{claimed_type}` was provided in `{curr_pred_line}`.\n"
                                f"- According to the action's parameter list, `{curr_param}` is of type `{claimed_type}`.\n\n"
                                f"Parsed line: {curr_pred_line}\n"
                                f"Extracted variables and types: {parsed_vars}\n\n"
                                f"Expected types for predicate `{pred_name}`: {target_param_types}\n\n"
                                f"Possible solutions:"
                                f"\n(1) Correct the variable used in the predicate to match the expected type."
                                f"\n(2) Update the action's parameter list if the type assignment is incorrect."
                            )
                            return False, feedback_msg

            # move to next token
            idx += 1
            
        available_preds = "\n - ".join([i['raw'] for i in predicates])

        for i in action_predicates.keys():
            if i not in pred_names.keys():
                feedback_msg = (
                    f"[ERROR]: Undeclared predicate `{i}` found in {part}.\n"
                    f"List of available predicates are:\n"
                    f" - {available_preds}\n\n"
                    f"Parsed line: {f'({i} {action_predicates[i]})'}"
                    f"\n\nMake sure this predicate is declared in the list of known predicates.\n"
                )
                return False, feedback_msg

        feedback_msg = "[PASS]: all correct use of predicates."
        return True, feedback_msg


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
            precond_str, curr_predicates, params_info, types, part="preconditions"
        )
        if not validation_info[0]:
            return validation_info

        # check effects
        if llm_response.split("Effects")[1].count("```\n") < 2:
            return True, "invalid_predicate_usage"
        eff_str = llm_response.split("Effects")[1].split("```\n")[1].strip()
        eff_str = eff_str.replace("\n", " ").replace("(", " ( ").replace(")", " ) ")
        return self.validate_pddl_usage_predicates(
            eff_str, curr_predicates, params_info, types, part="effects"
        )


    def validate_overflow_predicates(
        self, llm_response: str, limit: int
    ) -> tuple[bool, str]:
        """
        Checks if LLM output contains too many predicates in precondition/effects (based on users assigned limit)
        """
        assert "\nPreconditions:" in llm_response, llm_response
        precond_str = (
            llm_response.split("\nPreconditions:")[1].split("```\n")[1].strip()
        )
        if len(precond_str.split("\n")) > limit:
            feedback_msg = f"[ERROR]: You seem to have generated an action model with an unusually long list of preconditions. Please include only the relevant preconditions/effects and keep the action model concise.\n\nParameters:"
            return False, feedback_msg

        eff_str = llm_response.split("Effects")[1].split("```\n")[1].strip()
        if len(eff_str.split("\n")) > limit:
            feedback_msg = f"[ERROR]: You seem to have generated an action model with an unusually long list of effects. Please include only the relevant preconditions/effects and keep the action model concise.\n\nParameters:"
            return False, feedback_msg

        feedback_msg = "[PASS]: predicate output is fine."
        return True, feedback_msg


    def validate_task_objects(
        self, objects: dict[str, str], types: dict[str, str]
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

        valid = True
        feedback_msgs = []

        for obj_name, obj_type in objects.items():
            obj_type_found = False

            for type_key in types.keys():

                current_type, parent_type = type_key.split(" - ")

                # checks if obj_type is found in types
                if obj_type == current_type or obj_type == parent_type:
                    obj_type_found = True

                # checks if obj_name matches either current_type or parent_type
                if obj_name == current_type:
                    feedback_msgs.append(
                        f"[ERROR]: Object variable '{obj_name}' matches the type name '{current_type}', change it to be unique from types: {types.keys()}"
                    )
                    valid = False
                    break
                if obj_name == parent_type:
                    feedback_msgs.append(
                        f"[ERROR]: Object variable '{obj_name}' matches the type name '{parent_type}', change it to be unique from types: {types.keys()}"
                    )
                    valid = False
                    break

            # clause that checks if obj_type is found in types
            if not obj_type_found:
                feedback_msgs.append(
                    f"[ERROR]: Object variable '{obj_name}' has an invalid type '{obj_type}' not found in types: {types.keys()}"
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
            - parameters (OrderedDict): parameters of the current action.
            - types (dict[str,str]): a dictionary of the domain types.

        Returns:
            - check, feedback_msg (bool, str)

        Checks following cases:
            (i) if predicates in states are found in predicates in domain
            (ii) if object variables in states are found in task object list
        """

        valid = True
        feedback_msgs = []

        # loop through each state
        for state in states:

            # (i) check if predicates in states are found in predicates in domain
            matched_preds = False
            state_name = state["name"]  # retrieve predicate name from state

            # loop through each predicate name from domain
            for pred in predicates:
                # check if predicate in state is found in predicate domain
                if state_name == pred["name"]:
                    matched_preds = True

            # if no matches, then that state is missusing a predicate - not found in domain
            if matched_preds == False:
                feedback_msgs.append(
                    f"[ERROR]: In the {state_type} state, '({state['name']} {' '.join(state['params'])})' contains '{state_name}' predicate, which is not found in {[p['name'] for p in predicates]}, predicate in state is missused."
                )
                valid = False

            # (ii) check if object variables in states are found in task object list
            state_params = state["params"]  # retrieve variables from state

            # loop through each parameter in current state
            for state_p in state_params:

                matched_params = False
                for obj_name, obj_type in objects.items():
                    # check if parameter is found in object names
                    if state_p == obj_name:
                        matched_params = True

                if matched_params == False:
                    feedback_msgs.append(
                        f"[ERROR]: In the {state_type} state, '({state['name']} {' '.join(state['params'])})' contains parameter '{state_p}' not found in '{objects.keys()}'."
                    )
                    valid = False

        feedback_msg = (
            "\n".join(feedback_msgs) if not valid else "[PASS]: all objects are valid."
        )

        return valid, feedback_msg


    def validate_header(self, llm_response: str):
        """Checks if domain headers and formatted code block syntax are found in LLM output"""

        for header in ["Parameters", "Preconditions", "Effects", "New Predicates"]:
            if header not in llm_response:
                feedback_msg = f"[ERROR]: The header `{header}` is missing in the PDDL model. Please include the header `{header}` in the PDDL model."
                return False, feedback_msg
        for header in ["Parameters", "Preconditions", "Effects"]:
            if llm_response.split(f"{header}")[1].split("##")[0].count("```\n") < 2:
                feedback_msg = f'[ERROR]: The header `{header}` is missing in the formalised code block. Please include a "```" section in the {header} section.'
                return False, feedback_msg

        feedback_msg = "[PASS]: headers are identified properly in LLM output."
        return True, feedback_msg


    def validate_unsupported_keywords(
        self, llm_response: str, unsupported_keywords: list[str]
    ) -> tuple[bool, str]:
        """Checks whether PDDL model uses unsupported logic keywords"""

        for key in unsupported_keywords:
            if f"{key}" in llm_response:
                feedback_msg = (
                    f"[ERROR]: The precondition or effect contains the keyword {key}."
                )
                return False, feedback_msg

        feedback_msg = "[PASS]: Unsupported keywords not found in PDDL model."
        return True, feedback_msg


    def validate_keyword_usage(self, llm_response: str):
        """Checks if action effects uses unsupported universal condition keywords"""

        if not "Action Effects" in llm_response:
            feedback_msg = "[PASS]"
            return True, feedback_msg
        heading = llm_response.split("Action Effects")[1].split("```\n")[1].strip()
        for keyword in ["forall", "exists", "if "]:
            if keyword in heading:
                feedback_msg = (
                    f"The keyword `{keyword}` is not supported in the action effects."
                )
                return False, feedback_msg

        feedback_msg = "[PASS]: unsupported keywords are not found in the action effects."
        return True, feedback_msg


    def validate_new_action_creation(self, llm_response: str) -> tuple[bool, str]:
        """Checks if the LLM attempts to create a new action (so two or more actions defined in the same response)"""

        if (
            llm_response.count("## Action Parameters") > 1
            or llm_response.count("## Preconditions") > 1
            or llm_response.count("## Effects") > 1
            or llm_response.count("## New Predicates") > 1
        ):
            feedback_msg = "It's not possible to create new actions at this time. Please only define the requested action."
            return False, feedback_msg

        feedback_msg = "[PASS]: no new actions created"
        return True, feedback_msg


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
    
    
    def validate_format_types(self, types: dict[str,str] | list[dict[str,str]]) -> tuple[bool, str]: 
        types = format_types(types)

        invalid_types = []
        for t_name, _ in types.items():
            if t_name.startswith("?"):
                invalid_types.append(t_name)
        
        if invalid_types:
            invalid_types_str = "\n".join(invalid_types)
            feedback_msg = f"[ERROR]: There are type(s) with name(s) that start with character `?`. This is not allowed in PDDL."
            feedback_msg += f"\n\nRemove `?` from the following types:\n"
            feedback_msg += f"{invalid_types_str}"
            return False, feedback_msg
        
        feedback_msg = "[PASS]: all types are formatted correctly."
        return True, feedback_msg