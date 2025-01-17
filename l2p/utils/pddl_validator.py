"""
This file contains collection of functions PDDL syntax validations
"""

from collections import OrderedDict
from .pddl_parser import parse_params, parse_new_predicates, parse_predicates
from .pddl_types import Predicate


class SyntaxValidator:
    def __init__(self, error_types=None, unsupported_keywords=None):

        # current error types available
        default_error_types = [
            "invalid_header",
            "invalid_keyword_usage",
            "unsupported_keywords",
            "invalid_param_types",
            "invalid_predicate_name",
            "invalid_predicate_format",
            "invalid_predicate_usage",
        ]

        default_unsupported = [
            "forall",
            "when",
            "exists",
            "implies",
        ]  # current unsupported usage of PDDL
        self.error_types = default_error_types if error_types is None else error_types
        self.unsupported_keywords = (
            default_unsupported
            if unsupported_keywords is None
            else unsupported_keywords
        )

    # PARAMETER CHECKS

    def validate_params(
        self, parameters: OrderedDict, types: dict[str, str]
    ) -> tuple[bool, str]:
        """Checks whether a PDDL action parameter contains types found in object types."""

        for param_name in parameters:
            param_type = parameters[param_name]

            if not any(param_type in t for t in types.keys()):
                feedback_msg = f'There is an invalid object type `{param_type}` for the parameter {param_name} not found in the types {types.keys()}. If you need to use a new type, you can emulate it with an "is_{{type}} ?o - object" precondition. Please revise the PDDL model to fix this error.'
                return False, feedback_msg

        feedback_msg = "PASS: All parameter types found in object types."
        return True, feedback_msg

    # PREDICATE CHECKS

    def validate_types_predicates(
        self, predicates: list[dict], types: dict[str, str]
    ) -> tuple[bool, str]:
        """Check if predicate name is found within any type definitions"""

        invalid_predicates = list()
        for pred in predicates:
            pred_name = pred["name"].lower()

            for type_key in types.keys():
                # extract the actual type name, disregarding hierarchical or descriptive parts
                type_name = type_key.split(" - ")[0].strip().lower()

                # check if the predicate name is exactly the same as the type name
                if pred_name == type_name:
                    invalid_predicates.append(pred_name)

        if invalid_predicates:
            feedback_msg = "ERROR: The following predicate(s) have the same name(s) as existing object types:"
            for pred_i, pred_name in enumerate(invalid_predicates):
                feedback_msg += f"\n{pred_i + 1}. {pred_name}"
            feedback_msg += "\nPlease rename these predicates."
            return False, feedback_msg

        feedback_msg = "PASS: All predicate names are unique to object type names"
        return True, feedback_msg

    def validate_duplicate_predicates(
        self, curr_predicates: list[Predicate], new_predicates: list[Predicate]
    ) -> tuple[bool, str]:
        """Checks if predicates have the same name but different parameters"""

        curr_pred_dict = {pred["name"].lower(): pred for pred in curr_predicates}

        duplicated_predicates = list()
        for new_pred in new_predicates:
            # check if the name is already used
            if new_pred["name"].lower() in curr_pred_dict:

                curr = curr_pred_dict[new_pred["name"].lower()]

                if len(curr["params"]) != len(new_pred["params"]) or any(
                    [t1 != t2 for t1, t2 in zip(curr["params"], new_pred["params"])]
                ):
                    # if the params are the same, then it's not a problem
                    duplicated_predicates.append(
                        (
                            new_pred["raw"],
                            curr_pred_dict[new_pred["name"].lower()]["raw"],
                        )
                    )
        if len(duplicated_predicates) > 0:
            feedback_msg = f"The following predicate(s) have the same name(s) as existing predicate(s):"
            for pred_i, duplicated_pred_info in enumerate(duplicated_predicates):
                new_pred_full, existing_pred_full = duplicated_pred_info
                feedback_msg += f'\n{pred_i + 1}. {new_pred_full.replace(":", ",")}; existing predicate with the same name: {existing_pred_full.replace(":", ",")}'
            feedback_msg += "\n\nYou should reuse existing predicates whenever possible. If you are reusing existing predicate(s), you shouldn't list them under 'New Predicates'. If existing predicates are not enough and you are devising new predicate(s), please use names that are different from existing ones."
            feedback_msg += "\n\nPlease revise the PDDL model to fix this error.\n\n"
            feedback_msg += "Parameters:"
            return False, feedback_msg

        feedback_msg = "PASS: All predicates are unique to each other."
        return True, feedback_msg

    def validate_format_predicates(
        self, predicates: list[dict], types: dict[str, str]
    ) -> tuple[bool, str]:
        """Checks for any PDDL syntax found within predicates"""

        all_invalid_params = []

        for pred in predicates:
            pred_def = pred["raw"].split(": ")[0]
            pred_def = pred_def.strip(" ()`")  # discard parentheses and similar
            split_predicate = pred_def.split(" ")[1:]  # discard the predicate name
            split_predicate = [e for e in split_predicate if e != ""]

            for i, p in enumerate(split_predicate):
                if i % 3 == 0:
                    if "?" not in p:
                        feedback_msg = f"There are syntax errors in the definition of the new predicate {pred_def}. Check for any missing '?' variables, or missing type declarations. Please revise its definition and output the entire PDDL action model again. Note that you need to strictly follow the syntax of PDDL."
                        return False, feedback_msg
                    else:
                        if (
                            i + 1 >= len(split_predicate)
                            or split_predicate[i + 1] != "-"
                        ):
                            feedback_msg = f"There are syntax errors in the definition of the new predicate {pred_def}. Please revise its definition and output the entire PDDL action model again. Note that you need to define the object type of each parameter and strictly follow the syntax of PDDL."
                            return False, feedback_msg

                        if i + 2 >= len(split_predicate):
                            feedback_msg = f"There are syntax errors in the definition of the new predicate {pred_def}. Please revise its definition and output the entire PDDL action model again. Note that you need to define the object type of each parameter and strictly follow the syntax of PDDL."
                            return False, feedback_msg

                        param_obj_type = split_predicate[i + 2].lower()

                        # Extract the base type names from the keys in types
                        valid_types = {
                            type_key.split(" - ")[0].strip().lower()
                            for type_key in types.keys()
                        }

                        # Check if the parameter object type is in the set of valid types
                        if param_obj_type not in valid_types:
                            all_invalid_params.append((param_obj_type, p, pred_def))

        if all_invalid_params:
            feedback_msg = "There are invalid object types in the predicates:"
            for param_obj_type, p, pred_def in all_invalid_params:
                feedback_msg += f"\n- `{param_obj_type}` for the parameter `{p}` in the definition of the predicate `{pred_def}` not found in types: {valid_types}."
            feedback_msg += "\nPlease revise these definitions and output the entire PDDL action model again."
            return False, feedback_msg

        feedback_msg = "PASS: All predicates are formatted correctly."
        return True, feedback_msg

    def validate_pddl_usage_predicates(
        self,
        pddl: str,
        predicates: list[Predicate],
        action_params: list[str],
        types: dict[str, str],
        part="preconditions",
    ) -> tuple[bool, str]:
        """
        This function checks three types of errors:
            - (i) check if the num of params given matches the num of params in predicate definition
            - (ii) check if there is any param that is not listed under `Parameters:`
            - (iii) check if the param type matches that in the predicate definition
        """

        def get_ordinal_suffix(_num):
            return (
                {1: "st", 2: "nd", 3: "rd"}.get(_num % 10, "th")
                if _num not in (11, 12, 13)
                else "th"
            )

        pred_names = {predicates[i]["name"]: i for i in range(len(predicates))}
        pddl_elems = [e for e in pddl.split(" ") if e != ""]

        idx = 0
        while idx < len(pddl_elems):
            if pddl_elems[idx] == "(" and idx + 1 < len(pddl_elems):
                if pddl_elems[idx + 1] in pred_names:

                    curr_pred_name = pddl_elems[idx + 1]
                    curr_pred_params = list()
                    target_pred_info = predicates[pred_names[curr_pred_name]]

                    # read params
                    idx += 2
                    while idx < len(pddl_elems) and pddl_elems[idx] != ")":
                        curr_pred_params.append(pddl_elems[idx])
                        idx += 1
                    # (i) check if the num of params are correct
                    n_expected_param = len(target_pred_info["params"])
                    if n_expected_param != len(curr_pred_params):

                        feedback_msg = f'In the {part}, the predicate `{curr_pred_name}` requires {n_expected_param} parameters but {len(curr_pred_params)} parameters were provided. Object type should not be declared in the {part}, but just the variable. For example, "(drive ?a ?from)" does not contain its object types, just variables. Do not change the predicates. Please revise the PDDL model to fix this error.'
                        return False, feedback_msg

                    # (ii) check if there is any unknown param
                    for curr_param in curr_pred_params:

                        if curr_param not in action_params[0]:
                            feedback_msg = f"In the {part} and in the predicate `{curr_pred_name}`, there is an unknown parameter `{curr_param}`. You should define all parameters (i.e., name and type) under the `### Action Parameters` list. Please revise the PDDL model to fix this error (and other potentially similar errors)."
                            return False, feedback_msg

                    # (iii) check if the object types are correct
                    target_param_types = [
                        target_pred_info["params"][t_p]
                        for t_p in target_pred_info["params"]
                    ]
                    for param_idx, target_type in enumerate(target_param_types):
                        curr_param = curr_pred_params[param_idx]
                        claimed_type = action_params[0][curr_param]

                        if not self.validate_type(target_type, claimed_type, types):
                            feedback_msg = f"There is a syntax error in the {part.lower()}, the {param_idx+1}-{get_ordinal_suffix(param_idx+1)} parameter of `{curr_pred_name}` should be a `{target_type}` but a `{claimed_type}` was given. Please use the correct predicate or devise new one(s) if needed (but note that you should use existing predicates as much as possible)."
                            return False, feedback_msg
            idx += 1

        feedback_msg = "PASS: all correct use of predicates."
        return True, feedback_msg

    def validate_usage_predicates(
        self, llm_response: str, curr_predicates: list[Predicate], types: dict[str, str]
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
            feedback_msg = f"FAIL: You seem to have generated an action model with an unusually long list of preconditions. Please include only the relevant preconditions/effects and keep the action model concise.\n\nParameters:"
            return False, feedback_msg

        eff_str = llm_response.split("Effects")[1].split("```\n")[1].strip()
        if len(eff_str.split("\n")) > limit:
            feedback_msg = f"FAIL: You seem to have generated an action model with an unusually long list of effects. Please include only the relevant preconditions/effects and keep the action model concise.\n\nParameters:"
            return False, feedback_msg

        feedback_msg = "PASS: predicate output is fine."
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
                        f"ERROR: Object variable '{obj_name}' matches the type name '{current_type}', change it to be unique from types: {types.keys()}"
                    )
                    valid = False
                    break
                if obj_name == parent_type:
                    feedback_msgs.append(
                        f"ERROR: Object variable '{obj_name}' matches the type name '{parent_type}', change it to be unique from types: {types.keys()}"
                    )
                    valid = False
                    break

            # clause that checks if obj_type is found in types
            if not obj_type_found:
                feedback_msgs.append(
                    f"ERROR: Object variable '{obj_name}' has an invalid type '{obj_type}' not found in types: {types.keys()}"
                )
                valid = False

        feedback_msg = (
            "\n".join(feedback_msgs) if not valid else "PASS: all objects are valid."
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
                    f"ERROR: In the {state_type} state, '({state['name']} {' '.join(state['params'])})' contains '{state_name}' predicate, which is not found in {[p['name'] for p in predicates]}, predicate in state is missused."
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
                        f"ERROR: In the {state_type} state, '({state['name']} {' '.join(state['params'])})' contains parameter '{state_p}' not found in '{objects.keys()}'."
                    )
                    valid = False

        feedback_msg = (
            "\n".join(feedback_msgs) if not valid else "PASS: all objects are valid."
        )

        return valid, feedback_msg

    def validate_header(self, llm_response: str):
        """Checks if domain headers and formatted code block syntax are found in LLM output"""

        for header in ["Parameters", "Preconditions", "Effects", "New Predicates"]:
            if header not in llm_response:
                feedback_msg = f"FAIL: The header `{header}` is missing in the PDDL model. Please include the header `{header}` in the PDDL model."
                return False, feedback_msg
        for header in ["Parameters", "Preconditions", "Effects"]:
            if llm_response.split(f"{header}")[1].split("##")[0].count("```\n") < 2:
                feedback_msg = f'FAIL: The header `{header}` is missing in the formalised code block. Please include a "```" section in the {header} section.'
                return False, feedback_msg

        feedback_msg = "PASS: headers are identified properly in LLM output."
        return True, feedback_msg

    def validate_unsupported_keywords(
        self, llm_response: str, unsupported_keywords: list[str]
    ) -> tuple[bool, str]:
        """Checks whether PDDL model uses unsupported logic keywords"""

        for key in unsupported_keywords:
            if f"{key}" in llm_response:
                feedback_msg = (
                    f"ERROR: The precondition or effect contains the keyword {key}."
                )
                return False, feedback_msg

        feedback_msg = "PASS: Unsupported keywords not found in PDDL model."
        return True, feedback_msg

    def validate_keyword_usage(self, llm_response: str):
        """Checks if action effects uses unsupported universal condition keywords"""

        if not "Action Effects" in llm_response:
            feedback_msg = "PASS"
            return True, feedback_msg
        heading = llm_response.split("Action Effects")[1].split("```\n")[1].strip()
        for keyword in ["forall", "exists", "if "]:
            if keyword in heading:
                feedback_msg = (
                    f"The keyword `{keyword}` is not supported in the action effects."
                )
                return False, feedback_msg

        feedback_msg = "PASS: unsupported keywords are not found in the action effects."
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

        feedback_msg = "PASS: no new actions created"
        return True, feedback_msg

    def validate_type(self, target_type, claimed_type, types):
        """
        Check if the claimed_type is valid for the target_type according to the type hierarchy.

        Parameters:
            - target_type (str): The type that is expected for the parameter.
            - claimed_type (str): The type that is provided in the PDDL.
            - types (dict[str, str]): A dictionary mapping subtypes to their supertypes.

        Returns:
            - bool: True if claimed_type is valid, False otherwise.
        """
        # Check if the claimed type matches the target type
        if claimed_type == target_type:
            return True

        # Iterate through the types hierarchy to check if claimed_type is a subtype of target_type
        current_type = claimed_type

        # Extract all types from the keys in the types dictionary
        all_types = set()
        type_hierarchy = {}
        for key in types.keys():
            main_type, *subtype = key.split(" - ")
            all_types.add(main_type.strip())
            if subtype:
                all_types.add(subtype[0].strip())
                type_hierarchy[subtype[0].strip()] = main_type.strip()

        while current_type in all_types:
            # find the key that starts with the current type

            parent_type_entry = next(
                (k for k in types.keys() if k.startswith(f"{current_type} - ")), None
            )

            if parent_type_entry:
                # extract the parent type from the key
                super_type = parent_type_entry.split(" - ")[1].strip()

                if super_type == target_type:
                    return True
                current_type = super_type
            else:
                break

        return False
