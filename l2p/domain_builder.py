"""
This file contains collection of functions for PDDL domain generation purposes
"""

from .utils import *
from .llm import BaseLLM, require_llm
from collections import OrderedDict
import re, time

REQUIREMENTS = [
    ":strips",
    ":typing",
    ":equality",
    ":negative-preconditions",
    ":disjunctive-preconditions",
    ":universal-preconditions",
    ":conditional-effects",
]

class DomainBuilder:
    def __init__(
        self,
        types: dict[str, str] = None,
        type_hierarchy: dict[str, str] = None,
        predicates: list[Predicate] = None,
        nl_actions: dict[str, str] = None,
        pddl_actions: list[Action] = None,
    ):
        """
        Initializes a domain builder object

        Args:
            types (dict[str,str]): types dictionary with name: description key-value pair
            type_hierarchy (dict[str,str]): type hierarchy dictionary
            predicates (list[Predicate]): list of Predicate objects
            nl_actions (dict[str,str]): dictionary of extracted actions, where the keys are action names and values are action descriptions
            pddl_actions (list[Action]): list of Action objects
        """
        self.types = types
        self.type_hierarchy = type_hierarchy
        self.predicates = predicates
        self.nl_actions = nl_actions
        self.pddl_actions = pddl_actions

    """Extract functions"""

    @require_llm
    def extract_types(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str,str] | list[dict[str,str]] | None,
        check_invalid_obj_usage: bool = True,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[dict[str,str], str, tuple[bool, str]]:
        """
        Extracts types with domain given

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            types (dict[str,str]): current types in model, defaults to None
            check_invalid_obj_usage (bool): removes keyword `object` from types, defaults to True
            syntax_validator (SyntaxValidator): syntax checker for types, defaults to None
            max_retries (int): max # of retries if failure occurs, defaults to 3

        Returns:
            types (dict[str,str]): dictionary of types with {<name>: <description>} pair
            llm_output (str): the raw string BaseLLM response
            validation_info (dict[bool,str]): validation info containing pass flag and error message
        """

        prompt_data = {
            "domain_desc": domain_desc,
            "types": format_types_to_string(types) if types else "No types provided."
        }
        
        prompt = prompt_template.format(**prompt_data)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # parse LLM output into types
                types = parse_types(llm_output=llm_output)

                # flag that removes keyword 'object' if detected
                if check_invalid_obj_usage:
                    if types and "object" in types:
                        del types["object"]

                # run syntax validation if applicable
                validation_info = (True, "All validations passed.")
                if syntax_validator:
                    for error_type in syntax_validator.error_types:
                        validator = getattr(syntax_validator, f"{error_type}", None)
                        if not callable(validator):
                            continue

                        # dispatch based on expected arguments
                        if error_type == "validate_format_types":
                            validation_info = validator(types)
                        
                        if not validation_info[0]:
                            return types, llm_output, validation_info
                
                return types, llm_output, validation_info
            
            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract types.")

    @require_llm
    def extract_type_hierarchy(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str,str] | list[dict[str,str]] | None,
        check_invalid_obj_usage: bool = True,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[list[dict[str,str]], str, tuple[bool,str]]:
        """
        Extracts type hierarchy from types list and domain given

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            types (dict[str,str]): current types in model
            check_invalid_obj_usage (bool): removes keyword `object` from types, defaults to True
            syntax_validator (SyntaxValidator): syntax checker for types, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            type_hierarchy (list[dict[str,str]]): list of dictionaries containing the type hierarchy
            llm_output (str): the raw string BaseLLM response
            validation_info (dict[bool,str]): validation info containing pass flag and error message
        """
        
        prompt_data = prompt_data = {
            "domain_desc": domain_desc,
            "types": format_types_to_string(types) if types else "No types provided."
        }

        prompt = prompt_template.format(**prompt_data)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # extract respective types from response
                type_hierarchy = parse_type_hierarchy(llm_output=llm_output)

                # flag that removes keyword 'object' if detected
                if type_hierarchy is not None:
                    if check_invalid_obj_usage:
                        # promote children if top-level "object" type exists
                        new_hierarchy = []
                        for entry in type_hierarchy:
                            if "object" in entry:
                                children = entry.get("children", [])
                                new_hierarchy.extend(children)
                            else:
                                new_hierarchy.append(entry)
                        type_hierarchy = new_hierarchy
                
                # run syntax validation if applicable
                validation_info = (True, "All validations passed.")
                if syntax_validator:
                    for error_type in syntax_validator.error_types:
                        validator = getattr(syntax_validator, f"{error_type}", None)
                        if not callable(validator):
                            continue

                        # dispatch based on expected arguments
                        if error_type == "validate_format_types":
                            validation_info = validator(type_hierarchy)

                        if not validation_info[0]:
                            return type_hierarchy, llm_output, validation_info
                
                return type_hierarchy, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract types.")

    @require_llm
    def extract_nl_actions(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str,str]] = None,
        nl_actions: dict[str, str] = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str]:
        """
        Extract actions in natural language given domain description using BaseLLM.

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            types (dict[str,str]): current types in model
            nl_actions (dict[str, str]): NL actions currently in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            nl_actions (dict[str, str]): a dictionary of extracted actions {action name: action description}
            llm_output (str): the raw string BaseLLM response
        """

        prompt_data = prompt_data = {
            "domain_desc": domain_desc,
            "types": format_types_to_string(types) if types else "No types provided.",
            "nl_actions": "\n".join(f" - {name}: {desc}" for name, desc in nl_actions.items()) if nl_actions else "No actions provided."
        }

        prompt = prompt_template.format(**prompt_data)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # extract respective nl actions from response
                nl_actions = parse_types(llm_output=llm_output)

                if nl_actions is not None:
                    return nl_actions, llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract NL actions.")
    
    @require_llm
    def extract_pddl_action(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str = None,
        action_list: list[str] = None,
        predicates: list[Predicate] = None,
        types: dict[str,str] | list[dict[str,str]] = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3
    ) -> tuple[Action, list[Predicate], str, tuple[bool, str]]:
        """
        Extract an action and predicates from a given action description using BaseLLM

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): action construction prompt
            action_name (str): action name
            action_desc (str): action description, defaults to None
            action_list (dict[str,str]): dictionary of other actions to be translated, defaults to None
            predicates (list[Predicate]): list of predicates in current model, defaults to None
            types (dict[str,str] | list[dict[str,str]]): current types in model, defaults to None
            syntax_validator (SyntaxValidator): custom syntax validator, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            action (Action): constructed action class
            new_predicates (list[Predicate]): a list of new predicates
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool, str]): validation check information
        """
        
        prompt_data = {
            "domain_desc": domain_desc,
            "action_list": "\n".join([f"- {a}" for a in action_list]) if action_list else "No other actions provided.",
            "action_name": action_name,
            "action_desc": action_desc or "No description available.",
            "types": format_types_to_string(types) if types else "No types provided.",
            "predicates": "\n".join([f"- {pred['clean']}" for pred in predicates]) if predicates else "No predicates provided."
        }
        
        prompt = prompt_template.format(**prompt_data)
        
        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)
                
                # parse LLM output into action and predicates
                action = parse_action(llm_output=llm_output, action_name=action_name)
                new_predicates = parse_new_predicates(llm_output=llm_output)
                
                # run syntax validation if applicable
                validation_info = (True, "All validations passed.")
                if syntax_validator:
                    for error_type in syntax_validator.error_types:
                        validator = getattr(syntax_validator, f"{error_type}", None)
                        if not callable(validator):
                            continue
                        
                        # dispatch based on expected arguments
                        if error_type == "validate_header":
                            validation_info = validator(llm_output)
                        elif error_type == "validate_duplicate_headers":
                            validation_info = validator(llm_output)
                        elif error_type == "validate_unsupported_keywords":
                            validation_info = validator(llm_output)
                        elif error_type == "validate_params":
                            validation_info = validator(action["params"], types)
                        elif error_type == "validate_types_predicates":
                            validation_info = validator(new_predicates, types)
                        elif error_type == "validate_format_predicates":
                            validation_info = validator(new_predicates, types)
                        elif error_type == "validate_usage_predicates":
                            validation_info = validator(llm_output, predicates, types)
                        
                        if not validation_info[0]:
                            return action, new_predicates, llm_output, validation_info
                
                return action, new_predicates, llm_output, validation_info
            
            except Exception as e:
                print(
                    f"Error on attempt {attempt + 1}/{max_retries}: {e}\n"
                    f"LLM Output:\n{llm_output if 'llm_output' in locals() else 'None'}\nRetrying...\n"
                )
                time.sleep(2)

        raise RuntimeError("Max retries exceeded. Failed to extract PDDL action.")


    # CURRENTLY EXPERIMENTAL - in principal, this function is practically tasking LLM to generate whole domain actions
    @require_llm
    def extract_pddl_actions(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        nl_actions: dict[str, str] = None,
        predicates: list[Predicate] = None,
        types: dict[str, str] | list[dict[str,str]] = None,
    ) -> tuple[list[Action], list[Predicate], str]:
        """
        Extract all actions from a given action description using BaseLLM

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): action construction prompt
            nl_actions (dict[str, str]): NL actions currently in model
            predicates (list[Predicate]): list of predicates
            types (dict[str,str]): current types in model

        Returns:
            action (Action): constructed action class
            new_predicates (list[Predicate]): a list of new predicates
            llm_response (str): the raw string BaseLLM response
        """

        model.reset_tokens()

        # replace prompt placeholders
        types_str = pretty_print_dict(types) if types else "No types provided."
        predicates_str = (
            pretty_print_predicates(predicates) if predicates else "No predicates provided."
        )
        nl_actions_str = (
            pretty_print_dict(nl_actions) if nl_actions else "No actions provided."
        )

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{types}", types_str)
        prompt_template = prompt_template.replace("{predicates}", predicates_str)
        prompt_template = prompt_template.replace("{nl_actions}", nl_actions_str)

        llm_response = model.query(prompt=prompt_template)

        # extract respective types from response
        raw_actions = llm_response.split("## NEXT ACTION")

        actions = []
        for i in raw_actions:
            # define the regex patterns
            action_pattern = re.compile(r"\[([^\]]+)\]")
            rest_of_string_pattern = re.compile(r"\[([^\]]+)\](.*)", re.DOTALL)

            # search for the action name
            action_match = action_pattern.search(i)
            action_name = action_match.group(1) if action_match else None

            # extract the rest of the string
            rest_match = rest_of_string_pattern.search(i)
            rest_of_string = rest_match.group(2).strip() if rest_match else None

            actions.append(
                parse_action(llm_response=rest_of_string, action_name=action_name)
            )

        # if user queries predicate creation via BaseLLM
        try:
            new_predicates = parse_new_predicates(llm_response)

            if predicates:
                new_predicates = [
                    pred
                    for pred in new_predicates
                    if pred["name"] not in [p["name"] for p in predicates]
                ]  # remove re-defined predicates
        except Exception as e:
            # Log or print the exception if needed
            print(f"No new predicates: {e}")
            new_predicates = None

        return actions, new_predicates, llm_response

    @require_llm
    def extract_parameters(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str,
        types: dict[str, str] | list[dict[str,str]] = None,
        max_retries: int = 3,
    ) -> tuple[OrderedDict, list, str]:
        """
        Extracts parameters from single action description via BaseLLM

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            action_name (str): action name
            action_desc (str): action description
            types (dict[str,str]): current types in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            param (OrderedDict): ordered list of parameters
            param_raw (list()): list of raw parameters
            llm_response (str): the raw string BaseLLM response
        """

        # replace prompt placeholders
        types_str = pretty_print_dict(types) if types else "No types provided."

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{action_name}", action_name)
        prompt_template = prompt_template.replace("{action_desc}", action_desc)
        prompt_template = prompt_template.replace("{types}", types_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)  # get BaseLLM response

                # extract respective types from response
                param, param_raw = parse_params(llm_output=llm_response)

                return param, param_raw, llm_response

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_response if 'llm_response' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract parameters.")

    @require_llm
    def extract_preconditions(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str,
        params: list[str] = None,
        predicates: list[Predicate] = None,
        max_retries: int = 3,
    ) -> tuple[str, list[Predicate], str]:
        """
        Extracts preconditions from single action description via BaseLLM

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            action_name (str): action name
            action_desc (str): action description
            params (list[str]): list of parameters from action
            predicates (list[Predicate]): list of current predicates in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            preconditions (str): PDDL format of preconditions
            new_predicates (list[Predicate]): a list of new predicates
            llm_response (str): the raw string BaseLLM response
        """

        # replace prompt placeholders
        predicates_str = (
            pretty_print_predicates(predicates) if predicates else "No predicates provided."
        )
        params_str = "\n".join(params) if params else "No parameters provided."

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{action_name}", action_name)
        prompt_template = prompt_template.replace("{action_desc}", action_desc)
        prompt_template = prompt_template.replace("{parameters}", params_str)
        prompt_template = prompt_template.replace("{predicates}", predicates_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)  # get BaseLLM response

                # extract respective types from response
                preconditions = (
                    llm_response.split("Preconditions\n")[1]
                    .split("##")[0]
                    .split("```")[1]
                    .strip(" `\n")
                )
                new_predicates = parse_new_predicates(llm_output=llm_response)

                return preconditions, new_predicates, llm_response

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_response if 'llm_response' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract preconditions.")

    @require_llm
    def extract_effects(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str,
        params: list[str] = None,
        precondition: str = None,
        predicates: list[Predicate] = None,
        max_retries: int = 3,
    ) -> tuple[str, list[Predicate], str]:
        """
        Extracts effects from single action description via BaseLLM

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            action_name (str): action name
            action_desc (str): action description
            params (list[str]): list of parameters from action
            precondition (str): PDDL format of preconditions
            predicates (list[Predicate]): list of current predicates in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            effects (str): PDDL format of effects
            new_predicates (list[Predicate]): a list of new predicates
            llm_response (str): the raw string BaseLLM response
        """

        # replace prompt placeholders
        predicates_str = (
            pretty_print_predicates(predicates) if predicates else "No predicates provided."
        )
        params_str = "\n".join(params) if params else "No parameters provided."

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{action_name}", action_name)
        prompt_template = prompt_template.replace("{action_desc}", action_desc)
        prompt_template = prompt_template.replace("{parameters}", params_str)
        prompt_template = prompt_template.replace("{preconditions}", precondition)
        prompt_template = prompt_template.replace("{predicates}", predicates_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)  # get BaseLLM response

                # extract respective types from response
                effects = (
                    llm_response.split("Effects\n")[1]
                    .split("##")[0]
                    .split("```")[1]
                    .strip(" `\n")
                )
                new_predicates = parse_new_predicates(llm_output=llm_response)

                return effects, new_predicates, llm_response

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_response if 'llm_response' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract effects.")

    @require_llm
    def extract_predicates(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str,str]] = None,
        predicates: list[Predicate] = None,
        nl_actions: dict[str, str] = None,
        max_retries: int = 3,
    ) -> tuple[list[Predicate], str]:
        """
        Extracts predicates via BaseLLM

        Args:
            model (BaseLLM): BaseLLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            types (dict[str,str]): current types in model
            predicates (list[Predicate]): list of current predicates in model
            nl_actions (dict[str, str]): NL actions currently in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            new_predicates (list[Predicate]): a list of new predicates
            llm_response (str): the raw string BaseLLM response
        """

        # replace prompt placeholders
        types_str = pretty_print_dict(types) if types else "No types provided."
        predicates_str = (
            pretty_print_predicates(predicates) if predicates else "No predicates provided."
        )
        nl_actions_str = (
            "\n".join(f"{name}: {desc}" for name, desc in nl_actions.items())
            if nl_actions
            else "No actions provided."
        )

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{types}", types_str)
        prompt_template = prompt_template.replace("{predicates}", predicates_str)
        prompt_template = prompt_template.replace("{nl_actions}", nl_actions_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)  # prompt model

                # extract respective types from response
                new_predicates = parse_new_predicates(llm_output=llm_response)

                return new_predicates, llm_response

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_response if 'llm_response' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract predicates.")


    """Delete functions"""
    
    def delete_type(self, name: str):
        """Deletes a specific type from both `self.types` and `self.type_hierarchy`."""
        
        # remove from flat types dictionary if present
        if self.types is not None:
            self.types = {type_: desc for type_, desc in self.types.items() if type_ != name}

        def remove_and_promote(node_list):
            updated_list = []

            for node in node_list:
                # get the current node's type name and description
                type_name = next((k for k in node if k != "children"), None)
                if type_name is None:
                    continue

                # if this is the type to remove, promote its children to the current level
                if type_name == name:
                    children = node.get("children", [])
                    updated_list.extend(remove_and_promote(children))
                else:
                    # recursively clean the children
                    children = remove_and_promote(node.get("children", []))
                    updated_node = {
                        type_name: node[type_name],
                        "children": children
                    }
                    updated_list.append(updated_node)

            return updated_list

        # update the type_hierarchy if it exists
        if self.type_hierarchy is not None:
            self.type_hierarchy = remove_and_promote(self.type_hierarchy)

    def delete_nl_action(self, name: str):
        """Deletes specific NL action from current model"""
        if self.nl_actions is not None:
            self.nl_actions = {
                action_name: action_desc
                for action_name, action_desc in self.nl_actions.items()
                if action_name != name
            }

    def delete_pddl_action(self, name: str):
        """Deletes specific PDDL action from current model"""
        if self.pddl_actions is not None:
            self.pddl_actions = [
                action for action in self.pddl_actions if action["name"] != name
            ]

    def delete_predicate(self, name: str):
        """Deletes specific predicate from current model"""
        if self.predicates is not None:
            self.predicates = [
                predicate for predicate in self.predicates if predicate["name"] != name
            ]


    """Set functions"""

    def set_types(self, types: dict[str, str]):
        """Sets types for current model"""
        self.types = types

    def set_type_hierarchy(self, type_hierarchy: list[dict[str, str]]):
        """Sets type hierarchy for current model"""
        self.type_hierarchy = type_hierarchy

    def set_nl_actions(self, nl_actions: dict[str, str]):
        """Sets NL actions for current model"""
        self.nl_actions = nl_actions

    def set_pddl_action(self, pddl_action: Action):
        """Appends a PDDL action for current model"""
        self.pddl_actions.append(pddl_action)

    def set_predicate(self, predicate: Predicate):
        """Appends a predicate for current model"""
        self.predicates.append(predicate)


    """Get functions"""

    def get_types(self):
        """Returns types from current model"""
        return self.types

    def get_type_hierarchy(self):
        """Returns type hierarchy from current model"""
        return self.type_hierarchy

    def get_nl_actions(self):
        """Returns natural language actions from current model"""
        return self.nl_actions

    def get_pddl_actions(self):
        """Returns PDDL actions from current model"""
        return self.pddl_actions

    def get_predicates(self):
        """Returns predicates from current model"""
        return self.predicates


    def generate_domain(
        self,
        domain_name: str,
        types: dict[str,str] | list[dict[str,str]] | None = None,
        predicates: list[Predicate] | None = None,
        actions: list[Action] = [],
        requirements: list[str] = REQUIREMENTS,
        append_obj_type_to_parent: bool = True
    ) -> str:
        """
        Generates PDDL domain from given information

        Args:
            domain_name (str): domain name
            types (str | None): domain types
            predicates (str | None): domain predicates
            actions (list[Action]): domain actions
            requirements (list[str]): domain requirements

        Returns:
            desc (str): PDDL domain
        """

        desc = ""
        desc += f"(define (domain {domain_name})\n"
        desc += indent(string=f"(:requirements\n   {' '.join(requirements)})", level=1)
        if types:
            types_str = format_types_to_string(types, append_obj_type_to_parent)
            desc += f"\n\n   (:types \n{indent(string=types_str, level=2)}\n   )"
        if not predicates:
            print("[WARNING]: Domain has no predicates. This may cause planners to reject the domain or behave unexpectedly.")
        else:
            pred_str = format_predicates(predicates)
            desc += f"\n\n   (:predicates \n{indent(string=pred_str, level=2)}\n   )"
        if not actions:
            print("[WARNING]: Domain has no actions. The planner will not be able to generate any plan unless the goal is already satisfied.")
        else:
            desc += format_actions(actions)
        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc
