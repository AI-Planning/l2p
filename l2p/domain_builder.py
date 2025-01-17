"""
This file contains collection of functions for PDDL domain generation purposes
"""

from .utils import *
from .llm_builder import LLM, require_llm
from collections import OrderedDict
import re, time


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
    def extract_type(
        self,
        model: LLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str]:
        """
        Extracts types with domain given

        Args:
            model (LLM): LLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            types (dict[str,str]): current types in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            type_dict (dict[str,str]): dictionary of types with (name:description) pair
            llm_response (str): the raw string LLM response
        """

        # replace prompt placeholders
        types_str = format_dict(types) if types else "No types provided."

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{types}", types_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)  # prompt model

                # extract respective types from response
                types = convert_to_dict(llm_response=llm_response)

                if types is not None:
                    return types, llm_response

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_response if 'llm_response' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract types.")

    @require_llm
    def extract_type_hierarchy(
        self,
        model: LLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str]:
        """
        Extracts type hierarchy from types list and domain given

        Args:
            model (LLM): LLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            types (dict[str,str]): current types in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            type_hierarchy (dict[str,str]): dictionary of type hierarchy
            llm_response (str): the raw string LLM response
        """

        # replace prompt placeholders
        types_str = format_dict(types) if types else "No types provided."

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{types}", types_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:

                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)

                # extract respective types from response
                type_hierarchy = convert_to_dict(llm_response=llm_response)

                if type_hierarchy is not None:
                    return type_hierarchy, llm_response

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_response if 'llm_response' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract type hierarchy.")

    @require_llm
    def extract_nl_actions(
        self,
        model: LLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] = None,
        nl_actions: dict[str, str] = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str]:
        """
        Extract actions in natural language given domain description using LLM.

        Args:
            model (LLM): LLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            types (dict[str,str]): current types in model
            nl_actions (dict[str, str]): NL actions currently in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            nl_actions (dict[str, str]): a dictionary of extracted actions {action name: action description}
            llm_response (str): the raw string LLM response
        """

        # replace prompt placeholders
        types_str = format_dict(types) if types else "No types provided."
        nl_actions_str = (
            "\n".join(f"{name}: {desc}" for name, desc in nl_actions.items())
            if nl_actions
            else "No actions provided."
        )

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{types}", types_str)
        prompt_template = prompt_template.replace("{nl_actions}", nl_actions_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:

                model.reset_tokens()

                llm_response = model.query(
                    prompt=prompt_template
                )  # get LLM llm_response

                # extract respective types from response
                nl_actions = convert_to_dict(llm_response=llm_response)

                if nl_actions is not None:
                    return nl_actions, llm_response

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_response if 'llm_response' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract NL actions.")

    @require_llm
    def extract_pddl_action(
        self,
        model: LLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str = None,
        action_list: str = None,
        predicates: list[Predicate] = None,
        types: dict[str, str] = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[Action, list[Predicate], str, tuple[bool, str]]:
        """
        Extract an action and predicates from a given action description using LLM

        Args:
            model (LLM): LLM
            domain_desc (str): domain description
            prompt_template (str): action construction prompt
            action_name (str): action name
            action_desc (str): action description
            action_list (dict[str,str]): dictionary of other actions to be translated
            predicates (list[Predicate]): list of predicates in current model
            types (dict[str,str]): current types in model
            syntax_validator (SyntaxValidator): custom syntax validator, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            action (Action): constructed action class
            new_predicates (list[Predicate]): a list of new predicates
            llm_response (str): the raw string LLM response
            validation_info (tuple[bool, str]): validation check information
        """

        # replace prompt placeholders
        types_str = format_dict(types) if types else "No types provided."
        predicates_str = (
            "\n".join([f"- {pred['clean']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        action_list_str = action_list if action_list else "No other actions provided"

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{action_list}", action_list_str)
        prompt_template = prompt_template.replace("{action_name}", action_name)
        prompt_template = prompt_template.replace(
            "{action_desc}", action_desc if action_desc else "No description available."
        )
        prompt_template = prompt_template.replace("{types}", types_str)
        prompt_template = prompt_template.replace("{predicates}", predicates_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:

                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)

                # extract respective types from response
                action = parse_action(
                    llm_response=llm_response, action_name=action_name
                )
                new_predicates = parse_new_predicates(llm_response)

                # if syntax validator is enabled, run through checks, returns check message
                validation_info = [True, "All validations passed."]
                if syntax_validator:
                    for e in syntax_validator.error_types:
                        if e == "invalid_header":
                            validation_info = syntax_validator.validate_header(
                                llm_response
                            )
                        elif e == "invalid_keyword_usage":
                            validation_info = syntax_validator.validate_keyword_usage(
                                llm_response
                            )
                        elif e == "unsupported_keywords":
                            validation_info = (
                                syntax_validator.validate_unsupported_keywords(
                                    llm_response, syntax_validator.unsupported_keywords
                                )
                            )
                        elif e == "invalid_param_types":
                            validation_info = syntax_validator.validate_params(
                                action["params"], types
                            )
                        elif e == "invalid_predicate_name":
                            validation_info = (
                                syntax_validator.validate_types_predicates(
                                    new_predicates, types
                                )
                            )
                        elif e == "invalid_predicate_format":
                            validation_info = (
                                syntax_validator.validate_format_predicates(
                                    predicates, types
                                )
                            )
                        elif e == "invalid_predicate_usage":
                            validation_info = (
                                syntax_validator.validate_usage_predicates(
                                    llm_response, predicates, types
                                )
                            )
                        else:
                            raise NotImplementedError(
                                f"Validation type '{e}' is not implemented."
                            )

                        if not validation_info[0]:
                            return action, new_predicates, llm_response, validation_info

                if action is not None and new_predicates is not None:
                    return action, new_predicates, llm_response, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_response if 'llm_response' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract PDDL action.")

    @require_llm
    def extract_pddl_actions(
        self,
        model: LLM,
        domain_desc: str,
        prompt_template: str,
        nl_actions: dict[str, str] = None,
        predicates: list[Predicate] = None,
        types: dict[str, str] = None,
    ) -> tuple[list[Action], list[Predicate], str]:
        """
        Extract all actions from a given action description using LLM

        Args:
            model (LLM): LLM
            domain_desc (str): domain description
            prompt_template (str): action construction prompt
            nl_actions (dict[str, str]): NL actions currently in model
            predicates (list[Predicate]): list of predicates
            types (dict[str,str]): current types in model

        Returns:
            action (Action): constructed action class
            new_predicates (list[Predicate]): a list of new predicates
            llm_response (str): the raw string LLM response
        """

        model.reset_tokens()

        # replace prompt placeholders
        types_str = format_dict(types) if types else "No types provided."
        predicates_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        nl_actions_str = (
            format_dict(nl_actions) if nl_actions else "No actions provided."
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

        # if user queries predicate creation via LLM
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
        model: LLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str,
        types: dict[str, str] = None,
        max_retries: int = 3,
    ) -> tuple[OrderedDict, list, str]:
        """
        Extracts parameters from single action description via LLM

        Args:
            model (LLM): LLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            action_name (str): action name
            action_desc (str): action description
            types (dict[str,str]): current types in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            param (OrderedDict): ordered list of parameters
            param_raw (list()): list of raw parameters
            llm_response (str): the raw string LLM response
        """

        # replace prompt placeholders
        types_str = format_dict(types) if types else "No types provided."

        prompt_template = prompt_template.replace("{domain_desc}", domain_desc)
        prompt_template = prompt_template.replace("{action_name}", action_name)
        prompt_template = prompt_template.replace("{action_desc}", action_desc)
        prompt_template = prompt_template.replace("{types}", types_str)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_response = model.query(prompt=prompt_template)  # get LLM response

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
        model: LLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str,
        params: list[str] = None,
        predicates: list[Predicate] = None,
        max_retries: int = 3,
    ) -> tuple[str, list[Predicate], str]:
        """
        Extracts preconditions from single action description via LLM

        Args:
            model (LLM): LLM
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
            llm_response (str): the raw string LLM response
        """

        # replace prompt placeholders
        predicates_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
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

                llm_response = model.query(prompt=prompt_template)  # get LLM response

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
        model: LLM,
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
        Extracts effects from single action description via LLM

        Args:
            model (LLM): LLM
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
            llm_response (str): the raw string LLM response
        """

        # replace prompt placeholders
        predicates_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
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

                llm_response = model.query(prompt=prompt_template)  # get LLM response

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
        model: LLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] = None,
        predicates: list[Predicate] = None,
        nl_actions: dict[str, str] = None,
        max_retries: int = 3,
    ) -> tuple[list[Predicate], str]:
        """
        Extracts predicates via LLM

        Args:
            model (LLM): LLM
            domain_desc (str): domain description
            prompt_template (str): prompt template
            types (dict[str,str]): current types in model
            predicates (list[Predicate]): list of current predicates in model
            nl_actions (dict[str, str]): NL actions currently in model
            max_retries (int): max # of retries if failure occurs

        Returns:
            new_predicates (list[Predicate]): a list of new predicates
            llm_response (str): the raw string LLM response
        """

        # replace prompt placeholders
        types_str = format_dict(types) if types else "No types provided."
        predicates_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
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
        """Deletes specific type from current model"""
        if self.types is not None:
            self.types = {
                type_: desc for type_, desc in self.types.items() if type_ != name
            }

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

    def set_type_hierarchy(self, type_hierarchy: dict[str, str]):
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
        domain: str,
        types: str,
        predicates: str,
        actions: list[Action],
        requirements: list[str],
    ) -> str:
        """
        Generates PDDL domain from given information

        Args:
            domain (str): domain name
            types (str): domain types
            predicates (str): domain predicates
            actions (list[Action]): domain actions
            requirements (list[str]): domain requirements

        Returns:
            desc (str): PDDL domain
        """
        desc = ""
        desc += f"(define (domain {domain})\n"
        desc += (
            indent(string=f"(:requirements\n   {' '.join(requirements)})", level=1)
            + "\n\n"
        )
        desc += f"   (:types \n{indent(string=types, level=2)}\n   )\n\n"
        desc += f"   (:predicates \n{indent(string=predicates, level=2)}\n   )"
        desc += self.action_descs(actions)
        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc

    def action_desc(self, action: Action) -> str:
        """Helper function to format individual action descriptions"""
        param_str = "\n".join(
            [f"{name} - {type}" for name, type in action["params"].items()]
        )  # name includes ?
        desc = f"(:action {action['name']}\n"
        desc += f"   :parameters (\n{indent(string=param_str, level=2)}\n   )\n"
        desc += f"   :precondition\n{indent(string=action['preconditions'], level=2)}\n"
        desc += f"   :effect\n{indent(string=action['effects'], level=2)}\n"
        desc += ")"
        return desc

    def action_descs(self, actions) -> str:
        """Helper function to combine all action descriptions"""
        desc = ""
        for action in actions:
            desc += "\n\n" + indent(self.action_desc(action), level=1)
        return desc

    def format_predicates(self, predicates: list[Predicate]) -> str:
        """Helper function that formats predicate list into string"""
        return "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])
