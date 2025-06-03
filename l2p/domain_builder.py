"""
PDDL Domain Formalization/Generation Functions

This module defines the `DomainBuilder` class and related utilities for constructing
PDDL domain specifications programatically.

Refer to: https://marcustantakoun.github.io/l2p.github.io/l2p.html for more information
how to use class functions. Refer to /templates in: https://github.com/AI-Planning/l2p
for how to structurally prompt LLMs so they are compatible with class function parsing.
"""

import re
import time

from collections import OrderedDict
from typing import Any

from .llm import BaseLLM, require_llm
from .utils import *


class DomainBuilder:
    def __init__(
        self,
        requirements: list[str] = None,
        types: dict[str, str] = None,
        type_hierarchy: list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
        pddl_actions: list[Action] = None,
    ) -> None:
        """
        Initializes an L2P domain builder object.

        Args:
            requirements (list[str]): list of PDDL requirements
            types (dict[str,str]): flat types dictionary w/ {name: description} key-value pair (PDDL :types)
            type_hierarchy (list[dict[str,str]]): type hierarchy dictionary list (PDDL :types)
            constants (dict[str,str]): flat constant dictionary w/ {name: type} key-value pair (PDDL :constants)
            predicates (list[Predicate]): list of Predicate objects (PDDL :predicates)
            functions (list[Function]): list of Function objects (PDDL :functions)
            pddl_actions (list[Action]): list of Action objects (PDDL :action)
        """

        self.requirements = requirements or []
        self.types = types or {}
        self.type_hierarchy = type_hierarchy or []
        self.constants = constants or {}
        self.predicates = predicates or []
        self.functions = functions or []
        self.pddl_actions = pddl_actions or []

    """Formalize/generate functions"""

    @require_llm
    def formalize_types(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        check_invalid_obj_usage: bool = True,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str, tuple[bool, str]]:
        """
        Formalizes PDDL :types in singular flat hierarchy via LLM. It is recommended to use
        `formalize_type_hierarchy()` for sub-type support.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :types extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            check_invalid_obj_usage (bool): removes keyword `object` from types, defaults to True
            syntax_validator (SyntaxValidator): syntax checker for generated types, defaults to None
            max_retries (int): max # of retries if failure occurs, defaults to 3

        Returns:
            types (dict[str,str]): dictionary of types with {<name>: <description>} pair
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        types_str = pretty_print_dict(types) if types else "No types provided."

        prompt = prompt_template.replace("{domain_desc}", domain_desc).replace(
            "{types}", types_str
        )

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
    def formalize_type_hierarchy(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        check_invalid_obj_usage: bool = True,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[list[dict[str, str]], str, tuple[bool, str]]:
        """
        Formalizes PDDL :types in hierarchy format via LLM. Recommended to use over `formalize_types()`

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :types extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            check_invalid_obj_usage (bool): removes keyword `object` from types, defaults to True
            syntax_validator (SyntaxValidator): syntax checker for generated types, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            type_hierarchy (list[dict[str,str]]): list of dictionaries containing the type hierarchy
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        types_str = pretty_print_dict(types) if types else "No types provided."

        prompt = prompt_template.replace("{domain_desc}", domain_desc).replace(
            "{types}", types_str
        )

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
                        elif error_type == "validate_cyclic_types":
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
    def formalize_constants(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str, tuple[bool, str]]:
        """
        Formalizes PDDL :constants in flat dictionary format via LLM.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :constants extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            syntax_validator (SyntaxValidator): syntax checker for generated constants, defaults to None
            max_retries (int): max # of retries if failure occurs, defaults to 3

        Returns:
            constants (dict[str,str]): dictionary of constants with {<name>: <type>} pair
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # parse LLM output into constants
                constants = parse_constants(llm_output=llm_output)

                # run syntax validation if applicable
                validation_info = (True, "All validations passed.")
                if syntax_validator:
                    for error_type in syntax_validator.error_types:
                        validator = getattr(syntax_validator, f"{error_type}", None)
                        if not callable(validator):
                            continue

                        # dispatch based on expected arguments
                        if error_type == "validate_constant_types":
                            validation_info = validator(constants, types)

                        if not validation_info[0]:
                            return constants, llm_output, validation_info

                return constants, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract constants.")

    @require_llm
    def formalize_predicates(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[list[Predicate], str, tuple[bool, str]]:
        """
        Formalizes PDDL :predicates via LLM.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :predicates extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in specification, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            syntax_validator (SyntaxValidator): syntax checker for generated predicates, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            new_predicates (list[Predicate]): a list of new predicates
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool, str]): validation info containing pass flag and error message
        """

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)  # prompt model

                # extract new predicates from response
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
                        elif error_type == "validate_types_predicates":
                            validation_info = validator(new_predicates, types)
                        elif error_type == "validate_format_predicates":
                            validation_info = validator(new_predicates, types)
                        elif error_type == "validate_duplicate_predicates":
                            validation_info = validator(predicates, new_predicates)

                        if not validation_info[0]:
                            return new_predicates, llm_output, validation_info

                return new_predicates, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract predicates.")

    @require_llm
    def formalize_functions(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
        syntax_validator: SyntaxValidator = None,
        max_retries=3,
    ) -> tuple[list[Function], str, tuple[bool, str]]:
        """
        Formalizes PDDL :functions via LLM

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :functions extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in specification, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            syntax_validator (SyntaxValidator): syntax checker for generated functions, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            functions (list[Function]): a list of generated :functions
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # extract functions from response
                functions = parse_functions(llm_output=llm_output)

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
                        elif error_type == "validate_format_functions":
                            validation_info = validator(functions, types)

                        if not validation_info[0]:
                            return functions, llm_output, validation_info

                return functions, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract functions.")

    @require_llm
    def extract_nl_actions(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] = None,
        nl_actions: dict[str, str] = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str]:
        """
        Extract actions in natural language given domain description using BaseLLM.

        NOTE: This is not an official formalize function. It is inspired by the NL2PLAN framework
        (Gestrin et al., 2024) and is designed to guide the LLM in constructing appropriate actions.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for dictionary extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            nl_actions (dict[str, str]): NL actions currently in class object w/ {<name>: <description>} key-value pair
            max_retries (int): max # of retries if failure occurs

        Returns:
            nl_actions (dict[str, str]): a dictionary of extracted NL actions {<name>: <description>}
            llm_output (str): the raw string BaseLLM response
        """

        types_str = pretty_print_dict(types) if types else "No types provided."
        nl_act_str = (
            "\n".join(f" - {name}: {desc}" for name, desc in nl_actions.items())
            if nl_actions
            else "No actions provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{nl_actions}", nl_act_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # extract respective nl actions from response
                nl_actions = parse_types(llm_output=llm_output, heading="ACTIONS")

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
    def formalize_pddl_action(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str = None,
        action_list: list[str] = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] | None = None,
        functions: list[Function] | None = None,
        extract_new_preds=False,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[Action, list[Predicate], str, tuple[bool, str]]:
        """
        Formalizes an :action and new :predicates from a given action description using BaseLLM.
        Users can set `extract_new_preds (bool)` to True if tasking LLM to generate new predicates.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :action extraction
            action_name (str): action name
            action_desc (str): action description, defaults to None
            action_list (list[str]): list of other actions to be translated, defaults to None
            types (dict[str,str] | list[dict[str,str]]): types in current specification, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in specification, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            extract_new_preds (bool): flag for parsing new predicates generated from action, defaults to False
            syntax_validator (SyntaxValidator): syntax checker for generated actions
            max_retries (int): max # of retries if failure occurs

        Returns:
            action (Action): constructed action class containing :parameters, :preconditions, and :effects
            new_predicates (list[Predicate]): a list of new predicates, defaults to empty list
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool, str]): validation info containing pass flag and error message
        """

        act_list_str = (
            "\n".join([f"- {a}" for a in action_list])
            if action_list
            else "No other actions provided."
        )
        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{action_list}", act_list_str)
            .replace("{action_name}", action_name)
            .replace("{action_desc}", action_desc or "No description available.")
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # parse LLM output into action and predicates
                action = parse_action(llm_output=llm_output, action_name=action_name)

                if extract_new_preds:
                    new_predicates = parse_new_predicates(llm_output=llm_output)
                else:
                    new_predicates = []

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
                        elif error_type == "validate_duplicate_predicates":
                            validation_info == validator(predicates, new_predicates)
                        elif error_type == "validate_types_predicates":
                            validation_info = validator(new_predicates, types)
                        elif error_type == "validate_format_predicates":
                            validation_info = validator(new_predicates, types)
                        elif error_type == "validate_usage_action":
                            validation_info = validator(
                                llm_output,
                                predicates,
                                types,
                                functions,
                                extract_new_preds,
                            )

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

    # NOTE: This function is experimental and may be subject to change in future versions.
    @require_llm
    def formalize_pddl_actions(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_list: list[str] = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] | None = None,
        functions: list[Function] | None = None,
        extract_new_preds=False,
        max_retries: int = 3,
    ) -> tuple[list[Action], list[Predicate], str]:
        """
        Formalizes several :actions via LLM.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): domain description
            prompt_template (str): action construction prompt
            action_list (list[str]): list of other actions to be translated, defaults to None
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in specification, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            extract_new_preds (bool): flag for parsing new predicates generated from action, defaults to False
            max_retries (int): max # of retries if failure occurs

        Returns:
            action (Action): constructed action class
            new_predicates (list[Predicate]): a list of new predicates
            llm_output (str): the raw string BaseLLM response
        """

        act_list_str = (
            "\n".join([f"- {a}" for a in action_list])
            if action_list
            else "No other actions provided."
        )
        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{action_list}", act_list_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()

                llm_output = model.query(prompt=prompt)

                # extract respective types from response
                raw_actions = llm_output.split("## NEXT ACTION")

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
                        parse_action(llm_output=rest_of_string, action_name=action_name)
                    )

                # if user queries predicate creation via LLM
                try:
                    if extract_new_preds:
                        new_predicates = parse_new_predicates(llm_output)
                    else:
                        new_predicates = []

                    if predicates:
                        new_predicates = [
                            pred
                            for pred in new_predicates
                            if pred["name"] not in [p["name"] for p in predicates]
                        ]  # remove re-defined predicates
                except Exception as e:
                    print(f"No new predicates: {e}")
                    new_predicates = None

                return actions, new_predicates, llm_output

            except Exception as e:
                print(
                    f"Error on attempt {attempt + 1}/{max_retries}: {e}\n"
                    f"LLM Output:\n{llm_output if 'llm_output' in locals() else 'None'}\nRetrying...\n"
                )
                time.sleep(2)

        raise RuntimeError("Max retries exceeded. Failed to extract PDDL action.")

    @require_llm
    def formalize_parameters(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[OrderedDict, list, str, tuple[bool, str]]:
        """
        Formalizes PDDL :parameters for single action via LLM.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :parameters extraction
            action_name (str): action name
            action_desc (str): action description, defaults to None
            types (dict[str,str] | list(dict[str,str])): current types in specification, defaults to None
            syntax_validator (SyntaxValidator): syntax checker for generated params, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            param (OrderedDict): ordered list of parameters {<?var>: <type>}
            param_raw (list()): list of raw parameters
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        types_str = pretty_print_dict(types) if types else "No types provided."

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{action_name}", action_name)
            .replace("{action_desc}", action_desc or "No description available.")
            .replace("{types}", types_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)  # get BaseLLM response

                # extract respective types from response
                param, param_raw = parse_params(llm_output=llm_output)

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
                            validation_info = validator(param_raw)
                        elif error_type == "validate_params":
                            validation_info = validator(param, types)

                        if not validation_info[0]:
                            return param, param_raw, llm_output, validation_info

                return param, param_raw, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract parameters.")

    @require_llm
    def formalize_preconditions(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str = None,
        params: OrderedDict = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] | None = None,
        functions: list[Function] | None = None,
        extract_new_preds: bool = False,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[str, list[Predicate], str, tuple[bool, str]]:
        """
        Formalizes PDDL :preconditions from a single action via LLM.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :preconditions extraction
            action_name (str): action name
            action_desc (str): action description, defaults to None
            params (OrderedDict): dictionary of parameters from action, defaults to None
            types (dict[str,str] | list(dict[str,str])): current types in specification, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in specification, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            extract_new_preds (bool): flag for parsing new predicates generated from action, defaults to False
            syntax_validator (SyntaxValidator): syntax checker for generated preconditions, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            preconditions (str): PDDL format of :preconditions
            new_predicates (list[Predicate]): a list of new predicates, defaults to empty list
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        params_str = format_params(params) if params else "No parameters provided."
        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{action_name}", action_name)
            .replace("{action_desc}", action_desc or "No description available.")
            .replace("{parameters}", params_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)  # get BaseLLM response

                # extract respective preconditions from response
                preconditions = parse_preconditions(llm_output=llm_output)

                if extract_new_preds:
                    new_predicates = parse_new_predicates(llm_output=llm_output)
                else:
                    new_predicates = None

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
                            validation_info = validator(preconditions)
                        elif error_type == "validate_duplicate_predicates":
                            validation_info == validator(predicates, new_predicates)
                        elif error_type == "validate_pddl_action":
                            all_predicates = predicates
                            all_predicates.extend(new_predicates)
                            validation_info = validator(
                                preconditions,
                                all_predicates,
                                params,
                                types,
                                "preconditions",
                            )

                        if not validation_info[0]:
                            return (
                                preconditions,
                                new_predicates,
                                llm_output,
                                validation_info,
                            )

                return preconditions, new_predicates, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract preconditions.")

    @require_llm
    def formalize_effects(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        action_name: str,
        action_desc: str = None,
        params: OrderedDict = None,
        preconditions: str = None,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] | None = None,
        functions: list[Function] | None = None,
        extract_new_preds: bool = False,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[str, list[Predicate], str, tuple[bool, str]]:
        """
        Formalizes PDDL :effects from a single action via LLM

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain description
            prompt_template (str): structured prompt template for :effects extraction
            action_name (str): action name
            action_desc (str): action description, defaults to None
            params (list[str]): list of parameters from action, defaults to None
            precondition (str): PDDL format of preconditions, defaults to None
            types (dict[str,str] | list(dict[str,str])): current types in specification, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in specification, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            extract_new_preds (bool): flag for parsing new predicates generated from action, defaults to False
            syntax_validator (SyntaxValidator): syntax checker for generated effects, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            effects (str): PDDL format of :effects
            new_predicates (list[Predicate]): a list of new predicates, defaults to empty list
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        params_str = format_params(params) if params else "No parameters provided."
        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            prompt_template.replace("{domain_desc}", domain_desc)
            .replace("{action_name}", action_name)
            .replace("{action_desc}", action_desc or "No description available.")
            .replace("{parameters}", params_str)
            .replace("{preconditions}", preconditions or "No precondition provided.")
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)  # get BaseLLM response

                # extract respective effects from response
                effects = parse_effects(llm_output=llm_output)

                if extract_new_preds:
                    new_predicates = parse_new_predicates(llm_output=llm_output)
                else:
                    new_predicates = None

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
                            validation_info = validator(effects)
                        elif error_type == "validate_duplicate_predicates":
                            validation_info == validator(predicates, new_predicates)
                        elif error_type == "validate_pddl_action":
                            all_predicates = predicates
                            all_predicates.extend(new_predicates)
                            validation_info = validator(
                                effects, all_predicates, params, types, "effects"
                            )

                        if not validation_info[0]:
                            return effects, new_predicates, llm_output, validation_info

                return effects, new_predicates, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract effects.")

    # NOTE: This function is experimental and may be subject to change in future versions.
    @require_llm
    def formalize_domain_level_specs(
        self,
        model: BaseLLM,
        domain_desc: str,
        prompt_template: str,
        formalize_types: bool = False,
        formalize_constants: bool = False,
        formalize_predicates: bool = False,
        formalize_functions: bool = False,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, Any], str, tuple[bool, str]]:
        """
        Formalizes domain-level specifications (i.e. :types, :constants, :predicates, :functions) via LLM.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): domain description
            prompt_template (str): prompt template
            formalize_types (bool): flag for extracting :types, defaults to False
            formalize_constants (bool): flag for extracting :constants, defaults to False
            formalize_predicates (bool): flag for extracting :predicates, defaults to False
            formalize_functions (bool): flag for extracting :functions, defaults to False
            syntax_validator (SyntaxValidator): syntax checker for domain specs., defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            spec_results (dict[str, Any]): domain-level specifications of user requirements
            llm_output (str): the raw string BaseLLM response
        """

        spec_results = {}  # results dictionary of top-level PDDL domain specifications

        prompt = prompt_template.replace("{domain_desc}", domain_desc)

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                if formalize_types:
                    types = parse_type_hierarchy(llm_output=llm_output)
                if formalize_constants:
                    constants = parse_constants(llm_output=llm_output)
                if formalize_predicates:
                    predicates = parse_new_predicates(llm_output=llm_output)
                if formalize_functions:
                    functions = parse_functions(llm_output=llm_output)

                spec_results["types"] = types
                spec_results["constants"] = constants
                spec_results["predicates"] = predicates
                spec_results["functions"] = functions

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
                        elif error_type == "validate_cyclic_types":
                            validation_info = validator(types)
                        elif error_type == "validate_constant_types":
                            validation_info = validator(constants, types)
                        elif error_type == "validate_types_predicates":
                            validation_info = validator(predicates, types)
                        elif error_type == "validate_format_predicates":
                            validation_info = validator(predicates, types)
                        elif error_type == "validate_format_functions":
                            validation_info = validator(functions, types)

                        if not validation_info[0]:
                            return spec_results, llm_output, validation_info

                return spec_results, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError(
            "Max retries exceeded. Failed to extract domain specification."
        )

    """Delete functions"""

    def delete_type(self, name: str):
        """Deletes a specific type from both `self.types` and `self.type_hierarchy`."""

        # remove from flat types dictionary if present
        if self.types is not None:
            self.types = {
                type_: desc for type_, desc in self.types.items() if type_ != name
            }

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
                    updated_node = {type_name: node[type_name], "children": children}
                    updated_list.append(updated_node)

            return updated_list

        # update the type_hierarchy if it exists
        if self.type_hierarchy is not None:
            self.type_hierarchy = remove_and_promote(self.type_hierarchy)

    def delete_constants(self, name: str):
        """Deletes specific constant from current specification"""
        if self.constants is not None:
            self.constants = {
                cons_: type_ for cons_, type_ in self.constants.items() if cons_ != name
            }

    def delete_predicate(self, name: str):
        """Deletes specific predicate from current specification"""
        if self.predicates is not None:
            self.predicates = [
                predicate for predicate in self.predicates if predicate["name"] != name
            ]

    def delete_function(self, name: str):
        """Deletes specific function from current specification"""
        if self.functions is not None:
            self.functions = [
                function for function in self.functions if function["name"] != name
            ]

    def delete_pddl_action(self, name: str):
        """Deletes specific PDDL action from current specification"""
        if self.pddl_actions is not None:
            self.pddl_actions = [
                action for action in self.pddl_actions if action["name"] != name
            ]

    """Set functions"""

    def set_types(self, types: dict[str, str]):
        """Sets types for current specification"""
        self.types = types

    def set_type_hierarchy(self, type_hierarchy: list[dict[str, str]]):
        """Sets type hierarchy for current specification"""
        self.type_hierarchy = type_hierarchy

    def set_constants(self, constants: dict[str, str]):
        """Sets constants for current specification"""
        self.constants = constants

    def set_predicate(self, predicate: Predicate):
        """Appends a predicate for current specification"""
        self.predicates.append(predicate)

    def set_function(self, function: Function):
        """Appends a function for current specification"""
        self.functions.append(function)

    def set_pddl_action(self, pddl_action: Action):
        """Appends a PDDL action for current specification"""
        self.pddl_actions.append(pddl_action)

    """Get functions"""

    def get_types(self) -> dict[str, str]:
        """Returns types from current specification"""
        return self.types

    def get_type_hierarchy(self) -> list[dict[str, str]]:
        """Returns type hierarchy from current specification"""
        return self.type_hierarchy

    def get_constants(self) -> dict[str, str]:
        """Returns constants from current specification"""
        return self.constants

    def get_predicates(self) -> list[Predicate]:
        """Returns predicates from current specification"""
        return self.predicates

    def get_functions(self) -> list[Function]:
        """Returns functions from current specification"""
        return self.functions

    def get_pddl_actions(self) -> list[Action]:
        """Returns PDDL actions from current specification"""
        return self.pddl_actions

    def generate_requirements(
        self,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        functions: list[Function] | None = None,
        actions: list[Action] = None,
    ) -> list[str]:
        """
        Generates necessary PDDL requirements based off of rest of domain specification.
        Motivation was not needing LLMs to specify :requirements predeterminate of domain generation.

        Currently does not support :durative-actions

        Args:
            types (dict[str,str] | list(dict[str,str])): current types in specification, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            actions (list[Action]): domain :action(s), defaults to None

        Returns:
            requirements (list[str]): list of PDDL requirements
        """

        requirements = set()
        requirements.add(":strips")

        # check if each specification needs a :requirement
        if types:
            requirements.add(":typing")
        if functions:
            requirements.add(":numeric-fluents")

        # go through actions and checks if it needs a :requirement
        for action in actions:
            preconditions = "\n".join(
                line for line in action["preconditions"].splitlines() if line.strip()
            )
            effects = "\n".join(
                line for line in action["effects"].splitlines() if line.strip()
            )

            if "not" in preconditions:
                requirements.add(":negative-preconditions")
            if "or" in preconditions:
                requirements.add(":disjunctive-preconditions")
            if "=" in preconditions:
                requirements.add(":equality")
            if "exists" in preconditions and "forall" in preconditions:
                requirements.add(":quantified-preconditions")
            else:
                if "exists" in preconditions:
                    requirements.add(":existential-preconditions")
                if "forall" in preconditions:
                    requirements.add(":universal-preconditions")
            if "when" in effects:
                requirements.add(":conditional-effects")

        # replace ADL components with :adl
        adl_components = {
            ":strips",
            ":typing",
            ":disjunctive-preconditions",
            ":equality",
            ":quantified-preconditions",
            ":conditional-effects",
        }
        if adl_components.issubset(requirements):
            requirements -= adl_components
            requirements.add(":adl")

        requirements = list(sorted(requirements))  # convert set back into list
        return requirements

    def generate_domain(
        self,
        domain_name: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] | None = None,
        functions: list[Function] | None = None,
        actions: list[Action] = [],
        requirements: list[str] | None = None,
    ) -> str:
        """
        Generates PDDL domain from given information.

        Args:
            domain_name (str): domain name
            types (dict[str,str] | list[dict[str,str]] | None): domain :types, defaults to None
            constants (dict[str,str] | None): domain :constants, defaults to None
            predicates (list[Predicate] | None): domain :predicates, defaults to None
            functions (list[Function] | None): domain :functions, defaults to None
            actions (list[Action]): domain :action(s), defaults to None
            requirements (list[str]): domain :requirements, defaults to constant REQUIREMENTS

        Returns:
            desc (str): PDDL domain in string format
        """

        # generates requirements if not set
        if not requirements:
            requirements = self.generate_requirements(
                types=types, functions=functions, actions=actions
            )

        desc = ""
        desc += f"(define (domain {domain_name})\n"
        desc += indent(string=f"(:requirements\n   {' '.join(requirements)})", level=1)
        if types:
            types_str = format_types_to_string(types)
            desc += f"\n\n   (:types \n{indent(string=types_str, level=2)}\n   )"

        if constants:
            const_str = format_constants(constants)
            desc += f"\n\n   (:constants \n{indent(string=const_str, level=2)}\n   )"

        if not predicates:
            print(
                "[WARNING]: Domain has no predicates. This may cause planners to reject the domain or behave unexpectedly."
            )
        else:
            pred_str = format_expression(predicates)
            desc += f"\n\n   (:predicates \n{indent(string=pred_str, level=2)}\n   )"

        if functions:
            func_str = format_expression(functions)
            desc += f"\n\n   (:functions \n{indent(string=func_str, level=2)}\n   )"

        if not actions:
            print(
                "[WARNING]: Domain has no actions. The planner will not be able to generate any plan unless the goal is already satisfied."
            )
        else:
            desc += format_actions(actions)
        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc
