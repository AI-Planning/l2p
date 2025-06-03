"""
PDDL Problem Formalization/Generation Functions

This module defines the `TaskBuilder` class and related utilities for constructing
PDDL problem specifications programatically.

Refer to: https://marcustantakoun.github.io/l2p.github.io/l2p.html for more information
how to use class functions. Refer to /templates in: https://github.com/AI-Planning/l2p
for how to structurally prompt LLMs so they are compatible with class function parsing.
"""

import time
from .llm import BaseLLM, require_llm
from .utils import *


class TaskBuilder:
    def __init__(
        self,
        objects: dict[str, str] = None,
        initial: list[dict[str, str]] = None,
        goal: list[dict[str, str]] = None,
    ) -> None:
        """
        Initializes an L2P task builder object.

        Args:
            objects (dict[str,str]): current dictionary of task objects in specification
            initial (list[dict[str,str]]): current initial states in specification
            goal (list[dict[str,str]]): current goal states in specification
        """

        self.objects = objects or {}
        self.initial = initial or []
        self.goal = goal or []

    """Formalize/generate functions"""

    @require_llm
    def formalize_objects(
        self,
        model: BaseLLM,
        problem_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[dict[str, str], str, tuple[bool, str]]:
        """
        Formalizes PDDL :objects via LLM.

        Args:
            model (BaseLLM): LLM to query
            problem_desc (str): general problem description
            prompt_template (str): structured prompt template for :objects extraction
            types (dict[str,str] | list[dict[str,str]]): current types in specification, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            syntax_validator (SyntaxValidator): syntax checker for generated objects, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            objects (dict[str,str]): dictionary of object types {<name>: <type>}
            llm_output (str): the raw string BaseLLM response
            validation_info (tuple[bool,str]): validation info containing pass flag and error message
        """

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )

        prompt = (
            prompt_template.replace("{problem_desc}", problem_desc)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)  # get BaseLLM response

                # extract respective types from response
                objects = parse_objects(llm_output=llm_output)

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
                        elif error_type == "validate_task_objects":
                            validation_info = validator(objects, types)

                        if not validation_info[0]:
                            return objects, llm_output, validation_info

                return objects, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error on attempt {attempt + 1}/{max_retries}: {e}\n"
                    f"LLM Output:\n{llm_output if 'llm_output' in locals() else 'None'}\nRetrying...\n"
                )
                time.sleep(2)

        raise RuntimeError("Max retries exceeded. Failed to extract objects.")

    @require_llm
    def formalize_initial_state(
        self,
        model: BaseLLM,
        problem_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] | None = None,
        functions: list[Function] | None = None,
        objects: dict[str, str] | None = None,
        initial: list[dict[str, str]] | None = None,
        goal: list[dict[str, str]] | None = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[list[dict[str, str]], str, tuple[bool, str]]:
        """
        Formalizes PDDL :init states via LLM.

        Args:
            model (BaseLLM): LLM to query
            problem_desc (str): general problem description
            prompt_template (str): structured prompt template for :init extraction
            types (dict[str,str] | list[dict[str,str]]): current types in domain, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in domain, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            objects (dict[str,str]): current dictionary of task :objects in specification, defaults to None
            initial (list[dict[str,str]]): current :init states in specification, defaults to None
            goal (list[dict[str,str]]): current :goal states in specification, defaults to None
            syntax_validator (SyntaxValidator): syntax checker for generated initial states, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            initial (list[dict[str,str]]): list of dictionaries containing initial states consisting of:
                {<predicate_name>:<str>, <params>:<list[str]>, <neg>:<bool>} OR
                {<function_name>:<str>, <params>:<list[str]>, <value>:<int>, <op>:<str>}
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
        obj_str = format_objects(objects) if objects else "No objects provided."
        init_str = format_initial(initial) if initial else "No initial state provided."
        goal_str = format_goal(goal) if goal else "No goal state provided."

        prompt = (
            prompt_template.replace("{problem_desc}", problem_desc)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{objects}", obj_str)
            .replace("{initial_state}", init_str)
            .replace("{goal_state}", goal_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # extract respective types from response
                initial = parse_initial(llm_output=llm_output)

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
                        elif error_type == "validate_task_states":
                            validation_info = validator(
                                initial, objects, predicates, "initial"
                            )

                        if not validation_info[0]:
                            return initial, llm_output, validation_info

                return initial, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error on attempt {attempt + 1}/{max_retries}: {e}\n"
                    f"LLM Output:\n{llm_output if 'llm_output' in locals() else 'None'}\nRetrying...\n"
                )
                time.sleep(2)

        raise RuntimeError("Max retries exceeded. Failed to extract initial states.")

    @require_llm
    def formalize_goal_state(
        self,
        model: BaseLLM,
        problem_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] | None = None,
        functions: list[Function] | None = None,
        objects: dict[str, str] | None = None,
        initial: list[dict[str, str]] | None = None,
        goal: list[dict[str, str]] | None = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[list[dict[str, str]], str, tuple[bool, str]]:
        """
        Formalizes PDDL :goal states via LLM.

        Args:
            model (BaseLLM): LLM to query
            problem_desc (str): general problem description
            prompt_template (str): structured prompt template for :goal extraction
            types (dict[str,str] | list[dict[str,str]]): current :types in domain, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in domain, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            objects (dict[str,str]): current dictionary of task :objects in specification, defaults to None
            initial (list[dict[str,str]]): current :init states in specification, defaults to None
            goal (list[dict[str,str]]): current :goal states in specification, defaults to None
            syntax_validator (SyntaxValidator): syntax checker for generated goal states, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            goal (list[dict[str,str]]): list of dictionaries containing goal states consisting of:
                {<predicate_name>:<str>, <params>:<list[str]>, <neg>:<bool>} OR
                {<function_name>:<str>, <params>:<list[str]>, <value>:<int>, <op>:<str>}
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
        obj_str = format_objects(objects) if objects else "No objects provided."
        init_str = format_initial(initial) if initial else "No initial state provided."
        goal_str = format_goal(goal) if goal else "No goal state provided."

        prompt = (
            prompt_template.replace("{problem_desc}", problem_desc)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{objects}", obj_str)
            .replace("{initial_state}", init_str)
            .replace("{goal_state}", goal_str)
        )

        # iterate through attempts in case of extraction failure
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)

                # extract respective types from response
                goal = parse_goal(llm_output=llm_output)

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
                        elif error_type == "validate_task_states":
                            validation_info = validator(
                                goal, objects, predicates, "goal"
                            )

                        if not validation_info[0]:
                            return goal, llm_output, validation_info

                return goal, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error on attempt {attempt + 1}/{max_retries}: {e}\n"
                    f"LLM Output:\n{llm_output if 'llm_output' in locals() else 'None'}\nRetrying...\n"
                )
                time.sleep(2)

        raise RuntimeError("Max retries exceeded. Failed to extract goal states.")

    @require_llm
    def formalize_task(
        self,
        model: BaseLLM,
        problem_desc: str,
        prompt_template: str,
        types: dict[str, str] | list[dict[str, str]] | None = None,
        constants: dict[str, str] | None = None,
        predicates: list[Predicate] | None = None,
        functions: list[Function] | None = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> tuple[
        dict[str, str],
        list[dict[str, str]],
        list[dict[str, str]],
        str,
        tuple[bool, str],
    ]:
        """
        Formalizes whole task specification via LLM.

        Args:
            model (BaseLLM): LLM to query
            problem_desc (str): general problem description
            prompt_template (str): structured prompt template for :problem extraction
            types (dict[str,str] | list[dict[str,str]]): current :types in domain, defaults to None
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of current predicates in domain, defaults to None
            functions (list[Function]): list of current functions in specification, defaults to None
            syntax_validator (SyntaxValidator): syntax checker for generated :problem, defaults to None
            max_retries (int): max # of retries if failure occurs

        Returns:
            objects (dict[str,str]): dictionary of object names and assigned types
            initial (list[dict[str,str]]): list of dictionary of initial states
            goal (list[dict[str,str]]): list of dictionary of goal states
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
            prompt_template.replace("{problem_desc}", problem_desc)
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
                objects = parse_objects(llm_output=llm_output)
                initial = parse_initial(llm_output=llm_output)
                goal = parse_goal(llm_output=llm_output)

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
                        elif error_type == "validate_task_objects":
                            validation_info = validator(objects, types)
                        elif error_type == "validate_task_states":
                            validation_info = validator(
                                states=initial,
                                objects=objects,
                                predicates=predicates,
                                functions=functions,
                                state_type="initial",
                            )
                            if validation_info[0]:
                                validation_info = validator(
                                    states=goal,
                                    objects=objects,
                                    predicates=predicates,
                                    functions=functions,
                                    state_type="goal",
                                )

                        if not validation_info[0]:
                            return objects, initial, goal, llm_output, validation_info

                return objects, initial, goal, llm_output, validation_info

            except Exception as e:
                print(
                    f"Error on attempt {attempt + 1}/{max_retries}: {e}\n"
                    f"LLM Output:\n{llm_output if 'llm_output' in locals() else 'None'}\nRetrying...\n"
                )
                time.sleep(2)

        raise RuntimeError("Max retries exceeded. Failed to extract task.")

    """Delete functions"""

    def delete_objects(self, object: dict[str, str]):
        """Deletes specific item in :objects from current specification"""
        if self.objects is not None:
            self.objects = {
                var: type_ for var, type_ in self.objects.items() if var != object
            }

    def delete_initial_state(self, state: dict[str, str]):
        """Deletes specific :init state from current specification"""
        if self.initial is not None:
            self.initial = [s for s in self.initial if s != state]

    def delete_goal_state(self, state: dict[str, str]):
        """Deletes specific PDDL :goal state from current specification"""
        if self.goal is not None:
            self.goal = [s for s in self.goal if s != state]

    """Set functions"""

    def set_objects(self, objects: dict[str, str]):
        """Sets PDDL :objects for current specification"""
        self.objects = objects

    def set_initial(self, initial: list[dict[str, str]]):
        """Sets PDDL :init states for current specification"""
        self.initial = initial

    def set_goal(self, goal: list[dict[str, str]]):
        """Sets PDDL :goal states for current specification"""
        self.goal = goal

    """Get functions"""

    def get_objects(self) -> dict[str, str]:
        """Returns PDDL :objects from current specification"""
        return self.objects

    def get_initial(self) -> list[dict[str, str]]:
        """Returns PDDL :init states from current specification"""
        return self.initial

    def get_goal(self) -> list[dict[str, str]]:
        """Returns PDDL :goal states from current specification"""
        return self.goal

    def generate_task(
        self,
        domain_name: str,
        problem_name: str,
        objects: dict[str, str],
        initial: list[dict[str, str]],
        goal: list[dict[str, str]],
    ) -> str:
        """
        Generates PDDL problem from given information.

        Args:
            domain_name (str): domain name
            problem_name (str): specific task instance name
            objects (dict[str,str]): PDDL :objects
            initial (list[dict[str,str]]): PDDL :init states
            goal (list[dict[str,str]]): PDDL :goal states

        Returns:
            desc (str): PDDL problem in string format
        """

        desc = "(define\n"
        desc += f"   (problem {problem_name})\n"
        desc += f"   (:domain {domain_name})\n\n"
        desc += f"   (:objects \n{indent(format_objects(objects))}\n   )\n\n"
        desc += f"   (:init\n{indent(format_initial(initial))}\n   )\n\n"
        desc += f"   (:goal\n{indent(format_goal(goal))}\n   )\n"
        desc += ")"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc
