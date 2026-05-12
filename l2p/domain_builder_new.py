"""
PDDL Domain Formalization/Generation Functions

This module defines the `DomainBuilder` class and related utilities for constructing
PDDL domain specifications programmatically using structured Pydantic models.
"""

import time
from typing import Optional, Set, TypeVar, Union

from l2p.utils.pddl_types_new import (
    DomainDetails, ProblemDetails, Requirement, 
    PDDLType, Constant, Parameter, 
    Predicate, Function, DerivedPredicate,
    ActionPrecondition, ConditionalEffect, ActionEffect, Action,
    DurativeActionConditions, DurativeActionEffect, DurativeAction,
    Constraint, Event, Process, LogicalCondition)
from l2p.utils.pddl_format_new import *
from l2p.utils.pddl_parser_new import parse_xml_tags, parse_element

from l2p.llm import BaseLLM, require_llm
from l2p.utils import SyntaxValidator

T = TypeVar('T')

class DomainBuilder:
    def __init__(
            self, 
            domain_details: Optional[DomainDetails] = None,
            problem_details: Optional[ProblemDetails] = None,
            ) -> None:
        """
        Initializes an L2P domain builder object using the new Pydantic DomainDetails root model.

        Args:
            domain_details (DomainDetails | None): A fully validated Pydantic model 
            containing the entire PDDL domain structure.
        """
        if domain_details:
            self.domain_details = domain_details
        else: 
            self.domain_details = DomainDetails(name="domain-placeholder")


    # ---------------------------------------------------------------------------
    # PDDL DOMAIN GENERATE FUNCTIONS
    # ---------------------------------------------------------------------------

    @require_llm
    def formalize_domain(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
            ) -> tuple[DomainDetails, str, tuple[bool,str]]:
        """
        Formalizes a whole PDDL domain (:domain) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_requirements(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
            ) -> tuple[list[Requirement], str, tuple[bool,str]]:
        """
        Formalizes PDDL requirements (:requirements) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_types(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
            ) -> tuple[list[PDDLType], str, tuple[bool,str]]:
        """
        Formalizes PDDL types (:types) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_constants(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
            ) -> tuple[list[Constant], str, tuple[bool,str]]:
        """
        Formalizes PDDL constants (:constants) using BaseLLM.
        """
        pass
    
    @require_llm
    def formalize_predicates(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[Predicate], str, tuple[bool,str]]:
        """
        Formalizes PDDL predicates (:predicates) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_functions(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[Function], str, tuple[bool,str]]:
        """
        Formalizes PDDL functions (:functions) using BaseLLM.
        """
        pass
    
    @require_llm
    def formalize_constraints(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[Constraint], str, tuple[bool,str]]:
        """
        Formalizes PDDL constraints (:constraints) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_derived_predicates(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[DerivedPredicate], str, tuple[bool,str]]:
        """
        Formalizes PDDL derived predicates / axioms (:derived) using BaseLLM.
        """
        pass

    def formalize_action_parameters(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[Parameter], str, tuple[bool,str]]:
        """
        Formalizes a PDDL action parameters (:parameters) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_action_preconditions(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[ActionPrecondition, str, tuple[bool,str]]:
        """
        Formalizes a PDDL action precondition (:precondition) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_action_effects(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[ActionEffect, str, tuple[bool,str]]:
        """
        Formalizes a PDDL action effect (:effect) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_nl_actions(
            self,
            model: BaseLLM,
            prompt_template: str,
            max_retries: int = 3
        ) -> tuple[list[dict[str,str]], str]:
        """
        Extract actions in natural language given domain description using BaseLLM.
        """
        pass

    @require_llm
    def formalize_actions(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[Action], list[Predicate], str, tuple[bool,str]]:
        """
        Formalizes PDDL action instances (:action <n>) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_durative_conditions(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[DurativeActionConditions, str, tuple[bool,str]]:
        """
        Formalizes a PDDL durative action conditions (:condition) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_durative_effects(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[DurativeActionEffect, str, tuple[bool,str]]:
        """
        Formalizes a PDDL durative action effects (:effect) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_durative_actions(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[DurativeAction], str, tuple[bool,str]]:
        """
        Formalizes PDDL durative action instances (:durative-action <n>) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_events(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[Event], str, tuple[bool,str]]:
        """
        Formalizes PDDL event instances (:event <n>) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_processes(
            self,
            model: BaseLLM,
            prompt_template: str,
            validate_syntax: bool = True,
            max_retries: int = 3
        ) -> tuple[list[Process], str, tuple[bool,str]]:
        """
        Formalizes PDDL process instances (:process <n>) using BaseLLM.
        """
        pass
    
    
    def generate_requirements(
            self, 
            domain_details: DomainDetails, 
            problem_details: Optional[ProblemDetails] = None
        ) -> list[Requirement]:
        reqs: Set[str] = {":strips"}

        # basic structural requirements (Domain)
        if domain_details.types:
            reqs.add(":typing")
        if domain_details.functions:
            reqs.add(":numeric-fluents")
        if domain_details.derived_predicates:
            reqs.add(":derived-predicates")
        if domain_details.durative_actions:
            reqs.add(":durative-actions")
        if domain_details.events or domain_details.processes:
            reqs.add(":time")
        if domain_details.constraint:
            reqs.add(":constraints")

        # basic structural requirements (Problem)
        if problem_details:
            if problem_details.constraint:
                reqs.add(":constraints")

            for fact in problem_details.initial_state:
                if isinstance(fact, dict) and "time" in fact:
                    reqs.add(":timed-initial-literals")

            if problem_details.metric:
                if "is-violated" in problem_details.metric.expression:
                    reqs.add(":preferences")
            
        # helper to recursively check conditions/effects
        def check_logical_condition(condition: LogicalCondition):
            if isinstance(condition, str):
                if "=" in condition and "(=" not in condition:
                    reqs.add(":equality")
                return
                
            if isinstance(condition, dict):
                op = condition.get("operator", "").lower()
                quant = condition.get("quantifier", "").lower()
                if op == "not":
                    reqs.add(":negative-preconditions")
                elif op == "or":
                    reqs.add(":disjunctive-preconditions")
                elif op == "imply":
                    reqs.add(":disjunctive-preconditions")
                    
                if quant == "exists":
                    reqs.add(":existential-preconditions")
                elif quant == "forall":
                    reqs.add(":universal-preconditions")
                    
                if "preference" in condition:
                    reqs.add(":preferences")
                if op in ["always", "sometime", "at-most-once", "within", "hold-after", "hold-during", 
                        "sometime-after", "sometime-before", "always-within"]:
                    reqs.add(":constraints")
                    
                for val in condition.values():
                    if isinstance(val, (list, tuple)):
                        for item in val:
                            check_logical_condition(item)
                    elif isinstance(val, (dict, str)):
                        check_logical_condition(val)

        # check actions, events, and processes
        assignment_ops_used = False
        
        # combine all standard, durative, events, and processes to check their conditions
        for block in domain_details.actions + domain_details.events + domain_details.processes:
            # check preconditions
            for cond in block.preconditions.conditions:
                check_logical_condition(cond)
                
            # check effects
            for cond in block.effects.add + block.effects.delete:
                check_logical_condition(cond)
                
            if block.effects.conditional:
                reqs.add(":conditional-effects")
                for ce in block.effects.conditional:
                    for c in ce.condition:
                        check_logical_condition(c)
                    
            # check numeric assignments
            for num_eff in block.effects.numeric:
                assignment_ops_used = True
                check_logical_condition(num_eff)

        # check durative actions
        for d_act in domain_details.durative_actions:
            # Check duration strings for inequalities (< or >)
            for dur in d_act.duration:
                if "<" in dur or ">" in dur:
                    reqs.add(":durative-inequalities")

            # check durative conditions
            for cond in d_act.conditions.at_start + d_act.conditions.over_all + d_act.conditions.at_end:
                check_logical_condition(cond)

            # check durative effects
            if d_act.effects.at_start:
                for cond in d_act.effects.at_start.add + d_act.effects.at_start.delete + d_act.effects.at_start.numeric:
                    check_logical_condition(cond)
            if d_act.effects.at_end:
                for cond in d_act.effects.at_end.add + d_act.effects.at_end.delete + d_act.effects.at_end.numeric:
                    check_logical_condition(cond)
                    
            # check continuous effects (changes over time during the action)
            if d_act.effects.continuous:
                reqs.add(":continuous-effects")
                for cond in d_act.effects.continuous:
                    check_logical_condition(cond)

        # check problem file
        if problem_details:
            if problem_details.goal_state:
                check_logical_condition(problem_details.goal_state)
            for constraint in problem_details.constraints:
                check_logical_condition(constraint.condition)

        # handle Quantified Preconditions
        if ":existential-preconditions" in reqs and ":universal-preconditions" in reqs:
            reqs.discard(":existential-preconditions")
            reqs.discard(":universal-preconditions")
            reqs.add(":quantified-preconditions")

        # handle Action Costs (Global function + numeric assignments)
        has_global_function = any(len(f.params) == 0 for f in domain_details.functions)
        if has_global_function and assignment_ops_used:
            reqs.add(":action-costs")

        # handle ADL
        adl_components = {
            ":strips",
            ":typing",
            ":disjunctive-preconditions",
            ":equality",
            ":quantified-preconditions",
            ":conditional-effects",
        }
        if adl_components.issubset(reqs):
            reqs -= adl_components
            reqs.add(":adl")

        sorted_reqs = sorted(list(reqs))
        return [Requirement(name=r) for r in sorted_reqs]

    def generate_domain(self, domain_details: DomainDetails) -> str:
        """
        Assembles all formatted components into a complete PDDL domain string.
        """
        requirements = domain_details.requirements
        if not requirements:
            requirements = self.generate_requirements(domain_details=domain_details)

        desc = f"(define (domain {domain_details.name})\n"
        
        if requirements:
            reqs_str = format_requirements(reqs=requirements)
            desc += indent(string=f"(:requirements\n   {reqs_str})", level=1)
            
        if domain_details.types:
            types_str = format_types(domain_details.types)
            desc += f"\n\n   (:types \n{indent(string=types_str, level=2)}\n   )"

        if domain_details.constants:
            const_str = format_constants(domain_details.constants)
            desc += f"\n\n   (:constants \n{indent(string=const_str, level=2)}\n   )"

        if not domain_details.predicates:
            print(
                    "[WARNING]: Domain has no predicates. This may cause planners to reject the domain or behave unexpectedly."
            )
        else:
            pred_str = format_predicates(domain_details.predicates)
            desc += f"\n\n   (:predicates \n{indent(string=pred_str, level=2)}\n   )"

        if domain_details.functions:
            func_str = format_functions(domain_details.functions)
            desc += f"\n\n   (:functions \n{indent(string=func_str, level=2)}\n   )"

        if domain_details.constraint:
            constraints_str = format_constraints(domain_details.constraint)
            desc += f"\n\n   (:constraints \n{indent(string=constraints_str, level=2)}\n   )"

        if domain_details.derived_predicates:
            dp_str = format_derived_predicates(domain_details.derived_predicates)
            desc += f"\n\n{indent(string=dp_str, level=1)}"

        if not any([
            domain_details.actions, 
            domain_details.durative_actions, 
            domain_details.events, 
            domain_details.processes
        ]):
            print(
                "[WARNING]: Domain has no actions, events, or processes. "
                "The planner will not be able to generate any plan unless the goal is already satisfied."
            )
        else:
            if domain_details.actions:
                actions_str = format_actions(domain_details.actions)
                desc += f"\n\n{indent(string=actions_str, level=1)}"
                
            if domain_details.durative_actions:
                d_actions_str = format_durative_actions(domain_details.durative_actions)
                desc += f"\n\n{indent(string=d_actions_str, level=1)}"

            if domain_details.events:
                events_str = format_events(domain_details.events)
                desc += f"\n\n{indent(string=events_str, level=1)}"

            if domain_details.processes:
                processes_str = format_processes(domain_details.processes)
                desc += f"\n\n{indent(string=processes_str, level=1)}"

        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc

    # ---------------------------------------------------------------------------
    # PDDL DOMAIN SET FUNCTIONS
    # ---------------------------------------------------------------------------

    def _set_component(self, field_name: str, component: Union[list[T], T, None], append: bool = False):
        """
        Helper method to assign or append an L2P PDDL component to the domain_details model.
        Args:
            field_name (str): The exact attribute name on the DomainDetails Pydantic model 
                (e.g., "types", "actions", "constraint").
            component (Union[list[T], T, None]): The Pydantic model instance(s) to add. 
                Can be a single instance, a list of instances, or None.
            append (bool, optional): If True, concatenates the new items to the existing 
                attribute list. If False, overwrites the attribute entirely. Defaults to False.
        """
        if component is None:
            normalized_list = []
        elif not isinstance(component, list):
            normalized_list = [component]
        else:
            normalized_list = component
        clean_list = [item for item in normalized_list if item is not None]
        
        if append:
            existing_list = getattr(self.domain_details, field_name, [])
            updated_list = existing_list + clean_list
        else:
            updated_list = clean_list
        setattr(self.domain_details, field_name, updated_list)

    def set_domain_name(self, domain_name: str = None):
        """
        Sets PDDL domain name for current specification
        Args:
            domain_name (str = None): Name for domain.
        """
        if domain_name:
            self.domain_details.name = domain_name
        else:
            self.domain_details.name = "domain-placeholder"
        
    def set_domain_desc(self, domain_desc: str = None):
        """
        Sets PDDL domain description for current specification
        Args:
            domain_desc (str = None): Description for domain.
        """
        if domain_desc:
            self.domain_details.desc = domain_desc
        else:
            return
        
    def set_requirements(self, reqs: list[Requirement] | Requirement | None, append: bool = False):
        """
        Sets or appends PDDL requirements for current specification
        Args:
            reqs (list[Requirement] | Requirement | None): A single Requirement object, 
                a list of Requirement objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("requirements", reqs, append=append)

    def set_types(self, types: list[PDDLType] | PDDLType | None, append: bool = False):
        """
        Sets or appends PDDL types for current specification
        Args:
            types (list[PDDLType] | PDDLType | None): A single PDDLType object, 
                a list of PDDLType objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("types", types, append=append)

    def set_constants(self, constants: list[Constant] | Constant | None, append: bool = False):
        """
        Sets or appends PDDL constants for current specification
        Args:
            constants (list[Constant] | Constant | None): A single Constant object, 
                a list of Constant objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("constants", constants, append=append)

    def set_predicates(self, predicates: list[Predicate] | Predicate | None, append: bool = False):
        """
        Sets or appends PDDL predicates for current specification
        Args:
            predicates (list[Predicate] | Predicate | None): A single Predicate object, 
                a list of Predicate objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("predicates", predicates, append=append)

    def set_functions(self, functions: list[Function] | Function | None, append: bool = False):
        """
        Sets or appends PDDL functions for current specification
        Args:
            functions (list[Function] | Function | None): A single Function object, 
                a list of Function objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("functions", functions, append=append)

    def set_derived_predicates(self, d_predicates: list[DerivedPredicate] | DerivedPredicate | None, append: bool = False):
        """
        Sets or appends PDDL derived predicates for current specification
        Args:
            d_predicates (list[DerivedPredicate] | DerivedPredicate | None): A single DerivedPredicate object, 
                a list of DerivedPredicate objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("derived_predicates", d_predicates, append=append)

    def set_actions(self, actions: list[Action] | Action | None, append: bool = False):
        """
        Sets or appends standard PDDL actions for current specification
        Args:
            actions (list[Action] | Action | None): A single Action object, 
                a list of Action objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("actions", actions, append=append)

    def set_durative_actions(self, d_actions: list[DurativeAction] | DurativeAction | None, append: bool = False):
        """
        Sets or appends PDDL durative actions for current specification
        Args:
            d_actions (list[DurativeAction] | DurativeAction | None): A single DurativeAction object, 
                a list of DurativeAction objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("durative_actions", d_actions, append=append)

    def set_events(self, events: list[Event] | Event | None, append: bool = False):
        """
        Sets or appends PDDL events for current specification
        Args:
            events (list[Event] | Event | None): A single Event object, 
                a list of Event objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("events", events, append=append)

    def set_processes(self, processes: list[Process] | Process | None, append: bool = False):
        """
        Sets or appends PDDL processes for current specification
        Args:
            processes (list[Process] | Process | None): A single Process object, 
                a list of Process objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("processes", processes, append=append)

    def set_constraints(self, constraints: list[Constraint] | Constraint | None, append: bool = False):
        """
        Sets or appends PDDL constraints for current specification
        Args:
            constraints (list[Constraint] | Constraint | None): A single Constraint object, 
                a list of Constraint objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("constraint", constraints, append=append)

    # ---------------------------------------------------------------------------
    # PDDL DOMAIN DISPLAY FUNCTIONS
    # ---------------------------------------------------------------------------
    
if __name__ == "__main__":
    pass