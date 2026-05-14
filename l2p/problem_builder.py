"""
PDDL Problem Formalization/Generation Functions

This module defines the `ProblemBuilder` class and related utilities for constructing
PDDL problem specifications programatically.

Refer to: https://marcustantakoun.github.io/l2p.github.io/l2p.html for more information
how to use class functions. Refer to /templates in: https://github.com/AI-Planning/l2p
for how to structurally prompt LLMs so they are compatible with class function parsing.
"""

import time
from l2p.llm import BaseLLM, require_llm
from l2p.utils import *
from l2p.utils.pddl_format import *
from l2p.utils.pddl_types import DomainDetails, ProblemDetails

T = TypeVar('T')

class ProblemBuilder:
    def __init__(
            self, 
            problem_details: Optional[ProblemDetails] = None,
            domain_details: Optional[DomainDetails] = None,
            **kwargs
        ) -> None:
        """
        Initializes an L2P problem builder object using Pydantic ProblemDetails root model.

        Args:
            problem_details (ProblemDetails | None): A fully validated Problem Pydantic model.
            domain_details (DomainDetails | None): A fully validated Domain Pydantic model.
            **kwargs: Optional keyword arguments to initialize ProblemDetails directly 
                (e.g., name="my-problem", objects=[...]).
        """
        if problem_details:
            self.problem_details = problem_details
            if not self.problem_details.domain_name or self.problem_details.domain_name == "domain-placeholder":
                if domain_details:
                    self.problem_details.domain_name = domain_details.name
        else:
            problem_name = kwargs.pop("name", "problem-placeholder")
            
            if "domain_name" in kwargs:
                dom_name = kwargs.pop("domain_name")
            elif domain_details:
                dom_name = domain_details.name
            else:
                dom_name = "domain-placeholder"
                
            self.problem_details = ProblemDetails(
                name=problem_name, 
                domain_name=dom_name, 
                **kwargs
            )
        
        self.domain_details = domain_details

    
    # ---------------------------------------------------------------------------
    # PDDL DOMAIN GENERATE FUNCTIONS
    # ---------------------------------------------------------------------------

    @require_llm
    def formalize_problem(
            self,    
            model: BaseLLM,
            prompt_template: str,
            max_retries: int = 3,
            **ctx_kwargs
        ) -> tuple[ProblemDetails, str, tuple[bool,str]]:
        """
        Formalizes a whole PDDL problem (:problem) using BaseLLM.
        """
        pass

    @require_llm
    def formalize_objects(
            self,
            model: BaseLLM,
            prompt_template: str,
            max_retries: int = 3,
            **ctx_kwargs
        ) -> tuple[list[PDDLObject], str, tuple[bool,str]]:
        pass

    @require_llm
    def formalize_initial_states(
            self,
            model: BaseLLM,
            prompt_template: str,
            max_retries: int = 3,
            **ctx_kwargs
        ) -> tuple[InitialState, str, tuple[bool,str]]:
        pass

    @require_llm
    def formalize_goal_states(
            self,
            model: BaseLLM,
            prompt_template: str,
            max_retries: int = 3,
            **ctx_kwargs
        ) -> tuple[GoalState, str, tuple[bool,str]]:
        pass

    @require_llm
    def formalize_constraints(
            self,
            model: BaseLLM,
            prompt_template: str,
            max_retries: int = 3,
            **ctx_kwargs
        ) -> tuple[list[Constraint], str, tuple[bool,str]]:
        pass

    @require_llm
    def formalize_metric(
            self,
            model: BaseLLM,
            prompt_template: str,
            max_retries: int = 3,
            **ctx_kwargs
        ) -> tuple[Metric, str, tuple[bool,str]]:
        pass


    def generate_problem(self, problem_details: ProblemDetails) -> str:
        """
        Assembles all formatted components into a complete PDDL problem string.
        """
        desc = f"(define (problem {problem_details.name})\n"
        desc += f"   (:domain {problem_details.domain_name})\n"

        if problem_details.objects:
            objs_str = format_objects(problem_details.objects)
            desc += f"\n   (:objects \n{indent(string=objs_str, level=2)}\n   )"

        # check both standard and time facts lists inside initial state
        if not problem_details.initial_state.facts and not problem_details.initial_state.timed_facts:
            print("[WARNING]: Problem has no initial state.")
        else:
            init_str = format_initial_state(problem_details.initial_state)
            desc += f"\n\n   (:init \n{indent(string=init_str, level=2)}\n   )"

        # check the connditions list in goal state
        if not problem_details.goal_state.conditions:
            print("[WARNING]: Problem has no goals.")
        else:
            goal_str = format_goal_states(problem_details.goal_state)
            desc += f"\n\n   (:goal \n      {goal_str}\n   )"

        # check constraints
        if problem_details.constraint:
            constraints_str = format_constraints(problem_details.constraint)
            desc += f"\n\n   (:constraints \n{indent(string=constraints_str, level=2)}\n   )"

        # check metrics
        if problem_details.metric:
            metric_str = format_metric(problem_details.metric)
            desc += f"\n\n{indent(string=metric_str, level=1)}"

        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc
    

    # ---------------------------------------------------------------------------
    # PDDL PROBLEM SET FUNCTIONS
    # ---------------------------------------------------------------------------

    def _set_component(self, field_name: str, component: Union[list[T], T, None], append: bool = False):
        """
        Helper method to assign or append an L2P PDDL component to the problem_details model.
        Args:
            field_name (str): The exact attribute name on the DomainDetails Pydantic model 
                (e.g., "objects", "initial_states", "goal_states").
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
        setattr(self.problem_details, field_name, updated_list)

    def set_problem_name(self, problem_name: str = None):
        """
        Sets PDDL problem name for current specification
        Args:
            problem_name (str = None): Name for problem instance.
        """
        if problem_name:
            self.problem_details.name = problem_name
        else:
            self.problem_details.name = "problem-placeholder"
        
    def set_problem_desc(self, problem_desc: str = None):
        """
        Sets PDDL problem description for current specification
        Args:
            problem_desc (str = None): Description for problem instance.
        """
        if problem_desc:
            self.problem_details.desc = problem_desc
        else:
            return
        
    def set_objects(self, objects: list[PDDLObject] | PDDLObject | None, append: bool = False):
        """
        Sets or appends PDDL problem objects for current specification
        Args:
            objects (list[PDDLObject] | PDDLObject | None): A single PDDLObject object, 
                a list of PDDLObject objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("objects", objects, append=append)

    def set_initial_states(self, initial_states: InitialState | None, append: bool = False):
        """
        Sets or appends PDDL problem initial states for current specification
        Args:
            initial_states (InitialState | None): A single InitialState object, 
                or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("initial_state", initial_states, append=append)

    def set_goal_states(self, goal_states: GoalState | None, append: bool = False):
        """
        Sets or appends PDDL problem goal states for current specification
        Args:
            goal_states (GoalState | None): A single GoalState object, 
                or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("goal_state", goal_states, append=append)

    def set_constraints(self, constraints: list[Constraint] | Constraint | None, append: bool = False):
        """
        Sets or appends PDDL problem constraints for current specification
        Args:
            constraints (list[Constraint] | Constraint | None): A single Constraint object, 
                a list of Constraint objects, or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("constraint", constraints, append=append)

    def set_metric(self, metric: Metric | None, append: bool = False):
        """
        Sets or appends PDDL problem metric for current specification
        Args:
            metric (Metric | None): A single Metric object, 
                or None to clear the list.
            append (bool, optional): If True, appends the given types to the 
                existing list. If False, overwrites the existing list completely. 
                Defaults to False.
        """
        self._set_component("metric", metric, append=append)

    # ---------------------------------------------------------------------------
    # PDDL PROBLEM DISPLAY FUNCTIONS
    # ---------------------------------------------------------------------------

if __name__ == "__main__":
    pass