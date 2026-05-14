"""
PDDL Problem Formalization/Generation Functions

This module defines the `ProblemBuilder` class and related utilities for constructing
PDDL problem specifications programatically.

Refer to: https://marcustantakoun.github.io/l2p.github.io/l2p.html for more information
how to use class functions. Refer to /templates in: https://github.com/AI-Planning/l2p
for how to structurally prompt LLMs so they are compatible with class function parsing.
"""

import time
from typing import Optional, Set, TypeVar, Union, Type

from l2p.llm import BaseLLM, require_llm
from l2p.utils.pddl_types import DomainDetails, ProblemDetails
from l2p.utils.pddl_format import *
from l2p.utils.pddl_prompt import DEF_PROBLEM_PROMPTS, build_ctx, safe_format
from l2p.utils.pddl_parser import parse_xml_tags, parse_component

T = TypeVar('T', bound=BaseModel)

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
    def formalize_component(
        self,
        model: BaseLLM,
        component_class: Union[List[Type[T]], Type[T]],
        prompt_template: Optional[str] = None,
        problem_desc: Optional[str] = None,
        max_retries: int = 3,
        **ctx_kwargs
    ):
        """
        Formalizes L2P Domain components using BaseLLM
        Args:
            model (BaseLLM):
            prompt_template (str):
            component_class (List[Type[T]] | Type[T]):
            description (Optional[str] = None):
            max_retries (int = 3):
            **ctx_kwargs:
        """
        classes = component_class if isinstance(component_class, list) else [component_class]

        if not prompt_template:
            if len(classes) > 1:
                prompt_template = getattr(DEF_PROBLEM_PROMPTS, "combined", None)
            else:
                cls = classes[0]
                tag = cls.tag[0] if cls.tag else cls.__name__
                prompt_template = getattr(DEF_PROBLEM_PROMPTS, tag, 
                                          getattr(DEF_PROBLEM_PROMPTS, cls.__name__, None))

            if not prompt_template:
                raise ValueError(f"[ERROR] No prompt template provided and no default found for {classes}")

        context = build_ctx(**ctx_kwargs)
        prompt = safe_format(
            template=prompt_template,
            problem_desc=problem_desc,
            context_injection=context
        )
        
        for attempt in range(max_retries):
            try:
                model.reset_tokens()
                llm_output = model.query(prompt=prompt)
                
                extracted_results = {}

                for cls in classes:
                    raw_blocks, matched_tag = None, None

                    for t in cls.tag:
                        raw_blocks = parse_xml_tags(llm_output=llm_output,tag_name=t)
                        if raw_blocks:
                            matched_tag = t
                            break

                    if not raw_blocks:
                        raise ValueError(f"[ERROR] Missing expected XML block in LLM output. Looked for: {cls.tag}")
                    
                    parsed_items = parse_component(raw_blocks=raw_blocks, model_class=cls, tag_name=matched_tag)
                    extracted_results[cls] = parsed_items

                returned_classes = extracted_results if isinstance(component_class, list) else extracted_results[classes[0]]
                return returned_classes, llm_output

            except Exception as e:
                print(
                    f"Error encountered during attempt {attempt + 1}/{max_retries}: {e}. "
                    f"\nLLM Output: \n\n{llm_output if 'llm_output' in locals() else 'None'}\n\n Retrying..."
                )
                time.sleep(2)  # add a delay before retrying

        raise RuntimeError("Max retries exceeded. Failed to extract types.")


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