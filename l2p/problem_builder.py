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
from l2p.utils.pddl_format_new import *
from l2p.utils.pddl_types_new import DomainDetails, ProblemDetails


class ProblemBuilder:
    def __init__(
            self, 
            problem_details: Optional[ProblemDetails] = None,
            domain_details: Optional[DomainDetails] = None,
        ) -> None:
        """
        Initializes an L2P task builder object using the new Pydantic ProblemDetails root model.

        Args:
            problem_details (ProblemDetails | None): A fully validated Pydantic model 
            containing the entire PDDL problem structure.
            domain_details (DomainDetails | None): The corresponding domain details.
        """
        
        fallback_domain_name = domain_details.name if domain_details else "temp-domain"
        self.problem_details = problem_details or ProblemDetails(
            name="temp-problem", 
            domain_name=fallback_domain_name
        )

        if problem_details and getattr(self.problem_details, "domain_name", None) in [None, ""]:
            self.problem_details.domain_name = fallback_domain_name

    def generate_problem(self, problem_details: ProblemDetails) -> str:
        """
        Assembles all formatted components into a complete PDDL problem string.
        """
        desc = f"(define (problem {problem_details.name})\n"
        desc += f"   (:domain {problem_details.domain_name})\n"

        if problem_details.objects:
            objs_str = format_objects(problem_details.objects)
            desc += f"\n   (:objects \n{indent(string=objs_str, level=2)}\n   )"

        # 1. Check BOTH lists inside the new InitialState model
        if not problem_details.initial_state.facts and not problem_details.initial_state.timed_facts:
            print("[WARNING]: Problem has no initial state.")
        else:
            # 2. Pass the entire initial_state object, not the old 'initials' list
            init_str = format_initial_state(problem_details.initial_state)
            desc += f"\n\n   (:init \n{indent(string=init_str, level=2)}\n   )"

        # 3. Check the 'conditions' list inside the new GoalStates model
        if not problem_details.goal_state.conditions:
            print("[WARNING]: Problem has no goals.")
        else:
            # 4. Use the format_goal_states helper instead of format_logic directly
            goal_str = format_goal_states(problem_details.goal_state)
            desc += f"\n\n   (:goal \n      {goal_str}\n   )"

        # 5. Fix typo: problem_details.constraint -> problem_details.constraints
        if problem_details.constraint:
            constraints_str = format_constraints(problem_details.constraint)
            desc += f"\n\n   (:constraints \n{indent(string=constraints_str, level=2)}\n   )"

        if problem_details.metric:
            metric_str = format_metric(problem_details.metric)
            desc += f"\n\n{indent(string=metric_str, level=1)}"

        desc += "\n)"
        # Standardize logic operators
        desc = desc.replace("AND", "and").replace("OR", "or")
        return desc

if __name__ == "__main__":
    raw_problem_data = {
        "name": "p1",
        "domain_name": "advanced_rover_mission",
        "objects": [
            {"name": "rover1", "type": "rover"},
            {"name": "wp1", "type": "waypoint"},
            {"name": "wp2", "type": "waypoint"},
            {"name": "obj1", "type": "object"}
        ],
        # Replaced loose array with rigid InitialState structure
        "initial_state": {
            "facts": [
                "(at rover1 wp1)",
                "(= (battery-level rover1) 100)"
            ],
            "timed_facts": [
                {
                    "time": 10.0, 
                    "fact": "(solar-flare)"
                }
            ]
        },
        # Replaced single condition with rigid GoalStates structure
        "goal_state": {
            "conditions": [
                {
                    "preference": "P1",
                    "condition": "(data-transmitted)"
                },
                "(at rover1 wp2)"  # You can list multiple goals; format_goal_states will automatically (and ...) them!
            ]
        },
        "metric": {
            "optimization": "minimize",
            "expression": "(+ (* 10 (is-violated P1)) (total-cost))"
        }
    }

problem = ProblemDetails(**raw_problem_data)
builder = ProblemBuilder()
print(builder.generate_problem(problem))