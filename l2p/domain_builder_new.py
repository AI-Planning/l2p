"""
PDDL Domain Formalization/Generation Functions

This module defines the `DomainBuilder` class and related utilities for constructing
PDDL domain specifications programmatically using structured Pydantic models.
"""

import time
from typing import Optional, Union, Any, Set

from l2p.utils.pddl_types_new import DomainDetails, ProblemDetails, Requirement, LogicalCondition
from l2p.utils.pddl_format_new import *
from l2p.utils.pddl_parser_new import parse_xml_tags, parse_element

from l2p.llm import BaseLLM, require_llm
from l2p.utils import SyntaxValidator

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
        
        self.domain_details = domain_details or DomainDetails(name="temp-domain")
    

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
    
if __name__ == "__main__":

    raw_domain_data = {
        "name": "advanced_rover_mission",
        "types": [
            {"name": "waypoint", "parent": "object"},
            {"name": "rover", "parent": "object"},
            {"name": "camera", "parent": "object"}
        ],
        "constants": [
            {"name": "base_station", "type": "waypoint"}
        ],
        "predicates": [
            {
                "name": "at",
                "params": [
                    {"variable": "?r", "type": "rover"},
                    {"variable": "?w", "type": "waypoint"}
                ]
            },
            {
                "name": "safe-mode",
                "params": [
                    {"variable": "?r", "type": "rover"}
                ]
            },
            {
                "name": "can-transmit",
                "params": [
                    {"variable": "?r", "type": "rover"}
                ]
            }
        ],
        "functions": [
            {
                "name": "battery-level",
                "params": [
                    {"variable": "?r", "type": "rover"}
                ]
            },
            {
                "name": "total-cost",  # Global function
                "params": []
            }
        ],
        "constraints": [
            {
                "condition": {
                    "quantifier": "forall",
                    "parameters": [{"variable": "?r", "type": "rover"}],
                    "conditions": [
                        {
                            "operator": "always",
                            "condition": "(>= (battery-level ?r) 0)"
                        }
                    ]
                }
            }
        ],
        "derived_predicates": [
            {
                "name": "can-transmit",
                "params": [
                    {"variable": "?r", "type": "rover"}
                ],
                "condition": {
                    "operator": "and",
                    "conditions": [
                        "(at ?r base_station)",
                        "(>= (battery-level ?r) 50)"
                    ]
                }
            }
        ],
        "actions": [
            {
                "name": "navigate",
                "params": [
                    {"variable": "?r", "type": "rover"},
                    {"variable": "?from", "type": "waypoint"},
                    {"variable": "?to", "type": "waypoint"}
                ],
                "preconditions": {
                    "conditions": [
                        "(at ?r ?from)",
                        {"operator": "not", "condition": "(= ?from ?to)"},  # Triggers :equality & :negative-preconditions
                        {"operator": "or", "conditions": [                  # Triggers :disjunctive-preconditions
                            "(>= (battery-level ?r) 20)",
                            "(at ?r base_station)"
                        ]},
                        {
                            "quantifier": "forall",                         # Triggers :quantified-preconditions
                            "parameters": [{"variable": "?c", "type": "camera"}],
                            "conditions": ["(calibrated ?c)"]
                        }
                    ]
                },
                "effects": {
                    "add": ["(at ?r ?to)"],
                    "delete": ["(at ?r ?from)"],
                    "numeric": [
                        "(decrease (battery-level ?r) 10)",
                        "(increase (total-cost) 1)"                         # Triggers :action-costs
                    ],
                    "conditional": [                                        # Triggers :conditional-effects
                        {
                            "condition": ["(has-rock-sample ?r)"],
                            "effect": {
                                "add": ["(carrying-heavy-load ?r)"]
                            }
                        }
                    ]
                }
            }
        ],
        "durative_actions": [
            {
                "name": "transmit_data",
                "params": [
                    {"variable": "?r", "type": "rover"}
                ],
                "duration": ["(>= ?duration 5.0)"],                         # Triggers :durative-inequalities
                "conditions": {
                    "at_start": ["(at ?r base_station)"],
                    "over_all": [{"operator": "not", "condition": "(safe-mode ?r)"}],
                    "at_end": []
                },
                "effects": {
                    "at_start": {
                        "add": ["(transmitting ?r)"]
                    },
                    "at_end": {
                        "add": ["(data-transmitted)"],
                        "delete": ["(transmitting ?r)"]
                    },
                    "continuous": [
                        "(decrease (battery-level ?r) (* #t 2.0))"          # Triggers :continuous-effects
                    ]
                }
            }
        ],
        "events": [                                                         # Triggers :time
            {
                "name": "battery_depleted_event",
                "params": [{"variable": "?r", "type": "rover"}],
                "preconditions": {
                    "conditions": ["(<= (battery-level ?r) 0)"]
                },
                "effects": {
                    "add": ["(safe-mode ?r)"]
                }
            }
        ],
        "processes": [                                                      # Triggers :time
            {
                "name": "passive_battery_drain",
                "params": [{"variable": "?r", "type": "rover"}],
                "preconditions": {
                    "conditions": [
                        {"operator": "not", "condition": "(safe-mode ?r)"}
                    ]
                },
                "effects": {
                    "numeric": ["(decrease (battery-level ?r) (* #t 0.1))"] # Triggers :continuous-effects
                }
            }
        ]
    }

    # Optional: A problem payload to test problem-specific requirement generation
    raw_problem_data = {
        "name": "p1",
        "domain_name": "advanced_rover_mission",
        "initials": [
            {"time": 10.0, "fact": "(solar-flare)"}                         # Triggers :timed-initial-literals
        ],
        "goals": {
            "preference": "P1",                                             # Triggers :preferences
            "condition": "(data-transmitted)"
        },
        "metric": {
            "optimization": "minimize",
            "expression": "(+ (* 10 (is-violated P1)) (total-cost))"        # Triggers :preferences
        }
    }
    
    domain = DomainDetails(**raw_domain_data)
    problem = ProblemDetails(**raw_problem_data)
    db = DomainBuilder()
    print(db.generate_domain(domain))