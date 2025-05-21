"""
This file contains classes of PDDL types
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import TypedDict, NewType, Optional

ParameterList = NewType("ParameterList", OrderedDict[str, str]) # {param_name: param_type}
ObjectList = NewType("ObjectList", dict[str, str]) # {obj_name: obj_type}


class Predicate(TypedDict):
    name: str
    desc: Optional[str]
    raw: str
    params: ParameterList
    clean: str


class Action(TypedDict):
    name: str
    desc: Optional[str]
    raw: str
    params: ParameterList
    preconditions: str
    effects: str


class Function(TypedDict):
    name: str
    desc: Optional[str]
    raw: str
    params: ParameterList
    clean: str
    

# Domain details data class including predicates and actions
@dataclass
class DomainDetails:
    name: str
    domain_desc: str
    domain_pddl: str
    requirements: list[str]
    types: dict[str,str] | list[dict[str,str]]
    constants: dict[str,str]
    predicates: list[Predicate]  # List of Predicate objects
    functions: list[Function]
    actions: list[Action]  # List of Action objects


# Problem details data class
@dataclass
class ProblemDetails:
    name: str
    problem_desc: str
    problem_pddl: str
    objects: tuple[dict[str, str], str]
    initial: tuple[dict[str, str], str]
    goal: tuple[dict[str, str], str]


# Plan details data class
@dataclass
class PlanDetails:
    plan_pddl: str
    plan_nl: str