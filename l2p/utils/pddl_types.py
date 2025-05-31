"""
L2P PDDL Type and Data Structure Definitions

This module defines core types and data classes used for representing components of PDDL
domains, problems, and plans. These include structured representations for predicates, actions,
functions, domain/task metadata, and parameterized object lists.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import TypedDict, NewType, Optional

ParameterList = NewType(
    "ParameterList", OrderedDict[str, str]
)  # {param_name: param_type}
ObjectList = NewType("ObjectList", dict[str, str])  # {obj_name: obj_type}


class PDDLType(TypedDict):
    name: str
    parent: str
    desc: str


class Constant(TypedDict):
    name: str
    type: str


class Predicate(TypedDict):
    name: str
    desc: Optional[str]
    raw: str
    params: ParameterList
    clean: str


class Function(TypedDict):
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


# Domain details data class including predicates and actions
@dataclass
class DomainDetails:
    name: str
    domain_desc: str
    domain_pddl: str
    requirements: list[str]
    types: dict[str, str] | list[dict[str, str]]
    constants: dict[str, str]
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
    domain_pddl: str
    problem_pddl: str
    plan_pddl: str
    plan_nl: str
