"""
L2P PDDL Type and Data Structure Definitions.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# PDDL DOMAIN CLASSES
# ---------------------------------------------------------------------------

"""
LogicalCondition represents a single logical node in a PDDL formula. This type 
alias allows the LLM to output a mix of simple strings and recursive dictionaries:

    1. Simple Predicates & Numeric Checks (str): 
        "(at ?r ?l)"
        "(>= (battery-level ?r) 20)"

    2. Basic Logical Operators (Dict): 
        # NOT
        {
            "operator": "not",
            "condition": "(busy ?r)"
        }
        
        # AND / OR (Can contain nested conditions)
        {
            "operator": "and",  # or "or"
            "conditions": [
                "(has-power ?r)",
                {"operator": "not", "condition": "(busy ?r)"}
            ]
        }
        
        # IMPLY (Requires antecedent and consequent)
        {
            "operator": "imply",
            "antecedent": ["(at ?r ?l)"],
            "consequent": ["(can-transmit ?r)"]
        }
    
    3. Quantifiers (Dict):
        # FORALL / EXISTS
        {
            "quantifier": "forall",  # or "exists"
            "parameters": [{"variable": "?p", "type": "packet"}],
            "conditions": [
                "(transmitted ?p)"
            ]
        }

    4. PDDL 3.0 Trajectory Constraints (Dict):
        # ALWAYS / SOMETIME / AT-MOST-ONCE (Basic modal operators: 1 condition)
        {
            "operator": "always",  # or "sometime", "at-most-once"
            "condition": "(has-power ?r)"
        }
        
        # WITHIN / HOLD-AFTER (Time-bounded modal operators: 1 time value + 1 condition)
        {
            "operator": "within",  # or "hold-after"
            "time": 10.5,
            "condition": "(transmitted ?p)"
        }
        
        # HOLD-DURING (Interval modal operator: 2 time values + 1 condition)
        {
            "operator": "hold-during",
            "time_start": 5.0,
            "time_end": 15.0,
            "condition": "(transmitting ?r)"
        }
        
        # SOMETIME-AFTER / SOMETIME-BEFORE (Relational operators: 2 conditions)
        {
            "operator": "sometime-after",  # or "sometime-before"
            "antecedent": "(transmitted ?p)",
            "consequent": "(acknowledged ?p)"
        }
        
        # ALWAYS-WITHIN (Time-bounded relational: 1 time value + 2 conditions)
        {
            "operator": "always-within",
            "time": 5.0,
            "antecedent": "(error-detected ?r)",
            "consequent": "(safe-mode ?r)"
        }
    
    5. PDDL 3.0 Preferences (Dict):
        # PREFERENCE (Assigns a name to a condition for metric tracking)
        {
            "preference": "pref_transmit_early",
            "condition": {
                "operator": "sometime",
                "condition": "(transmitted ?p)"
            }
        }
"""
LogicalCondition = Union[str, Dict[str, Any]]

class PDDLType(BaseModel):
    name: str
    parent: str
    desc: Optional[str] = None

class Constant(BaseModel):
    name: str
    type: str
    desc: Optional[str] = None

class Parameter(BaseModel):
    variable: str   # e.g., "?r"
    type: str       # e.g., "rover"
    desc: Optional[str] = None

    @field_validator('variable')
    @classmethod
    def check_question_mark(cls, v: str) -> str:
        if not v.startswith('?'):
            raise ValueError(f"Parameter variable '{v}' must start with '?'")
        return v

class Predicate(BaseModel):
    name: str
    params: List[Parameter]
    desc: Optional[str] = None

class Function(BaseModel):
    name: str
    params: List[Parameter]
    desc: Optional[str] = None

# PDDL 2.2 Derived Predicates (Axioms)
class DerivedPredicate(BaseModel):
    name: str
    params: List[Parameter]
    condition: LogicalCondition
    desc: Optional[str] = None


class ActionPrecondition(BaseModel):
    conditions: List[LogicalCondition] = Field(default_factory=list)
    desc: Optional[str] = None

class ConditionalEffect(BaseModel):
    condition: List[LogicalCondition]
    effect: Dict[str, List[LogicalCondition]] # e.g., {"add": [], "delete": [], "numeric": []}
    desc: Optional[str] = None

class ActionEffect(BaseModel):
    add: List[LogicalCondition] = Field(default_factory=list)
    delete: List[LogicalCondition] = Field(default_factory=list)
    numeric: List[LogicalCondition] = Field(default_factory=list)
    conditional: List[ConditionalEffect] = Field(default_factory=list)
    desc: Optional[str] = None

class Action(BaseModel):
    name: str
    params: List[Parameter]
    preconditions: ActionPrecondition = Field(default_factory=ActionPrecondition)
    effects: ActionEffect = Field(default_factory=ActionEffect)
    desc: Optional[str] = None


class DurativeActionConditions(BaseModel):
    at_start: List[LogicalCondition] = Field(default_factory=list)
    at_end: List[LogicalCondition] = Field(default_factory=list)
    over_all: List[LogicalCondition] = Field(default_factory=list)
    desc: Optional[str] = None

class DurativeActionEffect(BaseModel):
    at_start: ActionEffect = Field(default_factory=ActionEffect)
    at_end: ActionEffect = Field(default_factory=ActionEffect)
    continuous: List[LogicalCondition] = Field(default_factory=list)  # added for PDDL 2.1/+ continuous numeric changes using #t
    desc: Optional[str] = None

class DurativeAction(BaseModel):
    name: str
    params: List[Parameter]
    duration: List[str]
    conditions: DurativeActionConditions
    effects: DurativeActionEffect
    desc: Optional[str] = None

# PDDL 3.0
class Constraint(BaseModel):
    condition: LogicalCondition
    desc: Optional[str] = None


# PDDL+
class Event(BaseModel):
    name: str
    params: List[Parameter]
    preconditions: ActionPrecondition = Field(default_factory=ActionPrecondition)
    effects: ActionEffect = Field(default_factory=ActionEffect)
    desc: Optional[str] = None

class Process(BaseModel):
    name: str
    params: List[Parameter]
    preconditions: ActionPrecondition = Field(default_factory=ActionPrecondition)
    effects: ActionEffect = Field(default_factory=ActionEffect)
    desc: Optional[str] = None


class DomainDetails(BaseModel):
    """Root model for parsing a complete PDDL domain."""
    name: str
    desc: Optional[str] = None
    domain_pddl: Optional[str] = None # optional raw PDDL string for whole domain
    
    # meta-data
    requirements: List[str] = Field(default_factory=list)
    types: List[PDDLType] = Field(default_factory=list)
    constants: List[Constant] = Field(default_factory=list)
    
    # state variables
    predicates: List[Predicate] = Field(default_factory=list)
    functions: List[Function] = Field(default_factory=list)
    derived_predicates: List[DerivedPredicate] = Field(default_factory=list)

    # standard and durative actions
    actions: List[Action] = Field(default_factory=list)
    durative_actions: List[DurativeAction] = Field(default_factory=list)

    # PDDL+ & PDDL 3.0 extensions
    events: List[Event] = Field(default_factory=list)
    processes: List[Process] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PDDL PROBLEM CLASSES
# ---------------------------------------------------------------------------

"""
InitFact represents a single condition in the initial state. 
This allows a mix of standard PDDL strings and PDDL 2.2 Timed Initial Literals (TILs).

    1. Standard Facts & Numeric Assignments (str):
        "(at rover1 waypoint1)"
        "(= (battery rover1) 100)"
        
    2. Timed Initial Literals / Fluents (Dict):
        # Triggers a fact at a specific time during the plan execution
        {
            "time": 15.5,
            "fact": "(communications-blackout)"
        }
"""
InitFact = Union[str, Dict[str, Any]]

class PDDLObject(BaseModel):
    name: str   # e.g., "rover1"
    type: str   # e.g., "rover"
    desc: Optional[str] = None

# PDDL 2.1 & PDDL 3.0 Plan Optimization Metrics
class Metric(BaseModel):
    optimization: str   # must be 'minimize' or 'maximize'
    expression: str     # e.g., 'total-time', '(* (fuel-used) 2)', or '(is-violated pref1)'
    desc: Optional[str] = None

    @field_validator('optimization')
    @classmethod
    def check_optimization_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ["minimize", "maximize"]:
            raise ValueError("Optimization must be 'minimize' or 'maximize'")
        return v

class ProblemDetails(BaseModel):
    name: str
    domain_name: str
    desc: Optional[str] = None
    problem_pddl: Optional[str] = None # optional raw PDDL string for whole problem

    # typed object instances
    objects: List[PDDLObject] = Field(default_factory=list)
    initial: List[InitFact] = Field(default_factory=list)
    goal: Optional[LogicalCondition] = None

    # PDDL 3.0 problem-specific trajectory constraints
    constraints: List[Constraint] = Field(default_factory=list)
    metric: Optional[Metric] = None


class PlanDetails(BaseModel):
    domain: DomainDetails
    problem: ProblemDetails
    plan: str
    desc: Optional[str] = None
