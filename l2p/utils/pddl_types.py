"""
L2P PDDL Type and Data Structure Definitions.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, ClassVar, Dict, List, Optional, Union

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

class Requirement(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": ":typing",
        "desc": "Enables typed variables"
    }
    """
    tag: ClassVar[tuple] = ("requirements", "requirement")
    name: str
    desc: Optional[str] = None

    @field_validator('name')
    @classmethod
    def check_colon(cls, r: str) -> str:
        if not r.startswith(':'):
            raise ValueError(f"Requirement name '{r}' must start with ':'")
        return r

class PDDLType(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "rover",
        "parent": "vehicle",
        "desc": "A planetary rover"
    }
    """
    tag: ClassVar[tuple] = ("types", "type")
    name: str
    parent: str
    desc: Optional[str] = None

class Constant(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "base_station",
        "type": "location",
        "desc": "The main hub"
    }
    """
    tag: ClassVar[tuple] = ("constants", "constant")
    name: str
    type: str
    desc: Optional[str] = None

class Parameter(BaseModel):
    """
    Expected JSON format (example):
    {
        "variable": "?r",
        "type": "rover",
        "desc": "A rover"
    }
    """
    tag: ClassVar[tuple] = ("parameters", "parameter")
    variable: str
    type: str
    desc: Optional[str] = None

    @field_validator('variable')
    @classmethod
    def check_question_mark(cls, v: str) -> str:
        if not v.startswith('?'):
            raise ValueError(f"Parameter variable '{v}' must start with '?'")
        return v

class Predicate(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "at",
        "params": [
            {"variable": "?r", "type": "rover"},
            {"variable": "?l", "type": "location"}
        ],
        "desc": "True if rover is at location"
    }
    """
    tag: ClassVar[tuple] = ("predicates", "predicate")
    name: str
    params: List[Parameter]
    desc: Optional[str] = None

class Function(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "battery-level",
        "params": [{"variable": "?r", "type": "rover"}],
        "desc": "Current battery level of the rover"
    }
    """
    tag: ClassVar[tuple] = ("functions", "function")
    name: str
    params: List[Parameter]
    desc: Optional[str] = None

# PDDL 2.2 Derived Predicates (Axioms)
class DerivedPredicate(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "can-move",
        "params": [{"variable": "?r", "type": "rover"}],
        "condition": "(> (battery-level ?r) 0.0)",
        "desc": "Derived from having positive battery"
    }
    """
    tag: ClassVar[tuple] = ("derived_predicates", "derived_predicate")
    name: str
    params: List[Parameter]
    condition: LogicalCondition
    desc: Optional[str] = None


class ActionPrecondition(BaseModel):
    """
    Expected JSON format (example):
    {
        "conditions": [
            "(at ?r ?from)",
            {"operator": "not", "condition": "(busy ?r)"}
        ],
        "desc": "Rover must be at the start location and not busy"
    }
    """
    tag: ClassVar[tuple] = ("preconditions", "precondition")
    conditions: List[LogicalCondition] = Field(default_factory=list)
    desc: Optional[str] = None

class ConditionalEffect(BaseModel):
    """
    Expected JSON format (example):
    {
        "condition": ["(has-payload ?r)"],
        "effect": {
            "add": ["(payload-delivered)"],
            "delete": [],
            "numeric": []
        },
        "desc": "If it has a payload, it is delivered"
    }
    """
    tag: ClassVar[tuple] = ("conditional_effects", "conditional_effect")
    condition: List[LogicalCondition]
    effect: Dict[str, List[LogicalCondition]] # e.g., {"add": [], "delete": [], "numeric": []}
    desc: Optional[str] = None

class ActionEffect(BaseModel):
    """
    Expected JSON format (example):
    {
        "add": ["(at ?r ?to)"],
        "delete": ["(at ?r ?from)"],
        "numeric": ["(decrease (battery-level ?r) 5.0)"],
        "conditional": [
            "condition": [],
            "effect": {
                "add": [],
                "delete": [],
                "numeric": []
            }
        ],
        "desc": "Optional description"
    }
    """
    tag: ClassVar[tuple] = ("effects", "effect")
    add: List[LogicalCondition] = Field(default_factory=list)
    delete: List[LogicalCondition] = Field(default_factory=list)
    numeric: List[LogicalCondition] = Field(default_factory=list)
    conditional: List[ConditionalEffect] = Field(default_factory=list)
    desc: Optional[str] = None

class Action(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "drive",
        "params": [
            {"variable": "?r", "type": "rover"},
            {"variable": "?from", "type": "location"}
        ],
        "preconditions": {
            "conditions": ["(at ?r ?from)"],
            "desc": null
        },
        "effects": {
            "add": [], 
            "delete": [], 
            "numeric": [], 
            "conditional": []
        },
        "desc": "Classical action to drive"
    }
    """
    tag: ClassVar[tuple] = ("actions", "action")
    name: str
    params: List[Parameter]
    preconditions: ActionPrecondition = Field(default_factory=ActionPrecondition)
    effects: ActionEffect = Field(default_factory=ActionEffect)
    desc: Optional[str] = None


class DurativeActionConditions(BaseModel):
    """
    Expected JSON format (example):
    {
        "at_start": ["(at ?r ?from)"],
        "over_all": [{"operator": "not", "condition": "(busy ?r)"}],
        "at_end": [],
        "desc": null
    }
    """
    tag: ClassVar[tuple] = ("durative_conditions", "durative_condition")
    at_start: List[LogicalCondition] = Field(default_factory=list)
    at_end: List[LogicalCondition] = Field(default_factory=list)
    over_all: List[LogicalCondition] = Field(default_factory=list)
    desc: Optional[str] = None

class DurativeActionEffect(BaseModel):
    """
    Expected JSON format (example):
    {
        "at_start": {
            "add": [], 
            "delete": [], 
            "numeric": [], 
            "conditional": []
        },
        "at_end": {
            "add": ["(at ?r ?to)"], 
            "delete": ["(at ?r ?from)"], 
            "numeric": [], 
            "conditional": []
        },
        "continuous": ["(decrease (battery-level ?r) (* #t 1.0))"],
        "desc": null
    }
    """
    tag: ClassVar[tuple] = ("durative_effects", "durative_effect")
    at_start: ActionEffect = Field(default_factory=ActionEffect)
    at_end: ActionEffect = Field(default_factory=ActionEffect)
    continuous: List[LogicalCondition] = Field(default_factory=list)  # added for PDDL 2.1/+ continuous numeric changes using #t
    desc: Optional[str] = None

class DurativeAction(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "transmit",
        "params": [{"variable": "?r", "type": "rover"}],
        "duration": ["(>= ?duration 5.0)"],
        "conditions": {
            "at_start": ["(at ?r base)"], "over_all": [], "at_end": []
        },
        "effects": {
            "at_start": {"add": [], "delete": [], "numeric": [], "conditional": []},
            "at_end": {"add": ["(data-transmitted)"], "delete": [], "numeric": [], "conditional": []},
            "continuous": ["(decrease (battery ?r) (* #t 2.0))"]
        },
        "desc": "Durative transmission action"
    }
    """
    tag: ClassVar[tuple] = ("durative_actions", "durative_action")
    name: str
    params: List[Parameter]
    duration: List[str]
    conditions: DurativeActionConditions
    effects: DurativeActionEffect
    desc: Optional[str] = None

# PDDL 3.0
class Constraint(BaseModel):
    """
    Expected JSON format (example):
    {
        "condition": {
            "operator": "always", 
            "condition": "(> (battery-level ?r) 0.0)"
        },
        "desc": "Battery must always be positive"
    }
    """
    tag: ClassVar[tuple] = ("constraints", "constraint")
    condition: LogicalCondition
    desc: Optional[str] = None


# PDDL+
class Event(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "battery-depleted",
        "params": [
            {"variable": "?r", "type": "rover"}
        ],
        "preconditions": {
            "conditions": [
                "(<= (battery-level ?r) 0)"
            ], 
            "desc": null
        },
        "effects": {
            "add": ["(dead ?r)"], 
            "delete": [], 
            "numeric": [], 
            "conditional": []
        },
        "desc": "Triggers when battery dies"
    }
    """
    tag: ClassVar[tuple] = ("events", "event")
    name: str
    params: List[Parameter]
    preconditions: ActionPrecondition = Field(default_factory=ActionPrecondition)
    effects: ActionEffect = Field(default_factory=ActionEffect)
    desc: Optional[str] = None

class Process(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "solar-charging",
        "params": [
            {"variable": "?r", "type": "rover"}
        ],
        "preconditions": {
            "conditions": [
                "(in-sun ?r)"
            ], 
            "desc": null
        },
        "effects": {
            "add": [], 
            "delete": [], 
            "numeric": [
                "(increase (battery-level ?r) (* #t 2.0))"
            ], 
            "conditional": []
        },
        "desc": "Charges continuously in sun"
    }
    """
    tag: ClassVar[tuple] = ("processes", "process")
    name: str
    params: List[Parameter]
    preconditions: ActionPrecondition = Field(default_factory=ActionPrecondition)
    effects: ActionEffect = Field(default_factory=ActionEffect)
    desc: Optional[str] = None


class DomainDetails(BaseModel):
    """
    Root model for parsing a complete PDDL domain.
    Expected JSON format (truncated example):
    {
        "name": "domain-name",
        "requirements": [":strips", ":typing"],
        "types": [
            {"name": "type_1", "parent": "object"}
        ],
        "constants": [...],
        "predicates": [...],
        "functions": [...],
        "derived_predicates": [...],
        "actions": [...],
        "durative_actions": [...],
        "events": [...],
        "processes": [...]
        "constraints": [...]
    }
    """
    tag: ClassVar[str] = "domain"
    name: str
    desc: Optional[str] = None
    domain_pddl: Optional[str] = None # optional raw PDDL string for whole domain
    
    # meta-data
    requirements: List[Requirement] = Field(default_factory=list)
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
    constraint: List[Constraint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PDDL PROBLEM CLASSES
# ---------------------------------------------------------------------------

"""
LogicalCondition represents a single condition in the initial state. 
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

class PDDLObject(BaseModel):
    """
    Expected JSON format (example):
    {
        "name": "rover1",
        "type": "rover",
        "desc": "Instance of a rover"
    }
    """
    tag: ClassVar[tuple] = ("objects", "object")
    name: str   # e.g., "r1"
    type: str   # e.g., "rover"
    desc: Optional[str] = None

class TimedFact(BaseModel):
    """
    Expected JSON format (example):
    {
        "time": 15.5,
        "fact": "(communications-blackout)",
        "desc": "Event triggers at t=15.5"
    }
    """
    tag: ClassVar[tuple] = ("timed_facts", "timed_fact")
    time: float
    fact: LogicalCondition
    desc: Optional[str] = None

class InitialState(BaseModel):
    """
    Expected JSON format (example):
    {
        "facts": [
            "(at rover1 loc1)", 
            "(= (battery-level rover1) 100)"
        ],
        "timed_facts": [],
        "desc": "Starting state"
    }
    """
    tag: ClassVar[tuple] = ("initial_states", "initial_state", "inital", "init")
    facts: List[LogicalCondition] = Field(default_factory=list)
    timed_facts: List[TimedFact] = Field(default_factory=list)
    desc: Optional[str] = None

class GoalState(BaseModel):
    """
    Expected JSON format (example):
    {
        "conditions": [
            "(at rover1 loc2)", 
            "(data-transmitted)"
        ],
        "desc": "Target state"
    }
    """
    tag: ClassVar[tuple] = ("goal_states", "goal_state", "goals", "goal")
    conditions: List[LogicalCondition] = Field(default_factory=list)
    desc: Optional[str] = None

# PDDL 2.1 & PDDL 3.0 Plan Optimization Metrics
class Metric(BaseModel):
    """
    Expected JSON format (example):
    {
        "optimization": "minimize",
        "expression": "total-time",
        "desc": "Minimize makespan"
    }
    """
    tag: ClassVar[tuple] = ("metrics", "metric")
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
    """
    Expected JSON format (truncated example):
    {
        "name": "prob1",
        "domain_name": "rover_domain",
        "objects": [{"name": "rover1", "type": "rover"}],
        "initial_state": {"facts": [...], "timed_facts": [...]},
        "goal_state": {"conditions": [...]}
    }
    """
    tag: ClassVar[str] = "problem"
    name: str
    domain_name: str
    desc: Optional[str] = None
    problem_pddl: Optional[str] = None # optional raw PDDL string for whole problem

    # typed object instances
    objects: List[PDDLObject] = Field(default_factory=list)
    initial_state: InitialState = Field(default_factory=InitialState)
    goal_state: GoalState = Field(default_factory=GoalState)

    # PDDL 3.0 problem-specific trajectory constraints
    constraint: List[Constraint] = Field(default_factory=list)
    metric: Optional[Metric] = None


# ---------------------------------------------------------------------------
# PDDL PLAN CLASSES
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    """
    Expected JSON format (example):
    {
        "action": "drive",
        "args": ["rover1", "loc1", "loc2"],
        "start_time": 0.0,
        "duration": 5.5,
        "desc": "Optional description"
    }
    """
    action: str                  # e.g., "drive"
    args: List[str]              # e.g., ["rover1", "loc1", "loc2"] (grounded objects)
    start_time: float = 0.0      # for durative actions / temporal planning
    duration: Optional[float] = None  # None for classical actions, float for durative
    desc: Optional[str] = None

class PlanDetails(BaseModel):
    """
    Expected JSON format (truncated example):
    {
        "domain": {...},
        "problem": {...},
        "raw_plan": "0.0: (drive rover1 loc1 loc2) [5.5]",
        "steps": [
            {"action": "drive", "args": ["rover1", "loc1", "loc2"]}
        ],
        "cost": 10.0,
        "makespan": 5.5
    }
    """
    domain: DomainDetails
    problem: ProblemDetails
    raw_plan: str                # keep the raw text for logging or standard VAL validation
    steps: List[PlanStep] = Field(default_factory=list)
    cost: Optional[float] = None # total plan cost (if using action costs)
    makespan: Optional[float] = None # total time to complete (for durative plans)
    desc: Optional[str] = None
