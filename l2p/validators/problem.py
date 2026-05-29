"""
PDDL Problem Validation Rules

This module provides a rule-based validation system for PDDL problem instances.
It extends the infrastructure in `base.py` with problem-specific rules:

Architecture
------------
- Each rule is a standalone function decorated with ``@problem_rule``, which
  automatically registers it into ``PROBLEM_REGISTRY``.
- ``ProblemValidator`` aggregates the registry and applies rules via
  ``validate_component(target, context)``.
- ``ProblemSemantics`` provides fast lookup for type hierarchies, predicate
  signatures, and constants to support type-checking across rules.

Validation Coverage
-------------------
- PDDL naming conventions for objects (no ``?`` prefix, no reserved keywords)
- Object type inheritance (declared types must exist in the domain)
- Initial state validation (objects exist, predicate arity matches, argument
  types are compatible, undeclared predicate symbols are flagged)
- Timed Initial Literal validation (same checks applied to timed facts)
- Goal state validation (same structural checks as initial state)
- Metric expression validation (declared functions, no variables, valid syntax)
- Duplicate object detection (no two objects with the same name)
- Unused object detection (warns if declared but never referenced in init/goal)
- Trivial goal detection (warns if goal has no conditions)

Usage
-----
    validator = ProblemValidator()
    result = validator.validate_component(
        initial_state,
        {PDDLType: [...], PDDLObject: [...], InitialState: [initial_state]}
    )
    if not result.valid:
        print(result.errors)

Custom rules can be added by decorating a function or passing a ``FunctionalRule``
instance to the constructor.
"""

import re
from typing import Any, Dict, List, Optional, Type, Callable, Set
from pydantic import BaseModel

from l2p.validators.base import (
    ValidationResult,
    SyntaxValidator,
    FunctionalRule,
    ValidationRule,
    _extract_symbols,
    get_ordinal,
    PDDL_KEYWORDS,
)
from l2p.utils.pddl_types import *

# ---------------------------------------------------------------------------
# GLOBAL REGISTRY & DECORATOR
# ---------------------------------------------------------------------------

PROBLEM_REGISTRY: List[FunctionalRule] = []  # master list of all default problem rules


def problem_rule(targets: List[Type[BaseModel]]):
    """
    Decorator that automatically registers a validation function
    into the PROBLEM_REGISTRY master list.
    """

    def decorator(func: Callable[[BaseModel, Dict[str, Any]], ValidationResult]):
        rule = FunctionalRule(name=func.__name__, targets=targets, func=func)
        PROBLEM_REGISTRY.append(rule)
        return func

    return decorator


class ProblemSemantics:
    """Helper class to rapidly look up inheritance, constants, and predicate signatures."""

    def __init__(self, context: Dict[str, Any]):
        self.signatures: Dict[str, List[str]] = {}
        self.constants: Dict[str, str] = {}
        self.type_parents: Dict[str, str] = {}
        self.has_domain_context: bool = False

        for cls, items in context.items():
            name = cls.__name__
            if name == "PDDLType":
                self.has_domain_context = True
                for t in items:
                    self.type_parents[t.name.lower()] = (
                        getattr(t, "parent", None) or "object"
                    ).lower()
            elif name == "Constant":
                for c in items:
                    self.constants[c.name.lower()] = (
                        getattr(c, "type", None) or "object"
                    ).lower()
            elif name in ["Predicate", "Function", "DerivedPredicate"]:
                self.has_domain_context = True
                for p in items:
                    # Map predicate name -> list of expected parameter types
                    self.signatures[p.name.lower()] = [
                        (getattr(param, "type", None) or "object").lower()
                        for param in getattr(p, "params", [])
                    ]

    def is_subtype(self, child: str, parent: str) -> bool:
        """Checks if 'child' is equal to or inherits from 'parent'."""
        child, parent = child.lower(), parent.lower()
        if parent == "object" or child == parent:
            return True

        visited = set()
        while child != "object" and child not in visited:
            visited.add(child)
            child = self.type_parents.get(child, "object")
            if child == parent:
                return True
        return False


def _check_problem_state_types(
    item: Any,
    allowed_objects: Dict[str, str],
    sem: ProblemSemantics,
    result: ValidationResult,
    location_desc: str,
):
    """
    Recursively checks object existence and semantic type matching for Problem States.
    Problem states look for concrete objects (e.g. rover1) rather than variables.
    """
    if isinstance(item, BaseModel):
        item = item.model_dump()

    if isinstance(item, str):
        # 1. Reject any variables (starting with ?)
        used_vars = set(re.findall(r"\?[a-zA-Z0-9_\-]+", item))
        if used_vars:
            result.add_error(
                f"[ERROR] {location_desc} '{item}' uses variable(s): {used_vars}. "
                f"Problem states cannot contain variables; they must use concrete objects or constants."
            )

        # 2. Semantic Signature & Type Checking
        for match in re.finditer(
            r"\(\s*([a-zA-Z][a-zA-Z0-9_\-]*)\s*([^\(\)]*)\)", item
        ):
            sym = match.group(1).lower()
            args_str = match.group(2).strip()
            args = args_str.split() if args_str else []

            if sym in sem.signatures:
                exp_types = sem.signatures[sym]

                # Check Arity
                if len(args) != len(exp_types):
                    result.add_error(
                        f"[ERROR] {location_desc}: '{sym}' expects {len(exp_types)} arguments, "
                        f"but got {len(args)} in '({sym} {args_str})'."
                    )
                    continue

                # Check Object Types & Constants
                for i, (arg, exp_type) in enumerate(zip(args, exp_types), 1):
                    arg_lower = arg.lower()
                    arg_type = None

                    if re.match(r"^-?\d+(\.\d+)?$", arg_lower):
                        arg_type = "number"
                    else:
                        # Check if arg is a declared problem Object OR a domain Constant
                        arg_type = allowed_objects.get(arg_lower) or sem.constants.get(
                            arg_lower
                        )

                        if not arg_type:
                            result.add_error(
                                f"[ERROR] {location_desc}: '{arg}' in '{item}' is not a declared object or constant."
                            )
                            continue

                    # Does the provided object match the expected predicate type?
                    if arg_type and not sem.is_subtype(arg_type, exp_type):
                        pos = get_ordinal(i)
                        result.add_error(
                            f"[ERROR] {location_desc}: The {pos} argument '{arg}' in '{item}' is of type "
                            f"'{arg_type}', which is not compatible with the expected type '{exp_type}'."
                        )
            elif sem.has_domain_context and sym not in PDDL_KEYWORDS:
                result.add_error(
                    f"[ERROR] {location_desc}: '{sym}' in '{item}' is neither a declared domain predicate/function "
                    f"nor a built-in PDDL keyword."
                )

    elif isinstance(item, dict):
        for val in item.values():
            _check_problem_state_types(val, allowed_objects, sem, result, location_desc)

    elif isinstance(item, list):
        for sub_item in item:
            _check_problem_state_types(
                sub_item, allowed_objects, sem, result, location_desc
            )


# ---------------------------------------------------------------------------
# STANDALONE VALIDATION FUNCTIONS
# ---------------------------------------------------------------------------


@problem_rule(targets=[PDDLObject])
def validate_pddl_naming(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Ensures names for follow strict PDDL syntax."""
    result = ValidationResult()
    name = getattr(target, "name", "")

    if not name:
        return result

    model_type = target.__class__.__name__

    # 1. character violations
    if name.startswith("?"):
        result.add_error(f"[ERROR] {model_type} name '{name}' cannot start with '?'.")
        return result
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_\-]*$", name):
        result.add_error(
            f"[ERROR] {model_type} name '{name}' contains invalid characters. "
            "It must start with a letter and contain only letters, numbers, hyphens, or underscores."
        )

    # 2. reserved PDDL keyword violations
    if name.lower() in PDDL_KEYWORDS:
        result.add_error(
            f"[ERROR] {model_type} name '{name}' is a reserved PDDL keyword and cannot be used."
        )

    # 3. warning check for uppercase sensitivity
    if not name.islower():
        result.add_warning(
            f"[WARNING] {model_type} '{name}' contains uppercase letters. "
            "PDDL is case-insensitive, so strictly lowercase names are recommended."
        )

    # 4. check duplicates
    duplicate_found = False
    conflict_type = None

    for _, items_list in context.items():
        for item in items_list:
            if item is not target:
                item_name = getattr(item, "name", "")

                if item_name and item_name.lower() == name.lower():
                    duplicate_found = True
                    conflict_type = item.__class__.__name__
                    break

        if duplicate_found:
            break

    if duplicate_found:
        result.add_error(
            f"[ERROR] {model_type} name '{name}' is already in use by a {conflict_type}. "
            "PDDL requires names to be unique across the domain to avoid ambiguity."
        )

    return result


@problem_rule(targets=[PDDLObject])
def check_obj_type_inheritance(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Checks if a type's declared parent actually exists in the domain."""
    result = ValidationResult()

    target_name = getattr(target, "name", None)
    target_type = getattr(target, "type", None) or "object"

    if not target_name or target_type == "object":
        return result

    all_types = context.get(PDDLType, [])
    declared_names = {t.name for t in all_types}

    if target_type not in declared_names:
        result.add_error(
            f"[ERROR] Object '{target_name}' declares type '{target_type}', "
            f"but '{target_type}' is not declared in existing types."
            f"\nList of existing type(s) are: [{', '.join(sorted(declared_names))}]."
        )

    return result


@problem_rule(targets=[InitialState])
def check_initial_state(target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
    """Validates that initial state facts use declared objects and respect predicate signatures."""
    result = ValidationResult()
    sem = ProblemSemantics(context)

    # Map all declared objects in the problem: name -> type
    allowed_objects = {}
    for obj in context.get(PDDLObject, []):
        allowed_objects[obj.name.lower()] = (
            getattr(obj, "type", None) or "object"
        ).lower()

    # 1. Check standard facts
    _check_problem_state_types(
        getattr(target, "facts", []),
        allowed_objects,
        sem,
        result,
        "Initial State facts",
    )

    # 2. Check Timed Initial Literals (timed_facts)
    for t_fact in getattr(target, "timed_facts", []):
        time = getattr(t_fact, "time", 0.0)
        _check_problem_state_types(
            getattr(t_fact, "fact", ""),
            allowed_objects,
            sem,
            result,
            f"Initial State timed_fact at t={time}",
        )

    return result


@problem_rule(targets=[GoalState])
def check_goal_state(target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
    """Validates that goal conditions use declared objects and respect predicate signatures."""
    result = ValidationResult()
    sem = ProblemSemantics(context)

    allowed_objects = {}
    for obj in context.get(PDDLObject, []):
        allowed_objects[obj.name.lower()] = (
            getattr(obj, "type", None) or "object"
        ).lower()

    # Goal states can contain complex logic (and, or, exists) just like actions
    _check_problem_state_types(
        getattr(target, "conditions", []),
        allowed_objects,
        sem,
        result,
        "Goal State conditions",
    )

    return result


@problem_rule(targets=[Metric])
def check_metric_syntax(target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
    """Ensures plan optimization metrics use declared functions and valid syntax."""
    result = ValidationResult()
    sem = ProblemSemantics(context)

    expr = getattr(target, "expression", "")
    if not expr:
        return result

    # 1. Check if the metric expression uses variables (which are invalid in metrics)
    used_vars = set(re.findall(r"\?[a-zA-Z0-9_\-]+", expr))
    if used_vars:
        result.add_error(
            f"[ERROR] Metric expression '{expr}' contains invalid variables {used_vars}."
        )

    # 2. Check that the functions used in the metric actually exist in the domain
    symbols = _extract_symbols(
        expr
    )  # Re-use the basic symbol extractor from the domain side!

    allowed_symbols = set(PDDL_KEYWORDS)
    allowed_symbols.update(f.name.lower() for f in context.get(Function, []))

    # 'total-time' is a special built-in PDDL keyword for makespan optimization
    allowed_symbols.add("total-time")

    for sym in symbols:
        if sym.lower() not in allowed_symbols:
            result.add_error(
                f"[ERROR] Metric expression uses undeclared function '{sym}'. "
                "Only declared numeric functions or 'total-time' can be used in metrics."
            )

    return result


# ---------------------------------------------------------------------------
# ADDITIONAL PROBLEM RULES
# ---------------------------------------------------------------------------


@problem_rule(targets=[PDDLObject])
def check_duplicate_objects(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Check that no two objects share the same name."""
    result = ValidationResult()
    target_name = getattr(target, "name", None)
    if not target_name:
        return result

    for item in context.get(PDDLObject, []):
        if item is target:
            continue
        other_name = getattr(item, "name", None)
        if other_name and other_name.lower() == target_name.lower():
            result.add_error(
                f"[ERROR] Duplicate object name '{target_name}'."
            )
            return result
    return result


@problem_rule(targets=[PDDLObject])
def check_unused_objects(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Warn if an object is declared but never referenced in init or goal."""
    result = ValidationResult()
    target_name = getattr(target, "name", None)
    if not target_name:
        return result
    target_lower = target_name.lower()

    # Collect all strings referenced in initial state and goal state
    referenced: Set[str] = set()
    _word_re = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]*")

    for init in context.get(InitialState, []):
        for fact in getattr(init, "facts", []):
            if isinstance(fact, str):
                referenced.update(
                    m.group(0).lower()
                    for m in _word_re.finditer(fact)
                )
        for tf in getattr(init, "timed_facts", []):
            f = getattr(tf, "fact", "")
            if isinstance(f, str):
                referenced.update(
                    m.group(0).lower()
                    for m in _word_re.finditer(f)
                )

    for goal in context.get(GoalState, []):
        for cond in getattr(goal, "conditions", []):
            if isinstance(cond, str):
                referenced.update(
                    m.group(0).lower()
                    for m in _word_re.finditer(cond)
                )

    if target_lower not in referenced:
        result.add_warning(
            f"[WARNING] Object '{target_name}' is declared but never referenced "
            f"in the initial state or goal."
        )
    return result


@problem_rule(targets=[GoalState])
def check_goal_not_trivially_satisfied(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Warn if the goal has no conditions (trivially satisfied)."""
    result = ValidationResult()
    conditions = getattr(target, "conditions", [])
    if not conditions:
        result.add_warning(
            "[WARNING] Goal state has no conditions — the problem is trivially satisfied."
        )
    return result


# ---------------------------------------------------------------------------
# ORCHESTRATOR FOR THE LLM PIPELINE
# ---------------------------------------------------------------------------


class ProblemValidator(SyntaxValidator):
    """Validator specifically for PDDL Problem components."""

    def __init__(
        self, use_defaults: bool = True, custom_rules: List[ValidationRule] = None
    ):
        """
        Args:
            use_defaults: If True, automatically loads all @problem_rule decorated functions.
            custom_rules: An optional list of user-defined ValidationRules to add.
        """
        super().__init__()

        if use_defaults:
            self.rules.extend(PROBLEM_REGISTRY)

        if custom_rules:
            for rule in custom_rules:
                self.register_rule(rule)

    def validate_problem(
        self,
        problem: ProblemDetails,
        domain: Optional[DomainDetails] = None,
    ) -> ValidationResult:
        """
        Validate every component in a ProblemDetails with full cross-component context.

        Iterates all component fields (objects, initial_state, goal_state, constraint,
        metric) and aggregates errors and warnings into a single result.

        When a domain is provided, additionally validates that:
          - Object types are declared in the domain
          - Predicate/function calls in init and goal match declared signatures
          - Metric expressions use declared functions

        Args:
            problem: A fully populated ProblemDetails model.
            domain: Optional DomainDetails for cross-component domain checks.

        Returns:
            ValidationResult with all errors and warnings collected across components.
        """
        context = {PDDLObject: problem.objects}
        if problem.initial_state:
            context[InitialState] = [problem.initial_state]
        if problem.goal_state:
            context[GoalState] = [problem.goal_state]
        if domain:
            context[PDDLType] = domain.types
            context[Predicate] = domain.predicates
            context[Function] = domain.functions
            
        result = ValidationResult()

        for item in problem.objects:
            r = self.validate_component(item, context)
            if not r.valid:
                for e in r.errors:
                    result.add_error(e)
            for w in r.warnings:
                result.add_warning(w)

        if problem.initial_state:
            r = self.validate_component(problem.initial_state, context)
            if not r.valid:
                for e in r.errors:
                    result.add_error(e)
            for w in r.warnings:
                result.add_warning(w)

        if problem.goal_state:
            r = self.validate_component(problem.goal_state, context)
            if not r.valid:
                for e in r.errors:
                    result.add_error(e)
            for w in r.warnings:
                result.add_warning(w)

        for item in problem.constraint:
            r = self.validate_component(item, context)
            if not r.valid:
                for e in r.errors:
                    result.add_error(e)
            for w in r.warnings:
                result.add_warning(w)

        if problem.metric:
            r = self.validate_component(problem.metric, context)
            if not r.valid:
                for e in r.errors:
                    result.add_error(e)
            for w in r.warnings:
                result.add_warning(w)

        return result
