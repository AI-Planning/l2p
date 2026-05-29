"""
PDDL Domain Validation Rules

This module provides a comprehensive rule-based validation system for PDDL domain
components. It extends the infrastructure in `base.py` with domain-specific rules:

Architecture
------------
- Each rule is a standalone function decorated with ``@domain_rule``, which
  automatically registers it into ``DOMAIN_REGISTRY``.
- ``DomainValidator`` aggregates the registry and applies rules via
  ``validate_component(target, context)``.
- ``DomainSemantics`` provides fast lookup for type hierarchies, predicate/function
  signatures, and constants to support type-checking across rules.

Validation Coverage
-------------------
- PDDL naming conventions (no ``?`` prefix, no reserved keywords, no duplicates)
- Type inheritance (parent types must exist)
- Type cycle detection (no circular ``:types`` hierarchies)
- Constant type validity (referenced types must be declared)
- Parameter type validity (parameter types must exist)
- Predicate/function symbol validation in preconditions, effects, conditions,
  durative conditions, durative effects, and constraints
- Variable scoping (forall/exists introduce local scope)
- Argument arity and type compatibility
- Action/event/process name uniqueness (no duplicates across all action types)
- Parameter name uniqueness (no duplicate ``?variable`` names within a component)
- Non-empty action effects (warns if an action has no add/delete/numeric/conditional effects)
- Orphaned predicate/function detection (warns if declared but never used in any action)
- Requirements consistency (e.g. ``:durative-actions`` implies ``:time``)
- Domain name format validation

Usage
-----
    validator = DomainValidator()
    result = validator.validate_component(
        action,
        {PDDLType: [...], Predicate: [...], Action: [action]}
    )
    if not result.valid:
        print(result.errors)

Custom rules can be added by decorating a function or passing a ``FunctionalRule``
instance to the constructor.
"""

import re
from typing import Any, Dict, List, Type, Callable, Set
from pydantic import BaseModel

from l2p.validators.base import (
    ValidationResult,
    SyntaxValidator,
    FunctionalRule,
    ValidationRule,
    _extract_symbols,
    _verify_symbols,
    get_ordinal,
    PDDL_KEYWORDS,
)
from l2p.utils.pddl_types import *

# ---------------------------------------------------------------------------
# GLOBAL REGISTRY & DECORATOR
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: List[FunctionalRule] = []  # master list of all default domain rules


def domain_rule(targets: List[Type[BaseModel]]):
    """
    Decorator that automatically registers a validation function
    into the DOMAIN_REGISTRY master list.
    """

    def decorator(func: Callable[[BaseModel, Dict[str, Any]], ValidationResult]):
        rule = FunctionalRule(name=func.__name__, targets=targets, func=func)
        DOMAIN_REGISTRY.append(rule)
        return func

    return decorator


class DomainSemantics:
    """Helper class to rapidly look up inheritance, constants, and predicate signatures."""

    def __init__(self, context: Dict[str, Any]):
        self.signatures: Dict[str, List[str]] = {}
        self.constants: Dict[str, str] = {}
        self.type_parents: Dict[str, str] = {}

        for cls, items in context.items():
            name = cls.__name__
            if name == "PDDLType":
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


def _check_variables(
    item: Any, allowed_vars: Set[str], result: ValidationResult, location_desc: str
):
    """Recursively checks that all variables starting with '?' are in the allowed_vars set."""
    if isinstance(item, BaseModel):
        item = item.model_dump()

    if isinstance(item, str):
        used_vars = set(re.findall(r"\?[a-zA-Z0-9_\-]+", item))
        for var in used_vars:
            if var not in allowed_vars:
                result.add_error(
                    f"[ERROR] {location_desc} uses undeclared variable '{var}'. "
                    f"It must be declared in the component's parameters."
                    f"\nAllowed variables: [{', '.join(sorted(allowed_vars))}]"
                )

    elif isinstance(item, dict):
        # create a local copy of allowed variables in case we enter a new scope
        local_vars = set(allowed_vars)

        # PDDL 'forall' and 'exists' introduce new local parameters
        is_quantifier = item.get("quantifier") in ["forall", "exists"]
        if is_quantifier and "parameters" in item:
            for p in item["parameters"]:
                if isinstance(p, dict) and "variable" in p:
                    local_vars.add(p["variable"])
                elif hasattr(p, "variable"):
                    local_vars.add(p.variable)

        # recurse deeper into the dictionary using the updated scope
        for val in item.values():
            _check_variables(val, local_vars, result, location_desc)

    elif isinstance(item, list):
        for sub_item in item:
            _check_variables(sub_item, allowed_vars, result, location_desc)


# ---------------------------------------------------------------------------
# STANDALONE VALIDATION FUNCTIONS
# ---------------------------------------------------------------------------


@domain_rule(
    targets=[
        PDDLType,
        Constant,
        Predicate,
        Function,
        DerivedPredicate,
        Action,
        DurativeAction,
        Event,
        Process,
    ]
)
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


@domain_rule(targets=[PDDLType])
def check_type_inheritance(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Checks if a type's declared parent actually exists in the domain."""
    result = ValidationResult()

    target_name = getattr(target, "name", None)
    target_parent = getattr(target, "parent", None) or "object"

    if not target_name or target_parent == "object":
        return result

    all_types = context.get(PDDLType, [])
    declared_names = {t.name for t in all_types}

    if target_parent not in declared_names:
        result.add_error(
            f"[ERROR] Type '{target_name}' declares parent '{target_parent}', "
            f"but '{target_parent}' is not declared in existing types."
            f"\nList of existing type(s) are: [{', '.join(sorted(declared_names))}]."
        )

    return result


@domain_rule(targets=[PDDLType])
def check_type_cycle(target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
    """Checks if the type forms a cyclic dependency in the hierarchy."""
    result = ValidationResult()

    target_name = getattr(target, "name", None)
    all_types = context.get(PDDLType, [])

    if not target_name or not all_types:
        return result

    type_map = {}
    for t in all_types:
        parent = getattr(t, "parent", None) or "object"
        type_map[t.name] = parent

    visited = set()
    current_type = target_name

    while current_type != "object":
        if current_type in visited:
            path = " -> ".join(list(visited) + [current_type])
            result.add_error(
                f"[ERROR] Invalid type hierarchy cycle detected involving '{target_name}'. "
                f"Inheritance path: {path}"
            )
            break

        visited.add(current_type)
        current_type = type_map.get(current_type, "object")

    return result


@domain_rule(targets=[Constant])
def check_constant_inheritance(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Checks if a constant's declared type actually exists in the domain."""
    result = ValidationResult()

    target_name = getattr(target, "name", None)
    target_type = getattr(target, "type", None)

    if not target_name or target_type == "object":
        return result

    all_types = context.get(PDDLType, [])
    declared_names = {t.name for t in all_types}

    if target_type not in declared_names:
        result.add_error(
            f"[ERROR] Constant '{target_name}' declares type '{target_type}', "
            f"but '{target_type}' is not declared in existing types."
            f"\nList of existing type(s) are: [{', '.join(sorted(declared_names))}]."
        )

    return result


@domain_rule(
    targets=[
        Predicate,
        Function,
        DerivedPredicate,
        Action,
        DurativeAction,
        Event,
        Process,
    ]
)
def check_parameter_types(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Checks if parameters use existing types or constants."""
    result = ValidationResult()

    allowed_types = {"object"}
    if PDDLType in context:
        allowed_types.update(t.name for t in context[PDDLType])
    if Constant in context:
        allowed_types.update(c.type for c in context[Constant])

    for idx, param in enumerate(getattr(target, "params", []), start=1):
        param_var = getattr(param, "variable", None)
        param_type = getattr(param, "type", None)
        pos = get_ordinal(idx)

        if param_var and not param_var.startswith("?"):
            result.add_error(
                f"[ERROR] {target.__class__.__name__} '{target.name}' has invalid {pos} parameter "
                f"'{param.variable}' that is missing '?' prefix (e.g., ?block). "
            )

        if param_type and param_type not in allowed_types:
            result.add_error(
                f"[ERROR] {target.__class__.__name__} '{target.name}' has invalid {pos} parameter "
                f"'{param.variable}' that uses undeclared type '{param_type}'. "
                f"\nList of existing type(s) are: [{', '.join(sorted(allowed_types))}]."
            )

    return result


@domain_rule(targets=[DerivedPredicate])
def check_derived_predicate(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Checks if the derived predicate's condition uses valid symbols."""
    name = getattr(target, "name", "Unknown")
    symbols = _extract_symbols(getattr(target, "condition", {}))
    return _verify_symbols(symbols, context, f"DerivedPredicate '{name}' condition")


@domain_rule(targets=[Action, ActionPrecondition])
def check_action_precondition(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Checks if the action preconditions use valid symbols."""
    name = getattr(target, "name", "ActionPrecondition")

    # handle if the target is the full Action or just the Precondition block
    preconds = getattr(target, "preconditions", target)
    conditions = getattr(preconds, "conditions", [])

    symbols = _extract_symbols(conditions)
    return _verify_symbols(symbols, context, f"Action '{name}' preconditions")


@domain_rule(targets=[Action, ActionEffect])
def check_action_effect(target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
    """Checks if the action effects use valid symbols."""
    name = getattr(target, "name", "ActionEffect")

    effects = getattr(target, "effects", target)
    # _extract_symbols safely traverses the 'add', 'delete', 'numeric', and 'conditional' dicts
    symbols = _extract_symbols(effects)

    return _verify_symbols(symbols, context, f"Action '{name}' effects")


@domain_rule(targets=[DurativeAction, DurativeActionConditions])
def check_dur_action_conditions(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Checks if the durative action conditions use valid symbols."""
    name = getattr(target, "name", "DurativeActionConditions")

    conditions = getattr(target, "conditions", target)
    symbols = _extract_symbols(conditions)

    return _verify_symbols(symbols, context, f"DurativeAction '{name}' conditions")


@domain_rule(targets=[DurativeAction, DurativeActionEffect])
def check_dur_action_effect(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Checks if the durative action effects use valid symbols."""
    name = getattr(target, "name", "DurativeActionEffect")

    effects = getattr(target, "effects", target)
    symbols = _extract_symbols(effects)

    return _verify_symbols(symbols, context, f"DurativeAction '{name}' effects")


@domain_rule(targets=[Constraint])
def check_constraint(target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
    """Checks if the global constraints use valid symbols."""
    desc = getattr(target, "desc", "Global Constraint")
    symbols = _extract_symbols(getattr(target, "condition", {}))

    return _verify_symbols(symbols, context, f"Constraint '{desc}'")


@domain_rule(targets=[Action, DurativeAction, Event, Process, DerivedPredicate])
def check_component_variables(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Ensures all variables are declared AND semantically match predicate signatures."""
    result = ValidationResult()
    name = getattr(target, "name", "Unknown")
    model_type = target.__class__.__name__

    # Initialize the rapid lookup semantics engine
    sem = DomainSemantics(context)

    # Gather the base parameters declared globally for this component (var -> type)
    base_vars = {}
    for param in getattr(target, "params", []):
        var_name = getattr(param, "variable", "")
        var_type = getattr(param, "type", "object")
        if var_name:
            base_vars[var_name.lower()] = var_type.lower()

    # Check Preconditions / Conditions
    if hasattr(target, "preconditions"):
        _check_variables_and_types(
            getattr(target, "preconditions"),
            base_vars,
            sem,
            result,
            f"{model_type} '{name}' preconditions",
        )
    elif hasattr(target, "conditions"):  # DurativeActions
        _check_variables_and_types(
            getattr(target, "conditions"),
            base_vars,
            sem,
            result,
            f"{model_type} '{name}' conditions",
        )
    elif hasattr(target, "condition"):  # DerivedPredicates
        _check_variables_and_types(
            getattr(target, "condition"),
            base_vars,
            sem,
            result,
            f"{model_type} '{name}' condition",
        )

    # Check Effects
    if hasattr(target, "effects"):
        _check_variables_and_types(
            getattr(target, "effects"),
            base_vars,
            sem,
            result,
            f"{model_type} '{name}' effects",
        )

    # Check Duration
    if hasattr(target, "duration"):
        dur_vars = dict(base_vars)
        dur_vars["?duration"] = "number"
        _check_variables_and_types(
            getattr(target, "duration"),
            dur_vars,
            sem,
            result,
            f"{model_type} '{name}' duration",
        )

    return result


def _check_variables_and_types(
    item: Any,
    var_types: Dict[str, str],
    sem: DomainSemantics,
    result: ValidationResult,
    location_desc: str,
):
    """Recursively checks variable existence AND semantic type matching."""
    if isinstance(item, BaseModel):
        item = item.model_dump()

    if isinstance(item, str):
        # 1. Basic Variable Existence
        used_vars = set(re.findall(r"\?[a-zA-Z0-9_\-]+", item))
        for var in used_vars:
            if var.lower() not in var_types:
                result.add_error(
                    f"[ERROR] {location_desc} uses undeclared variable '{var}'. "
                    f"It must be declared in the component's parameters."
                    f"\nAllowed variables: [{', '.join(sorted(var_types))}]"
                )
                continue  # Skip type checking for this broken variable

        # 2. Semantic Signature & Type Checking
        # Regex finds inner (predicate ?arg1 ?arg2) calls
        for match in re.finditer(
            r"\(\s*([a-zA-Z][a-zA-Z0-9_\-]*)\s*([^\(\)]*)\)", item
        ):
            sym = match.group(1).lower()
            args_str = match.group(2).strip()
            args = args_str.split() if args_str else []

            # If the predicate exists in our domain context, check its arguments!
            if sym in sem.signatures:
                exp_types = sem.signatures[sym]

                # Check Arity (Argument Count)
                if len(args) != len(exp_types):
                    result.add_error(
                        f"[ERROR] {location_desc}: '{sym}' expects {len(exp_types)} arguments, "
                        f"but got {len(args)} in '({sym} {args_str})'."
                    )
                    continue

                # Check Types (Order and Inheritance)
                for i, (arg, exp_type) in enumerate(zip(args, exp_types), 1):
                    arg_lower = arg.lower()
                    arg_type = None

                    if arg_lower.startswith("?"):
                        arg_type = var_types.get(arg_lower)
                    elif re.match(r"^-?\d+(\.\d+)?$", arg_lower):
                        arg_type = "number"
                    else:
                        arg_type = sem.constants.get(arg_lower)
                        if not arg_type:
                            result.add_error(
                                f"[ERROR] {location_desc}: '{arg}' is not a recognized variable, number, or constant."
                            )
                            continue

                    if arg_type and not sem.is_subtype(arg_type, exp_type):
                        pos = get_ordinal(i)
                        result.add_error(
                            f"[ERROR] {location_desc} '({sym} {args_str})': The {pos} argument '{arg}' in '{sym}' is of type "
                            f"'{arg_type}', which is not compatible with the expected type '{exp_type}'."
                        )

    elif isinstance(item, dict):
        local_vars = dict(var_types)

        # Scope resolution for forall/exists (supports both "operator" and "quantifier" keys)
        is_quantifier = item.get("operator") in ["forall", "exists"] or item.get(
            "quantifier"
        ) in ["forall", "exists"]
        if is_quantifier and "parameters" in item:
            for p in item["parameters"]:
                v = (
                    p.get("variable")
                    if isinstance(p, dict)
                    else getattr(p, "variable", None)
                )
                t = (
                    p.get("type")
                    if isinstance(p, dict)
                    else getattr(p, "type", "object")
                )
                if v:
                    local_vars[v.lower()] = t.lower()

        for val in item.values():
            _check_variables_and_types(val, local_vars, sem, result, location_desc)

    elif isinstance(item, list):
        for sub_item in item:
            _check_variables_and_types(sub_item, var_types, sem, result, location_desc)


# ---------------------------------------------------------------------------
# ADDITIONAL DOMAIN RULES
# ---------------------------------------------------------------------------


@domain_rule(targets=[Action, DurativeAction, Event, Process])
def check_action_name_uniqueness(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Check that no other action/event/process shares the same name."""
    result = ValidationResult()
    target_name = getattr(target, "name", None)
    if not target_name:
        return result

    for cls in [Action, DurativeAction, Event, Process]:
        for item in context.get(cls, []):
            if item is target:
                continue
            other_name = getattr(item, "name", None)
            if other_name and other_name.lower() == target_name.lower():
                result.add_error(
                    f"[ERROR] Duplicate action/event/process name '{target_name}'. "
                    f"Already used by {cls.__name__} '{other_name}'."
                )
                return result
    return result


@domain_rule(targets=[Action, DurativeAction, Event, Process])
def check_parameter_name_uniqueness(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Check that parameters within a component don't share the same ?variable name."""
    result = ValidationResult()
    params = getattr(target, "params", [])
    seen = set()
    for param in params:
        var = getattr(param, "variable", None)
        if var:
            if var.lower() in seen:
                result.add_error(
                    f"[ERROR] {target.__class__.__name__} '{getattr(target, 'name', '')}' "
                    f"has duplicate parameter '{var}'."
                )
            seen.add(var.lower())
    return result


@domain_rule(targets=[Action])
def check_action_has_effect(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Warn if an action declares no effects (does nothing)."""
    result = ValidationResult()
    effects = getattr(target, "effects", None)
    if effects is None:
        result.add_warning(f"[WARNING] Action '{target.name}' has no effects block.")
        return result
    if (
        not effects.add
        and not effects.delete
        and not effects.numeric
        and not effects.conditional
    ):
        result.add_warning(
            f"[WARNING] Action '{target.name}' has empty effects (add/delete/numeric/conditional all empty)."
        )
    return result


@domain_rule(targets=[Requirement])
def check_requirements_consistency(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Check that requirements are consistent.

    e.g. :durative-actions implies :time, :numeric-fluents needed for effects, etc.
    """
    result = ValidationResult()
    reqs = {r.name.lower() for r in context.get(Requirement, [])}
    target_req = getattr(target, "name", "").lower()

    implied = {
        ":durative-actions": ":time",
        ":continuous-effects": ":durative-actions",
        ":action-costs": ":numeric-fluents",
    }
    if target_req in implied and implied[target_req] not in reqs:
        result.add_warning(
            f"[WARNING] Requirement '{target_req}' is present but implied "
            f"requirement '{implied[target_req]}' is missing."
        )
    return result


@domain_rule(targets=[Predicate, Function, DerivedPredicate])
def check_orphaned_predicate(
    target: BaseModel, context: Dict[str, Any]
) -> ValidationResult:
    """Warn if a declared predicate/function is never used in any action/event/process."""
    from l2p.validators.base import _extract_symbols as _extract

    result = ValidationResult()
    target_name = getattr(target, "name", None)
    if not target_name:
        return result
    target_name_lower = target_name.lower()

    # Collect all symbols used across all actions/events/processes
    used_symbols: Set[str] = set()
    for cls in [Action, DurativeAction, Event, Process]:
        for item in context.get(cls, []):
            syms = _extract(item)
            used_symbols.update(s.lower() for s in syms)

    if target_name_lower not in used_symbols:
        result.add_warning(
            f"[WARNING] {target.__class__.__name__} '{target_name}' is declared "
            f"but never used in any action, event, or process."
        )
    return result


# ---------------------------------------------------------------------------
# ORCHESTRATOR FOR THE LLM PIPELINE
# ---------------------------------------------------------------------------


class DomainValidator(SyntaxValidator):
    """Validator specifically for PDDL Domain components."""

    def __init__(
        self, use_defaults: bool = True, custom_rules: List[ValidationRule] = None
    ):
        """
        Args:
            use_defaults: If True, automatically loads all @domain_rule decorated functions.
            custom_rules: An optional list of user-defined ValidationRules to add.
        """
        super().__init__()

        if use_defaults:
            self.rules.extend(DOMAIN_REGISTRY)

        if custom_rules:
            for rule in custom_rules:
                self.register_rule(rule)

    def validate_domain(self, domain: DomainDetails) -> ValidationResult:
        """
        Validate every component in a DomainDetails with full cross-component context.

        Iterates all 11 component fields (types, constants, predicates, functions,
        derived_predicates, actions, durative_actions, events, processes, constraint,
        requirements) and aggregates errors and warnings into a single result.

        Args:
            domain: A fully populated DomainDetails model.

        Returns:
            ValidationResult with all errors and warnings collected across components.
        """
        context = {
            PDDLType: domain.types,
            Predicate: domain.predicates,
            Function: domain.functions,
            Action: domain.actions,
            DurativeAction: domain.durative_actions,
            Event: domain.events,
            Process: domain.processes,
        }
        result = ValidationResult()

        # Domain name validation
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_\-]*$", domain.name):
            result.add_error(
                f"[ERROR] Domain name '{domain.name}' contains invalid characters. "
                "Must start with a letter and contain only letters, numbers, hyphens, or underscores."
            )

        fields = [
            ("types", domain.types),
            ("constants", domain.constants),
            ("predicates", domain.predicates),
            ("functions", domain.functions),
            ("derived_predicates", domain.derived_predicates),
            ("actions", domain.actions),
            ("durative_actions", domain.durative_actions),
            ("events", domain.events),
            ("processes", domain.processes),
            ("constraint", domain.constraint),
            ("requirements", domain.requirements),
        ]

        for name, items in fields:
            for item in items:
                r = self.validate_component(item, context)
                if not r.valid:
                    for e in r.errors:
                        result.add_error(f"[{name}] {e}")
                for w in r.warnings:
                    result.add_warning(f"[{name}] {w}")

        return result
