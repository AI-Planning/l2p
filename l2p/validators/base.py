"""
Base Orchestrator and Rule Engine for L2P Component Validation.
This module for validates generated PDDL components generated.

Classes:
    ValidationResult: Stores the outcome of a rule check (errors and warnings).
    ValidationRule: Abstract base class defining the contract for all rules.
    FunctionalRule: Wrapper that converts standalone functions into ValidationRules.
    SyntaxValidator: The core orchestrator that manages and executes rules.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Callable, Set
from pydantic import BaseModel

PDDL_KEYWORDS = {
    # ---------------------------------------------------------
    # Core Logic & Math (PDDL 1.2 / 2.1)
    # ---------------------------------------------------------
    "and",
    "or",
    "not",
    "imply",
    "exists",
    "forall",
    "when",
    ">",
    "<",
    "=",
    ">=",
    "<=",
    "+",
    "-",
    "*",
    "/",
    "assign",
    "scale-up",
    "scale-down",
    "increase",
    "decrease",
    # ---------------------------------------------------------
    # Core Domain & Problem Structure
    # ---------------------------------------------------------
    "define",
    "domain",
    "problem",
    "requirements",
    "types",
    "constants",
    "predicates",
    "functions",
    "objects",
    "init",
    "goal",
    "action",
    "parameters",
    "precondition",
    "effect",
    # ---------------------------------------------------------
    # Built-in Types
    # ---------------------------------------------------------
    "object",
    "number",
    # ---------------------------------------------------------
    # Temporal Planning (PDDL 2.1 / PDDL+)
    # ---------------------------------------------------------
    "durative-action",
    "duration",
    "condition",
    "at start",
    "at end",
    "over all",
    "continuous",
    "event",
    "process",
    "derived",
    # ---------------------------------------------------------
    # Optimization Metrics (PDDL 2.1+)
    # ---------------------------------------------------------
    "metric",
    "minimize",
    "maximize",
    "total-time",
    "is-violated",
    "length",
    "serial-length",
    "parallel-length",
    # ---------------------------------------------------------
    # Trajectory Constraints & Preferences (PDDL 3.0)
    # ---------------------------------------------------------
    "preference",
    "always",
    "sometime",
    "within",
    "at-most-once",
    "sometime-after",
    "sometime-before",
    "always-within",
    "hold-during",
    "hold-after",
}


class ValidationResult:
    """Stores the outcome of a validation run."""

    def __init__(self):
        self.valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, msg: str):
        """Adds a fatal error that fails validation."""
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        """Adds a non-fatal warning for the user."""
        self.warnings.append(msg)


# type alias for a standalone validation function
ValidationFunc = Callable[[BaseModel, Dict[str, Any]], ValidationResult]


class ValidationRule(ABC):
    """Abstract base class for all L2P validation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the rule (for debugging purposes)."""
        pass

    @property
    @abstractmethod
    def target_models(self) -> List[Type[BaseModel]]:
        """List of Pydantic model classes this rule applies to."""
        pass

    @abstractmethod
    def validate(self, target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
        """Runs the validation logic on the target model."""
        pass


class FunctionalRule(ValidationRule):
    """Wraps a standalone Python function into a formal ValidationRule."""

    def __init__(self, name: str, targets: List[Type[BaseModel]], func: ValidationFunc):
        self._name = name
        self._targets = targets
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    @property
    def target_models(self) -> List[Type[BaseModel]]:
        return self._targets

    def validate(self, target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
        return self._func(target, context)


class SyntaxValidator(ABC):
    """Base orchestrator for running validation rules in the LLM pipeline."""

    def __init__(self):
        self.rules: List[ValidationRule] = []

    def register_rule(self, rule: ValidationRule):
        """Adds a new rule to the validation pipeline."""
        self.rules.append(rule)

    def validate_component(
        self, target: BaseModel, context: Dict[str, Any]
    ) -> ValidationResult:
        """Runs all applicable rules against the target component based on its type."""
        final_result = ValidationResult()

        for rule in self.rules:
            if type(target) in rule.target_models:
                result = rule.validate(target, context)

                if not result.valid:
                    final_result.valid = False
                    final_result.errors.extend(result.errors)

                final_result.warnings.extend(result.warnings)

        return final_result


def _extract_symbols(item: Any) -> Set[str]:
    """Recursively extracts PDDL predicate and function names from LogicalConditions."""
    symbols = set()

    if isinstance(item, BaseModel):
        item = item.model_dump()

    if isinstance(item, str):
        # matches any word immediately following an open parenthesis e.g. "(at ?r ?l)" -> "at"
        matches = re.findall(r"\(\s*([a-zA-Z][a-zA-Z0-9_\-]*)", item)
        symbols.update(matches)
    elif isinstance(item, dict):
        for val in item.values():
            symbols.update(_extract_symbols(val))
    elif isinstance(item, list):
        for sub_item in item:
            symbols.update(_extract_symbols(sub_item))

    return symbols


def _verify_symbols(
    symbols: Set[str], context: Dict[str, Any], location_desc: str
) -> ValidationResult:
    """Compares extracted symbols against the domain context to flag undeclared usage."""
    result = ValidationResult()
    allowed_symbols = set(PDDL_KEYWORDS)

    class_names = []

    for cls in context:
        if cls.__name__ in ["Predicate", "Function", "DerivedPredicate"]:
            class_names.extend([c.name.lower() for c in context[cls]])
            allowed_symbols.update(class_names)

    for sym in symbols:
        if sym.lower() not in allowed_symbols:
            result.add_error(
                f"[ERROR] {location_desc} uses undeclared predicate/function keyword '{sym}'. "
                f"Please ensure '{sym}' is properly defined in the domain."
                f"\nAllowed predicates/functions: [{', '.join(sorted(class_names))}]"
            )

    return result


def get_ordinal(n: int) -> str:
    """Converts a 1-based integer to its ordinal string representation (1st, 2nd...)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
