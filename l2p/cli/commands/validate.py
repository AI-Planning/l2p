"""Validate PDDL components against L2P semantic rules.

Two modes:

1. **Component validation** — validate a JSON snippet against the
   Pydantic model + L2P rules:

       l2p validate types       --data '[{"name":"block","parent":"object"}]'
       l2p validate predicates  --file preds.json

   Available component targets:
       types, constants, predicates, functions, derived-predicates,
       actions, durative-actions, events, processes, constraints,
       objects

2. **File validation** — parse a .pddl file and run all rules:

       l2p validate domain  domain.pddl
       l2p validate problem problem.pddl

   This reads the file, converts it to L2P's internal models via
   parse_domain_pddl() / parse_problem_pddl(), then runs every
   applicable DomainValidator or ProblemValidator rule.

3. **Full JSON validation** — validate a complete DomainDetails or
   ProblemDetails JSON (with cross-component checks):

       l2p validate domain  --data '{"name":"bw","types":[...],...}'
       l2p validate problem --data '{"name":"p","objects":[...],...}'

Validation rules applied:
  - Naming: letters/digits/hyphens only, no PDDL keywords
  - Type hierarchy: parents must exist, no cycles
  - Parameter types: every ?var has a declared type
  - Variable scope: all ?vars in preconditions/effects are declared
  - Symbol reference: no undeclared predicates/functions in conditions
  - Arity: correct number of arguments per predicate/function call
  - Uppercase warnings (PDDL is case-insensitive)

Returns:
  - SUCCESS / FAIL with error details
  - Exit code 1 on any error
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Any

from pydantic import BaseModel

from l2p.utils.pddl_types import *
from l2p.utils.pddl_parser import parse_domain_pddl, parse_problem_pddl
from l2p.validators.domain import DomainValidator
from l2p.validators.problem import ProblemValidator

# Mapping: component name → (Pydantic model class, validator class)
VALIDATABLE: Dict[str, Tuple[Any, Any]] = {
    "types": (PDDLType, DomainValidator),
    "constants": (Constant, DomainValidator),
    "predicates": (Predicate, DomainValidator),
    "functions": (Function, DomainValidator),
    "derived-predicates": (DerivedPredicate, DomainValidator),
    "actions": (Action, DomainValidator),
    "durative-actions": (DurativeAction, DomainValidator),
    "events": (Event, DomainValidator),
    "processes": (Process, DomainValidator),
    "constraints": (Constraint, DomainValidator),
    "objects": (PDDLObject, ProblemValidator),
}


def _resolve_path(args) -> Optional[Path]:
    if getattr(args, "path", None):
        p = Path(args.path)
        if p.exists():
            return p
        print(f"[ERROR] File not found: {args.path}", file=sys.stderr)
        sys.exit(1)
    return None


def _load_data(args) -> str:
    if args.data:
        return args.data
    if args.file:
        return Path(args.file).read_text()
    raise ValueError("No input provided. Use --data, --file, or <path>.")


def _build_context(items: List[BaseModel]) -> Dict[Type, List[Any]]:
    """Build a validation context from the validated items."""
    context: Dict[Type, List[Any]] = {}
    for item in items:
        cls = type(item)
        context.setdefault(cls, []).append(item)
    return context


def _validate_component_json(
    raw: str, model_cls: Type[BaseModel], validator_cls
) -> dict:
    data = json.loads(raw)

    if isinstance(data, list):
        items = [model_cls.model_validate(item) for item in data]
    else:
        items = [model_cls.model_validate(data)]

    validator = validator_cls()
    context = _build_context(items)
    errors = []
    warnings = []

    for item in items:
        result = validator.validate_component(item, context)
        if not result.valid:
            errors.extend(result.errors)
        warnings.extend(result.warnings)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "count": len(items),
    }


def _validate_full_domain(raw: str) -> dict:
    data = json.loads(raw)
    details = DomainDetails.model_validate(data)
    validator = DomainValidator()
    errors = []
    warnings = []

    fields = [
        ("types", details.types),
        ("constants", details.constants),
        ("predicates", details.predicates),
        ("functions", details.functions),
        ("derived_predicates", details.derived_predicates),
        ("actions", details.actions),
    ]
    context = {
        PDDLType: details.types,
        Predicate: details.predicates,
        Function: details.functions,
    }

    for name, items in fields:
        for item in items:
            result = validator.validate_component(item, context)
            if not result.valid:
                for e in result.errors:
                    errors.append(f"[{name}] {e}")
            warnings.extend(f"[{name}] {w}" for w in result.warnings)

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def _validate_full_problem(raw: str) -> dict:
    data = json.loads(raw)
    details = ProblemDetails.model_validate(data)
    validator = ProblemValidator()
    errors = []
    warnings = []

    context = {PDDLObject: details.objects}

    for item in details.objects:
        result = validator.validate_component(item, context)
        if not result.valid:
            errors.extend(result.errors)
        warnings.extend(result.warnings)

    if details.initial_state:
        result = validator.validate_component(details.initial_state, context)
        if not result.valid:
            errors.extend(result.errors)
        warnings.extend(result.warnings)

    if details.goal_state:
        result = validator.validate_component(details.goal_state, context)
        if not result.valid:
            errors.extend(result.errors)
        warnings.extend(result.warnings)

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "validate",
        help="Validate PDDL components against semantic rules",
        description="Check PDDL components or .pddl files against L2P's rule engine: naming, type hierarchy, parameter types, variable scope, symbol references, and arity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Validate a component JSON snippet
  l2p validate types --data '[{"name":"block","parent":"object"}]'
  l2p validate predicates --file preds.json

  # Validate a .pddl file (parses PDDL, then checks rules)
  l2p validate domain domain.pddl
  l2p validate problem problem.pddl

  # Validate a full domain/problem JSON
  l2p validate domain --data '{"name":"bw","types":[...],...}'
        """,
    )

    subparsers = parser.add_subparsers(
        dest="validate_command",
        title="validate targets",
        description="What to validate",
        metavar="TARGET",
        required=True,
    )

    # Individual components
    for name in VALIDATABLE:
        vp = subparsers.add_parser(name, help=f"Validate {name}")
        vp.add_argument(
            "--data", type=str, default=None, help="JSON string of the component."
        )
        vp.add_argument("--file", type=str, default=None, help="Path to JSON file.")
        vp.set_defaults(func=validate_component_command, component_name=name)

    # Full domain
    dom_parser = subparsers.add_parser("domain", help="Validate a full domain")
    dom_parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to .pddl domain file (alternative to --data/--file).",
    )
    dom_parser.add_argument(
        "--data", type=str, default=None, help="DomainDetails JSON string."
    )
    dom_parser.add_argument(
        "--file", type=str, default=None, help="Path to DomainDetails JSON file."
    )
    dom_parser.set_defaults(func=validate_domain_command)

    # Full problem
    prob_parser = subparsers.add_parser("problem", help="Validate a full problem")
    prob_parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to .pddl problem file (alternative to --data/--file).",
    )
    prob_parser.add_argument(
        "--data", type=str, default=None, help="ProblemDetails JSON string."
    )
    prob_parser.add_argument(
        "--file", type=str, default=None, help="Path to ProblemDetails JSON file."
    )
    prob_parser.set_defaults(func=validate_problem_command)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def validate_component_command(args):
    try:
        raw = _load_data(args)
        model_cls, validator_cls = VALIDATABLE[args.component_name]
        result = _validate_component_json(raw, model_cls, validator_cls)
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    _print_result(result)


def _validate_domain_from_pddl(path: Path) -> dict:
    raw = path.read_text()
    try:
        details = parse_domain_pddl(raw)
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Failed to parse PDDL domain: {e}"],
            "warnings": [],
        }

    validator = DomainValidator()
    errors = []
    warnings = []

    fields = [
        ("types", details.types),
        ("constants", details.constants),
        ("predicates", details.predicates),
        ("functions", details.functions),
        ("derived_predicates", details.derived_predicates),
        ("actions", details.actions),
    ]
    context = {
        PDDLType: details.types,
        Predicate: details.predicates,
        Function: details.functions,
    }

    for name, items in fields:
        for item in items:
            result = validator.validate_component(item, context)
            if not result.valid:
                for e in result.errors:
                    errors.append(f"[{name}] {e}")
            warnings.extend(f"[{name}] {w}" for w in result.warnings)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "name": details.name,
    }


def _validate_problem_from_pddl(path: Path) -> dict:
    raw = path.read_text()
    try:
        details = parse_problem_pddl(raw)
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Failed to parse PDDL problem: {e}"],
            "warnings": [],
        }

    validator = ProblemValidator()
    errors = []
    warnings = []
    context = {PDDLObject: details.objects}

    for item in details.objects:
        result = validator.validate_component(item, context)
        if not result.valid:
            errors.extend(result.errors)
        warnings.extend(result.warnings)

    if details.initial_state:
        result = validator.validate_component(details.initial_state, context)
        if not result.valid:
            errors.extend(result.errors)
        warnings.extend(result.warnings)

    if details.goal_state:
        result = validator.validate_component(details.goal_state, context)
        if not result.valid:
            errors.extend(result.errors)
        warnings.extend(result.warnings)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "name": details.name,
    }


def validate_domain_command(args):
    p = _resolve_path(args)
    if p and p.suffix == ".pddl":
        result = _validate_domain_from_pddl(p)
        name = result.get("name", p.stem)
        if result["valid"]:
            print(f'[SUCCESS] Domain "{name}" is valid')
        else:
            print(
                f'[FAIL] Domain "{name}" has {len(result["errors"])} error(s):',
                file=sys.stderr,
            )
            for e in result["errors"]:
                print(f"  {e}", file=sys.stderr)
        if result.get("warnings"):
            for w in result["warnings"]:
                print(f"  [WARN] {w}")
        if not result["valid"]:
            sys.exit(1)
        return

    try:
        raw = _load_data(args)
        result = _validate_full_domain(raw)
    except Exception as e:
        print(f"[ERROR] Domain validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    _print_result(result)


def validate_problem_command(args):
    p = _resolve_path(args)
    if p and p.suffix == ".pddl":
        result = _validate_problem_from_pddl(p)
        name = result.get("name", p.stem)
        if result["valid"]:
            print(f'[SUCCESS] Problem "{name}" is valid')
        else:
            print(
                f'[FAIL] Problem "{name}" has {len(result["errors"])} error(s):',
                file=sys.stderr,
            )
            for e in result["errors"]:
                print(f"  {e}", file=sys.stderr)
        if result.get("warnings"):
            for w in result["warnings"]:
                print(f"  [WARN] {w}")
        if not result["valid"]:
            sys.exit(1)
        return

    try:
        raw = _load_data(args)
        result = _validate_full_problem(raw)
    except Exception as e:
        print(f"[ERROR] Problem validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    _print_result(result)


def _print_result(result: dict):
    if result["valid"]:
        print(f"[SUCCESS] Valid ({result.get('count', '?')} items checked)")
    else:
        print(f"[FAIL] Found {len(result['errors'])} error(s):", file=sys.stderr)
        for e in result["errors"]:
            print(f"  {e}", file=sys.stderr)

    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"  [WARN] {w}")

    if not result["valid"]:
        sys.exit(1)
