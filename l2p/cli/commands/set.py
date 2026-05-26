"""Inject and validate individual PDDL components from JSON.

Use this to set one PDDL sub-component at a time (types, predicates,
actions, objects, etc.).  Each call parses the JSON, validates the
result against L2P's semantic rules, and optionally outputs the
validated model as JSON or PDDL.

The JSON must match the Pydantic model for that component.  To see the
exact schema an LLM should follow:

    l2p set types --schema
    l2p set predicates --schema
    l2p set actions --schema

Components are **stateless** — this command validates and returns; it
does not maintain a session.  Pipe validated JSON between calls::

    l2p set types --data '[...]' --json | l2p set predicates --stdin --json

Available components and their Pydantic JSON shape:

  DOMAIN:
    requirements        List[{"name": ":strips"}]            (no validation)
    types               List[{"name":"block","parent":"object"}]
    constants           List[{"name":"base","type":"location"}]
    predicates          List[{"name":"on","params":[{"variable":"?x","type":"block"}]}]
    functions           List[{"name":"battery","params":[...]}]
    derived-predicates  List[{"name":"can-move","params":[...],"condition":"..."}]
    actions             List[{"name":"drive","params":[...],"preconditions":{...},"effects":{...}}]
    durative-actions    List[{"name":"transmit","duration":["..."],"conditions":{...},"effects":{...}}]
    events              List[{"name":"battery-dead",...}]
    processes           List[{"name":"solar-charging",...}]
    constraints         List[{"condition":{...}}]

  PROBLEM:
    objects             List[{"name":"rover1","type":"rover"}]
    initial-state       {"facts":["(at rover1 loc1)"],"timed_facts":[]}
    goal-state          {"conditions":["(at rover1 loc2)"]}
    metric              {"optimization":"minimize","expression":"total-time"}

Validation rules applied per component:
  - Naming: valid PDDL characters, no reserved keywords, no duplicates
  - Type inheritance: parent types must exist
  - Parameters: ?vars must start with "?", types must be declared
  - Symbol references: all predicates/functions used in conditions
    and effects must be declared (only checked when --context provided)

Returns:
  - Without flags: success/error message + item count
  - With --json:    validated JSON array of models (exit 1 on failure)
  - With --pddl:    PDDL-formatted string (exit 1 on validation failure)
  - With --schema:  Pydantic JSON Schema for LLM reference
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, TypeAdapter

from l2p.utils.pddl_types import *
from l2p.utils.pddl_format import *
from l2p.validators.domain import DomainValidator
from l2p.validators.problem import ProblemValidator


# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------
# Each entry: (pydantic_model, is_list, format_func, validator_class)


def _register(
    model: Type[BaseModel],
    is_list: bool,
    fmt: Optional[Callable] = None,
    validator: Optional[Any] = None,
):
    return {"model": model, "is_list": is_list, "format": fmt, "validator": validator}


DOMAIN_COMPONENTS: Dict[str, dict] = {
    "requirements": _register(Requirement, True, format_requirements, None),
    "types": _register(PDDLType, True, format_types, DomainValidator),
    "constants": _register(Constant, True, format_constants, DomainValidator),
    "predicates": _register(Predicate, True, format_predicates, DomainValidator),
    "functions": _register(Function, True, format_functions, DomainValidator),
    "derived-predicates": _register(
        DerivedPredicate, True, format_derived_predicates, DomainValidator
    ),
    "actions": _register(Action, True, format_actions, DomainValidator),
    "durative-actions": _register(
        DurativeAction, True, format_durative_actions, DomainValidator
    ),
    "events": _register(Event, True, format_events, DomainValidator),
    "processes": _register(Process, True, format_processes, DomainValidator),
    "constraints": _register(Constraint, True, format_constraints, DomainValidator),
}

PROBLEM_COMPONENTS: Dict[str, dict] = {
    "objects": _register(PDDLObject, True, format_objects, ProblemValidator),
    "initial-state": _register(
        InitialState, False, format_initial_state, ProblemValidator
    ),
    "goal-state": _register(GoalState, False, format_goal_states, ProblemValidator),
    "metric": _register(Metric, False, format_metric, ProblemValidator),
}

ALL_COMPONENTS: Dict[str, dict] = {**DOMAIN_COMPONENTS, **PROBLEM_COMPONENTS}


def _resolve_data(args) -> str:
    if args.data:
        return args.data
    if args.file:
        return Path(args.file).read_text()
    if args.stdin:
        return sys.stdin.read()
    raise ValueError("No input provided. Use --data, --file, or --stdin.")


def _parse_json(data: str, component: dict) -> Any:
    model = component["model"]
    is_list = component["is_list"]
    raw = json.loads(data)
    if is_list:
        if not isinstance(raw, list):
            raw = [raw]
        adapter = TypeAdapter(List[model])
        return adapter.validate_python(raw)
    else:
        if isinstance(raw, list):
            if len(raw) != 1:
                raise ValueError(
                    f"Expected a single object, got a list of {len(raw)} items."
                )
            raw = raw[0]
        return model.model_validate(raw)


def _run_validation(parsed: Any, component: dict) -> Tuple[bool, List[str], List[str]]:
    validator_cls = component.get("validator")
    if not validator_cls:
        return True, [], []

    validator = validator_cls()
    items = parsed if isinstance(parsed, list) else [parsed]
    errors = []
    warnings = []
    for item in items:
        result = validator.validate_component(item, {})
        if not result.valid:
            errors.extend(result.errors)
        warnings.extend(result.warnings)

    return len(errors) == 0, errors, warnings


def _format_output(parsed: Any, component: dict) -> str:
    fmt = component.get("format")
    if not fmt:
        items = parsed if isinstance(parsed, list) else [parsed]
        return json.dumps(
            [i.model_dump(exclude_none=True) for i in items], indent=2
        )
    items = parsed if isinstance(parsed, list) else [parsed]
    return fmt(items)


# ---------------------------------------------------------------------------
# Shared argparse
# ---------------------------------------------------------------------------


def add_component_arguments(parser: argparse.ArgumentParser, components: Dict):
    parser.add_argument(
        "--data", type=str, default=None, help="JSON string of the component."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to JSON file containing the component.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        default=False,
        help="Read JSON from stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output validated model as JSON.",
    )
    parser.add_argument(
        "--pddl",
        action="store_true",
        default=False,
        help="Output the PDDL-formatted component.",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        default=False,
        help="Output the Pydantic JSON Schema for this component.",
    )


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "set",
        help="Set a PDDL component from JSON",
        description="Validate and inject a PDDL component (types, predicates, actions, etc.) from structured JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  l2p set types --data '[{"name":"block","parent":"object"}]'
  l2p set predicates --file preds.json
  l2p set actions --stdin --json
  l2p set types --schema
  
  # Pipe between commands:
  l2p set types --data '[...]' --json | l2p set predicates --stdin --json
        """,
    )

    subparsers = parser.add_subparsers(
        dest="set_component",
        title="components",
        description="PDDL component to set",
        metavar="COMPONENT",
        required=True,
    )

    for name in ALL_COMPONENTS:
        comp_parser = subparsers.add_parser(name, help=f"Set {name}")
        add_component_arguments(comp_parser, {name: ALL_COMPONENTS[name]})
        comp_parser.set_defaults(func=set_command)


def set_command(args):
    """Execute set command for a single component."""
    name = args.set_component
    component = ALL_COMPONENTS.get(name)
    if not component:
        print(f"[ERROR] Unknown component: {name}", file=sys.stderr)
        sys.exit(1)

    if args.schema:
        schema = component["model"].model_json_schema()
        print(json.dumps(schema, indent=2))
        return

    try:
        data = _resolve_data(args)
        parsed = _parse_json(data, component)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Failed to parse JSON: {e}", file=sys.stderr)
        sys.exit(1)

    valid, errors, warnings = _run_validation(parsed, component)

    if errors:
        print("[VALIDATION FAILED]", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}", file=sys.stderr)

    if args.json:
        items = parsed if isinstance(parsed, list) else [parsed]
        print(json.dumps(
            [i.model_dump(exclude_none=True) for i in items], indent=2
        ))
    elif args.pddl:
        print(_format_output(parsed, component))
    else:
        count = len(parsed) if isinstance(parsed, list) else 1
        status = "[SUCCESS]" if valid else "[SET WITH ERRORS]"
        print(f"{status} Set {count} {name}(s)")

    if not valid:
        sys.exit(1)
