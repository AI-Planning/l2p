"""Output the Pydantic JSON Schema for any PDDL component.

Use this to tell an LLM what JSON structure it should produce for
l2p set and l2p build commands.

Each schema describes the exact fields, types, and constraints of the
Pydantic model underlying that PDDL component.  This includes field
descriptions, default values, validators, and nested model definitions.

Available components:
    types, constants, predicates, functions, derived-predicates,
    actions, durative-actions, events, processes, constraints,
    parameters, objects, initial-state, goal-state, metric,
    domain, problem, requirements

Use --examples to include a concrete JSON example alongside the schema.

Example output shape::

    {
      "component": "types",
      "schema": {
        "$defs": {...},
        "properties": {
          "name": {"type": "string"},
          "parent": {"type": "string"},
          "desc": {"type": "string"}
        }
      },
      "example": [{"name": "block", "parent": "object"}]
    }
"""

import argparse
import json
from typing import Type, Dict

from pydantic import BaseModel

from l2p.utils.pddl_types import *

SCHEMAS: Dict[str, Type[BaseModel]] = {
    "requirements": Requirement,
    "types": PDDLType,
    "constants": Constant,
    "predicates": Predicate,
    "functions": Function,
    "derived-predicates": DerivedPredicate,
    "actions": Action,
    "durative-actions": DurativeAction,
    "events": Event,
    "processes": Process,
    "constraints": Constraint,
    "parameters": Parameter,
    "objects": PDDLObject,
    "initial-state": InitialState,
    "goal-state": GoalState,
    "metric": Metric,
    "domain": DomainDetails,
    "problem": ProblemDetails,
}

EXAMPLES: Dict[str, str] = {
    "types": '[{"name": "block", "parent": "object"}]',
    "predicates": '[{"name": "on", "params": [{"variable": "?x", "type": "block"}, {"variable": "?y", "type": "block"}]}]',
    "actions": '[{"name": "pickup", "params": [{"variable": "?b", "type": "block"}], "preconditions": {"conditions": ["(clear ?b)"]}, "effects": {"add": ["(holding ?b)"], "delete": ["(clear ?b)"]}}]',
    "domain": '{"name": "blocksworld", "types": [...], "predicates": [...], "actions": [...]}',
    "problem": '{"name": "pb1", "domain_name": "blocksworld", "objects": [...], "initial_state": {...}, "goal_state": {...}}',
}

DOMAIN_EXAMPLE = """{
  "name": "blocksworld",
  "types": [{"name": "block", "parent": "object"}],
  "predicates": [
    {"name": "on", "params": [{"variable": "?x", "type": "block"}, {"variable": "?y", "type": "block"}]},
    {"name": "clear", "params": [{"variable": "?x", "type": "block"}]}
  ],
  "actions": [
    {
      "name": "pickup",
      "params": [{"variable": "?b", "type": "block"}],
      "preconditions": {"conditions": ["(clear ?b)", "(on-table ?b)"]},
      "effects": {"add": ["(holding ?b)"], "delete": ["(clear ?b)", "(on-table ?b)"]}
    }
  ]
}"""

PROBLEM_EXAMPLE = """{
  "name": "pb1",
  "domain_name": "blocksworld",
  "objects": [
    {"name": "b1", "type": "block"},
    {"name": "b2", "type": "block"}
  ],
  "initial_state": {
    "facts": ["(on b1 b2)", "(on-table b2)", "(clear b1)"]
  },
  "goal_state": {
    "conditions": ["(on b2 b1)"]
  }
}"""

EXAMPLES_FULL = {
    "domain": DOMAIN_EXAMPLE,
    "problem": PROBLEM_EXAMPLE,
}


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "schema",
        help="Output JSON Schema for PDDL components",
        description="Print the Pydantic JSON Schema for any PDDL component. Use --examples for sample JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  l2p schema types
  l2p schema predicates
  l2p schema domain
  l2p schema domain --examples
        """,
    )

    parser.add_argument(
        "component",
        type=str,
        nargs="?",
        choices=list(SCHEMAS.keys()) + ["list"],
        default="list",
        help="Component name (types, predicates, domain, problem, ...) or 'list' to show all.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        default=False,
        help="Include a concrete JSON example alongside the schema.",
    )
    parser.set_defaults(func=schema_command)


def schema_command(args):
    if args.component == "list":
        print("Available components for schema:")
        for name in sorted(SCHEMAS.keys()):
            print(f"  {name}")
        return

    model_cls = SCHEMAS.get(args.component)
    if not model_cls:
        print(f"[ERROR] Unknown component: {args.component}", file=sys.stderr)
        return

    schema = model_cls.model_json_schema()
    output = {"component": args.component, "schema": schema}

    if args.examples:
        ex = EXAMPLES_FULL.get(args.component) or EXAMPLES.get(args.component)
        if ex:
            try:
                output["example"] = json.loads(ex) if isinstance(ex, str) else ex
            except json.JSONDecodeError:
                output["example"] = ex

    print(json.dumps(output, indent=2))
