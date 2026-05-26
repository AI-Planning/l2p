"""Assemble and render a complete PDDL domain or problem from JSON.

Two input modes:

1. **Full JSON** (recommended for agents) — pass the complete
   DomainDetails or ProblemDetails JSON via --data:

       l2p build domain --data '{
         "name":"blocksworld",
         "types":[{"name":"block","parent":"object"}],
         "predicates":[...],
         "actions":[...]
       }'

       l2p build problem --data '{
         "name":"pb1",
         "domain_name":"blocksworld",
         "objects":[{"name":"b1","type":"block"}],
         "initial_state":{"facts":["(on-table b1)"]},
         "goal_state":{"conditions":["(holding b1)"]}
       }'

2. **Component files** — pass individual JSON files or strings:

       l2p build domain --name bw --types types.json --predicates preds.json
       l2p build problem --name pb1 --objects @objects.json

DomainDetails expected fields:
    name (str), types (list), constants (list), predicates (list),
    functions (list), derived_predicates (list), actions (list),
    durative_actions (list), events (list), processes (list),
    constraint (list)

ProblemDetails expected fields:
    name (str), domain_name (str), objects (list),
    initial_state ({facts, timed_facts}), goal_state ({conditions}),
    constraint (list), metric ({optimization, expression})

Output:
  - Default:  PDDL string written to stdout or --output file
  - --json:  assembled DomainDetails/ProblemDetails as JSON
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from l2p.domain_builder import DomainBuilder
from l2p.problem_builder import ProblemBuilder
from l2p.utils.pddl_types import *
from l2p.utils.pddl_format import *


def _load_json(path_or_raw: str) -> Any:
    """Load JSON from a file path (prefixed with @) or raw string."""
    if path_or_raw.startswith("@"):
        return json.loads(Path(path_or_raw[1:]).read_text())
    return json.loads(path_or_raw)


def _resolve_component(args, key: str, default=None):
    """Resolve a component from --data or individual flag."""
    val = getattr(args, key, None)
    if val is not None:
        if isinstance(val, str):
            return _load_json(val)
        return val
    return default


def _build_domain_from_data(data: dict) -> str:
    details = DomainDetails.model_validate(data)
    builder = DomainBuilder(domain_details=details)
    return builder.generate_domain(details)


def _build_problem_from_data(data: dict) -> str:
    details = ProblemDetails.model_validate(data)
    builder = ProblemBuilder(problem_details=details)
    return builder.generate_problem(details)


def _build_domain_from_components(args) -> str:
    types = _resolve_component(args, "types", [])
    constants = _resolve_component(args, "constants", [])
    predicates = _resolve_component(args, "predicates", [])
    functions = _resolve_component(args, "functions", [])
    derived_predicates = _resolve_component(args, "derived_predicates", [])
    actions = _resolve_component(args, "actions", [])
    durative_actions = _resolve_component(args, "durative_actions", [])
    events = _resolve_component(args, "events", [])
    processes = _resolve_component(args, "processes", [])
    constraints = _resolve_component(args, "constraints", [])

    details = DomainDetails(
        name=args.name or "domain",
        types=types,
        constants=constants,
        predicates=predicates,
        functions=functions,
        derived_predicates=derived_predicates,
        actions=actions,
        durative_actions=durative_actions,
        events=events,
        processes=processes,
        constraint=constraints,
    )
    builder = DomainBuilder(domain_details=details)
    return builder.generate_domain(details)


def _build_problem_from_components(args) -> str:
    objects = _resolve_component(args, "objects", [])
    initial_state = _resolve_component(args, "initial_state")
    goal_state = _resolve_component(args, "goal_state")

    details = ProblemDetails(
        name=args.name or "problem",
        domain_name=args.domain_name or "domain",
        objects=objects,
        initial_state=initial_state or InitialState(),
        goal_state=goal_state or GoalState(),
    )
    builder = ProblemBuilder(problem_details=details)
    return builder.generate_problem(details)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "build",
        help="Assemble and render PDDL domain/problem",
        description="Assemble PDDL domain or problem from components and output the final PDDL string.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  l2p build domain --data '{"name":"bw","types":[...],"predicates":[...],...}' -o domain.pddl
  l2p build domain --name bw --types types.json --predicates preds.json -o domain.pddl
  l2p build problem --data '{"name":"p","domain_name":"bw","objects":[...]}' -o problem.pddl
        """,
    )

    subparsers = parser.add_subparsers(
        dest="build_command",
        title="build commands",
        description="What to build",
        metavar="COMMAND",
        required=True,
    )

    # --- domain ---
    dom_parser = subparsers.add_parser("domain", help="Build a PDDL domain")
    dom_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Domain name (used when not passing full --data).",
    )
    dom_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Full DomainDetails JSON string, or @path/to/file.json.",
    )
    dom_parser.add_argument(
        "--types", type=str, default=None, help="Types JSON array or @file."
    )
    dom_parser.add_argument(
        "--constants", type=str, default=None, help="Constants JSON array or @file."
    )
    dom_parser.add_argument(
        "--predicates", type=str, default=None, help="Predicates JSON array or @file."
    )
    dom_parser.add_argument(
        "--functions", type=str, default=None, help="Functions JSON array or @file."
    )
    dom_parser.add_argument(
        "--derived-predicates",
        type=str,
        default=None,
        help="Derived predicates JSON array or @file.",
    )
    dom_parser.add_argument(
        "--actions", type=str, default=None, help="Actions JSON array or @file."
    )
    dom_parser.add_argument(
        "--durative-actions",
        type=str,
        default=None,
        help="Durative actions JSON array or @file.",
    )
    dom_parser.add_argument(
        "--events", type=str, default=None, help="Events JSON array or @file."
    )
    dom_parser.add_argument(
        "--processes", type=str, default=None, help="Processes JSON array or @file."
    )
    dom_parser.add_argument(
        "--constraints", type=str, default=None, help="Constraints JSON array or @file."
    )
    dom_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout).",
    )
    dom_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output assembled DomainDetails as JSON instead of PDDL.",
    )
    dom_parser.set_defaults(func=build_domain_command)

    # --- problem ---
    prob_parser = subparsers.add_parser("problem", help="Build a PDDL problem")
    prob_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Problem name (used when not passing full --data).",
    )
    prob_parser.add_argument(
        "--domain-name", type=str, default=None, help="Domain this problem belongs to."
    )
    prob_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Full ProblemDetails JSON string, or @path/to/file.json.",
    )
    prob_parser.add_argument(
        "--objects", type=str, default=None, help="Objects JSON array or @file."
    )
    prob_parser.add_argument(
        "--initial-state",
        type=str,
        default=None,
        help="InitialState JSON object or @file.",
    )
    prob_parser.add_argument(
        "--goal-state", type=str, default=None, help="GoalState JSON object or @file."
    )
    prob_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout).",
    )
    prob_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output assembled ProblemDetails as JSON instead of PDDL.",
    )
    prob_parser.set_defaults(func=build_problem_command)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def build_domain_command(args):
    try:
        if args.data:
            data = _load_json(args.data)
            pddl = _build_domain_from_data(data)
            domain_json = data
        else:
            pddl = _build_domain_from_components(args)
            domain_json = None

        if args.json and domain_json:
            output = json.dumps(domain_json, indent=2)
        else:
            output = pddl

        if args.output:
            Path(args.output).write_text(output)
            print(f"[SUCCESS] Domain written to {args.output}")
        else:
            print(output)

    except Exception as e:
        print(f"[ERROR] Failed to build domain: {e}", file=sys.stderr)
        sys.exit(1)


def build_problem_command(args):
    try:
        if args.data:
            data = _load_json(args.data)
            pddl = _build_problem_from_data(data)
            problem_json = data
        else:
            pddl = _build_problem_from_components(args)
            problem_json = None

        if args.json and problem_json:
            output = json.dumps(problem_json, indent=2)
        else:
            output = pddl

        if args.output:
            Path(args.output).write_text(output)
            print(f"[SUCCESS] Problem written to {args.output}")
        else:
            print(output)

    except Exception as e:
        print(f"[ERROR] Failed to build problem: {e}", file=sys.stderr)
        sys.exit(1)
