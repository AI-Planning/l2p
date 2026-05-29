"""Run a classical planner on PDDL domain/problem strings.

Accepts PDDL as raw strings or via @filepath.  No need to write
temporary files manually — this command handles that internally.

Output with --json returns a structured PlanningResult::

    {
      "is_successful": true,
      "plan": ["(pickup b1)", "(stack b1 b2)"],
      "error_message": null,
      "raw_output": "...",
      "metrics": {}
    }

Planner backends:
  - fast-downward (default) — uses the Fast Downward classical planner
    via the downward/fast-downward.py submodule.  Supports --alias
    (e.g. lama-first, seq-opt-fdss-1, etc.).
  - unified — uses the Unified Planning Python library.  Supports
    --engine (e.g. aries, pyperplan, etc.).  Requires:
        pip install unified-planning unified-planning[engines]

Examples:
  # Raw PDDL strings (no files needed)
  l2p plan --domain '(define (domain test) ...)' --problem '(define (problem p) ...)'

  # Read from files
  l2p plan --domain @domain.pddl --problem @problem.pddl --planner fast-downward

  # Machine-readable output
  l2p plan --domain @d.pddl --problem @p.pddl --json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional


def _resolve_pddl(value: Optional[str]) -> Optional[str]:
    """Resolve PDDL from raw string or @filepath."""
    if value is None:
        return None
    if value.startswith("@"):
        return Path(value[1:]).read_text()
    return value


def _write_temp_pddl(content: str, suffix: str = ".pddl") -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "plan",
        help="Run a planner on domain/problem",
        description="Execute an automated planner on PDDL domain and problem files or raw strings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  l2p plan --domain '(define (domain test) (:requirements :strips))' --problem '(define (problem p) (:domain test) (:init) (:goal (and)))'
  l2p plan --domain @domain.pddl --problem @problem.pddl --planner fast-downward --alias lama-first
  l2p plan --domain @d.pddl --problem @p.pddl --json
        """,
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="PDDL domain as a raw string, or @path/to/domain.pddl to read from file.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="PDDL problem as a raw string, or @path/to/problem.pddl to read from file.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="fast-downward",
        choices=["fast-downward", "unified"],
        help="Planner backend: 'fast-downward' (default) or 'unified'.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="lama-first",
        help="Fast Downward search alias (default: lama-first).  Other options: seq-opt-fdss-1, seq-opt-bjolp, etc.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Unified Planning engine name (default: aries).  Requires: pip install unified-planning[engines].",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Planner timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--executable",
        type=str,
        default=None,
        help="Path to Fast Downward executable (default: downward/fast-downward.py).  Only for --planner fast-downward.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to write plan text (only on success).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output PlanningResult as structured JSON instead of human-readable text.",
    )

    parser.set_defaults(func=plan_command)


def plan_command(args):
    domain_pddl = _resolve_pddl(args.domain)
    problem_pddl = _resolve_pddl(args.problem)

    if not domain_pddl or not problem_pddl:
        print("[ERROR] Both --domain and --problem are required.", file=sys.stderr)
        sys.exit(1)

    domain_path = _write_temp_pddl(domain_pddl)
    problem_path = _write_temp_pddl(problem_pddl)

    try:
        if args.planner == "fast-downward":
            from l2p.planner_builder import FastDownward

            executable = args.executable or "downward/fast-downward.py"
            if not Path(executable).exists():
                print(
                    f"[ERROR] FastDownward executable not found: {executable}",
                    file=sys.stderr,
                )
                print(
                    "  Ensure the submodule is initialized: "
                    "git submodule update --init --recursive",
                    file=sys.stderr,
                )
                print(
                    "  Or specify a custom path: --executable /path/to/fast-downward.py",
                    file=sys.stderr,
                )
                sys.exit(1)
            planner = FastDownward(executable_path=executable)
            result = planner.run_planner(
                domain_path=domain_path,
                problem_path=problem_path,
                alias=args.alias,
                timeout=args.timeout,
            )
        elif args.planner == "unified":
            from l2p.planner_builder import UnifiedPlanning

            planner = UnifiedPlanning()
            result = planner.run_planner(
                domain_path=domain_path,
                problem_path=problem_path,
                engine=args.engine or "aries",
                timeout=args.timeout,
            )
        else:
            print(f"[ERROR] Unknown planner: {args.planner}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            import dataclasses

            output = json.dumps(dataclasses.asdict(result), indent=2, default=str)
            print(output)
        else:
            if result.is_successful:
                print("[SUCCESS] Plan found:")
                if result.plan:
                    for i, step in enumerate(result.plan, 1):
                        print(f"  {i}: {step}")
            else:
                print(
                    f"[FAIL] {result.error_message or 'No plan found.'}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if args.output and result.plan:
                plan_text = "\n".join(result.plan)
                Path(args.output).write_text(plan_text)
                print(f"\nPlan written to {args.output}")

    except Exception as e:
        print(f"[ERROR] Planner failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        Path(domain_path).unlink(missing_ok=True)
        Path(problem_path).unlink(missing_ok=True)
