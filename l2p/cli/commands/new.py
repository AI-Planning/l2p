"""
New command for L2P CLI.

Creates blank PDDL domain/problem files.
"""

import argparse
import sys
from pathlib import Path


def add_subparser(subparsers):
    """Add new subparser."""
    parser = subparsers.add_parser(
        "new",
        help="Create a blank PDDL file",
        description="Create a blank PDDL domain or problem file from a template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  l2p new domain.pddl
  l2p new problem.pddl --type problem
  l2p new my_domain.pddl --domain-name blocksworld
        """,
    )

    parser.add_argument(
        "filename",
        type=str,
        help="Name of the PDDL file to create"
    )

    parser.add_argument(
        "--type",
        choices=["domain", "problem"],
        default="domain",
        help="Type of PDDL file (default: domain)"
    )

    parser.add_argument(
        "--domain-name",
        type=str,
        default="my-domain",
        help="Domain name (default: my-domain)"
    )

    parser.add_argument(
        "--problem-name",
        type=str,
        default="my-problem",
        help="Problem name (default: my-problem)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing file"
    )

    parser.set_defaults(func=new_command)


def new_command(args):
    """Execute new command."""
    filepath = Path(args.filename)

    if filepath.exists() and not args.force:
        print(f"Error: {filepath} already exists. Use --force to overwrite.")
        sys.exit(1)

    if args.type == "domain":
        content = _domain_template(args.domain_name)
    else:
        content = _problem_template(args.problem_name, args.domain_name)

    filepath.write_text(content)
    print(f"Created {args.type} file: {filepath.resolve()}")


def _domain_template(name: str) -> str:
    return (
        f"(define (domain {name})\n"
        "    (:requirements :strips :typing)\n"
        "    (:types)\n"
        "    (:predicates)\n"
        ")\n"
    )


def _problem_template(problem_name: str, domain_name: str) -> str:
    return (
        f"(define (problem {problem_name})\n"
        f"    (:domain {domain_name})\n"
        "    (:objects)\n"
        "    (:init)\n"
        "    (:goal (and ))\n"
        ")\n"
    )
