"""
Problem generator for L2P CLI.

Generates a complete PDDL problem using the new ProblemBuilder API.
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from l2p.utils.pddl_types import (
    ProblemDetails,
    PDDLObject,
    PDDLType,
    Predicate,
    Function,
    Parameter,
    Constant,
    InitialState,
    GoalState,
)
from l2p.cli.commands.generate import GeneratorBase
from l2p.cli.utils.errors import handle_error
from l2p.cli.utils.helpers import _input_or_exit, BOLD, GREEN, CYAN, YELLOW, RESET


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "problem",
        help="Generate complete PDDL problem",
        description="Generate complete PDDL problem using a pipeline approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  l2p generate problem
  l2p generate problem --max-retries 5
        """,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for LLM (default: 3)",
    )
    parser.set_defaults(func=generate_problem_command)


def generate_problem_command(args):
    try:
        ProblemGenerator().generate(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


def _parse_domain_file(filepath: str) -> Dict[str, Any]:
    """Parse a PDDL domain file and return components as Pydantic model lists."""
    from pddl import parse_domain as pddl_parse_domain

    path = Path(filepath).expanduser().resolve()
    domain = pddl_parse_domain(str(path))

    types_list: List[PDDLType] = []
    for type_name, parent in domain.types.items():
        types_list.append(
            PDDLType(
                name=str(type_name),
                parent=str(parent) if parent else "object",
            )
        )

    constants_list: List[Constant] = []
    for c in domain.constants:
        constants_list.append(
            Constant(
                name=str(c.name),
                type=str(c.type_tag) if c.type_tag else "object",
            )
        )

    predicates_list: List[Predicate] = []
    for pred in domain.predicates:
        params = [
            Parameter(
                variable=f"?{v.name}",
                type=str(next(iter(v.type_tags)) if v.type_tags else "object"),
            )
            for v in pred.terms
        ]
        predicates_list.append(Predicate(name=str(pred.name), params=params))

    functions_list: List[Function] = []
    for func, _return_type in domain.functions.items():
        params = [
            Parameter(
                variable=f"?{v.name}",
                type=str(next(iter(v.type_tags)) if v.type_tags else "object"),
            )
            for v in func.terms
        ]
        functions_list.append(Function(name=str(func.name), params=params))

    if not types_list:
        types_list = [PDDLType(name="object", parent="object")]

    return {
        "name": str(domain.name),
        "requirements": [str(r) for r in domain.requirements],
        "types": types_list,
        "constants": constants_list if constants_list else [],
        "predicates": predicates_list if predicates_list else [],
        "functions": functions_list if functions_list else [],
    }


class ProblemGenerator(GeneratorBase):
    """Generator for a complete PDDL problem."""

    def generate(self, args):
        print(
            f"{BOLD}{'=' * 60}{RESET}\n"
            f"{BOLD}  L2P Interactive Problem Generator{RESET}\n"
            f"{BOLD}{'=' * 60}{RESET}\n"
            f"  Type {YELLOW}/exit{RESET} at any prompt to quit\n"
        )

        # ---- domain source ----
        domain_info = self._prompt_domain_source()
        if domain_info is None:
            return
        types = domain_info.get("types", [])
        constants = domain_info.get("constants", [])
        predicates = domain_info.get("predicates", [])
        functions = domain_info.get("functions", [])
        domain_name = domain_info.get("name", "unnamed")

        problem_name = self._prompt_problem_name()
        problem_desc = self._prompt_problem_desc()

        # --- objects ---
        print(f"\n{BOLD}--- Objects ---{RESET}")
        include, manual = self._prompt_component("objects", default_include=True)
        if include:
            obj_desc = _input_or_exit("  Describe the objects (optional): ").strip()
            objects = self._confirm_stage(
                label="objects",
                llm_func=lambda feedback="": self._generate_objects(
                    problem_desc,
                    types,
                    constants,
                    args.max_retries,
                    obj_desc,
                    feedback,
                ),
                manual_func=self._manual_objects(types) if manual else None,
            )
        else:
            objects = []

        # --- initial state ---
        print(f"\n{BOLD}--- Initial State ---{RESET}")
        init = self._generate_initial(
            problem_desc,
            types,
            predicates,
            functions,
            objects,
            constants,
            args.max_retries,
        )

        # --- goal state ---
        print(f"\n{BOLD}--- Goal State ---{RESET}")
        goal = self._generate_goal(
            problem_desc,
            types,
            predicates,
            functions,
            objects,
            constants,
            args.max_retries,
        )

        # --- assemble problem ---
        print(
            f"\n\n{BOLD}{'=' * 60}{RESET}"
            f"\n{BOLD}  Assembling Problem{RESET}"
            f"\n{BOLD}{'=' * 60}{RESET}"
        )

        details = ProblemDetails(
            name=problem_name,
            domain_name=domain_name,
            objects=objects or [],
            initial_state=init or InitialState(),
            goal_state=goal or GoalState(),
        )
        problem_pddl = self.problem_builder.generate_problem(details)

        print(
            f"\nOUTPUT:\n\n{BOLD}{'=' * 60}{RESET}\n"
            f"{problem_pddl}\n"
            f"{BOLD}{'=' * 60}{RESET}"
        )
        self._prompt_save(problem_pddl, problem_name)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _prompt_domain_source(self) -> Optional[Dict[str, Any]]:
        print(f"\n{GREEN}How do you want to provide the domain?{RESET}")
        print(f"  {CYAN}1{RESET} - Enter a natural-language description")
        print(f"  {CYAN}2{RESET} - Load an existing PDDL domain file")
        choice = _input_or_exit("Choice (default: 1): ").strip()
        if choice == "2":
            path = _input_or_exit("Path to domain PDDL file: ").strip()
            try:
                info = _parse_domain_file(path)
                print(
                    f"  {GREEN}Loaded domain '{info['name']}' with "
                    f"{len(info['types'])} types, {len(info['predicates'])} predicates{RESET}"
                )
                return info
            except Exception as e:
                print(f"  {YELLOW}Failed to parse domain file: {e}{RESET}")
                return None
        else:
            desc = _input_or_exit("Describe the domain (optional): ").strip()
            return {
                "name": "from_description",
                "types": [],
                "constants": [],
                "predicates": [],
                "functions": [],
                "_description": desc or "A planning domain",
            }

    def _prompt_problem_name(self) -> str:
        while True:
            name = (
                _input_or_exit(f"{GREEN}Enter problem name:{RESET} ")
                .strip()
                .lower()
                .replace(" ", "-")
            )
            if name:
                return re.sub(r"[^a-z0-9-]", "", name)
            print("Problem name cannot be empty.")

    def _prompt_problem_desc(self) -> str:
        print(f"\n{GREEN}Enter a brief description of the problem:{RESET}")
        return _input_or_exit().strip() or "A planning problem instance."

    def _prompt_component(self, name: str, default_include: bool) -> Tuple[bool, bool]:
        resp = (
            _input_or_exit(f"Include {name}? ({'Y/n' if default_include else 'y/N'}): ")
            .strip()
            .lower()
        )
        include = default_include if not resp else resp == "y"
        if not include:
            return False, False
        resp = _input_or_exit(f"  Configure {name} manually? (y/N): ").strip().lower()
        return True, resp == "y"

    # ------------------------------------------------------------------
    # Confirmation / display helpers
    # ------------------------------------------------------------------

    def _confirm_stage(self, label: str, llm_func, manual_func=None):
        while True:
            if manual_func:
                result = manual_func()
            else:
                result = llm_func()
            if result is None:
                return result
            self._display_component(label, result)
            resp = _input_or_exit(f"\n  Is this correct? (y/N): ").strip().lower()
            if resp == "y":
                return result
            if manual_func:
                print(f"  {YELLOW}Restarting {label} entry...{RESET}")
            else:
                fix = _input_or_exit("  Describe what to fix: ").strip()
                if not fix:
                    return result

    def _display_component(self, label: str, items):
        if items is None:
            print(f"  ({label} omitted)")
            return
        print(f"\n  {BOLD}{label.capitalize()}:{RESET}")
        if label == "objects":
            for o in items:
                print(f"    {CYAN}{o.name}{RESET} - {o.type}")
        elif label == "initial_state" or label == "goal_state":
            if isinstance(items, InitialState):
                for f in items.facts:
                    print(f"    {CYAN}{f}{RESET}")
            elif isinstance(items, GoalState):
                for c in items.conditions:
                    print(f"    {CYAN}{c}{RESET}")
        else:
            print(f"    ({len(items)} items)")

    # ------------------------------------------------------------------
    # Manual data entry
    # ------------------------------------------------------------------

    def _manual_objects(self, types: List[PDDLType]):
        type_names = ["object"] + [t.name for t in types]
        objects: List[PDDLObject] = []
        print(f"  Available types: {CYAN}{', '.join(type_names)}{RESET}")
        print("  Enter objects. Name or 'done' to finish.\n")
        state = 0
        name = ""
        while True:
            if state == 0:
                raw = _input_or_exit(f"\n  {GREEN}Object name:{RESET} ").strip().lower()
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return objects if objects else None
                if any(o.name == raw for o in objects):
                    print(f"  {YELLOW}'{raw}' already exists.{RESET}")
                    continue
                name = raw
                state = 1
            elif state == 1:
                t = _input_or_exit(f"    Type (default: object): ").strip().lower()
                if t == "..":
                    state = 0
                    continue
                t = t if t in type_names else "object"
                objects.append(PDDLObject(name=name, type=t))
                print(f"  {GREEN}Added:{RESET} {name} - {t}")
                state = 0
        return objects if objects else None

    # ------------------------------------------------------------------
    # LLM generation wrappers
    # ------------------------------------------------------------------

    def _generate_objects(
        self,
        problem_desc: str,
        types: List[PDDLType],
        constants: List[Constant],
        max_retries: int,
        comp_desc: str = "",
        feedback: str = "",
    ) -> List[PDDLObject]:
        desc = problem_desc
        if comp_desc:
            desc += f"\n\n[Additional context for objects]\n{comp_desc}"
        if feedback:
            desc += f"\n\n[Feedback]\n{feedback}"
        what = "Re-generating" if feedback else "Generating"
        print(f"  {what} objects from description...")
        result, _ = self.problem_builder.formalize_component(
            model=self.llm,
            component_class=PDDLObject,
            description=desc,
            types=types,
            constants=constants,
            max_retries=max_retries,
        )
        return result if isinstance(result, list) else [result]

    def _generate_initial(
        self,
        problem_desc: str,
        types: List[PDDLType],
        predicates: List[Predicate],
        functions: List[Function],
        objects: List[PDDLObject],
        constants: List[Constant],
        max_retries: int,
    ) -> Optional[InitialState]:
        print("  Generating initial state from description...")
        try:
            result, _ = self.problem_builder.formalize_component(
                model=self.llm,
                component_class=InitialState,
                description=problem_desc,
                types=types,
                predicates=predicates,
                functions=functions,
                objects=objects,
                constants=constants,
                max_retries=max_retries,
            )
            self._display_component("initial_state", result)
            resp = _input_or_exit("  Is this correct? (Y/n): ").strip().lower()
            if resp == "n":
                print("  Keeping generated initial state.")
            return result
        except Exception as e:
            print(f"  {YELLOW}Failed to generate initial state: {e}{RESET}")
            return None

    def _generate_goal(
        self,
        problem_desc: str,
        types: List[PDDLType],
        predicates: List[Predicate],
        functions: List[Function],
        objects: List[PDDLObject],
        constants: List[Constant],
        max_retries: int,
    ) -> Optional[GoalState]:
        print("  Generating goal state from description...")
        try:
            result, _ = self.problem_builder.formalize_component(
                model=self.llm,
                component_class=GoalState,
                description=problem_desc,
                types=types,
                predicates=predicates,
                functions=functions,
                objects=objects,
                constants=constants,
                max_retries=max_retries,
            )
            self._display_component("goal_state", result)
            resp = _input_or_exit("  Is this correct? (Y/n): ").strip().lower()
            if resp == "n":
                print("  Keeping generated goal state.")
            return result
        except Exception as e:
            print(f"  {YELLOW}Failed to generate goal state: {e}{RESET}")
            return None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _prompt_save(self, content: str, problem_name: str):
        default_path = f"{problem_name}-problem.pddl"
        path_str = _input_or_exit(
            f"\n{GREEN}Enter output filename (default: {default_path}):{RESET} "
        ).strip()
        if not path_str:
            path_str = default_path
        output_path = Path(path_str)
        if output_path.exists():
            resp = (
                _input_or_exit(f"  {YELLOW}File exists. Overwrite? (y/N):{RESET} ")
                .strip()
                .lower()
            )
            if resp != "y":
                print(f"\n{BOLD}Generated problem:{RESET}\n{content}\n")
                return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"\n{GREEN}[SUCCESS] Problem saved to: {output_path.resolve()}{RESET}")
