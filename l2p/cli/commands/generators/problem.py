"""
Problem generator for L2P CLI.

Generates complete PDDL problem using pipeline approach.
"""

import sys
import re
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple

from l2p import load_file
from l2p.cli.commands.generate import GeneratorBase
from l2p.cli.utils.errors import handle_error
from l2p.cli.utils.helpers import _input_or_exit, BOLD, GREEN, CYAN, YELLOW, RESET


def add_subparser(subparsers):
    """Add problem generator subparser."""
    parser = subparsers.add_parser(
        "problem",
        help="Generate complete PDDL problem",
        description="Generate complete PDDL problem using pipeline approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive problem generation
  l2p generate problem

  # Interactive problem generation w/ max retries
  l2p generate problem --max-retries <n>
        """,
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for LLM (default: 3)"
    )

    parser.set_defaults(func=generate_problem_command)


def generate_problem_command(args):
    """Execute problem generation command."""
    try:
        generator = ProblemGenerator()
        generator.generate(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


def _input_or_exit(prompt: str = "") -> str:
    value = input(prompt).strip()
    if value == "/exit":
        print("Operation cancelled.")
        sys.exit(0)
    return value


def _parse_domain_file(filepath: str) -> dict:
    """Parse a PDDL domain file and extract components into L2P internal format."""
    from pddl import parse_domain as pddl_parse_domain

    path = Path(filepath).expanduser().resolve()
    domain = pddl_parse_domain(str(path))

    # extract types
    types_list = []
    for type_name, parent in domain.types.items():
        name = str(type_name)
        parent_str = str(parent) if parent else "object"
        types_list.append({"name": name, "parent": parent_str, "desc": ""})

    # extract constants
    constants_dict = {}
    for c in domain.constants:
        constants_dict[str(c.name)] = str(c.type_tag) if c.type_tag else "object"

    # extract predicates
    predicates_list = []
    for pred in domain.predicates:
        params = OrderedDict()
        for var in pred.terms:
            param_name = f"?{var.name}"
            param_type = next(iter(var.type_tags)) if var.type_tags else ""
            params[param_name] = param_type

        clean_parts = []
        for pname, ptype in params.items():
            if ptype:
                clean_parts.append(f"{pname} - {ptype}")
            else:
                clean_parts.append(pname)
        clean = f"({pred.name} {' '.join(clean_parts)})"

        predicates_list.append({
            "name": str(pred.name),
            "desc": "",
            "raw": clean,
            "params": params,
            "clean": clean,
        })

    # extract functions
    functions_list = []
    for func, return_type in domain.functions.items():
        params = OrderedDict()
        for var in func.terms:
            param_name = f"?{var.name}"
            param_type = next(iter(var.type_tags)) if var.type_tags else ""
            params[param_name] = param_type

        clean_parts = []
        for pname, ptype in params.items():
            if ptype:
                clean_parts.append(f"{pname} - {ptype}")
            else:
                clean_parts.append(pname)
        clean = f"({func.name} {' '.join(clean_parts)})"
        if return_type:
            clean += f" - {return_type}"

        functions_list.append({
            "name": str(func.name),
            "desc": "",
            "raw": clean,
            "params": params,
            "clean": clean,
        })

    if not types_list:
        types_list = [{"name": "object", "parent": "", "desc": ""}]

    return {
        "name": str(domain.name),
        "requirements": [str(r) for r in domain.requirements],
        "types": types_list,
        "constants": constants_dict if constants_dict else None,
        "predicates": predicates_list if predicates_list else None,
        "functions": functions_list if functions_list else None,
    }


class ProblemGenerator(GeneratorBase):
    """Generator for complete PDDL problems."""

    def generate(self, args):
        """Interactive problem generation."""
        print(
            f"{BOLD}{'=' * 60}{RESET}\n"
            f"{BOLD}  L2P Interactive Problem Generator{RESET}\n"
            f"{BOLD}{'=' * 60}{RESET}\n"
            f"  Type {YELLOW}/exit{RESET} at any prompt to quit\n")

        domain_info = self._prompt_domain_file()
        domain_name = domain_info["name"]
        types = domain_info["types"]
        constants = domain_info["constants"]
        predicates = domain_info["predicates"]
        functions = domain_info["functions"]

        problem_name = self._prompt_problem_name()
        problem_desc = self._prompt_problem_desc()

        context = {
            "types": types,
            "constants": constants,
            "predicates": predicates,
            "functions": functions,
        }

        # generate objects
        print(f"\n{BOLD}--- Objects ---{RESET}")
        include, manual = self._prompt_component("objects", default_include=True)
        if include:
            if manual:
                context["objects"] = self._confirm_stage(
                    "objects", context,
                    manual_func=lambda: self._manual_objects(types),
                    llm_func=None
                )
            else:
                obj_desc = _input_or_exit("  Describe the objects (optional): ").strip()
                context["objects"] = self._confirm_stage(
                    "objects", context,
                    manual_func=None,
                    llm_func=lambda feedback="":
                        self._generate_objects(problem_desc, types, constants, args.max_retries, feedback, obj_desc)
                )
        else:
            context["objects"] = None

        # generate initial states
        print(f"\n{BOLD}--- Initial States ---{RESET}")
        include, manual = self._prompt_component("initial", default_include=True)
        if include:
            if manual:
                context["initial"] = self._confirm_stage(
                    "initial", context,
                    manual_func=lambda: self._manual_states("initial", predicates, context.get("objects")),
                    llm_func=None
                )
            else:
                init_desc = _input_or_exit("  Describe the initial states (optional): ").strip()
                context["initial"] = self._confirm_stage(
                    "initial", context,
                    manual_func=None,
                    llm_func=lambda feedback="":
                        self._generate_initial_states(
                            problem_desc, types, constants, predicates, functions,
                            context.get("objects"), None, None, args.max_retries, feedback, init_desc
                        )
                )
        else:
            context["initial"] = None

        # generate goal states
        print(f"\n{BOLD}--- Goal States ---{RESET}")
        include, manual = self._prompt_component("goal", default_include=True)
        if include:
            if manual:
                context["goal"] = self._confirm_stage(
                    "goal", context,
                    manual_func=lambda: self._manual_states("goal", predicates, context.get("objects")),
                    llm_func=None
                )
            else:
                goal_desc = _input_or_exit("  Describe the goal states (optional): ").strip()
                context["goal"] = self._confirm_stage(
                    "goal", context,
                    manual_func=None,
                    llm_func=lambda feedback="":
                        self._generate_goal_states(
                            problem_desc, types, constants, predicates, functions,
                            context.get("objects"), context.get("initial"), None, args.max_retries, feedback, goal_desc
                        )
                )
        else:
            context["goal"] = None

        # assemble problem
        print(
            f"\n\n{BOLD}{'=' * 60}{RESET}"
            f"\n{BOLD}  Assembling Problem{RESET}"
            f"\n{BOLD}{'=' * 60}{RESET}"
        )
        problem_pddl = self.problem_builder.generate_task(
            domain_name=domain_name,
            problem_name=problem_name,
            objects=context.get("objects"),
            initial=context.get("initial"),
            goal=context.get("goal"),
        )

        # print output
        print(
            f"\nOUTPUT:"
            f"\n\n{BOLD}{'=' * 60}{RESET}"
            f"\n{problem_pddl}"
            f"\n{BOLD}{'=' * 60}{RESET}"
        )

        self._prompt_save(problem_pddl, problem_name)

    
    # PROMPTING HELPERS
    def _prompt_domain_file(self) -> dict:
        """Prompt for domain file and parse it."""
        while True:
            path_str = _input_or_exit(f"{GREEN}Enter domain file path:{RESET} ").strip()
            path = Path(path_str).expanduser().resolve()

            if not path.exists():
                print(f"  {YELLOW}File not found: {path}{RESET}")
                continue
            if path.suffix.lower() != ".pddl":
                print(f"  {YELLOW}Expected a '.pddl' file{RESET}")
                continue

            try:
                domain_info = _parse_domain_file(str(path))
                print(f"  Domain: {CYAN}{domain_info['name']}{RESET}")
                type_count = len(domain_info.get("types") or [])
                if type_count:
                    print(f"  Types: {CYAN}{type_count}{RESET}")
                const_count = len(domain_info.get("constants") or {})
                if const_count:
                    print(f"  Constants: {CYAN}{const_count}{RESET}")
                pred_count = len(domain_info.get("predicates") or [])
                if pred_count:
                    print(f"  Predicates: {CYAN}{pred_count}{RESET}")
                func_count = len(domain_info.get("functions") or [])
                if func_count:
                    print(f"  Functions: {CYAN}{func_count}{RESET}")

                resp = _input_or_exit(f"  Use this domain? ({GREEN}Y{RESET}/n): ").strip().lower()
                if resp != "n":
                    return domain_info
            except Exception as e:
                print(f"  {YELLOW}Failed to parse domain: {e}{RESET}")
                print("  Make sure it's a valid PDDL domain file.")

    def _prompt_problem_name(self) -> str:
        while True:
            name = _input_or_exit(f"{GREEN}Enter problem name:{RESET} ").strip().lower().replace(" ", "-")
            if name:
                return re.sub(r"[^a-z0-9-]", "", name)
            print("Problem name cannot be empty.")

    def _prompt_problem_desc(self) -> str:
        print(f"\n{GREEN}Enter a brief description of your problem:{RESET}")
        desc = _input_or_exit().strip()
        return desc or "A general planning problem."

    def _prompt_component(self, name: str, default_include: bool) -> Tuple[bool, bool]:
        resp = _input_or_exit(f"Include {name}? ({'Y/n' if default_include else 'y/N'}): ").strip().lower()
        include = default_include if not resp else resp == "y"

        if not include:
            return False, False

        resp = _input_or_exit(f"  Configure {name} manually? (y/N): ").strip().lower()
        manual = resp == "y"
        return True, manual

    
    # MANUAL BUILDERS
    def _manual_objects(self, types) -> Optional[dict]:
        """Interactive objects builder."""
        available = self._collect_type_names(types)
        objects = {}
        print(
            f"  Available types: {CYAN}{', '.join(available)}{RESET}\n"
            f"  Use {YELLOW}'..'{RESET} to go back, {YELLOW}'done'{RESET} to finish.\n")

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
                if raw in objects:
                    print(f"  {YELLOW}Object '{raw}' already exists.{RESET}")
                    continue
                name = raw
                state = 1
            elif state == 1:
                raw = _input_or_exit("    Type: ").strip().lower()
                if raw == "..":
                    state = 0
                    continue
                if raw in available:
                    objects[name] = raw
                    print(f"  {GREEN}Added:{RESET} {name} - {raw}")
                    state = 0
                else:
                    print(f"  {YELLOW}Invalid type. Available: {', '.join(available)}{RESET}")

        return objects if objects else None

    def _manual_states(self, kind: str, predicates, objects) -> Optional[list]:
        """Interactive initial/goal states builder."""
        available_preds = [p["name"] for p in (predicates or [])]
        available_objs = list(objects.keys()) if objects else []
        states = []
        print(
            f"  Available predicates: {CYAN}{', '.join(available_preds)}{RESET}\n"
            f"  Available objects: {CYAN}{', '.join(available_objs)}{RESET}\n"
            f"  Use {YELLOW}'..'{RESET} to go back, {YELLOW}'done'{RESET} to finish.\n")

        state = 0
        pred_name = ""
        pred_obj = None
        negated = False
        while True:
            if state == 0:
                raw = _input_or_exit(f"\n  {GREEN}Predicate name:{RESET} ").strip().lower()
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return states if states else None
                match = next((p for p in (predicates or []) if p["name"] == raw), None)
                if not match:
                    print(f"  {YELLOW}Unknown predicate '{raw}'. Available: {', '.join(available_preds)}{RESET}")
                    continue
                pred_name = raw
                pred_obj = match
                state = 1

            elif state == 1:
                raw = _input_or_exit("    Negated? (y/N): ").strip().lower()
                if raw == "..":
                    state = 0
                    continue
                negated = raw == "y"
                state = 2

            elif state == 2:
                params = pred_obj["params"] if pred_obj else OrderedDict()
                param_names = list(params.keys())
                if param_names:
                    print(f"    Parameters: {CYAN}{', '.join(param_names)}{RESET} (types: {', '.join(params.values())})")
                    raw = _input_or_exit("    Enter object values (comma-separated): ").strip()
                    if raw == "..":
                        state = 1
                        continue
                    values = [v.strip() for v in raw.split(",") if v.strip()]
                    if len(values) != len(param_names):
                        print(f"  {YELLOW}Expected {len(param_names)} values, got {len(values)}.{RESET}")
                        continue
                else:
                    values = []

                states.append({
                    "pred_name": pred_name,
                    "params": values,
                    "neg": negated,
                })

                if negated:
                    print(f"  {GREEN}Added:{RESET} (not ({pred_name} {' '.join(values)}))")    
                else:
                    print(f"  {GREEN}Added:{RESET} ({pred_name} {' '.join(values)})")
                state = 0

        return states if states else None

    def _prompt_save(self, content: str, problem_name: str):
        """Prompt for output file and save."""
        default_path = f"{problem_name}-problem.pddl"
        path_str = _input_or_exit(
            f"\n{GREEN}Enter output filename (default: {default_path}):{RESET} "
        ).strip()
        if not path_str:
            path_str = default_path

        output_path = Path(path_str)
        if output_path.exists():
            resp = _input_or_exit(
                f"  {YELLOW}File already exists. Overwrite? (y/N):{RESET} "
            ).strip().lower()
            if resp != "y":
                print(
                    f"  {YELLOW}Skipping file save.{RESET}\n"
                    f"\n{BOLD}Generated problem:{RESET}\n"
                    f"{content}\n")
                return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"\n{GREEN}[SUCCESS] Problem saved to: {output_path.resolve()}{RESET}")


    # DISPLAY & CONFIRM
    def _collect_type_names(self, types) -> list:
        if not types:
            return ["object"]
        names = ["object"]
        for t in types:
            if isinstance(t, dict):
                names.append(t.get("name", ""))
        return [n for n in names if n]

    def _display_component(self, label: str, items):
        if items is None:
            print(f"  ({label} omitted)")
            return
        print(f"\n  {BOLD}{label.capitalize()}:{RESET}")
        if label == "objects":
            if isinstance(items, dict):
                for name, typ in items.items():
                    print(f"    {CYAN}{name}{RESET} - {typ}")
        elif label in ("initial", "goal"):
            for s in items:
                if isinstance(s, dict):
                    params_str = " ".join(s.get("params", []))
                    if s.get("neg"):
                        print(f"    {CYAN}(not ({s.get('pred_name', '?')} {params_str})){RESET}")
                    else:
                        print(f"    {CYAN}({s.get('pred_name', '?')} {params_str}){RESET}")
        else:
            if isinstance(items, (list, dict)):
                print(f"    ({len(items)} {label})")

    def _confirm_stage(self, label: str, context: dict, manual_func, llm_func):
        """Run a component stage with confirmation and fix loop."""
        while True:
            if manual_func:
                result = manual_func()
            else:
                feedback = context.pop("_feedback", "")
                result = llm_func(feedback) if feedback else llm_func()

            if result is None:
                return None

            self._display_component(label, result)

            resp = _input_or_exit("\n  Is this correct? (y/N): ").strip().lower()
            if resp == "y":
                return result

            if manual_func:
                print(f"  {YELLOW}Restarting {label} entry...{RESET}")
            else:
                fix = _input_or_exit("  Describe what to fix: ").strip()
                if not fix:
                    return result
                context["_feedback"] = fix


    # LLM-BASED GENERATION WRAPPERS
    def _generate_objects(self, problem_desc: str, types, constants, max_retries: int, feedback: str = "", component_desc: str = ""):
        template = load_file("l2p/cli/commands/generators/templates/problem/formalize_objects.txt")
        prompt = problem_desc
        if component_desc:
            prompt = f"{problem_desc}\n\n[Additional context for objects]\n{component_desc}"
        if feedback:
            prompt = f"{prompt}\n\n[Feedback to apply to the generated objects]\n{feedback}"
            print("  Re-generating objects with your feedback...")
        else:
            print("  Generating objects from description...")
        result = self.problem_builder.formalize_objects(
            model=self.llm, problem_desc=prompt,
            prompt_template=template, types=types,
            constants=constants, max_retries=max_retries
        )
        obj_result, _, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return obj_result

    def _generate_initial_states(
        self, problem_desc: str, types, constants, predicates, functions,
        objects, initial, goal, max_retries: int, feedback: str = "", component_desc: str = ""
    ):
        template = load_file("l2p/cli/commands/generators/templates/problem/formalize_initial.txt")
        prompt = problem_desc
        if component_desc:
            prompt = f"{problem_desc}\n\n[Additional context for initial states]\n{component_desc}"
        if feedback:
            prompt = f"{prompt}\n\n[Feedback to apply to the generated initial states]\n{feedback}"
            print("  Re-generating initial states with your feedback...")
        else:
            print("  Generating initial states from description...")
        result = self.problem_builder.formalize_initial_state(
            model=self.llm, problem_desc=prompt,
            prompt_template=template, types=types,
            constants=constants, predicates=predicates,
            functions=functions, objects=objects,
            initial=initial, goal=goal,
            max_retries=max_retries
        )
        init_result, _, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return init_result

    def _generate_goal_states(
        self, problem_desc: str, types, constants, predicates, functions,
        objects, initial, goal, max_retries: int, feedback: str = "", component_desc: str = ""
    ):
        template = load_file("l2p/cli/commands/generators/templates/problem/formalize_goal.txt")
        prompt = problem_desc
        if component_desc:
            prompt = f"{problem_desc}\n\n[Additional context for goal states]\n{component_desc}"
        if feedback:
            prompt = f"{prompt}\n\n[Feedback to apply to the generated goal states]\n{feedback}"
            print("  Re-generating goal states with your feedback...")
        else:
            print("  Generating goal states from description...")
        result = self.problem_builder.formalize_goal_state(
            model=self.llm, problem_desc=prompt,
            prompt_template=template, types=types,
            constants=constants, predicates=predicates,
            functions=functions, objects=objects,
            initial=initial, goal=goal,
            max_retries=max_retries
        )
        goal_result, _, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return goal_result
