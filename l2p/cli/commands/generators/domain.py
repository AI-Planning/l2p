"""
Domain generator for L2P CLI.

Generates complete PDDL domain using pipeline approach.
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import Any, Optional, List, Tuple

from ..generate import GeneratorBase
from ...utils.config import CLIError
from ...utils.errors import handle_error


BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

PDDL_REQUIREMENTS = [
    ":strips",
    ":typing",
    ":negative-preconditions",
    ":disjunctive-preconditions",
    ":equality",
    ":existential-preconditions",
    ":universal-preconditions",
    ":quantified-preconditions",
    ":conditional-effects",
    ":numeric-fluents",
    ":adl",
    ":durative-actions",
    ":derived-predicates",
    ":timed-initial-literals",
    ":action-costs",
    ":multi-agent",
    ":constraints",
    ":preferences",
]


def add_subparser(subparsers):
    """Add domain generator subparser."""
    parser = subparsers.add_parser(
        "domain",
        help="Generate complete PDDL domain",
        description="Generate complete PDDL domain using pipeline approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive domain generation
  l2p generate domain
  
  # Generate complete domain with pipeline
  l2p generate domain --desc "blocksworld domain" --pipeline
  
  # Generate domain with specific actions
  l2p generate domain --desc "blocksworld" --actions "pick-up, put-down, stack, unstack"
        """,
    )

    parser.add_argument(
        "--desc",
        type=str,
        help="Domain description (text or path to file). Omit for interactive mode."
    )

    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use automatic pipeline to generate all components"
    )

    parser.add_argument(
        "--actions",
        type=str,
        help="Comma-separated list of action names to generate"
    )

    parser.add_argument(
        "--requirements",
        type=str,
        default=":strips,:typing",
        help="PDDL requirements (default: :strips,:typing)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output PDDL file (default: stdout)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for intermediate files (used with --save-intermediate)"
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate component files"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for LLM (default: 3)"
    )

    parser.set_defaults(func=generate_domain_command)


def generate_domain_command(args):
    """Execute domain generation command."""
    try:
        generator = DomainGenerator()
        generator.generate(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


def _input_or_exit(prompt: str = "") -> str:
    value = input(prompt).strip()
    if value == "'exit":
        print("Operation cancelled.")
        sys.exit(0)
    return value


class DomainGenerator(GeneratorBase):
    """Generator for complete PDDL domains."""

    def generate(self, args):
        """Generate complete domain based on command line arguments."""
        if args.pipeline or args.actions:
            self._pipeline_generate(args)
        else:
            self._interactive_generate(args)

    def _pipeline_generate(self, args):
        """Non-interactive pipeline generation (existing behavior)."""
        domain_desc = self._load_description(args.desc)

        if not args.pipeline and not args.actions:
            raise CLIError(
                "[ERROR] Must specify either --pipeline or --actions",
                [
                    "Use --pipeline for automatic component generation",
                    "Or --actions to specify which actions to generate"
                ]
            )

        output_dir = None
        if args.output_dir or args.save_intermediate:
            output_dir = Path(args.output_dir if args.output_dir else ".")
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating domain from description...")
        print(f"Description: {domain_desc[:100]}..." if len(domain_desc) > 100 else f"Description: {domain_desc}")

        print("\n" + "=" * 60)
        print("Step 1: Generating types...")
        types = self._generate_types(domain_desc, output_dir, args.max_retries)

        print("\n" + "=" * 60)
        print("Step 2: Generating constants...")
        constants = self._generate_constants(domain_desc, types, output_dir, args.max_retries)

        print("\n" + "=" * 60)
        print("Step 3: Generating predicates...")
        predicates = self._generate_predicates(domain_desc, types, constants, output_dir, args.max_retries)

        print("\n" + "=" * 60)
        print("Step 4: Generating actions...")

        action_names = []
        if args.actions:
            action_names = [name.strip() for name in args.actions.split(",")]
        else:
            action_names = self._extract_action_names(domain_desc, types, args.max_retries)

        actions = self._generate_actions(
            domain_desc, action_names, types, constants, predicates,
            output_dir, args.max_retries
        )

        print("\n" + "=" * 60)
        print("Step 5: Generating complete PDDL domain...")

        domain_pddl = self._generate_domain_pddl(
            domain_desc=domain_desc,
            requirements=args.requirements,
            types=types,
            constants=constants,
            predicates=predicates,
            actions=actions
        )

        self.save_output(domain_pddl, args.output, "pddl")

        print("\n" + "=" * 60)
        print("✅ Domain generation complete!")

        if output_dir:
            print(f"Intermediate files saved to: {output_dir.resolve()}")

    # ------------------------------------------------------------------ #
    #  Interactive generation
    # ------------------------------------------------------------------ #

    def _confirm_stage(self, label: str, context: dict,
                       manual_func, llm_func):
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

            resp = _input_or_exit(f"\n  Is this correct? (y/N): ").strip().lower()
            if resp == "y":
                return result

            if manual_func:
                print(f"  {YELLOW}Restarting {label} entry...{RESET}")
            else:
                fix = _input_or_exit(f"  Describe what to fix: ").strip()
                if not fix:
                    return result
                context["_feedback"] = fix

    def _interactive_generate(self, args):
        """Interactive domain generation."""
        print()
        print(f"{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}  L2P Interactive Domain Generator{RESET}")
        print(f"{BOLD}{'=' * 60}{RESET}")
        print(f"  Type {YELLOW}'exit{RESET} at any prompt to quit")
        print()

        domain_name = self._prompt_domain_name()
        domain_desc = self._prompt_domain_description()
        requirements = self._prompt_requirements()

        context = {}

        # --- Types ---
        print(f"\n{BOLD}--- Types ---{RESET}")
        include, manual = self._prompt_component("types", default_include=True)
        if include:
            if manual:
                context["types"] = self._confirm_stage(
                    "types", context,
                    manual_func=lambda: self._manual_type_hierarchy(),
                    llm_func=None
                )
            else:
                context["types"] = self._confirm_stage(
                    "types", context,
                    manual_func=None,
                    llm_func=lambda feedback="":
                        self._generate_types_from_desc(domain_desc, args.max_retries, feedback)
                )
        else:
            context["types"] = None

        # --- Constants ---
        print(f"\n{BOLD}--- Constants ---{RESET}")
        include, manual = self._prompt_component("constants", default_include=False)
        if include:
            if manual:
                context["constants"] = self._confirm_stage(
                    "constants", context,
                    manual_func=lambda: self._manual_constants(context.get("types")),
                    llm_func=None
                )
            else:
                context["constants"] = self._confirm_stage(
                    "constants", context,
                    manual_func=None,
                    llm_func=lambda feedback="":
                        self._generate_constants_from_desc(
                            domain_desc, context.get("types"), args.max_retries, feedback
                        )
                )
        else:
            context["constants"] = None

        # --- Predicates ---
        print(f"\n{BOLD}--- Predicates ---{RESET}")
        include, manual = self._prompt_component("predicates", default_include=True)
        if include:
            if manual:
                context["predicates"] = self._confirm_stage(
                    "predicates", context,
                    manual_func=lambda: self._manual_predicates(
                        context.get("types"), context.get("constants")
                    ),
                    llm_func=None
                )
            else:
                context["predicates"] = self._confirm_stage(
                    "predicates", context,
                    manual_func=None,
                    llm_func=lambda feedback="":
                        self._generate_predicates_from_desc(
                            domain_desc, context.get("types"), context.get("constants"),
                            args.max_retries, feedback
                        )
                )
        else:
            context["predicates"] = None

        # --- Functions ---
        print(f"\n{BOLD}--- Functions ---{RESET}")
        include, manual = self._prompt_component("functions", default_include=False)
        if include:
            if manual:
                context["functions"] = self._confirm_stage(
                    "functions", context,
                    manual_func=lambda: self._manual_functions(
                        context.get("types"), context.get("constants")
                    ),
                    llm_func=None
                )
            else:
                context["functions"] = self._confirm_stage(
                    "functions", context,
                    manual_func=None,
                    llm_func=lambda feedback="":
                        self._generate_functions_from_desc(
                            domain_desc, context.get("types"), context.get("constants"),
                            context.get("predicates"), args.max_retries, feedback
                        )
                )
        else:
            context["functions"] = None

        # --- Actions ---
        print(f"\n{BOLD}--- Actions ---{RESET}")
        self._handle_actions_interactive(domain_desc, context, args.max_retries)

        # --- Assemble ---
        print(f"\n{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}  Assembling Domain{RESET}")
        print(f"{BOLD}{'=' * 60}{RESET}")

        domain_pddl = self._build_domain_pddl(
            domain_name=domain_name,
            requirements=requirements,
            context=context,
        )

        self._prompt_save(domain_pddl, domain_name)

    # ------------------------------------------------------------------ #
    #  Interactive helpers
    # ------------------------------------------------------------------ #

    def _prompt_domain_name(self) -> str:
        while True:
            name = _input_or_exit(f"{GREEN}Enter domain name:{RESET} ").strip().lower().replace(" ", "-")
            if name:
                return re.sub(r"[^a-z0-9-]", "", name)
            print("Domain name cannot be empty.")

    def _prompt_domain_description(self) -> str:
        print(f"\n{GREEN}Enter a brief description of your domain:{RESET}")
        print(f"  (This helps the LLM generate appropriate PDDL components)")
        desc = _input_or_exit().strip()
        return desc or "A general planning domain."

    def _prompt_requirements(self) -> str:
        print(f"\n{BOLD}PDDL Requirements (comma-separated, default: :strips,:typing){RESET}")
        print(f"  Available:")
        for req in PDDL_REQUIREMENTS:
            print(f"    {req}")
        user_input = _input_or_exit("Requirements: ").strip()
        if not user_input:
            return ":strips,:typing"
        parts = [r.strip() for r in user_input.split(",") if r.strip()]
        valid = {r for r in parts if r in PDDL_REQUIREMENTS}
        invalid = [r for r in parts if r not in PDDL_REQUIREMENTS]
        if invalid:
            print(f"  {YELLOW}Ignoring unknown: {', '.join(invalid)}{RESET}")
        return ",".join(valid) if valid else ":strips,:typing"

    def _prompt_component(self, name: str, default_include: bool) -> Tuple[bool, bool]:
        resp = _input_or_exit(f"Include {name}? ({'Y/n' if default_include else 'y/N'}): ").strip().lower()
        include = default_include if not resp else resp == "y"

        if not include:
            return False, False

        resp = _input_or_exit(f"  Configure {name} manually? (y/N): ").strip().lower()
        manual = resp == "y"
        return True, manual

    def _manual_type_hierarchy(self) -> list:
        """Interactive type hierarchy builder."""
        types = []
        print(f"  Enter types. Type name or {YELLOW}'done'{RESET} to finish.")
        print(f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.")
        print(f"  Parent defaults to {CYAN}object{RESET} (root).")

        state = 0
        name = parent = desc = ""
        while True:
            if state == 0:
                raw = _input_or_exit(f"\n  {GREEN}Type name:{RESET} ").strip().lower()
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return None
                if any(t["name"] == raw for t in types):
                    print(f"  {YELLOW}Type '{raw}' already exists.{RESET}")
                    continue
                name = raw
                state = 1
            elif state == 1:
                raw = _input_or_exit(f"    Parent (default: object): ").strip().lower()
                if raw == "..":
                    state = 0
                    continue
                parent = raw or "object"
                state = 2
            elif state == 2:
                raw = _input_or_exit(f"    Description (optional): ").strip()
                if raw == "..":
                    state = 1
                    continue
                desc = raw
                types.append({"name": name, "parent": parent, "desc": desc})
                print(f"  {GREEN}Added:{RESET} {name} : {parent}")
                self._show_type_tree(types)
                state = 0

        if not types:
            return None
        return types

    def _show_type_tree(self, types: list):
        """Display the type hierarchy as an indented tree."""
        tree = {"object": {"children": {}}}
        for t in types:
            parent = t["parent"]
            if parent not in tree:
                tree[parent] = {"children": {}}
            if t["name"] not in tree:
                tree[t["name"]] = {"children": {}}
            tree[parent]["children"][t["name"]] = tree[t["name"]]

        def _print(node, indent_level):
            for name, child in node.items():
                print(f"    {'  ' * indent_level}{CYAN}{name}{RESET}")
                _print(child["children"], indent_level + 1)

        print(f"  {BOLD}Current hierarchy:{RESET}")
        _print(tree, 0)

    def _manual_constants(self, types) -> dict:
        """Interactive constant builder."""
        available = self._collect_type_names(types)
        constants = {}
        print(f"  Available types: {CYAN}{', '.join(available)}{RESET}")
        print(f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.")
        print(f"  Enter constants. Name or {YELLOW}'done'{RESET} to finish.")

        state = 0
        name = ""
        while True:
            if state == 0:
                raw = _input_or_exit(f"\n  {GREEN}Constant name:{RESET} ").strip().lower()
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return constants if constants else None
                if raw in constants:
                    print(f"  {YELLOW}Constant '{raw}' already exists.{RESET}")
                    continue
                name = raw
                state = 1
            elif state == 1:
                raw = _input_or_exit(f"    Type: ").strip().lower()
                if raw == "..":
                    state = 0
                    continue
                if raw in available:
                    constants[name] = raw
                    print(f"  {GREEN}Added:{RESET} {name} - {raw}")
                    state = 0
                else:
                    print(f"  {YELLOW}Invalid type. Available: {', '.join(available)}{RESET}")

        return constants if constants else None

    def _manual_predicates(self, types, constants) -> list:
        """Interactive structured predicate builder."""
        available = self._collect_type_names(types)
        predicates = []
        print(f"  Available types: {CYAN}{', '.join(available)}{RESET}")
        print(f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.")
        print(f"  Enter predicates. Name or {YELLOW}'done'{RESET} to finish.")

        state = 0
        name = desc = ""
        params = {}

        while True:
            if state == 0:
                raw = _input_or_exit(f"\n  {GREEN}Predicate name:{RESET} ").strip().lower()
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return predicates if predicates else None
                if any(p["name"] == raw for p in predicates):
                    print(f"  {YELLOW}Predicate '{raw}' already exists.{RESET}")
                    continue
                name = raw
                params = {}
                state = 1

            elif state == 1:
                raw = _input_or_exit(f"    Description (optional): ").strip()
                if raw == "..":
                    state = 0
                    continue
                desc = raw
                state = 2

            elif state == 2:
                raw = _input_or_exit(f"    Add a parameter? (Y/n): ").strip().lower()
                if raw == "..":
                    state = 1
                    continue
                if raw == "n":
                    self._save_predicate(predicates, name, desc, params)
                    state = 0
                else:
                    state = 3

            elif state == 3:
                raw = _input_or_exit(f"      Parameter name (e.g. ?b): ").strip()
                if not raw:
                    continue
                if raw == "..":
                    if params:
                        state = 2
                    else:
                        state = 1
                    continue
                pname = raw if raw.startswith("?") else f"?{raw}"
                state = 4
                _param_name_holder = pname

            elif state == 4:
                raw = _input_or_exit(f"      Type (default: object): ").strip().lower()
                if raw == "..":
                    state = 3
                    continue
                ptype = raw if raw in available else "object"
                params[_param_name_holder] = ptype
                print(f"      {GREEN}Added parameter:{RESET} {_param_name_holder} - {ptype}")
                state = 2

        return predicates if predicates else None

    def _manual_functions(self, types, constants) -> list:
        """Interactive structured function builder."""
        available = self._collect_type_names(types)
        functions = []
        print(f"  Available types: {CYAN}{', '.join(available)}{RESET}")
        print(f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.")
        print(f"  Enter functions. Name or {YELLOW}'done'{RESET} to finish.")

        state = 0
        name = desc = ""
        params = {}

        while True:
            if state == 0:
                raw = _input_or_exit(f"\n  {GREEN}Function name:{RESET} ").strip().lower()
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return functions if functions else None
                if any(f["name"] == raw for f in functions):
                    print(f"  {YELLOW}Function '{raw}' already exists.{RESET}")
                    continue
                name = raw
                params = {}
                state = 1

            elif state == 1:
                raw = _input_or_exit(f"    Description (optional): ").strip()
                if raw == "..":
                    state = 0
                    continue
                desc = raw
                state = 2

            elif state == 2:
                raw = _input_or_exit(f"    Add a parameter? (Y/n): ").strip().lower()
                if raw == "..":
                    state = 1
                    continue
                if raw == "n":
                    self._save_predicate(functions, name, desc, params)
                    state = 0
                else:
                    state = 3

            elif state == 3:
                raw = _input_or_exit(f"      Parameter name (e.g. ?b): ").strip()
                if not raw:
                    continue
                if raw == "..":
                    if params:
                        state = 2
                    else:
                        state = 1
                    continue
                pname = raw if raw.startswith("?") else f"?{raw}"
                state = 4
                _param_name_holder = pname

            elif state == 4:
                raw = _input_or_exit(f"      Type (default: object): ").strip().lower()
                if raw == "..":
                    state = 3
                    continue
                ptype = raw if raw in available else "object"
                params[_param_name_holder] = ptype
                print(f"      {GREEN}Added parameter:{RESET} {_param_name_holder} - {ptype}")
                state = 2

        return functions if functions else None

    def _save_predicate(self, collection: list, name: str, desc: str, params: dict):
        """Build and append a Predicate dict, then print it."""
        parts = [f"{p} - {t}" for p, t in params.items()]
        raw = f"({name} {' '.join(parts)})" if parts else f"({name})"
        entry = {
            "name": name,
            "desc": desc,
            "raw": raw,
            "params": dict(params),
            "clean": raw,
        }
        collection.append(entry)
        print(f"  {GREEN}Added:{RESET} {raw}")

    def _handle_actions_interactive(self, domain_desc: str, context: dict, max_retries: int):
        """Interactive action definition with optional LLM extraction."""
        while True:
            print(f"  How do you want to define actions?")
            print(f"    {CYAN}1{RESET} - Let the LLM extract action names from the domain description")
            print(f"    {CYAN}2{RESET} - Specify them manually")
            print(f"  Type {YELLOW}'..'{RESET} to go back, {YELLOW}'skip'{RESET} to skip actions.")

            choice = _input_or_exit(f"  Choice (default: 1): ").strip()
            if choice == ".." or choice == "skip":
                print(f"  {YELLOW}Skipping actions.{RESET}")
                context["actions"] = []
                return

            if not choice or choice == "1":
                action_names, action_descs = self._interactive_extract_actions(
                    domain_desc, context, max_retries
                )
                if action_names is None:
                    continue
                break
            elif choice == "2":
                action_names, action_descs = self._interactive_manual_actions()
                break
            else:
                print(f"  {YELLOW}Invalid choice. Enter 1 or 2.{RESET}")

        if not action_names:
            print(f"  {YELLOW}No actions defined.{RESET}")
            context["actions"] = []
            return

        context["actions"] = self._generate_actions_from_list(
            domain_desc, action_names, action_descs, context, max_retries
        )

    def _interactive_extract_actions(self, domain_desc: str, context: dict, max_retries: int) -> Tuple[Optional[list], Optional[list]]:
        """Use LLM to extract action names from domain description.
        Returns (None, None) to signal going back to the choice menu."""
        from l2p import DomainBuilder
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        template = self.template_manager.get_template("extract_nl_actions.txt", "domain")

        print(f"\n  Extracting action names from description...")
        result = domain_builder.extract_nl_actions(
            model=llm,
            domain_desc=domain_desc,
            prompt_template=template,
            types=context.get("types"),
            max_retries=max_retries
        )
        nl_actions, llm_output = result

        if not nl_actions:
            print(f"  {YELLOW}No actions extracted.{RESET}")
            return None, None

        action_names = list(nl_actions.keys())
        action_descs = list(nl_actions.values())

        print(f"\n  LLM extracted {len(action_names)} actions:")
        for i, (name, desc) in enumerate(zip(action_names, action_descs), 1):
            d = f" - {desc}" if desc else ""
            print(f"    {i}. {name}{d}")

        resp = _input_or_exit(f"  Accept these? ({YELLOW}Y{RESET}/n/{YELLOW}..{RESET}): ").strip().lower()
        if resp == "..":
            return None, None
        if resp and resp != "y":
            return self._interactive_manual_actions()

        return action_names, action_descs

    def _interactive_manual_actions(self) -> Tuple[list, list]:
        """Manually specify action names and descriptions."""
        names = []
        descs = []
        print(f"  Enter actions. Name or {YELLOW}'done'{RESET} to finish.")
        print(f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.")

        state = 0
        while True:
            if state == 0:
                raw = _input_or_exit(f"\n  {GREEN}Action name:{RESET} ").strip().lower().replace(" ", "-")
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return names, descs
                if not re.match(r"^[a-z][a-z0-9_-]*$", raw):
                    print(f"  {YELLOW}Invalid action name. Use lowercase letters, numbers, hyphens, underscores.{RESET}")
                    continue
                if raw in names:
                    print(f"  {YELLOW}Action '{raw}' already added.{RESET}")
                    continue
                state = 1
                _action_name_holder = raw
            elif state == 1:
                desc = _input_or_exit(f"    Description (optional): ").strip()
                if desc == "..":
                    state = 0
                    continue
                names.append(_action_name_holder)
                descs.append(desc)
                print(f"  {GREEN}Added:{RESET} {_action_name_holder}")
                state = 0

        return names, descs

    def _prompt_save(self, content: str, domain_name: str):
        """Prompt for output file and save."""
        default_path = f"{domain_name}-domain.pddl"
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
                print(f"  {YELLOW}Skipping file save.{RESET}")
                print(f"\n{BOLD}Generated domain:{RESET}")
                print(content)
                return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"\n{GREEN}✅ Domain saved to: {output_path.resolve()}{RESET}")

    def _display_component(self, label: str, items):
        if items is None:
            print(f"  ({label} omitted)")
            return
        print(f"\n  {BOLD}{label.capitalize()}:{RESET}")
        if label == "types":
            for t in items:
                if isinstance(t, dict):
                    name = next(iter(t))
                    p = t.get("parent", "object")
                    print(f"    {CYAN}{name}{RESET} : {p}")
        elif label == "constants":
            if isinstance(items, dict):
                for name, typ in items.items():
                    print(f"    {CYAN}{name}{RESET} - {typ}")
        elif label in ("predicates", "functions"):
            for p in items:
                if isinstance(p, dict):
                    print(f"    {CYAN}{p.get('raw', p.get('name', '?'))}{RESET}")
        else:
            if isinstance(items, list):
                print(f"    ({len(items)} {label})")
            elif isinstance(items, dict):
                print(f"    ({len(items)} {label})")

    def _collect_type_names(self, types) -> list:
        if not types:
            return ["object"]
        names = ["object"]
        for t in types:
            if isinstance(t, dict):
                names.append(t.get("name", ""))
        return [n for n in names if n]

    # ------------------------------------------------------------------ #
    #  LLM-based generation wrappers (with context support)
    # ------------------------------------------------------------------ #

    def _generate_types_from_desc(self, domain_desc: str, max_retries: int, feedback: str = ""):
        from l2p import DomainBuilder
        llm = self.load_llm()
        builder = DomainBuilder()
        template = self.template_manager.get_template("formalize_type_hierarchy.txt", "domain")
        prompt = domain_desc
        if feedback:
            prompt = f"{domain_desc}\n\n[Feedback to apply to the generated types]\n{feedback}"
            print(f"  Re-generating types with your feedback...")
        else:
            print(f"  Generating types from description...")
        result = builder.formalize_type_hierarchy(
            model=llm, domain_desc=prompt,
            prompt_template=template, max_retries=max_retries
        )
        types_result, llm_output, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return types_result

    def _generate_constants_from_desc(self, domain_desc: str, types, max_retries: int, feedback: str = ""):
        from l2p import DomainBuilder
        llm = self.load_llm()
        builder = DomainBuilder()
        template = self.template_manager.get_template("formalize_constants.txt", "domain")
        prompt = domain_desc
        if feedback:
            prompt = f"{domain_desc}\n\n[Feedback to apply to the generated constants]\n{feedback}"
            print(f"  Re-generating constants with your feedback...")
        else:
            print(f"  Generating constants from description...")
        result = builder.formalize_constants(
            model=llm, domain_desc=prompt,
            prompt_template=template, types=types, max_retries=max_retries
        )
        const_result, llm_output, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return const_result

    def _generate_predicates_from_desc(self, domain_desc: str, types, constants, max_retries: int, feedback: str = ""):
        from l2p import DomainBuilder
        llm = self.load_llm()
        builder = DomainBuilder()
        template = self.template_manager.get_template("formalize_predicates.txt", "domain")
        prompt = domain_desc
        if feedback:
            prompt = f"{domain_desc}\n\n[Feedback to apply to the generated predicates]\n{feedback}"
            print(f"  Re-generating predicates with your feedback...")
        else:
            print(f"  Generating predicates from description...")
        result = builder.formalize_predicates(
            model=llm, domain_desc=prompt,
            prompt_template=template, types=types,
            constants=constants, max_retries=max_retries
        )
        pred_result, llm_output, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return pred_result

    def _generate_functions_from_desc(self, domain_desc: str, types, constants, predicates, max_retries: int, feedback: str = ""):
        from l2p import DomainBuilder
        llm = self.load_llm()
        builder = DomainBuilder()
        template = self.template_manager.get_template("formalize_functions.txt", "domain")
        prompt = domain_desc
        if feedback:
            prompt = f"{domain_desc}\n\n[Feedback to apply to the generated functions]\n{feedback}"
            print(f"  Re-generating functions with your feedback...")
        else:
            print(f"  Generating functions from description...")
        result = builder.formalize_functions(
            model=llm, domain_desc=prompt,
            prompt_template=template, types=types,
            constants=constants, predicates=predicates,
            max_retries=max_retries
        )
        func_result, llm_output, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return func_result

    def _generate_actions_from_list(
        self, domain_desc: str, action_names: list, action_descs: list,
        context: dict, max_retries: int
    ) -> list:
        """Generate full PDDL actions with new-predicate merging."""
        from l2p import DomainBuilder
        llm = self.load_llm()
        builder = DomainBuilder()
        actions = []
        all_new_predicates = []

        total = len(action_names)
        for i, (name, desc) in enumerate(zip(action_names, action_descs), 1):
            print(f"  [{i}/{total}] Generating action: {name}")
            template = self.template_manager.get_template("formalize_pddl_action.txt", "domain")
            result = builder.formalize_pddl_action(
                model=llm, domain_desc=domain_desc,
                prompt_template=template, action_name=name,
                action_desc=desc or f"{name} action",
                types=context.get("types"),
                constants=context.get("constants"),
                predicates=context.get("predicates"),
                functions=context.get("functions"),
                max_retries=max_retries
            )
            action_result, new_preds, llm_output, validation_info = result
            if validation_info and not validation_info[0]:
                print(f"    {YELLOW}Validation: {validation_info[1]}{RESET}")
            actions.append(action_result)
            if new_preds:
                existing_names = {
                    p["name"] for p in (context.get("predicates") or [])
                }
                for np in new_preds:
                    if np["name"] not in existing_names:
                        all_new_predicates.append(np)
                        existing_names.add(np["name"])

        if all_new_predicates:
            if context.get("predicates") is None:
                context["predicates"] = []
            context["predicates"].extend(all_new_predicates)
            print(f"  {GREEN}Merged {len(all_new_predicates)} new predicates from actions.{RESET}")

        return actions

    def _build_domain_pddl(self, domain_name: str, requirements: str, context: dict) -> str:
        """Assemble final domain PDDL from context."""
        from l2p import DomainBuilder
        builder = DomainBuilder()
        req_list = [r.strip() for r in requirements.split(",") if r.strip()]
        return builder.generate_domain(
            domain_name=domain_name,
            requirements=req_list,
            types=context.get("types"),
            constants=context.get("constants"),
            predicates=context.get("predicates"),
            functions=context.get("functions"),
            actions=context.get("actions", []),
        )

    # ------------------------------------------------------------------ #
    #  Non-interactive pipeline helpers (unchanged)
    # ------------------------------------------------------------------ #

    def _generate_types(self, domain_desc: str, output_dir: Optional[Path], max_retries: int) -> Any:
        from l2p import DomainBuilder
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        template_content = self.template_manager.get_template("formalize_type_hierarchy.txt", "domain")
        print("  Using hierarchical type generation...")
        result = domain_builder.formalize_type_hierarchy(
            model=llm, domain_desc=domain_desc,
            prompt_template=template_content, max_retries=max_retries
        )
        types_result, llm_output, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  [WARNING] Validation warning: {validation_info[1]}")
        if output_dir:
            types_file = output_dir / "types.json"
            with open(types_file, 'w') as f:
                json.dump(types_result, f, indent=2)
            print(f"  Types saved to: {types_file}")
        return types_result

    def _generate_constants(self, domain_desc: str, types: Any, output_dir: Optional[Path], max_retries: int) -> Any:
        from l2p import DomainBuilder
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        template_content = self.template_manager.get_template("formalize_constants.txt", "domain")
        print("  Generating constants...")
        result = domain_builder.formalize_constants(
            model=llm, domain_desc=domain_desc,
            prompt_template=template_content, types=types, max_retries=max_retries
        )
        constants_result, llm_output, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  [WARNING] Validation warning: {validation_info[1]}")
        if output_dir and constants_result:
            constants_file = output_dir / "constants.json"
            with open(constants_file, 'w') as f:
                json.dump(constants_result, f, indent=2)
            print(f"  Constants saved to: {constants_file}")
        return constants_result

    def _generate_predicates(self, domain_desc: str, types: Any, constants: Any,
                             output_dir: Optional[Path], max_retries: int) -> Any:
        from l2p import DomainBuilder
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        template_content = self.template_manager.get_template("formalize_predicates.txt", "domain")
        print("  Generating predicates...")
        result = domain_builder.formalize_predicates(
            model=llm, domain_desc=domain_desc,
            prompt_template=template_content, types=types,
            constants=constants, max_retries=max_retries
        )
        predicates_result, llm_output, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  [WARNING] Validation warning: {validation_info[1]}")
        if output_dir:
            predicates_file = output_dir / "predicates.json"
            serializable_predicates = []
            for pred in predicates_result:
                if hasattr(pred, 'to_dict'):
                    serializable_predicates.append(pred.to_dict())
                elif isinstance(pred, dict):
                    serializable_predicates.append(pred)
                else:
                    serializable_predicates.append({"raw": str(pred)})
            with open(predicates_file, 'w') as f:
                json.dump(serializable_predicates, f, indent=2)
            print(f"  Predicates saved to: {predicates_file}")
        return predicates_result

    def _extract_action_names(self, domain_desc: str, types: Any, max_retries: int) -> List[str]:
        from l2p import DomainBuilder
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        template_content = self.template_manager.get_template("extract_nl_actions.txt", "domain")
        print("  Extracting action names from description...")
        result = domain_builder.extract_nl_actions(
            model=llm, domain_desc=domain_desc,
            prompt_template=template_content, types=types, max_retries=max_retries
        )
        nl_actions, llm_output = result
        if not nl_actions:
            print("  [WARNING] No actions extracted, using default action names")
            return ["action1", "action2", "action3"]
        action_names = list(nl_actions.keys())
        print(f"  Extracted {len(action_names)} actions: {', '.join(action_names)}")
        return action_names

    def _generate_actions(self, domain_desc: str, action_names: List[str], types: Any,
                          constants: Any, predicates: Any, output_dir: Optional[Path],
                          max_retries: int) -> List[Any]:
        from l2p import DomainBuilder
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        actions = []
        for i, action_name in enumerate(action_names, 1):
            print(f"  [{i}/{len(action_names)}] Generating action: {action_name}")
            template_content = self.template_manager.get_template("formalize_pddl_action.txt", "domain")
            result = domain_builder.formalize_pddl_action(
                model=llm, domain_desc=domain_desc,
                prompt_template=template_content, action_name=action_name,
                action_desc=f"{action_name} action for the domain",
                types=types, constants=constants, predicates=predicates,
                max_retries=max_retries
            )
            action_result, new_predicates, llm_output, validation_info = result
            if validation_info and not validation_info[0]:
                print(f"    [WARNING] Validation warning: {validation_info[1]}")
            actions.append(action_result)
            if output_dir:
                action_file = output_dir / f"action_{action_name}.json"
                with open(action_file, 'w') as f:
                    json.dump(action_result, f, indent=2)
        return actions

    def _generate_domain_pddl(self, domain_desc: str, requirements: str, types: Any,
                              constants: Any, predicates: Any, actions: List[Any]) -> str:
        from l2p import DomainBuilder
        domain_builder = DomainBuilder()
        requirements_list = [req.strip() for req in requirements.split(",") if req.strip()]
        domain_pddl = domain_builder.generate_domain(
            domain_name=self._extract_domain_name(domain_desc),
            requirements=requirements_list,
            types=types, constants=constants,
            predicates=predicates, actions=actions
        )
        return domain_pddl

    def _extract_domain_name(self, domain_desc: str) -> str:
        words = domain_desc.split()[:3]
        name = "-".join(words).lower()
        name = re.sub(r'[^a-z0-9-]', '', name)
        return name or "generated-domain"

    def _load_description(self, desc_input: str) -> str:
        desc_path = Path(desc_input)
        if desc_path.exists() and desc_path.is_file():
            try:
                return desc_path.read_text().strip()
            except Exception as e:
                raise CLIError(
                    f"[ERROR] Failed to read description file: {e}",
                    ["Check file permissions and encoding"]
                )
        return desc_input
