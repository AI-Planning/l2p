"""
Domain generator for L2P CLI.

Generates a complete PDDL domain using the new DomainBuilder API.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from l2p.utils.pddl_types import (
    DomainDetails,
    PDDLType,
    Constant,
    Predicate,
    Function,
    Action,
    Parameter,
)
from l2p.cli.commands.generate import GeneratorBase
from l2p.cli.utils.errors import handle_error
from l2p.cli.utils.helpers import _input_or_exit, BOLD, GREEN, CYAN, YELLOW, RESET


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "domain",
        help="Generate complete PDDL domain",
        description="Generate complete PDDL domain using a pipeline approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  l2p generate domain
  l2p generate domain --max-retries 5
        """,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for LLM (default: 3)",
    )
    parser.set_defaults(func=generate_domain_command)


def generate_domain_command(args):
    try:
        DomainGenerator().generate(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


class DomainGenerator(GeneratorBase):
    """Generator for a complete PDDL domain."""

    def generate(self, args):
        print(
            f"{BOLD}{'=' * 60}{RESET}\n"
            f"{BOLD}  L2P Interactive Domain Generator{RESET}\n"
            f"{BOLD}{'=' * 60}{RESET}\n"
            f"  Type {YELLOW}/exit{RESET} at any prompt to quit\n"
        )

        domain_name = self._prompt_domain_name()
        domain_desc = self._prompt_domain_desc()
        context: Dict[str, Any] = {}

        # --- types ---
        print(f"\n{BOLD}--- Types ---{RESET}")
        include, manual = self._prompt_component("types", default_include=True)
        if include:
            types_desc = _input_or_exit("  Describe the types (optional): ").strip()
            context["types"] = self._confirm_stage(
                label="types",
                llm_func=lambda feedback="": self._generate_types(
                    domain_desc, args.max_retries, types_desc, feedback
                ),
                manual_func=self._manual_types if manual else None,
            )
        else:
            context["types"] = []

        # --- constants ---
        print(f"\n{BOLD}--- Constants ---{RESET}")
        include, manual = self._prompt_component("constants", default_include=False)
        if include:
            const_desc = _input_or_exit("  Describe the constants (optional): ").strip()
            context["constants"] = self._confirm_stage(
                label="constants",
                llm_func=lambda feedback="": self._generate_constants(
                    domain_desc,
                    context["types"],
                    args.max_retries,
                    const_desc,
                    feedback,
                ),
                manual_func=self._manual_constants if manual else None,
            )
        else:
            context["constants"] = []

        # --- predicates ---
        print(f"\n{BOLD}--- Predicates ---{RESET}")
        include, manual = self._prompt_component("predicates", default_include=True)
        if include:
            pred_desc = _input_or_exit("  Describe the predicates (optional): ").strip()
            context["predicates"] = self._confirm_stage(
                label="predicates",
                llm_func=lambda feedback="": self._generate_predicates(
                    domain_desc,
                    context["types"],
                    context["constants"],
                    args.max_retries,
                    pred_desc,
                    feedback,
                ),
                manual_func=self._manual_predicates if manual else None,
            )
        else:
            context["predicates"] = []

        # --- functions ---
        print(f"\n{BOLD}--- Functions ---{RESET}")
        include, manual = self._prompt_component("functions", default_include=False)
        if include:
            func_desc = _input_or_exit("  Describe the functions (optional): ").strip()
            context["functions"] = self._confirm_stage(
                label="functions",
                llm_func=lambda feedback="": self._generate_functions(
                    domain_desc,
                    context["types"],
                    context["constants"],
                    context["predicates"],
                    args.max_retries,
                    func_desc,
                    feedback,
                ),
                manual_func=self._manual_functions if manual else None,
            )
        else:
            context["functions"] = []

        # --- actions ---
        print(f"\n{BOLD}--- Actions ---{RESET}")
        self._handle_actions_interactive(domain_desc, context, args.max_retries)

        # --- assemble domain ---
        print(
            f"\n\n{BOLD}{'=' * 60}{RESET}"
            f"\n{BOLD}  Assembling Domain{RESET}"
            f"\n{BOLD}{'=' * 60}{RESET}"
        )

        details = DomainDetails(
            name=domain_name,
            types=context.get("types") or [],
            constants=context.get("constants") or [],
            predicates=context.get("predicates") or [],
            functions=context.get("functions") or [],
            actions=context.get("actions") or [],
        )
        domain_pddl = self.domain_builder.generate_domain(details)

        print(
            f"\nOUTPUT:\n\n{BOLD}{'=' * 60}{RESET}\n"
            f"{domain_pddl}\n"
            f"{BOLD}{'=' * 60}{RESET}"
        )
        self._prompt_save(domain_pddl, domain_name)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _prompt_domain_name(self) -> str:
        while True:
            name = (
                _input_or_exit(f"{GREEN}Enter domain name:{RESET} ")
                .strip()
                .lower()
                .replace(" ", "-")
            )
            if name:
                return re.sub(r"[^a-z0-9-]", "", name)
            print("Domain name cannot be empty.")

    def _prompt_domain_desc(self) -> str:
        print(f"\n{GREEN}Enter a brief description of your domain:{RESET}")
        desc = _input_or_exit().strip()
        return desc or "A general planning domain."

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
    # Confirmation / fix loop
    # ------------------------------------------------------------------

    def _confirm_stage(
        self,
        label: str,
        llm_func,
        manual_func=None,
    ):
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

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_component(self, label: str, items):
        if items is None:
            print(f"  ({label} omitted)")
            return
        print(f"\n  {BOLD}{label.capitalize()}:{RESET}")
        if label == "types":
            for t in items:
                print(f"    {CYAN}{t.name}{RESET} - {t.parent}")
        elif label == "constants":
            for c in items:
                print(f"    {CYAN}{c.name}{RESET} - {c.type}")
        elif label in ("predicates", "functions"):
            for item in items:
                params = " ".join(f"{p.variable} - {p.type}" for p in item.params)
                print(f"    {CYAN}{item.name}{RESET}({params})")
        elif label == "actions":
            for a in items:
                print(f"    {CYAN}{a.name}{RESET}")
        else:
            print(f"    ({len(items)} {label})")

    # ------------------------------------------------------------------
    # Manual data entry (returns Pydantic models)
    # ------------------------------------------------------------------

    def _collect_type_names(self, types: List[PDDLType]) -> List[str]:
        names = ["object"]
        for t in types:
            names.append(t.name)
        return names

    def _manual_types(self) -> Optional[List[PDDLType]]:
        types: List[PDDLType] = []
        print(
            "  Enter types. Type name or 'done' to finish.\n"
            "  Use '..' to go back. Parent defaults to object.\n"
        )
        state = 0
        name = parent = ""
        while True:
            if state == 0:
                raw = _input_or_exit(f"\n  {GREEN}Type name:{RESET} ").strip().lower()
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return None
                if any(t.name == raw for t in types):
                    print(f"  {YELLOW}Type '{raw}' already exists.{RESET}")
                    continue
                name = raw
                state = 1
            elif state == 1:
                raw = _input_or_exit("    Parent (default: object): ").strip().lower()
                if raw == "..":
                    state = 0
                    continue
                parent = raw or "object"
                types.append(PDDLType(name=name, parent=parent))
                print(f"  {GREEN}Added:{RESET} {name} : {parent}")
                state = 0
        return types if types else None

    def _manual_constants(self) -> Optional[List[Constant]]:
        available = self._collect_type_names([])
        constants: List[Constant] = []
        print(f"  Available types: {CYAN}{', '.join(available)}{RESET}")
        print("  Enter constants. Name or 'done' to finish.\n")
        state = 0
        name = ""
        while True:
            if state == 0:
                raw = (
                    _input_or_exit(f"\n  {GREEN}Constant name:{RESET} ").strip().lower()
                )
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return constants if constants else None
                if any(c.name == raw for c in constants):
                    print(f"  {YELLOW}Constant '{raw}' already exists.{RESET}")
                    continue
                name = raw
                state = 1
            elif state == 1:
                t = _input_or_exit("    Type: ").strip().lower()
                if t == "..":
                    state = 0
                    continue
                constants.append(Constant(name=name, type=t if t else "object"))
                print(f"  {GREEN}Added:{RESET} {name} - {t or 'object'}")
                state = 0
        return constants if constants else None

    def _manual_predicates(self) -> Optional[List[Predicate]]:
        predicates: List[Predicate] = []
        print("  Enter predicates. Name or 'done' to finish.\n")
        state = 0
        name = ""
        params: List[Parameter] = []
        while True:
            if state == 0:
                raw = (
                    _input_or_exit(f"\n  {GREEN}Predicate name:{RESET} ")
                    .strip()
                    .lower()
                )
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return predicates if predicates else None
                if any(p.name == raw for p in predicates):
                    print(f"  {YELLOW}Predicate '{raw}' already exists.{RESET}")
                    continue
                name = raw
                params = []
                state = 1
            elif state == 1:
                resp = _input_or_exit("    Add a parameter? (Y/n): ").strip().lower()
                if resp == "..":
                    state = 0
                    continue
                if resp == "n":
                    predicates.append(Predicate(name=name, params=params))
                    print(f"  {GREEN}Added:{RESET} {name}")
                    state = 0
                else:
                    state = 2
            elif state == 2:
                raw = _input_or_exit("      Variable (e.g. ?b): ").strip()
                if not raw or raw == "..":
                    state = 1
                    continue
                var = raw if raw.startswith("?") else f"?{raw}"
                state = 3
                _v = var
            elif state == 3:
                t = _input_or_exit("      Type (default: object): ").strip().lower()
                if t == "..":
                    state = 2
                    continue
                params.append(Parameter(variable=_v, type=t if t else "object"))
                print(f"      {GREEN}Added parameter:{RESET} {_v} - {t or 'object'}")
                state = 1
        return predicates if predicates else None

    def _manual_functions(self) -> Optional[List[Function]]:
        functions: List[Function] = []
        print("  Enter functions. Name or 'done' to finish.\n")
        state = 0
        name = ""
        params: List[Parameter] = []
        while True:
            if state == 0:
                raw = (
                    _input_or_exit(f"\n  {GREEN}Function name:{RESET} ").strip().lower()
                )
                if not raw:
                    continue
                if raw == "done":
                    break
                if raw == "..":
                    return functions if functions else None
                if any(f.name == raw for f in functions):
                    print(f"  {YELLOW}Function '{raw}' already exists.{RESET}")
                    continue
                name = raw
                params = []
                state = 1
            elif state == 1:
                resp = _input_or_exit("    Add a parameter? (Y/n): ").strip().lower()
                if resp == "..":
                    state = 0
                    continue
                if resp == "n":
                    functions.append(Function(name=name, params=params))
                    print(f"  {GREEN}Added:{RESET} {name}")
                    state = 0
                else:
                    state = 2
            elif state == 2:
                raw = _input_or_exit("      Variable (e.g. ?r): ").strip()
                if not raw or raw == "..":
                    state = 1
                    continue
                var = raw if raw.startswith("?") else f"?{raw}"
                state = 3
                _v = var
            elif state == 3:
                t = _input_or_exit("      Type (default: object): ").strip().lower()
                if t == "..":
                    state = 2
                    continue
                params.append(Parameter(variable=_v, type=t if t else "object"))
                print(f"      {GREEN}Added parameter:{RESET} {_v} - {t or 'object'}")
                state = 1
        return functions if functions else None

    # ------------------------------------------------------------------
    # Action handling  (single call to generate ALL actions at once)
    # ------------------------------------------------------------------

    def _handle_actions_interactive(
        self, domain_desc: str, context: dict, max_retries: int
    ):
        while True:
            print(
                "  How do you want to define actions?\n"
                f"    {CYAN}1{RESET} - LLM extracts action names, you confirm, then generates all\n"
                f"    {CYAN}2{RESET} - You type action names, then the LLM generates all\n"
                f"  Type {YELLOW}'..'{RESET} to go back, {YELLOW}'skip'{RESET} to skip.\n"
            )
            choice = _input_or_exit("  Choice (default: 1): ").strip()
            if choice in ("..", "skip"):
                context["actions"] = []
                return

            action_names: Optional[List[str]] = None

            if not choice or choice == "1":
                action_names = self._llm_extract_names(
                    domain_desc, context, max_retries
                )
            elif choice == "2":
                action_names = self._manual_action_names()
            else:
                print(f"  {YELLOW}Invalid choice.{RESET}")
                continue

            if action_names is None:
                continue
            if not action_names:
                print(f"  {YELLOW}No actions defined.{RESET}")
                context["actions"] = []
                return

            # --- generate all actions in a single call ---
            print(f"  Generating {len(action_names)} action(s) in one call...")
            actions = self._confirm_stage(
                label="actions",
                llm_func=lambda: self._generate_all_actions(
                    domain_desc,
                    action_names,
                    context,
                    max_retries,
                ),
            )
            if actions is not None:
                context["actions"] = actions
            else:
                context["actions"] = []
            return

    def _llm_extract_names(
        self, domain_desc: str, context: dict, max_retries: int
    ) -> Optional[List[str]]:
        """Extract action names from the NL description and confirm with the user."""
        print("\n  Extracting action names from description...")
        try:
            result, _ = self.domain_builder.extract_nl(
                model=self.llm,
                template_key="nl_actions",
                description=domain_desc,
                types=context.get("types"),
                constants=context.get("constants"),
                predicates=context.get("predicates"),
                functions=context.get("functions"),
                max_retries=max_retries,
            )
        except Exception as e:
            print(f"  {YELLOW}Failed to extract actions: {e}{RESET}")
            return None

        if not result:
            print(f"  {YELLOW}No actions extracted.{RESET}")
            return None

        names = list(result.keys())
        print(f"  LLM extracted {len(names)} action(s):")
        for i, n in enumerate(names, 1):
            desc = result.get(n, "")
            if desc:
                print(f"    {i}. {n} — {desc}")
            else:
                print(f"    {i}. {n}")

        resp = _input_or_exit(f"  Accept these names? (Y/n/..): ").strip().lower()
        if resp == "..":
            return None
        if resp and resp != "y":
            return self._manual_action_names()
        return names

    def _manual_action_names(self) -> Optional[List[str]]:
        """Let the user type action names manually."""
        names: List[str] = []
        print("  Enter action names. Name or 'done' to finish.\n")
        while True:
            raw = (
                _input_or_exit(f"\n  {GREEN}Action name:{RESET} ")
                .strip()
                .lower()
                .replace(" ", "-")
            )
            if not raw:
                continue
            if raw == "done":
                break
            if raw == "..":
                return names if names else None
            if not re.match(r"^[a-z][a-z0-9_-]*$", raw):
                print(f"  {YELLOW}Invalid name.{RESET}")
                continue
            if raw in names:
                print(f"  {YELLOW}'{raw}' already added.{RESET}")
                continue
            names.append(raw)
            print(f"  {GREEN}Added:{RESET} {raw}")
        return names if names else None

    def _generate_all_actions(
        self,
        domain_desc: str,
        action_names: List[str],
        context: dict,
        max_retries: int,
    ) -> Optional[List[Action]]:
        """Generate all actions in a single LLM call."""
        print("  Generating all actions from description...")
        try:
            result, _ = self.domain_builder.formalize_component(
                model=self.llm,
                component_class=Action,
                description=domain_desc,
                types=context.get("types"),
                constants=context.get("constants"),
                predicates=context.get("predicates"),
                functions=context.get("functions"),
                max_retries=max_retries,
            )
            actions = result.get(Action, [])
            if actions:
                print(f"  {GREEN}Generated {len(actions)} action(s).{RESET}")
            else:
                print(f"  {YELLOW}No actions were generated.{RESET}")
            return actions
        except Exception as e:
            print(f"  {YELLOW}Failed to generate actions: {e}{RESET}")
            return None

    # ------------------------------------------------------------------
    # LLM-based generation wrappers
    # ------------------------------------------------------------------

    def _generate_types(
        self,
        domain_desc: str,
        max_retries: int,
        comp_desc: str = "",
        feedback: str = "",
    ) -> List[PDDLType]:
        desc = domain_desc
        if comp_desc:
            desc = f"{domain_desc}\n\n[Additional context for types]\n{comp_desc}"
        if feedback:
            desc = f"{desc}\n\n[Feedback to apply to the generated types]\n{feedback}"
        what = "Re-generating" if feedback else "Generating"
        print(f"  {what} types from description...")
        result, _ = self.domain_builder.formalize_component(
            model=self.llm,
            component_class=PDDLType,
            description=desc,
            max_retries=max_retries,
        )
        return result.get(PDDLType, [])

    def _generate_constants(
        self,
        domain_desc: str,
        types: List[PDDLType],
        max_retries: int,
        comp_desc: str = "",
        feedback: str = "",
    ) -> List[Constant]:
        desc = domain_desc
        if comp_desc:
            desc = f"{domain_desc}\n\n[Additional context for constants]\n{comp_desc}"
        if feedback:
            desc = f"{desc}\n\n[Feedback to apply]\n{feedback}"
        what = "Re-generating" if feedback else "Generating"
        print(f"  {what} constants from description...")
        result, _ = self.domain_builder.formalize_component(
            model=self.llm,
            component_class=Constant,
            description=desc,
            types=types,
            max_retries=max_retries,
        )
        return result.get(Constant, [])

    def _generate_predicates(
        self,
        domain_desc: str,
        types: List[PDDLType],
        constants: List[Constant],
        max_retries: int,
        comp_desc: str = "",
        feedback: str = "",
    ) -> List[Predicate]:
        desc = domain_desc
        if comp_desc:
            desc = f"{domain_desc}\n\n[Additional context]\n{comp_desc}"
        if feedback:
            desc = f"{desc}\n\n[Feedback]\n{feedback}"
        what = "Re-generating" if feedback else "Generating"
        print(f"  {what} predicates from description...")
        result, _ = self.domain_builder.formalize_component(
            model=self.llm,
            component_class=Predicate,
            description=desc,
            types=types,
            constants=constants,
            max_retries=max_retries,
        )
        return result.get(Predicate, [])

    def _generate_functions(
        self,
        domain_desc: str,
        types: List[PDDLType],
        constants: List[Constant],
        predicates: List[Predicate],
        max_retries: int,
        comp_desc: str = "",
        feedback: str = "",
    ) -> List[Function]:
        desc = domain_desc
        if comp_desc:
            desc = f"{domain_desc}\n\n[Additional context]\n{comp_desc}"
        if feedback:
            desc = f"{desc}\n\n[Feedback]\n{feedback}"
        what = "Re-generating" if feedback else "Generating"
        print(f"  {what} functions from description...")
        result, _ = self.domain_builder.formalize_component(
            model=self.llm,
            component_class=Function,
            description=desc,
            types=types,
            constants=constants,
            predicates=predicates,
            max_retries=max_retries,
        )
        return result.get(Function, [])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _prompt_save(self, content: str, domain_name: str):
        default_path = f"{domain_name}-domain.pddl"
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
                print(f"\n{BOLD}Generated domain:{RESET}\n{content}\n")
                return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"\n{GREEN}[SUCCESS] Domain saved to: {output_path.resolve()}{RESET}")
