"""
Domain generator for L2P CLI. Generates complete PDDL domain using pipeline approach (simplified NL2PLAN).
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple

from l2p.utils.pddl_format import format_types_to_string
from l2p.utils.pddl_parser import load_file
from l2p.cli.commands.generate import GeneratorBase
from l2p.cli.utils.errors import handle_error
from l2p.cli.utils.helpers import _input_or_exit, BOLD, GREEN, CYAN, YELLOW, RESET


PDDL_REQUIREMENTS = {
    "PDDL Core": [
        ":strips", ":typing", ":disjunctive-preconditions", ":equality",
        ":existential-preconditions", ":universal-preconditions",
        ":quantified-preconditions", ":conditional-effects", ":adl"
    ],
    "PDDL Extended": [
        ":action-expansions", ":foreach-expansions", ":dag-expansions",
        ":domain-axioms", ":subgoals-through-axioms", ":safety-constraints",
        ":expression-evaluation", ":open-world", ":true-negation", ":ucpop"
    ],
    "PDDL 2.1": [
        ":fluents", ":numeric-fluents", ":durative-actions",
        ":duration-inequalities", ":durative-inequalities",
        ":continuous-effects", ":negative-preconditions",
        ":timed-effects", ":action-costs"
    ],
    "PDDL 2.2": [
        ":derived-predicates", ":derived-functions",
        ":timed-initial-literals", ":timed-initial-fluents"
    ],
    "PDDL 3.0": [
        ":constraints", ":preferences"
    ],
    "PDDL 3.1/+": [
        ":object-fluents", ":time"
    ]
}

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

  # Interactive domain generation w/ max retries
  l2p generate domain --max_retries <n>
        """,
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


class DomainGenerator(GeneratorBase):
    """Generator for complete PDDL domains."""

    def generate(self, args):
        """Interactive domain generation."""
        print(
            f"{BOLD}{'=' * 60}{RESET}\n"
            f"{BOLD}  L2P Interactive Domain Generator{RESET}\n"
            f"{BOLD}{'=' * 60}{RESET}\n"
            f"  Type {YELLOW}/exit{RESET} at any prompt to quit\n")

        domain_name = self._prompt_domain_name()
        domain_desc = self._prompt_domain_desc()
        requirements = self._prompt_requirements()

        context = {}

        # generate types
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
                        self._generate_types(domain_desc, args.max_retries, feedback)
                )
        else:
            context["types"] = None

        # generate constants
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
                        self._generate_constants(
                            domain_desc, context.get("types"), args.max_retries, feedback
                        )
                )
        else:
            context["constants"] = None

        # generate predicates
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
                        self._generate_predicates(
                            domain_desc, context.get("types"), context.get("constants"),
                            args.max_retries, feedback
                        )
                )
        else:
            context["predicates"] = None

        # generate functions
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
                        self._generate_functions(
                            domain_desc, context.get("types"), context.get("constants"),
                            context.get("predicates"), args.max_retries, feedback
                        )
                )
        else:
            context["functions"] = None

        # generate actions
        print(f"\n{BOLD}--- Actions ---{RESET}")
        self._handle_actions_interactive(domain_desc, context, args.max_retries)

        # assemble domain
        print(
            f"\n\n{BOLD}{'=' * 60}{RESET}"
            f"\n{BOLD}  Assembling Domain{RESET}"
            f"\n{BOLD}{'=' * 60}{RESET}"
        )

        # print(context.get("types"))

        req_list = [r.strip() for r in requirements.split(",") if r.strip()]
        domain_pddl = self.domain_builder.generate_domain(
            domain_name=domain_name,
            requirements=req_list,
            types=context.get("types"),
            constants=context.get("constants"),
            predicates=context.get("predicates"),
            functions=context.get("functions"),
            actions=context.get("actions", []),
        )

        # print output
        print(
            f"\nOUTPUT:"
            f"\n\n{BOLD}{'=' * 60}{RESET}"
            f"\n{domain_pddl}"
            f"\n{BOLD}{'=' * 60}{RESET}"
        )

        self._prompt_save(domain_pddl, domain_name)

    
    # HELPER METHODS
    def _prompt_domain_name(self) -> str:
        while True:
            name = _input_or_exit(f"{GREEN}Enter domain name:{RESET} ").strip().lower().replace(" ", "-")
            if name:
                return re.sub(r"[^a-z0-9-]", "", name)
            print("Domain name cannot be empty.")

    def _prompt_domain_desc(self) -> str:
        print(f"\n{GREEN}Enter a brief description of your domain:{RESET}")
        print(f"  (This helps the LLM generate appropriate PDDL components)")
        desc = _input_or_exit().strip()
        return desc or "A general planning domain."

    def _prompt_requirements(self) -> str:
        print(f"\n{BOLD}PDDL Requirements (comma-separated, default: :strips,:typing){RESET}")
        print(f"  Available:")
    
        columns = 3
        col_width = 30
        for level, reqs in PDDL_REQUIREMENTS.items():
            print(f"    -------- {level} --------")
            for i in range(0, len(reqs), columns):
                chunk = reqs[i:i + columns]
                row_str = "".join(f"{req:<{col_width}}" for req in chunk)
                print(f"      {row_str}")
        print()
        
        user_input = _input_or_exit("Requirements: ").strip()
        if not user_input or user_input == "default" or user_input == ":default":
            return ":strips,:typing"
            
        parts = [r.strip() for r in user_input.split(",") if r.strip()]
        valid_reqs_set = {req for req_list in PDDL_REQUIREMENTS.values() for req in req_list}
        
        valid = []
        for r in parts:
            if r in valid_reqs_set and r not in valid:
                valid.append(r)
                
        invalid = [r for r in parts if r not in valid_reqs_set]
        
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
        print(
            f"  Enter types. Type name or {YELLOW}'done'{RESET} to finish.\n"
            f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.\n"
            f"  Parent defaults to {CYAN}object{RESET} (root).\n"
        )

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
        formatted_tree = self._format_to_nested_tree(types)
        
        return formatted_tree
    
    def _format_to_nested_tree(self, flat_types: list) -> list:
        """Converts a flat list of types into a nested dictionary structure."""
        nodes = {}
        for t in flat_types:
            name = t["name"]
            desc = t["desc"]
            # format: { "type_name": "description", "children": [] }
            nodes[name] = {
                name: desc,
                "children": []
            }
            
        tree = []
        for t in flat_types:
            name = t["name"]
            parent = t["parent"]
            if parent == "object" or parent not in nodes:
                tree.append(nodes[name])
            else:
                nodes[parent]["children"].append(nodes[name])
                
        return tree

    def _show_type_tree(self, types: list):
        """Display the type hierarchy as an indented tree with lines."""
        nodes = {"object": {"children": {}}}
        for t in types:
            parent = t["parent"]
            name = t["name"]
            if parent not in nodes:
                nodes[parent] = {"children": {}}
            if name not in nodes:
                nodes[name] = {"children": {}}
            nodes[parent]["children"][name] = nodes[name]
        print(f"  {BOLD}Current hierarchy:{RESET}")

        def _print_tree(node_name, node, prefix="", is_last=True, is_root=False):
            # print the current node
            if is_root:
                print(f"  {CYAN}{node_name}{RESET}")
                child_prefix = prefix
            else:
                connector = "└── " if is_last else "├── "
                print(f"  {prefix}{connector}{CYAN}{node_name}{RESET}")
                child_prefix = prefix + ("    " if is_last else "│   ")

            children = list(node["children"].items())
            for i, (child_name, child_node) in enumerate(children):
                is_last_child = (i == len(children) - 1)
                _print_tree(child_name, child_node, child_prefix, is_last_child, is_root=False)
        _print_tree("object", nodes["object"], is_root=True)

    def _manual_constants(self, types) -> dict:
        """Interactive constant builder."""
        available = self._collect_type_names(types)
        constants = {}
        print(
            f"  Available types: {CYAN}{', '.join(available)}{RESET}\n"
            f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.\n"
            f"  Enter constants. Name or {YELLOW}'done'{RESET} to finish.\n"
            )

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
        print(
            f"  Available types: {CYAN}{', '.join(available)}{RESET}\n"
            f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.\n"
            f"  Enter predicates. Name or {YELLOW}'done'{RESET} to finish.\n"
            )

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
        print(
            f"  Available types: {CYAN}{', '.join(available)}{RESET}\n"
            f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.\n"
            f"  Enter functions. Name or {YELLOW}'done'{RESET} to finish.\n")

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
            print(
                f"  How do you want to define actions?\n"
                f"    {CYAN}1{RESET} - Let the LLM extract action names from the domain description\n"
                f"    {CYAN}2{RESET} - Specify them manually\n"
                f"  Type {YELLOW}'..'{RESET} to go back, {YELLOW}'skip'{RESET} to skip actions.\n"
                )

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
        
        template = self.template_manager.get_template("extract_nl_actions.txt", "domain")

        print(f"\n  Extracting action names from description...")
        result = self.domain_builder.extract_nl_actions(
            model=self.llm,
            domain_desc=domain_desc,
            prompt_template=template,
            types=context.get("types"),
            max_retries=max_retries
        )
        nl_actions, _ = result

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
        names, descs = [], []
        print(
            f"  Enter actions. Name or {YELLOW}'done'{RESET} to finish.\n"
            f"  Use {YELLOW}'..'{RESET} to go back to the previous prompt.\n")

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
                print(
                    f"  {YELLOW}Skipping file save.{RESET}\n"
                    f"\n{BOLD}Generated domain:{RESET}\n"
                    f"{content}\n")
                return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"\n{GREEN}[SUCCESS] Domain saved to: {output_path.resolve()}{RESET}")

    def _display_component(self, label: str, items):
        if items is None:
            print(f"  ({label} omitted)")
            return
        print(f"\n  {BOLD}{label.capitalize()}:{RESET}")
        if label == "types":
            types_string = format_types_to_string(items)
            print(f"{CYAN}{types_string}{RESET}")
        elif label == "constants":
            if isinstance(items, dict):
                for name, typ in items.items():
                    print(f"    {CYAN}{name}{RESET} - {typ}")
        elif label in ("predicates", "functions"):
            for p in items:
                if isinstance(p, dict):
                    print(f"    {CYAN}{p.get('raw', p.get('name', '?'))}{RESET}")
        else:
            if isinstance(items, list) or isinstance(items, dict):
                print(f"    ({len(items)} {label})")

    def _collect_type_names(self, types) -> list:
        if not types:
            return ["object"]
        names = ["object"]
        def _extract_names(node_list):
            for node in node_list:
                # find the key that is not "children"
                for key in node.keys():
                    if key != "children":
                        names.append(key)
                if "children" in node and node["children"]:
                    _extract_names(node["children"])
                    
        _extract_names(types)
        return names
    
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


    # LLM-BASED GENERATION WRAPPERS
    def _generate_types(self, domain_desc: str, max_retries: int, feedback: str = ""):
    
        template = self.template_manager.get_template("formalize_type_hierarchy.txt", "domain")
        prompt = domain_desc
        if feedback:
            prompt = f"{domain_desc}\n\n[Feedback to apply to the generated types]\n{feedback}"
            print(f"  Re-generating types with your feedback...")
        else:
            print(f"  Generating types from description...")
        result = self.domain_builder.formalize_type_hierarchy(
            model=self.llm, domain_desc=prompt,
            prompt_template=template, max_retries=max_retries
        )
        types_result, _, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return types_result

    def _generate_constants(self, domain_desc: str, types, max_retries: int, feedback: str = ""):
        template = load_file("l2p/cli/commands/generators/templates/domain/formalize_constants.txt")
        prompt = domain_desc
        if feedback:
            prompt = f"{domain_desc}\n\n[Feedback to apply to the generated constants]\n{feedback}"
            print(f"  Re-generating constants with your feedback...")
        else:
            print(f"  Generating constants from description...")
        result = self.domain_builder.formalize_constants(
            model=self.llm, domain_desc=prompt,
            prompt_template=template, types=types, max_retries=max_retries
        )
        const_result, _, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return const_result

    def _generate_predicates(self, domain_desc: str, types, constants, max_retries: int, feedback: str = ""):
        template = load_file("l2p/cli/commands/generators/templates/domain/formalize_predicates.txt")
        prompt = domain_desc
        if feedback:
            prompt = f"{domain_desc}\n\n[Feedback to apply to the generated predicates]\n{feedback}"
            print(f"  Re-generating predicates with your feedback...")
        else:
            print(f"  Generating predicates from description...")
        result = self.domain_builder.formalize_predicates(
            model=self.llm, domain_desc=prompt,
            prompt_template=template, types=types,
            constants=constants, max_retries=max_retries
        )
        pred_result, _, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return pred_result

    def _generate_functions(self, domain_desc: str, types, constants, predicates, max_retries: int, feedback: str = ""):
        template = load_file("l2p/cli/commands/generators/templates/domain/formalize_functions.txt")
        
        prompt = domain_desc
        if feedback:
            prompt = f"{domain_desc}\n\n[Feedback to apply to the generated functions]\n{feedback}"
            print(f"  Re-generating functions with your feedback...")
        else:
            print(f"  Generating functions from description...")
        result = self.domain_builder.formalize_functions(
            model=self.llm, domain_desc=prompt,
            prompt_template=template, types=types,
            constants=constants, predicates=predicates,
            max_retries=max_retries
        )
        func_result, _, validation_info = result
        if validation_info and not validation_info[0]:
            print(f"  {YELLOW}Validation: {validation_info[1]}{RESET}")
        return func_result

    def _generate_actions_from_list(
        self, domain_desc: str, action_names: list, action_descs: list,
        context: dict, max_retries: int
    ) -> list:
        """Generate full PDDL actions with new-predicate merging."""
        
        actions = []
        all_new_predicates = []

        total = len(action_names)
        for i, (name, desc) in enumerate(zip(action_names, action_descs), 1):
            print(f"  [{i}/{total}] Generating action: {name}")
            template = load_file("l2p/cli/commands/generators/templates/domain/formalize_pddl_action.txt")

            result = self.domain_builder.formalize_pddl_action(
                model=self.llm, domain_desc=domain_desc,
                prompt_template=template, action_name=name,
                action_desc=desc or f"{name} action",
                types=context.get("types"),
                constants=context.get("constants"),
                predicates=context.get("predicates"),
                functions=context.get("functions"),
                max_retries=max_retries
            )
            action_result, new_preds, _, validation_info = result
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