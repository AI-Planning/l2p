"""
Chat command for L2P CLI.

Interactive single-exchange chat with the configured LLM model:
    - Run command: `l2p chat`
"""

import os
import re
import sys
import tempfile
import argparse
from pathlib import Path

from ..utils.config import get_config_manager
from ..utils.errors import handle_error

BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

EDIT_SYSTEM_PROMPT = (
    "You are a PDDL editing assistant. The user will provide you with the contents of a PDDL file "
    "and a description of the edit they want to make.\n\n"
    "Your task is to apply the requested edit and return the ENTIRE updated PDDL file content "
    "inside a single fenced code block with the 'pddl' language tag:\n\n"
    "```pddl\n(define (domain ...)\n  ...\n)\n```\n\n"
    "Rules:\n"
    "- Preserve all existing content that the user did not ask to change.\n"
    "- Ensure the PDDL syntax is correct: balanced parentheses, proper indentation, valid requirement declarations.\n"
    "- Use :typing if the domain uses typed parameters, :strips is always implied.\n"
    "- Do NOT include any explanatory text outside the code block.\n"
    "- Always output the full file content, never a diff or partial snippet.\n"
    "- If the requested edit would break the PDDL syntax, suggest an alternative that preserves correctness."
)

# system prompt for L2P context
SYSTEM_PROMPT = (
    "You are an assistant for L2P (Library to connect LLMs and Planning), a CLI tool that uses LLMs "
    "to generate PDDL (Planning Domain Definition Language) components for AI planning tasks.\n\n"
    "Available CLI commands:\n"
    "  l2p init              - Configure the LLM provider, model, and API key\n"
    "  l2p models list       - List available models for the configured provider\n"
    "  l2p models switch     - Interactively pick a different model\n"
    "  l2p models test       - Test the connection to the configured model\n"
    "  l2p generate types    - Generate PDDL type definitions from a description\n"
    "  l2p generate predicates - Generate PDDL predicate definitions\n"
    "  l2p generate action   - Generate a PDDL action definition\n"
    "  l2p generate domain   - Generate a full PDDL domain (pipeline)\n"
    "  l2p generate task     - Generate a PDDL problem/task\n"
    "  l2p config show       - Display current configuration\n"
    "  l2p config edit       - Open configuration in editor\n"
    "  l2p config validate   - Validate configuration\n"
    "  l2p templates list    - List available PDDL templates\n"
    "  l2p new               - Create a blank PDDL file\n"
    "  l2p chat              - Start this interactive chat session\n\n"
    "Documentation: https://marcustantakoun.github.io/l2p.github.io/\n\n"
    "When users ask about generating PDDL components, guide them on using the appropriate "
    "l2p generate command. When they ask about configuration issues, refer them to l2p init "
    "or l2p config edit. Keep responses concise and focused on L2P usage."
    "If they ask how to exit, tell them it is `/exit`."
)


def add_subparser(subparsers):
    """Add chat command subparser."""
    parser = subparsers.add_parser(
        "chat",
        help="Chat with the configured LLM model",
        description="Start an interactive chat session with the configured LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start chat session
  l2p chat
  
  # Chat with a different model
  l2p chat --model gpt-4o
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (default: use configured model)"
    )

    parser.set_defaults(func=chat_command)


def _check_pddl_syntax(content: str) -> str | None:
    """Run PDDL syntax check on content. Returns None if valid, error string if not."""
    from pddl import parse_domain as pddl_parse_domain, parse_problem as pddl_parse_problem

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False)
    try:
        tmp.write(content)
        tmp.close()
        if content.strip().startswith("(define (domain"):
            pddl_parse_domain(tmp.name)
        elif content.strip().startswith("(define (problem"):
            pddl_parse_problem(tmp.name)
        else:
            return "Unknown PDDL type — must start with (define (domain ... or (define (problem ..."
        return None
    except Exception as e:
        return str(e)
    finally:
        os.unlink(tmp.name)


def _handle_edit_command(llm, backend: str, filepath: str):
    """Handle /edit <filepath> — load, prompt for edit, send to LLM, confirm overwrite."""
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        print(f"  {YELLOW}File not found:{RESET} {path}")
        return

    content = path.read_text()
    print(f"  Loaded {CYAN}{path}{RESET} ({len(content)} chars)")

    edit_desc = input(f"  {GREEN}Describe your edit:{RESET} ").strip()
    if not edit_desc:
        print(f"  {YELLOW}No edit provided.{RESET}")
        return

    prompt = (
        f"[PDDL File: {path.name}]\n\n"
        f"```pddl\n{content}\n```\n\n"
        f"Requested edit: {edit_desc}\n\n"
        f"Apply this edit and return the entire updated file in a ```pddl block."
    )

    try:
        if backend == "openai":
            messages = [
                {"role": "system", "content": EDIT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            response = llm.query(prompt, messages=messages, max_retry=1)
        else:
            prefixed = f"[System Instructions]\n{EDIT_SYSTEM_PROMPT}\n\n[User]\n{prompt}"
            response = llm.query(prefixed, max_retry=1)
    except Exception as e:
        print(f"\n{YELLOW}LLM error: {e}{RESET}")
        return

    # Extract ```pddl ... ``` block
    m = re.search(r"```pddl\s*\n(.*?)```", response, re.DOTALL)
    if not m:
        print(f"\n{CYAN}{response}{RESET}\n")
        print(f"  {YELLOW}Could not find a ```pddl block in the response. Edit aborted.{RESET}")
        return

    new_content = m.group(1).strip()
    print(f"\n{CYAN}```pddl{RESET}")
    print(new_content)
    print(f"{CYAN}```{RESET}")

    # Run PDDL syntax check (warning only, never reformats)
    print(f"\n  Checking PDDL syntax...")
    error = _check_pddl_syntax(new_content)
    if error:
        print(f"  {YELLOW}[WARNING] PDDL syntax issue: {error}{RESET}")
        print(f"  {YELLOW}The file may not be valid — proceed with caution.{RESET}")
    else:
        print(f"  {GREEN}PDDL syntax is valid.{RESET}")

    confirm = input(f"\n  {GREEN}Overwrite file? (y/N):{RESET} ").strip().lower()
    if confirm != "y":
        print(f"  {YELLOW}Edit cancelled.{RESET}")
        return

    path.write_text(new_content + "\n")
    print(f"  {GREEN}File saved to:{RESET} {path}")


def chat_command(args):
    """Execute chat command."""
    try:
        config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
        model_config = config_manager.get_model_config().copy()

        provider = model_config.get("provider")
        model = args.model or model_config.get("model")
        config_path = model_config.get("config_path")
        api_key = model_config.get("api_key", "")
        backend = model_config.get("backend", "unified")

        if not provider or not model:
            print("Error: No model configured.")
            print("Run 'l2p init' to configure a model first or 'l2p config edit' to edit an existing config file.")
            sys.exit(1)

        # resolve API key
        if api_key and api_key.endswith("_API_KEY"):
            api_key = os.getenv(api_key, "")

        llm_name = "OPENAI" if backend == "openai" else "UnifiedLLM"

        if backend == "openai":
            from l2p.llm.openai import OPENAI as llm_class
        else:
            from l2p.llm.unified import UnifiedLLM as llm_class

        print()
        print(f"{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}  L2P Chat{RESET}")
        print(f"  {CYAN}{provider}/{model}{RESET}")
        print(f"{BOLD}{'=' * 60}{RESET}")
        print(f"  {YELLOW}/exit{RESET}       Quit")
        print(f"  {YELLOW}/edit <file>{RESET}  Edit a PDDL file with LLM assistance")
        print()

        llm = llm_class(
            provider=provider,
            model=model,
            config_path=config_path,
            api_key=api_key,
        )

        while True:
            # get input from user
            try:
                user_input = input(f"{GREEN}>>>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            # if not user input continue
            if not user_input:
                continue
            
            if user_input == "/exit":
                break

            if user_input.startswith("/edit "):
                filepath = user_input[len("/edit "):].strip()
                _handle_edit_command(llm, backend, filepath)
                continue

            try:
                # insert L2P system prompt to query with
                if backend == "openai":
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_input},
                    ]
                    response = llm.query(user_input, messages=messages, max_retry=1)
                else:
                    prefixed = f"[System Instructions]\n{SYSTEM_PROMPT}\n\n[User]\n{user_input}"
                    response = llm.query(prefixed, max_retry=1)
                print(f"\n{CYAN}{response}{RESET}\n")
            except Exception as e:
                print(f"\n{YELLOW}Error: {e}{RESET}\n")

        # exit message
        print(f"\n{BOLD}Goodbye!{RESET}")

    # fail catch message returns
    except ImportError as e:
        print(f"\nFailed to import {llm_name}: {e}")
        print("Troubleshooting:")
        if backend == "openai":
            print("  pip install openai tiktoken")
        else:
            print("  pip install llm tiktoken")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n")
        sys.exit(130)
    except Exception as e:
        handle_error(e)
        sys.exit(1)
