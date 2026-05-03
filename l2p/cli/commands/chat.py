"""
Chat command for L2P CLI.

Interactive single-exchange chat with the configured LLM model.
"""

import os
import sys
import argparse

from ..utils.config import CLIError, get_config_manager
from ..utils.errors import handle_error


BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

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
            print("Run 'l2p init' to configure a model first.")
            sys.exit(1)

        # Resolve API key
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
        print(f"  Type {YELLOW}/exit{RESET} to quit")
        print()

        llm = llm_class(
            provider=provider,
            model=model,
            config_path=config_path,
            api_key=api_key,
        )

        while True:
            try:
                user_input = input(f"{GREEN}>>>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input == "/exit":
                break

            try:
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

        print(f"\n{BOLD}Goodbye!{RESET}")

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
