#!/usr/bin/env python3
"""
L2P Command Line Interface

This module provides a CLI for generating PDDL domains/problems using LLMs.
Users must configure an LLM model before using generation commands.
"""

import sys
import argparse

from l2p.cli.utils.errors import handle_error


def main():
    """Main CLI entry point."""
    try:
        # create the main parser
        parser = argparse.ArgumentParser(
            description="L2P: Library to connect LLMs and planning tasks",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  l2p init --provider openai --model gpt-4o-mini
  l2p generate domain --max-retries <n>
  l2p generate problem --max-retries <n>
  
For more information on a specific command, use:
  l2p <command> --help
            """,
        )

        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output for debugging",
        )

        parser.add_argument(
            "--config",
            type=str,
            help="Path to configuration file (default: ~/.l2p/config.yaml)",
        )

        # Create subparsers
        subparsers = parser.add_subparsers(
            dest="command",
            title="commands",
            description="Available commands",
            metavar="COMMAND",
            required=True,
        )

        # Initialize commands
        try:
            from .commands.init import add_subparser as add_init_parser
            from .commands.models import add_subparser as add_models_parser
            from .commands.generate import add_subparser as add_generate_parser
            from .commands.config import add_subparser as add_config_parser
            from .commands.templates import add_subparser as add_templates_parser
            from .commands.new import add_subparser as add_new_parser
            from .commands.chat import add_subparser as add_chat_parser
            from .commands.set import add_subparser as add_set_parser
            from .commands.build import add_subparser as add_build_parser
            from .commands.validate import add_subparser as add_validate_parser
            from .commands.plan import add_subparser as add_plan_parser
            from .commands.schema import add_subparser as add_schema_parser
            from .commands.mcp import add_subparser as add_mcp_parser

            add_init_parser(subparsers)
            add_models_parser(subparsers)
            add_generate_parser(subparsers)
            add_config_parser(subparsers)
            add_templates_parser(subparsers)
            add_new_parser(subparsers)
            add_chat_parser(subparsers)
            add_set_parser(subparsers)
            add_build_parser(subparsers)
            add_validate_parser(subparsers)
            add_plan_parser(subparsers)
            add_schema_parser(subparsers)
            add_mcp_parser(subparsers)

        except ImportError as e:
            print(f"[ERROR] Failed to load CLI commands: {e}")
            print("\nTroubleshooting:")
            print(" > Ensure L2P CLI is properly installed")
            print(" > Check Python path and module structure")
            sys.exit(1)

        args = parser.parse_args()

        # set up logging/verbose output
        if args.verbose:
            import logging

            logging.basicConfig(level=logging.DEBUG)

        # execute command
        # import command functions dynamically to avoid circular imports
        if args.command == "init":
            from l2p.cli.commands.init import init_command

            init_command(args)
        elif args.command == "models":
            from l2p.cli.commands.models import models_command

            models_command(args)
        elif args.command == "generate":
            from l2p.cli.commands.generate import generate_command

            generate_command(args)
        elif args.command == "config":
            from l2p.cli.commands.config import config_command

            config_command(args)
        elif args.command == "templates":
            from l2p.cli.commands.templates import templates_command

            templates_command(args)
        elif args.command == "new":
            from l2p.cli.commands.new import new_command

            new_command(args)
        elif args.command == "chat":
            from l2p.cli.commands.chat import chat_command

            chat_command(args)
        elif args.command == "set":
            from l2p.cli.commands.set import set_command

            set_command(args)
        elif args.command in ("build", "validate"):
            if hasattr(args, "func"):
                args.func(args)
            else:
                print(f"Error: No {args.command} subcommand specified.", file=sys.stderr)
                print(f"Use `l2p {args.command} --help` for usage.", file=sys.stderr)
                sys.exit(1)
        elif args.command == "plan":
            from l2p.cli.commands.plan import plan_command

            plan_command(args)
        elif args.command == "schema":
            from l2p.cli.commands.schema import schema_command

            schema_command(args)
        elif args.command == "mcp":
            from l2p.cli.commands.mcp import mcp_command

            mcp_command(args)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
