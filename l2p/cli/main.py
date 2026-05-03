#!/usr/bin/env python3
"""
L2P Command Line Interface

This module provides a CLI for generating PDDL components using LLMs.
Users must configure an LLM model before using generation commands.
"""

import os
import sys
import argparse
from typing import Optional

# Import CLI utilities
from .utils.errors import handle_error


def main():
    """Main CLI entry point."""
    try:
        # Create the main parser
        parser = argparse.ArgumentParser(
            description="L2P: Library to connect LLMs and planning tasks",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  l2p init --provider openai --model gpt-4o-mini
  l2p generate types --desc "blocksworld domain"
  l2p generate action --name pick-up --desc "pick up a block"
  
For more information on a specific command, use:
  l2p <command> --help
            """,
        )
        
        # Global arguments
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output for debugging"
        )
        
        parser.add_argument(
            "--config",
            type=str,
            help="Path to configuration file (default: ~/.l2p/config.yaml)"
        )
        
        # Create subparsers
        subparsers = parser.add_subparsers(
            dest="command",
            title="commands",
            description="Available commands",
            metavar="COMMAND",
            required=True
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
            
            add_init_parser(subparsers)
            add_models_parser(subparsers)
            add_generate_parser(subparsers)
            add_config_parser(subparsers)
            add_templates_parser(subparsers)
            add_new_parser(subparsers)
            add_chat_parser(subparsers)
            
        except ImportError as e:
            print(f"ERROR: Failed to load CLI commands: {e}")
            print("\nTroubleshooting:")
            print("• Ensure L2P CLI is properly installed")
            print("• Check Python path and module structure")
            sys.exit(1)
        
        # Parse arguments
        args = parser.parse_args()
        
        # Set up logging/verbose output
        if args.verbose:
            import logging
            logging.basicConfig(level=logging.DEBUG)
        
        # Execute command
        # Import command functions dynamically to avoid circular imports
        if args.command == "init":
            from .commands.init import init_command
            init_command(args)
        elif args.command == "models":
            from .commands.models import models_command
            models_command(args)
        elif args.command == "generate":
            from .commands.generate import generate_command
            generate_command(args)
        elif args.command == "config":
            from .commands.config import config_command
            config_command(args)
        elif args.command == "templates":
            from .commands.templates import templates_command
            templates_command(args)
        elif args.command == "new":
            from .commands.new import new_command
            new_command(args)
        elif args.command == "chat":
            from .commands.chat import chat_command
            chat_command(args)
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