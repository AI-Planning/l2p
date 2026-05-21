"""
Template management commands for L2P CLI.

List and manage prompt templates.
"""

import argparse
import sys

from l2p.cli.utils.config import get_config_manager
from l2p.cli.utils.errors import handle_error
from l2p.cli.utils.templates import get_template_manager


def add_subparser(subparsers):
    """Add templates command subparser."""
    parser = subparsers.add_parser(
        "templates",
        help="Manage templates",
        description="List and manage prompt templates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available templates
  l2p templates list
  
  # List templates for a specific category
  l2p templates list --category domain
  
  # Show template content
  l2p templates show --name formalize_type.txt --category domain
  
  # Find template file location
  l2p templates find --name formalize_predicates.txt
        """,
    )

    subparsers = parser.add_subparsers(
        dest="templates_command",
        title="templates commands",
        description="Available template subcommands",
        metavar="COMMAND",
        required=True,
    )

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List templates", description="List available templates."
    )
    list_parser.add_argument(
        "--category",
        type=str,
        choices=["domain", "task", "feedback", "all"],
        default="all",
        help="Template category to list (default: all)",
    )
    list_parser.add_argument(
        "--details", action="store_true", help="Show detailed template information"
    )
    list_parser.set_defaults(func=templates_list_command)

    # Show command
    show_parser = subparsers.add_parser(
        "show", help="Show template content", description="Show template content."
    )
    show_parser.add_argument(
        "--name", type=str, required=True, help="Template file name"
    )
    show_parser.add_argument(
        "--category",
        type=str,
        choices=["domain", "task", "feedback"],
        default="domain",
        help="Template category (default: domain)",
    )
    show_parser.set_defaults(func=templates_show_command)

    # Find command
    find_parser = subparsers.add_parser(
        "find",
        help="Find template location",
        description="Find template file location.",
    )
    find_parser.add_argument(
        "--name", type=str, required=True, help="Template file name"
    )
    find_parser.add_argument(
        "--category",
        type=str,
        choices=["domain", "task", "feedback"],
        default="domain",
        help="Template category (default: domain)",
    )
    find_parser.set_defaults(func=templates_find_command)


def templates_command(args):
    """Execute templates command."""
    try:
        args.func(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


def templates_list_command(args):
    """List available templates."""
    config_manager = get_config_manager(
        args.config if hasattr(args, "config") else None
    )
    template_manager = get_template_manager(config_manager)

    category = args.category if args.category != "all" else None
    templates = template_manager.list_templates(category)

    print(f"Available Templates:\n{"=" * 60}")

    for cat, files in templates.items():
        print(f"\n{cat.upper()} Templates ({len(files)}):")
        print("-" * 40)

        if args.details:
            for file in files:
                template_path = template_manager.get_template_path(file, cat)
                source = (
                    "Package"
                    if "l2p" in str(template_path)
                    else "Custom" if template_path else "Not found"
                )
                print(f"  > {file}")
                print(f"    Source: {source}")
                if template_path:
                    print(f"    Path: {template_path}")
        else:
            # simple list
            for file in files:
                print(f"  > {file}")

    # show custom template path if configured
    templates_config = config_manager.get_templates_config()
    custom_path = templates_config.get("custom_path")
    if custom_path:
        print(f"\nCustom template path: {custom_path}")

    print(f"\nTotal templates: {sum(len(files) for files in templates.values())}")


def templates_show_command(args):
    """Show template content."""
    config_manager = get_config_manager(
        args.config if hasattr(args, "config") else None
    )
    template_manager = get_template_manager(config_manager)

    try:
        template_content = template_manager.get_template(args.name, args.category)
        print(
            f"Template: {args.category}/{args.name}"
            f"\n{"=" * 60}\n"
            f"\n{template_content}"
        )

        # show template location
        template_path = template_manager.get_template_path(args.name, args.category)
        if template_path:
            print(f"\nSource: {template_path}")

    except Exception as e:
        print(f"[ERROR] Error loading template: {e}", file=sys.stderr)

        # show available templates in category
        templates = template_manager.list_templates(args.category)
        if args.category in templates:
            print(f"\nAvailable templates in '{args.category}' category:")
            for template in templates[args.category]:
                print(f"  > {template}")


def templates_find_command(args):
    """Find template file location."""
    config_manager = get_config_manager(
        args.config if hasattr(args, "config") else None
    )
    template_manager = get_template_manager(config_manager)

    template_path = template_manager.get_template_path(args.name, args.category)

    if template_path:
        print(f"Template found: {args.category}/{args.name}")
        print(f"Path: {template_path}")

        # show if it is package or custom
        if "l2p/templates" in str(template_path):
            print("Source: Package templates")
        else:
            print("Source: Custom templates")

        # show file info
        try:
            stat = template_path.stat()
            import datetime

            mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
            print(f"Size: {stat.st_size} bytes")
            print(f"Modified: {mod_time}")

            # Show preview
            with open(template_path, "r") as f:
                preview = f.read(500)
                if len(preview) >= 500:
                    preview = preview[:497] + "..."
                print(f"\nPreview:")
                print("-" * 40)
                print(preview)

        except Exception as e:
            print(f"Note: Could not read file details: {e}")
    else:
        print(f"Template not found: {args.category}/{args.name}")

        # Show search locations
        templates_config = config_manager.get_templates_config()
        custom_path = templates_config.get("custom_path")

        print("\nSearch locations:")
        print("1. Package templates: l2p/templates/")
        if custom_path:
            print(f"2. Custom templates: {custom_path}")

        # Show available templates
        templates = template_manager.list_templates(args.category)
        if args.category in templates:
            print(f"\nAvailable templates in '{args.category}' category:")
            for template in templates[args.category]:
                print(f"  > {template}")
