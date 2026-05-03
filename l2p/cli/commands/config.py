"""
Configuration management commands for L2P CLI.

Show, edit, and reset configuration.
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any

from ..utils.config import ConfigManager, CLIError, get_config_manager
from ..utils.errors import handle_error


def add_subparser(subparsers):
    """Add config command subparser."""
    parser = subparsers.add_parser(
        "config",
        help="Manage configuration",
        description="Show, edit, or reset L2P configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current configuration
  l2p config show
  
  # Edit configuration with default editor
  l2p config edit
  
  # Reset to defaults
  l2p config reset --force
  
  # Validate configuration
  l2p config validate
        """,
    )
    
    subparsers = parser.add_subparsers(
        dest="config_command",
        title="config commands",
        description="Available configuration subcommands",
        metavar="COMMAND",
        required=True
    )
    
    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show current configuration",
        description="Display current configuration."
    )
    show_parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw YAML without formatting"
    )
    show_parser.set_defaults(func=config_show_command)
    
    # Edit command
    edit_parser = subparsers.add_parser(
        "edit",
        help="Edit configuration",
        description="Edit configuration with default editor."
    )
    edit_parser.set_defaults(func=config_edit_command)
    
    # Reset command
    reset_parser = subparsers.add_parser(
        "reset",
        help="Reset configuration",
        description="Reset configuration to defaults."
    )
    reset_parser.add_argument(
        "--force",
        action="store_true",
        help="Reset without confirmation"
    )
    reset_parser.set_defaults(func=config_reset_command)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration",
        description="Validate current configuration."
    )
    validate_parser.set_defaults(func=config_validate_command)


def config_command(args):
    """Execute config command."""
    try:
        args.func(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


def config_show_command(args):
    """Show current configuration."""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    config_file = config_manager.get_config_path()
    print(f"Configuration file: {config_file}")
    print("=" * 60)
    
    if args.raw:
        # Show raw YAML
        try:
            with open(config_file, 'r') as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading config file: {e}")
    else:
        # Show formatted configuration
        config = config_manager.config
        
        # Model configuration
        print("\nModel Configuration:")
        print("-" * 40)
        model_config = config.get("model", {})
        for key, value in model_config.items():
            if key == "api_key" and isinstance(value, str) and len(value) > 8:
                masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
                print(f"  {key}: {masked}")
            else:
                print(f"  {key}: {value}")
        
        # Generation configuration
        print("\nGeneration Configuration:")
        print("-" * 40)
        gen_config = config.get("generation", {})
        for key, value in gen_config.items():
            print(f"  {key}: {value}")
        
        # Templates configuration
        print("\nTemplates Configuration:")
        print("-" * 40)
        templates_config = config.get("templates", {})
        for key, value in templates_config.items():
            print(f"  {key}: {value}")
        
        # Validation status
        print("\nValidation:")
        print("-" * 40)
        is_valid, message = config_manager.validate_model_config()
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  Status: {status}")
        if not is_valid:
            print(f"  Message: {message}")


def config_edit_command(args):
    """Edit configuration with default editor."""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    config_file = config_manager.get_config_path()
    
    # Determine editor
    editor = os.environ.get('EDITOR')
    if not editor:
        # Try common editors
        for candidate in ['nano', 'vim', 'vi', 'code', 'subl']:
            if subprocess.run(['which', candidate], capture_output=True).returncode == 0:
                editor = candidate
                break
    
    if not editor:
        raise CLIError(
            "No editor found.",
            [
                "Set EDITOR environment variable: export EDITOR='your-editor'",
                "Or install a common editor like nano, vim, or VS Code"
            ]
        )
    
    print(f"Opening configuration in {editor}: {config_file}")
    
    try:
        subprocess.run([editor, config_file], check=True)
        
        # Reload and validate
        config_manager.load_config()
        
        # Detect backend from config_path and fix if misaligned
        model_cfg = config_manager.config.get("model", {})
        config_path = model_cfg.get("config_path", "")
        current_backend = model_cfg.get("backend", "")
        suggested_backend = None
        if config_path.endswith("openaiSDK.yaml") and current_backend != "openai":
            suggested_backend = "openai"
        elif config_path.endswith("llm.yaml") and current_backend != "unified":
            suggested_backend = "unified"
        if suggested_backend:
            config_manager.config["model"]["backend"] = suggested_backend
            config_manager.save_config()
            print(f"ℹ Auto-updated backend from '{current_backend}' to '{suggested_backend}' to match config_path")
        
        is_valid, message = config_manager.validate_model_config()
        
        if is_valid:
            print("✅ Configuration updated and valid.")
        else:
            print(f"⚠ Configuration updated but has issues: {message}")
            
    except subprocess.CalledProcessError as e:
        raise CLIError(
            f"Editor command failed: {e}",
            ["Check EDITOR environment variable", "Ensure editor is properly installed"]
        )
    except Exception as e:
        raise CLIError(
            f"Failed to edit configuration: {e}",
            ["Check file permissions", "Try editing manually: {config_file}"]
        )


def config_reset_command(args):
    """Reset configuration to defaults."""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    config_file = config_manager.get_config_path()
    
    if not args.force:
        print(f"Warning: This will reset configuration to defaults.")
        print(f"Current configuration at: {config_file}")
        response = input("Are you sure? (y/N): ").strip().lower()
        if response != 'y':
            print("Reset cancelled.")
            return
    
    try:
        config_manager.reset_to_defaults()
        print("✅ Configuration reset to defaults.")
        print(f"\nNew configuration at: {config_file}")
        
        # Show new configuration
        config = config_manager.config
        print("\nDefault configuration:")
        print("-" * 40)
        print("Model: ", config.get("model", {}).get("provider"), "/", config.get("model", {}).get("model"))
        print("API Key: Using environment variable reference")
        print("\nRun 'l2p init' to customize configuration.")
        
    except Exception as e:
        raise CLIError(
            f"Failed to reset configuration: {e}",
            ["Check file permissions", "Try removing config file manually: rm {config_file}"]
        )


def config_validate_command(args):
    """Validate current configuration."""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    config_file = config_manager.get_config_path()
    print(f"Validating configuration: {config_file}")
    print("=" * 60)
    
    # Validate model configuration
    print("\n1. Model Configuration:")
    print("-" * 40)
    model_config = config_manager.get_model_config()
    
    checks = [
        ("Provider", model_config.get("provider"), bool(model_config.get("provider"))),
        ("Model", model_config.get("model"), bool(model_config.get("model"))),
        ("Config Path", model_config.get("config_path"), bool(model_config.get("config_path"))),
        ("API Key", "Set" if model_config.get("api_key") else "Missing", bool(model_config.get("api_key"))),
    ]
    
    all_passed = True
    for name, value, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {value}")
        if not passed:
            all_passed = False
    
    # Validate config file exists
    config_path = model_config.get("config_path")
    if config_path:
        if config_path.startswith("l2p/"):
            print(f"  📦 Using package config: {config_path}")
        else:
            config_file_path = Path(config_path).expanduser().resolve()
            if config_file_path.exists():
                print(f"  ✅ Config file exists: {config_file_path}")
            else:
                print(f"  ❌ Config file not found: {config_path}")
                all_passed = False
    
    # Overall validation
    print("\n2. Overall Validation:")
    print("-" * 40)
    is_valid, message = config_manager.validate_model_config()
    
    if is_valid and all_passed:
        print("✅ Configuration is valid and ready to use.")
        print(f"\nNext steps:")
        print("  • Test connection: l2p models test")
        print("  • Generate components: l2p generate types --desc \"your domain\"")
    else:
        print("❌ Configuration has issues.")
        if message:
            print(f"\nIssues: {message}")
        print(f"\nTo fix:")
        print("  • Run 'l2p init' to reconfigure")
        print("  • Or edit configuration: l2p config edit")