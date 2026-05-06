"""
Configuration management commands for L2P CLI: l2p config <sub-arg>
    - `l2p config show`
    - `l2p config edit`
    - `l2p config reset`
    - `l2p config validate`
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

from l2p.cli.utils.config import CLIError, get_config_manager
from l2p.cli.utils.errors import handle_error


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
    
    # show command
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
    
    # edit command
    edit_parser = subparsers.add_parser(
        "edit",
        help="Edit configuration",
        description="Edit configuration with default editor."
    )
    edit_parser.set_defaults(func=config_edit_command)
    
    # reset command
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
    
    # validate command
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
    """Show current configuration. COMMAND: `l2p config show`"""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    config_file = config_manager.get_config_path()
    print(f"Configuration file: {config_file}\n{"=" * 60}")
    
    if args.raw:
        try:
            with open(config_file, 'r') as f:
                print(f.read())
        except Exception as e:
            print(f"[ERROR] Reading config file: {e}")
    else:
        # show formatted configuration
        config = config_manager.config
        
        # model configuration
        print(f"\nModel Configuration:\n{"-" * 40}")
        model_config = config.get("model", {})
        for key, value in model_config.items():
            if key == "api_key" and isinstance(value, str) and len(value) > 8:
                masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
                print(f"  {key}: {masked}")
            else:
                print(f"  {key}: {value}")
        
        # generation configuration
        print(f"\nGeneration Configuration:\n{"-" * 40}")
        gen_config = config.get("generation", {})
        for key, value in gen_config.items():
            print(f"  {key}: {value}")
        
        # templates configuration
        print(f"\nTemplates Configuration:\n{"-" * 40}")
        templates_config = config.get("templates", {})
        for key, value in templates_config.items():
            print(f"  {key}: {value}")
        
        # validation status
        print(f"\nValidation:\n{"-" * 40}")
        is_valid, message = config_manager.validate_model_config()
        status = "[SUCCESS] Valid" if is_valid else "[FAIL] Invalid"
        print(f"  Status: {status}")
        if not is_valid:
            print(f"  Message: {message}")


def config_edit_command(args):
    """Edit configuration with default editor. COMMAND: `l2p config edit`"""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    config_file = config_manager.get_config_path()
    
    # determine editor
    editor = os.environ.get('EDITOR')
    if not editor:
        # try common editors
        for candidate in ['nano', 'vim', 'vi', 'code', 'subl']:
            if subprocess.run(['which', candidate], capture_output=True).returncode == 0:
                editor = candidate
                break
    
    if not editor:
        raise CLIError(
            "[ERROR] No editor found.",
            [
                "Set EDITOR environment variable: export EDITOR='your-editor'",
                "Or install a common editor like nano, vim, or VSCode"
            ]
        )
    
    print(f"Opening configuration in {editor}: {config_file}")
    
    try:
        subprocess.run([editor, config_file], check=True)
        
        # reload and validate
        config_manager.load_config()
        
        # detect backend from config_path and fix if misaligned
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
            print(f"(i) Auto-updated backend from '{current_backend}' to '{suggested_backend}' to match config_path")
        
        is_valid, message = config_manager.validate_model_config()
        
        if is_valid:
            print("[SUCCESS] Configuration updated and valid.")
        else:
            print(f"[WARNING] Configuration updated but has issues: {message}")
            
    except subprocess.CalledProcessError as e:
        raise CLIError(
            f"[ERROR] Editor command failed: {e}",
            ["Check EDITOR environment variable", "Ensure editor is properly installed"]
        )
    except Exception as e:
        raise CLIError(
            f"[ERROR] Failed to edit configuration: {e}",
            ["Check file permissions", "Try editing manually: {config_file}"]
        )


def config_reset_command(args):
    """Reset configuration to defaults. COMMAND: `l2p config reset`"""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    config_file = config_manager.get_config_path()
    
    if not args.force:
        print(f"[WARNING] This will reset configuration to defaults.")
        print(f"Current configuration at: {config_file}")
        response = input("Are you sure? (y/N): ").strip().lower()
        if response != 'y':
            print("Reset cancelled.")
            return
    
    try:
        config_manager.reset_to_defaults()
        print("[SUCCESS] Configuration reset to defaults.")
        print(f"\nNew configuration at: {config_file}")
        
        # show new configuration
        config = config_manager.config
        print("\nDefault configuration:")
        print("-" * 40)
        print("Model: ", config.get("model", {}).get("provider"), "/", config.get("model", {}).get("model"))
        print("API Key: Using environment variable reference")
        print("\nRun 'l2p init' to customize configuration.")
        
    except Exception as e:
        raise CLIError(
            f"[ERROR] Failed to reset configuration: {e}",
            ["Check file permissions", "Try removing config file manually: rm {config_file}"]
        )


def config_validate_command(args):
    """Validate current configuration. COMMAND: `l2p config validate`"""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    config_file = config_manager.get_config_path()
    print(f"Validating configuration: {config_file}\n{"=" * 60}")
    
    # validate model configuration
    print(f"\n1. Model Configuration:\n{"-" * 40}")
    model_config = config_manager.get_model_config()
    
    checks = [
        ("Provider", model_config.get("provider"), bool(model_config.get("provider"))),
        ("Model", model_config.get("model"), bool(model_config.get("model"))),
        ("Config Path", model_config.get("config_path"), bool(model_config.get("config_path"))),
        ("API Key", "Set" if model_config.get("api_key") else "Missing", bool(model_config.get("api_key"))),
    ]
    
    all_passed = True
    for name, value, passed in checks:
        status = "[SUCCESS]" if passed else "[FAIL]"
        print(f"  {status} {name}: {value}")
        if not passed:
            all_passed = False
    
    # validate config file exists
    config_path = model_config.get("config_path")
    if config_path:
        if config_path.startswith("l2p/"):
            print(f"  Using package config: {config_path}")
        else:
            config_file_path = Path(config_path).expanduser().resolve()
            if config_file_path.exists():
                print(f"  [SUCCESS] Config file exists: {config_file_path}")
            else:
                print(f"  [FAIL] Config file not found: {config_path}")
                all_passed = False
    
    # overall validation
    print(f"\n2. Overall Validation:\n{"-" * 40}")
    is_valid, message = config_manager.validate_model_config()
    
    if is_valid and all_passed:
        print(
            f"\n[SUCCESS] Configuration is valid and ready to use."
            f"\n\nNext steps:"
            f"\n    > Test connection: `l2p models test`"
            f"\n    > Generate components: `l2p generate types --desc \"your domain\"`"
        )
    else:
        print("[FAIL] Configuration has issues.")
        if message:
            print(f"\nIssues: {message}")
        print(
            f"\nTo fix:"
            f"\n    > Run 'l2p init' to reconfigure"
            f"\n    > Or edit configuration: l2p config edit"
            )