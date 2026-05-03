"""
Model management commands for L2P CLI.

List available models, test connections, and show model information.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.config import ConfigManager, CLIError, get_config_manager
from ..utils.errors import handle_error
from ...llm.base import resolve_config_path


def _input_or_exit(prompt: str = "") -> str:
    value = input(prompt).strip()
    if value == "'exit":
        print("Operation cancelled.")
        sys.exit(0)
    return value


def add_subparser(subparsers):
    """Add models command subparser."""
    parser = subparsers.add_parser(
        "models",
        help="Manage LLM models",
        description="List available models and test connections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models from configured provider
  l2p models list
  
  # List models for a specific provider
  l2p models list --provider openai
  
  # Interactively switch to a different model
  l2p models switch
  
  # Test connection to configured model
  l2p models test
  
  # Test specific model
  l2p models test --provider openai --model gpt-4o-mini
        """,
    )
    
    subparsers = parser.add_subparsers(
        dest="models_command",
        title="models commands",
        description="Available models subcommands",
        metavar="COMMAND",
        required=True
    )
    
    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List available models",
        description="List models available for a provider."
    )
    list_parser.add_argument(
        "--provider",
        type=str,
        help="Filter by provider (default: use configured provider)"
    )
    list_parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed model information"
    )
    list_parser.set_defaults(func=list_models_command)
    
    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test model connection",
        description="Test connection to a model."
    )
    test_parser.add_argument(
        "--provider",
        type=str,
        help="Provider to test (default: use configured provider)"
    )
    test_parser.add_argument(
        "--model",
        type=str,
        help="Model to test (default: use configured model)"
    )
    test_parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple test without loading full model"
    )
    test_parser.set_defaults(func=test_model_command)
    
    # Switch command
    switch_parser = subparsers.add_parser(
        "switch",
        help="Switch to a different model",
        description="Interactively select a model for the configured provider."
    )
    switch_parser.set_defaults(func=switch_model_command)


def models_command(args):
    """Execute models command."""
    try:
        args.func(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


def list_models_command(args):
    """List available models for a provider."""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    # Determine provider
    provider = args.provider
    if not provider:
        model_config = config_manager.get_model_config()
        provider = model_config.get("provider")
        if not provider:
            raise CLIError(
                "No provider configured.",
                ["Run 'l2p init' to configure a provider first."]
            )
    
    # Load provider configuration
    model_config = config_manager.get_model_config()
    config_path = model_config.get("config_path", "l2p/llm/utils/llm.yaml")
    
    try:
        # Load config file
        if config_path.startswith("l2p/"):
            # Extract the YAML filename from config path
            # e.g. "l2p/llm/utils/openaiSDK.yaml" -> "openaiSDK.yaml"
            parts = config_path.replace("\\", "/").split("/")
            yaml_filename = parts[-1] if parts else "llm.yaml"
            
            # Try to load from package
            try:
                import importlib.resources
                config_content = importlib.resources.read_text("l2p.llm.utils", yaml_filename)
                config = yaml.safe_load(config_content)
            except Exception:
                # Fall back to file system
                package_root = Path(__file__).parent.parent.parent.parent
                config_file = package_root / "l2p" / "llm" / "utils" / yaml_filename
                if not config_file.exists():
                    # Try the other YAML as last resort
                    fallback = "openaiSDK.yaml" if yaml_filename == "llm.yaml" else "llm.yaml"
                    config_file = package_root / "l2p" / "llm" / "utils" / fallback
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
        else:
            # Load from custom path
            config_file = Path(config_path).expanduser().resolve()
            if not config_file.exists():
                raise CLIError(
                    f"Config file not found: {config_path}",
                    ["Check config path in your configuration", "Run 'l2p config show' to see current config"]
                )
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        
        # Get models for provider
        provider_config = config.get(provider, {})
        if not provider_config:
            available_providers = list(config.keys())
            raise CLIError(
                f"Provider '{provider}' not found in configuration.",
                [
                    f"Available providers: {', '.join(available_providers)}",
                    f"Check config file: {config_path}",
                    "Run 'l2p init' to reconfigure with valid provider"
                ]
            )
        
        # return models excluding `base_url` argument
        models = list(provider_config.keys() - {"base_url"})
        
        # Display results
        print(f"\nAvailable models for '{provider}' provider:")
        print("=" * 60)
        
        if not models:
            print("No models found for this provider.")
            return
        
        if args.details:
            for model_name in sorted(models):
                model_info = provider_config[model_name]
                print(f"\n{model_name}:")
                print(f"  Family: {model_info.get('model_family', 'N/A')}")
                print(f"  Alias: {model_info.get('model_alias', model_name)}")
                print(f"  Context: {model_info.get('model_context_length', 'N/A')} tokens")
                
                # Show cost if available
                cost = model_info.get('cost_usd_mtok', {})
                if cost:
                    input_cost = cost.get('input', 0)
                    output_cost = cost.get('output', 0)
                    if input_cost or output_cost:
                        print(f"  Cost: ${input_cost}/M input, ${output_cost}/M output")
                
                # Show parameters
                params = model_info.get('model_params', {})
                if params:
                    print(f"  Parameters: {', '.join(params.keys())}")
        else:
            for i, model_name in enumerate(sorted(models), 1):
                model_info = provider_config[model_name]
                family = model_info.get('model_family', '')
                alias = model_info.get('model_alias', '')
                
                display = model_name
                if alias and alias != model_name:
                    display += f" (alias: {alias})"
                if family:
                    display += f" [{family}]"
                
                print(f"{i:2}. {display}")
        
        print(f"\nTotal: {len(models)} models")
        
        # Show current configuration
        model_config = config_manager.get_model_config()
        current_provider = model_config.get("provider")
        current_model = model_config.get("model")
        
        if current_provider == provider and current_model in models:
            print(f"\n✓ Currently configured: {current_provider}/{current_model}")
        elif current_provider == provider:
            print(f"\n⚠ Configured model '{current_model}' not found in list")
        
    except Exception as e:
        raise CLIError(
            f"Failed to list models: {e}",
            [
                f"Check config file: {config_path}",
                "Ensure the file has valid YAML format",
                "Run 'l2p init' to reconfigure"
            ]
        )


def test_model_command(args):
    """Test connection to a model."""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    # Get model configuration
    model_config = config_manager.get_model_config().copy()
    
    # Override with command line arguments if provided
    if args.provider:
        model_config["provider"] = args.provider
    if args.model:
        model_config["model"] = args.model
    
    provider = model_config.get("provider")
    model = model_config.get("model")
    config_path = model_config.get("config_path")
    api_key = model_config.get("api_key", "")
    
    if not provider or not model:
        raise CLIError(
            "Provider and model not configured.",
            ["Run 'l2p init' to configure model settings", "Or specify with --provider and --model flags"]
        )
    
    print(f"\nTesting connection to: {provider}/{model}")
    print("=" * 60)
    
    # Check API key
    if api_key.endswith("_API_KEY"):
        env_key = os.getenv(api_key)
        if not env_key:
            if provider == "ollama":
                print("ℹ No API key needed for local Ollama")
                api_key_status = True
            else:
                print(f"⚠ API key not set: environment variable {api_key} is empty")
                print(f"  Set it with: export ${api_key}=\"your-key\"")
                api_key_status = False
        else:
            print(f"✓ API key: Using environment variable {api_key}")
            api_key_status = True
            model_config["api_key"] = env_key
    elif api_key:
        if len(api_key) > 8:
            masked = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            print(f"✓ API key: {masked}")
        else:
            print(f"✓ API key: Provided")
        api_key_status = True
    elif provider == "ollama":
        print("ℹ No API key needed for local Ollama")
        api_key_status = True
    else:
        print("⚠ No API key configured")
        api_key_status = False
    
    # Check config file
    config_file = None
    try:
        resolved_path = resolve_config_path(config_path)
        config_file = Path(resolved_path)
        print(f"✓ Config: Found at {config_file}")
        config_exists = True
    except FileNotFoundError as e:
        print(f"✗ Config: {e}")
        config_exists = False
    
    if not api_key_status or not config_exists:
        print("\n✗ Pre-checks failed. Please fix issues above.")
        return
    
    # Simple test (just config validation)
    if args.simple:
        print("\n✅ Configuration looks valid.")
        print("\nTo perform actual connection test, run without --simple flag.")
        return
    
    # Actual connection test
    print("\nAttempting to connect to model...")
    
    # Determine backend from config (safe defaults for error handling)
    backend = model_config.get("backend", "unified")
    llm_name = "OPENAI" if backend == "openai" else "UnifiedLLM"
    
    try:
        if backend == "openai":
            from l2p.llm.openai import OPENAI
            llm_class = OPENAI
        else:
            from l2p.llm.unified import UnifiedLLM
            llm_class = UnifiedLLM
        
        print(f"Initializing {llm_name} for {provider}/{model}...")
        llm = llm_class(
            provider=provider,
            model=model,
            config_path=config_path,
            api_key=model_config.get("api_key")
        )
        
        print("✓ Model initialized successfully")
        
        # Test with a simple prompt
        test_prompt = "Respond with exactly: OK"
        print(f"Sending test prompt: '{test_prompt}'")
        
        response = llm.query(test_prompt, max_retry=1)
        
        if response.strip() == "OK":
            print("✓ Test successful: Model responded correctly")
        else:
            print(f"✓ Model responded: '{response[:50]}...'")
        
        # Show token usage if available
        input_tokens, output_tokens = llm.get_tokens()
        if input_tokens or output_tokens:
            print(f"Token usage: {input_tokens} input, {output_tokens} output")
        
        print("\n✅ Connection test successful!")
        
    except ImportError as e:
        print(f"\n✗ Failed to import {llm_name}: {e}")
        print("\nTroubleshooting:")
        if backend == "openai":
            print("• Install required packages: pip install openai tiktoken")
        else:
            print("• Install required packages: pip install llm tiktoken")
        print("• You can also use other LLM providers directly")
        
    except Exception as e:
        print(f"\n✗ Connection test failed: {e}")
        print("\nTroubleshooting:")
        print("• Check API key validity and permissions")
        print("• Verify model name is correct")
        print("• Check network connectivity")
        print("• Ensure provider service is available")
        print("• Run with --verbose flag for more details")


def _load_models_for_provider(config_manager, provider=None):
    """Load available models for a provider from the YAML config.
    
    Returns:
        Tuple of (provider, models_list, provider_config_dict)
    """
    model_config = config_manager.get_model_config()
    if not provider:
        provider = model_config.get("provider")
        if not provider:
            raise CLIError(
                "No provider configured.",
                ["Run 'l2p init' to configure a provider first."]
            )
    
    config_path = model_config.get("config_path", "l2p/llm/utils/llm.yaml")
    
    # Load config file
    if config_path.startswith("l2p/"):
        parts = config_path.replace("\\", "/").split("/")
        yaml_filename = parts[-1] if parts else "llm.yaml"
        try:
            import importlib.resources
            config_content = importlib.resources.read_text("l2p.llm.utils", yaml_filename)
            config = yaml.safe_load(config_content)
        except Exception:
            package_root = Path(__file__).parent.parent.parent.parent
            config_file = package_root / "l2p" / "llm" / "utils" / yaml_filename
            if not config_file.exists():
                fallback = "openaiSDK.yaml" if yaml_filename == "llm.yaml" else "llm.yaml"
                config_file = package_root / "l2p" / "llm" / "utils" / fallback
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
    else:
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.exists():
            raise CLIError(
                f"Config file not found: {config_path}",
                ["Check config path in your configuration", "Run 'l2p config show' to see current config"]
            )
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    
    provider_config = config.get(provider, {})
    if not provider_config:
        available_providers = list(config.keys())
        raise CLIError(
            f"Provider '{provider}' not found in configuration.",
            [
                f"Available providers: {', '.join(available_providers)}",
                f"Check config file: {config_path}",
                "Run 'l2p init' to reconfigure with valid provider"
            ]
        )
    
    models = sorted(provider_config.keys() - {"base_url"})
    return provider, models, provider_config


def switch_model_command(args):
    """Interactively switch to a different model."""
    config_manager = get_config_manager(args.config if hasattr(args, 'config') else None)
    
    try:
        provider, models, provider_config = _load_models_for_provider(config_manager)
    except CLIError as e:
        print(f"\n{e}")
        sys.exit(1)
    
    if not models:
        print(f"\nNo models found for provider '{provider}'.")
        return
    
    current_model = config_manager.get_model_config().get("model", "")
    
    print(f"\nAvailable models for '{provider}' provider:")
    print("=" * 60)
    for i, model_name in enumerate(models, 1):
        model_info = provider_config[model_name]
        alias = model_info.get('model_alias', '')
        display = model_name
        if alias and alias != model_name:
            display += f" (alias: {alias})"
        marker = " ← current" if model_name == current_model else ""
        print(f"{i:2}. {display}{marker}")
    print()
    
    while True:
        choice = _input_or_exit("Select model by number or name: ").strip()
        if not choice:
            continue
        
        # Try as number
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                selected = models[idx - 1]
                break
            print(f"Invalid number. Enter 1-{len(models)}.")
            continue
        except ValueError:
            pass
        
        # Try as name
        matching = [m for m in models if m.lower() == choice.lower()]
        if matching:
            selected = matching[0]
            break
        
        print(f"Invalid model '{choice}'. Valid options: {', '.join(models)}")
    
    if selected == current_model:
        print(f"\nAlready configured: {provider}/{selected}")
        return
    
    config_manager.update_config({"model": {"model": selected}})
    print(f"\n✅ Switched to {provider}/{selected}")