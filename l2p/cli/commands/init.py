"""
Initialization command for L2P CLI.

Sets up model configuration and creates config file.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.config import ConfigManager, CLIError, get_config_manager
from ..utils.errors import handle_error


def add_subparser(subparsers):
    """Add init command subparser."""
    parser = subparsers.add_parser(
        "init",
        help="Initialize L2P configuration",
        description="Set up model configuration and create config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive initialization
  l2p init
  
  # Non-interactive setup for OpenAI
  l2p init --provider openai --model gpt-4o-mini
  
  # With custom config path
  l2p init --config ~/.l2p/custom-config.yaml
        """,
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        help="LLM provider (openai, google, anthropic, ollama, etc.)"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        help="Model name (e.g., gpt-4o-mini, gemini-pro, claude-3-haiku)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set via environment variable like OPENAI_API_KEY)"
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to LLM configuration YAML (default: l2p/llm/utils/llm.yaml)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration without prompting"
    )
    
    parser.set_defaults(func=init_command)


def init_command(args):
    """Execute init command."""
    try:
        config_manager = get_config_manager(args.config)
        
        # Check if config already exists
        config_file = config_manager.get_config_path()
        if config_file.exists() and not args.force:
            print(f"Configuration already exists at: {config_file}")
            response = input("Overwrite? (y/N): ").strip().lower()
            if response != 'y':
                print("Init cancelled.")
                return
        
        # Non-interactive mode when provider and model are given via CLI args
        non_interactive = bool(args.provider and args.model)
        
        config_updates = {"model": {}}
        
        # Provider
        provider = args.provider
        if not provider:
            print("\n" + "="*50)
            print("L2P Configuration Setup")
            print("="*50)
            print("\nAvailable LLM providers:")
            print("• openai - OpenAI models (GPT-4o, o1, etc.)")
            print("• google - Google models (Gemini)")
            print("• anthropic - Anthropic models (Claude)")
            print("• deepseek - DeepSeek models")
            print("• mistral - Mistral models")
            print("• ollama - Local Ollama models")
            print("\nEnter provider name: ", end="")
            provider = input().strip().lower()
        
        if not provider:
            raise CLIError("Provider is required.")
        
        config_updates["model"]["provider"] = provider
        
        # Model
        model = args.model
        if not model:
            print(f"\nEnter model name for {provider} (e.g., {get_example_model(provider)}): ", end="")
            model = input().strip()
        
        if not model:
            raise CLIError("Model name is required.")
        
        config_updates["model"]["model"] = model
        
        # Config path
        config_path = args.config_path
        if not config_path:
            if non_interactive:
                config_path = "l2p/llm/utils/llm.yaml"
            else:
                default_config = "l2p/llm/utils/llm.yaml"
                print(f"\nPath to LLM configuration YAML (default: {default_config}): ", end="")
                user_input = input().strip()
                config_path = user_input if user_input else default_config
        
        config_updates["model"]["config_path"] = config_path
        
        # API key
        api_key = args.api_key
        if not api_key:
            if provider == "ollama":
                api_key = ""
            else:
                env_var = get_api_key_env_var(provider)
                print(f"\nAPI key for {provider}:")
                print(f"• Set environment variable: export {env_var}=\"your-key\"")
                print(f"• Or enter API key directly (will be stored in config): ", end="")
                user_input = input().strip()
                
                if user_input:
                    api_key = user_input
                else:
                    api_key = f"${{{env_var}}}"
                    print(f"Using environment variable reference: {api_key}")
        
        config_updates["model"]["api_key"] = api_key
        
        # Update configuration
        config_manager.update_config(config_updates)
        
        print(f"\n✅ Configuration saved to: {config_file}")
        
        # Test configuration
        print("\nTesting configuration...")
        is_valid, message = config_manager.validate_model_config()
        
        if is_valid:
            print("✅ Configuration is valid.")
            
            # Try to initialize model
            try:
                from l2p.llm.unified import UnifiedLLM
                model_config = config_manager.get_model_config()
                
                print(f"\nInitializing {provider}/{model}...")
                # Note: We don't actually connect here, just validate config
                print("✅ Model configuration ready.")
                
                print("\nNext steps:")
                if provider == "ollama":
                    print("1. Make sure the Ollama plugin is installed:")
                    print("   llm install llm-ollama")
                    print("2. Verify Ollama server is running:")
                    print("   ollama list")
                else:
                    print("1. Set your API key if using environment variable:")
                    print(f"   export {get_api_key_env_var(provider)}=\"your-key\"")
                steps_offset = 3 if provider == "ollama" else 2
                print(f"{steps_offset}. Test connection:")
                print("   l2p models test")
                print(f"{steps_offset + 1}. Generate your first PDDL component:")
                print("   l2p generate types --desc \"blocksworld domain\"")
                
            except ImportError as e:
                print(f"⚠️  Could not import UnifiedLLM: {e}")
                print("\nTroubleshooting:")
                print("• Install CLI dependencies: pip install llm tiktoken")
                print("• Or use a different provider")
                
        else:
            print(f"⚠️  Configuration issue: {message}")
            print("\nYou can edit configuration later with:")
            print("  l2p config edit")
        
    except KeyboardInterrupt:
        print("\n\nInit cancelled.")
        sys.exit(0)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


def get_example_model(provider: str) -> str:
    """Get example model name for a provider."""
    examples = {
        "openai": "gpt-4o-mini, o1-mini, gpt-4o",
        "google": "gemini-2.0-flash, gemini-2.5-pro",
        "anthropic": "claude-3-haiku, claude-3.5-sonnet",
        "deepseek": "deepseek-chat, deepseek-reasoner",
        "mistral": "mistral-small, mistral-large",
        "ollama": "llama3.1:8b, deepseek-r1:32b",
    }
    return examples.get(provider, "model-name")


def get_api_key_env_var(provider: str) -> str:
    """Get environment variable name for provider's API key."""
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "ollama": "OLLAMA_API_KEY",  # Usually not needed for local Ollama
    }
    return env_vars.get(provider, f"{provider.upper()}_API_KEY")