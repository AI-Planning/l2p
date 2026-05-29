"""
Initialization command for L2P CLI.

Sets up model configuration and creates config file.
"""

import argparse
import sys

from l2p.cli.utils.config import get_config_manager
from l2p.cli.utils.errors import handle_error
from l2p.cli.utils.helpers import _input_or_exit, YELLOW, RESET

VALID_BACKENDS = {"unified", "openai"}
VALID_PROVIDERS = [
    "openai",
    "google",
    "anthropic",
    "deepseek",
    "mistral",
    "ollama",
    "ollama-cloud",
]


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
   
  # Non-interactive setup for Unified backend (simonw/llm)
  l2p init --backend unified --provider openai --model gpt-4o-mini
   
  # Non-interactive setup for OpenAI SDK backend
  l2p init --backend openai --provider deepseek --model deepseek-chat
   
  # With custom config path
  l2p init --config ~/.l2p/custom-config.yaml
        """,
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["unified", "openai"],
        help="LLM backend: unified (simonw/llm) or openai (direct OpenAI SDK)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        help="LLM provider (openai, google, anthropic, ollama, ollama-cloud, etc.)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., gpt-4o-mini, gemini-pro, claude-3-haiku)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set via environment variable like OPENAI_API_KEY)",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to LLM configuration YAML (default: l2p/llm/utils/llm.yaml)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration without prompting",
    )

    parser.set_defaults(func=init_command)


def init_command(args):
    """Execute init command."""

    print(f"Type {YELLOW}/exit{RESET} at any prompt to quit\n")

    try:
        config_manager = get_config_manager(args.config)

        # check if config already exists
        config_file = config_manager.get_config_path()
        if config_file.exists() and not args.force:
            print(f"Configuration already exists at: {config_file}")
            response = _input_or_exit("Overwrite? (y/N): ").strip().lower()
            if response != "y":
                print("Init cancelled.")
                return

        # non-interactive mode when provider and model are given via CLI args
        non_interactive = bool(args.provider and args.model)

        config_updates = {"model": {}}

        # 1) Backend
        backend = args.backend
        if not backend:
            print(
                f"\n{"="*50}"
                f"\nL2P Configuration Setup"
                f"\n{"="*50}"
                f"\nAvailable LLM backends:"
                f"\n     > unified - Use simonw/llm (supports all providers via plugins)"
                f"\n     > openai - Use OpenAI SDK directly (for OpenAI-compatible APIs)"
            )
            while True:
                backend = _input_or_exit("\nEnter backend (default: unified): ").lower()
                if not backend:
                    backend = "unified"
                    break
                if backend in VALID_BACKENDS:
                    break
                print(f"Invalid backend '{backend}'. Valid options: unified, openai")

        config_updates["model"]["backend"] = backend

        # 2) Providers
        provider = args.provider
        if not provider:
            print(
                f"\n{"="*50}"
                f"\nL2P Configuration Setup\n{"="*50}"
                f"\nAvailable LLM providers:"
                f"\n    > openai         - OpenAI models"
                f"\n    > google         - Google models (Gemini)"
                f"\n    > anthropic      - Anthropic models (Claude)"
                f"\n    > deepseek       - DeepSeek models"
                f"\n    > mistral        - Mistral models"
                f"\n    > ollama         - Local Ollama models (currently only available on UNIFIED backend)"
                f"\n    > ollama-cloud   - Ollama cloud models (currently only available on OPENAI backend)"
            )
            while True:
                provider = _input_or_exit("\nEnter provider name: ").lower()
                if provider in VALID_PROVIDERS:
                    break
                print(
                    f"[ERROR] Invalid provider '{provider}'. Valid options: {', '.join(VALID_PROVIDERS)}"
                )

        config_updates["model"]["provider"] = provider

        model = args.model
        if not model:
            while True:
                model = _input_or_exit(
                    f"\nEnter model name for {provider} (e.g., {get_example_model(provider)}): "
                ).strip()
                if model:
                    break
                print(f"[ERROR] Model name is required.")

        config_updates["model"]["model"] = model

        # 3. Configuration Path
        config_path = args.config_path
        if not config_path:
            if non_interactive:
                config_path = (
                    "l2p/llm/utils/llm.yaml"
                    if backend == "unified"
                    else "l2p/llm/utils/openaiSDK.yaml"
                )
            else:
                default_config = (
                    "l2p/llm/utils/llm.yaml"
                    if backend == "unified"
                    else "l2p/llm/utils/openaiSDK.yaml"
                )
                user_input = _input_or_exit(
                    f"\nPath to LLM configuration YAML (default: {default_config}): "
                ).strip()
                config_path = user_input if user_input else default_config

        config_updates["model"]["config_path"] = config_path

        # auto-detect backend from config_path if there's a mismatch
        final_backend = config_updates["model"].get("backend", backend)
        if config_path.endswith("openaiSDK.yaml") and final_backend != "openai":
            print(f"(i) Auto-updating backend to 'openai' to match config_path")
            config_updates["model"]["backend"] = "openai"
        elif config_path.endswith("llm.yaml") and final_backend != "unified":
            print(f"(i) Auto-updating backend to 'unified' to match config_path")
            config_updates["model"]["backend"] = "unified"

        # 4. API key
        api_key = args.api_key
        if not api_key:
            if provider == "ollama":
                api_key = ""
            else:
                env_var = get_api_key_env_var(provider)
                print(f"\nAPI key for {provider}:")
                print(f'> Set environment variable: export {env_var}="your-key"')
                print(
                    f"> Or enter API key directly (will be stored in config): ", end=""
                )
                user_input = _input_or_exit().strip()
                if user_input:
                    api_key = user_input
                else:
                    api_key = env_var
                    print(f"Using environment variable reference: {api_key}")

        config_updates["model"]["api_key"] = api_key
        config_manager.update_config(config_updates)  # update configuration

        print(
            f"\n[SUCCESS] Configuration saved to: {config_file}\n\nTesting configuration..."
        )
        is_valid, message = config_manager.validate_model_config()

        if is_valid:
            print("[SUCCESS] Configuration is valid.")
            llm_name = "OPENAI" if backend == "openai" else "UnifiedLLM"
            try:
                # NOTE: we don't actually connect here, just validate config
                print(
                    f"\nInitializing {provider}/{model} via {llm_name}..."
                    f"\n\nNext steps:"
                )
                if backend == "unified" and provider == "ollama":
                    print(
                        f"\n1. Make sure the Ollama plugin is installed:"
                        f"\n   `llm install llm-ollama`"
                        f"\n2. Verify Ollama server is running:"
                        f"\n   `ollama list`"
                    )
                    steps_offset = 3
                elif backend == "unified":
                    print(
                        f"\n1. Set your API key if using environment variable:"
                        f'\n   export {get_api_key_env_var(provider)}="your-key"'
                    )
                    steps_offset = 2
                else:
                    print(
                        f"\n1. Set your API key if using environment variable:"
                        f'\n   export {get_api_key_env_var(provider)}="your-key"'
                    )
                    steps_offset = 2

                print(
                    f"\n{steps_offset}. Test connection:"
                    f"\n   `l2p models test`"
                    f"\n{steps_offset + 1}. Generate your first PDDL domain:"
                    f"\n   `l2p generate domain`"
                )

            except ImportError as e:
                print(f"[ERROR]  Could not import {llm_name}: {e}\nTroubleshooting:")
                if backend == "unified":
                    print(" > Install CLI dependencies: pip install llm tiktoken")
                else:
                    print(" > Install CLI dependencies: pip install openai tiktoken")

        else:
            print(f"[WARNING]  Configuration issue: {message}")
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
        "openai": "gpt-4o-mini, o1-mini, gpt-5",
        "google": "gemini-2.0-flash, gemini-2.5-pro",
        "anthropic": "claude-3-haiku, claude-3.5-sonnet",
        "deepseek": "deepseek-chat, deepseek-reasoner",
        "mistral": "mistral-small, mistral-large",
        "ollama": "llama3.1:8b, deepseek-r1:32b",
        "ollama-cloud": "gemma4:31b-cloud, deepseek-v4-pro:cloud",
    }
    return examples.get(provider, "model-name")


def get_api_key_env_var(provider: str) -> str:
    """Get environment variable name for provider's API key."""
    NO_KEY_PROVIDERS = {"ollama"}
    if provider in NO_KEY_PROVIDERS:
        return ""
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "ollama-cloud": "OLLAMA_API_KEY",
    }
    return env_vars.get(provider, f"{provider.upper()}_API_KEY")
