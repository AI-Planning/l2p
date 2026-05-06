"""
Generation commands for L2P CLI.

Generate PDDL components using configured LLM models.
"""

import sys
import argparse

from l2p.domain_builder import DomainBuilder
from l2p.task_builder import TaskBuilder
from l2p.cli.utils.config import CLIError, get_config_manager
from l2p.cli.utils.templates import get_template_manager
from l2p.cli.utils.errors import handle_error
from l2p.llm.base import resolve_config_path


def add_subparser(subparsers):
    """Add generate command subparser."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate PDDL domain or tasks",
        description="Generate PDDL domain and task components using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:  
  # Generate domain
  l2p generate domain --max-retries <n>

  # Generate problem/task
  l2p generate problem --max-retries <n>
        """,
    )
    
    subparsers = parser.add_subparsers(
        dest="generate_command",
        title="generate commands",
        description="Available generation subcommands",
        metavar="COMMAND",
        required=True
    )
    
    from .generators.domain import add_subparser as add_domain_parser
    from .generators.problem import add_subparser as add_problem_parser

    add_domain_parser(subparsers)
    add_problem_parser(subparsers)


def generate_command(args):
    """Execute generate command."""
    try:
        # this function is just a dispatcher - the actual work is done
        # by the subcommand functions set via set_defaults
        if hasattr(args, 'func'):
            args.func(args)
        else:
            print("Error: No generate subcommand specified.", file=sys.stderr)
            print("Use `l2p generate --help` for usage information.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        handle_error(e)
        sys.exit(1)


class GeneratorBase:
    """Base class for PDDL generators."""
    
    def __init__(self, config_manager=None, template_manager=None):
        """Initialize generator.
        
        Args:
            config_manager: ConfigManager instance.
            template_manager: TemplateManager instance.
        """
        self.config_manager = config_manager or get_config_manager()
        self.template_manager = template_manager or get_template_manager(self.config_manager)
        self.domain_builder = DomainBuilder()
        self.problem_builder = TaskBuilder()
        self.llm = None
        self.llm = self.load_llm()
    
    def load_llm(self):
        """Load and initialize LLM from configuration.
        
        Returns:
            Initialized LLM instance.
            
        Raises:
            CLIError: If LLM cannot be loaded.
        """
        if self.llm is not None:
            return self.llm
        
        try:
            model_config = self.config_manager.get_model_config()
            
            backend = model_config.get("backend", "unified")
            provider = model_config.get("provider")
            model = model_config.get("model")
            config_path = model_config.get("config_path")
            api_key = model_config.get("api_key")
            
            if not provider or not model:
                raise CLIError(
                    "[ERROR] Model not configured.",
                    ["Run 'l2p init' to configure model settings first."]
                )
            
            try:
                config_path = resolve_config_path(config_path)
            except FileNotFoundError as e:
                raise CLIError(str(e), [
                    "[ERROR] Check config_path in your configuration",
                    "Run 'l2p config show' to see current config",
                    "Run 'l2p init' to reconfigure"
                ])
            
            if backend == "openai":
                from l2p.llm.openai import OPENAI
                print(f"Loading LLM: {provider}/{model} (OpenAI SDK backend)")
                self.llm = OPENAI(
                    provider=provider,
                    model=model,
                    config_path=config_path,
                    api_key=api_key
                )
            else:
                from l2p.llm.unified import UnifiedLLM
                print(f"Loading LLM: {provider}/{model} (Unified backend)")
                self.llm = UnifiedLLM(
                    provider=provider,
                    model=model,
                    config_path=config_path,
                    api_key=api_key
                )
            
            return self.llm
            
        except ImportError as e:
            raise CLIError(
                f"[ERROR] Failed to import LLM class: {e}",
                [
                    "For Unified backend: pip install llm tiktoken",
                    "For OpenAI SDK backend: pip install openai tiktoken"
                ]
            )
        except Exception as e:
            raise CLIError(
                f"[ERROR] Failed to load LLM: {e}",
                [
                    "Check model configuration with 'l2p config show'",
                    "Test connection with 'l2p models test'",
                    "Ensure API keys are valid and set"
                ]
            )