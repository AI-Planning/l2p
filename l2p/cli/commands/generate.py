"""
Generation commands for L2P CLI.

Generate PDDL components using configured LLM models.
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.config import ConfigManager, CLIError, get_config_manager
from ..utils.templates import TemplateManager, get_template_manager
from ..utils.errors import handle_error
from ...llm.base import resolve_config_path


def add_subparser(subparsers):
    """Add generate command subparser."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate PDDL components",
        description="Generate PDDL domain and task components using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate types from domain description
  l2p generate types --desc "blocksworld domain"
  
  # Generate predicates with existing types
  l2p generate predicates --desc "blocksworld" --types-file types.json
  
  # Generate a specific action
  l2p generate action --name "pick-up" --desc "pick up a block"
  
  # Generate full domain (pipeline)
  l2p generate domain --desc "blocksworld domain" --pipeline
  
  # Generate problem/task
  l2p generate task --desc "stack blocks A and B" --domain-file domain.pddl
  
  # Output in different formats
  l2p generate types --desc "domain" --format json --output types.json
  l2p generate types --desc "domain" --format yaml --output types.yaml
  l2p generate types --desc "domain" --format pddl --output types.pddl
        """,
    )
    
    subparsers = parser.add_subparsers(
        dest="generate_command",
        title="generate commands",
        description="Available generation subcommands",
        metavar="COMMAND",
        required=True
    )
    
    # Add individual generator subparsers
    from .generators.types import add_subparser as add_types_parser
    from .generators.predicates import add_subparser as add_predicates_parser
    from .generators.constants import add_subparser as add_constants_parser
    from .generators.action import add_subparser as add_action_parser
    from .generators.domain import add_subparser as add_domain_parser
    from .generators.task import add_subparser as add_task_parser
    
    add_types_parser(subparsers)
    add_predicates_parser(subparsers)
    add_constants_parser(subparsers)
    add_action_parser(subparsers)
    add_domain_parser(subparsers)
    add_task_parser(subparsers)


def generate_command(args):
    """Execute generate command."""
    try:
        # This function is just a dispatcher - the actual work is done
        # by the subcommand functions set via set_defaults
        if hasattr(args, 'func'):
            args.func(args)
        else:
            print("Error: No generate subcommand specified.", file=sys.stderr)
            print("Use 'l2p generate --help' for usage information.", file=sys.stderr)
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
        self.llm = None
    
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
                    "Model not configured.",
                    ["Run 'l2p init' to configure model settings first."]
                )
            
            # Resolve config path to catch errors early
            try:
                config_path = resolve_config_path(config_path)
            except FileNotFoundError as e:
                raise CLIError(str(e), [
                    "Check config_path in your configuration",
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
                f"Failed to import LLM class: {e}",
                [
                    "For Unified backend: pip install llm tiktoken",
                    "For OpenAI SDK backend: pip install openai tiktoken"
                ]
            )
        except Exception as e:
            raise CLIError(
                f"Failed to load LLM: {e}",
                [
                    "Check model configuration with 'l2p config show'",
                    "Test connection with 'l2p models test'",
                    "Ensure API keys are valid and set"
                ]
            )
    
    def load_input_file(self, file_path: str, description: str = "input") -> Any:
        """Load input file (JSON, YAML, or text).
        
        Args:
            file_path: Path to input file.
            description: Description for error messages.
            
        Returns:
            Parsed content.
        """
        path = Path(file_path).expanduser().resolve()
        
        if not path.exists():
            raise CLIError(
                f"{description.capitalize()} file not found: {file_path}",
                [
                    f"Check file path: {file_path}",
                    "Use absolute path for better reliability",
                    "Ensure file exists and is readable"
                ]
            )
        
        try:
            suffix = path.suffix.lower()
            
            if suffix == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            elif suffix in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Assume text file
                with open(path, 'r') as f:
                    return f.read().strip()
                
        except Exception as e:
            raise CLIError(
                f"Failed to load {description} file: {e}",
                [
                    f"Check file format: {path.suffix}",
                    "Ensure file has valid format (JSON, YAML, or text)",
                    "Check file permissions and encoding"
                ]
            )
    
    def save_output(self, content: Any, output_path: Optional[str], output_format: str = "pddl"):
        """Save output to file or stdout.
        
        Args:
            content: Content to save.
            output_path: Output file path (None for stdout).
            output_format: Output format (pddl, json, yaml).
        """
        # Convert content to string based on format
        if output_format == "json":
            if isinstance(content, (dict, list)):
                output = json.dumps(content, indent=2)
            else:
                # Try to parse as JSON first, then treat as string
                try:
                    parsed = json.loads(str(content))
                    output = json.dumps(parsed, indent=2)
                except:
                    output = str(content)
        elif output_format == "yaml":
            if isinstance(content, (dict, list)):
                output = yaml.dump(content, default_flow_style=False, sort_keys=False)
            else:
                output = str(content)
        else:  # pddl or default
            output = str(content)
        
        # Write to file or stdout
        if output_path:
            path = Path(output_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(path, 'w') as f:
                    f.write(output)
                print(f"Output written to: {path}")
            except Exception as e:
                raise CLIError(
                    f"Failed to write output: {e}",
                    [
                        f"Check write permissions for: {path.parent}",
                        "Ensure enough disk space",
                        "Try different output path"
                    ]
                )
        else:
            print(output)
    
    def format_for_output(self, content: Any, output_format: str) -> Any:
        """Format content for specified output format.
        
        Args:
            content: Content to format.
            output_format: Desired output format.
            
        Returns:
            Formatted content.
        """
        if output_format == "pddl":
            # For PDDL output, content should already be string
            return str(content)
        elif output_format == "json":
            if isinstance(content, (dict, list)):
                return content
            else:
                # Try to parse as JSON
                try:
                    return json.loads(str(content))
                except:
                    return {"raw": str(content)}
        elif output_format == "yaml":
            if isinstance(content, (dict, list)):
                return content
            else:
                # Try to parse as JSON first
                try:
                    parsed = json.loads(str(content))
                    return parsed
                except:
                    return {"raw": str(content)}
        else:
            return content