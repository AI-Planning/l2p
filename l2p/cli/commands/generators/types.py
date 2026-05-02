"""
Types generator for L2P CLI.

Generates PDDL types from domain descriptions.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from ..generate import GeneratorBase
from ...utils.config import get_config_manager, CLIError
from ...utils.templates import get_template_manager
from ...utils.errors import handle_error


def add_subparser(subparsers):
    """Add types generator subparser."""
    parser = subparsers.add_parser(
        "types",
        help="Generate PDDL types",
        description="Generate PDDL types (flat or hierarchical) from domain description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate flat types
  l2p generate types --desc "blocksworld domain"
  
  # Generate hierarchical types  
  l2p generate types --desc "blocksworld" --hierarchy
  
  # With existing types for refinement
  l2p generate types --desc "domain" --types-file existing_types.json
  
  # Output in different formats
  l2p generate types --desc "domain" --format json --output types.json
  l2p generate types --desc "domain" --format yaml --output types.yaml
  l2p generate types --desc "domain" --format pddl --output types.pddl
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--desc",
        type=str,
        required=True,
        help="Domain description (text or path to file)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--hierarchy",
        action="store_true",
        help="Generate hierarchical types instead of flat"
    )
    
    parser.add_argument(
        "--types-file",
        type=str,
        help="Existing types file (JSON/YAML) for refinement"
    )
    
    parser.add_argument(
        "--template-file",
        type=str,
        help="Custom template file (default: formalize_type.txt or formalize_type_hierarchy.txt)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["pddl", "json", "yaml"],
        default="pddl",
        help="Output format (default: pddl)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (default: stdout)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for LLM (default: 3)"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable syntax validation"
    )
    
    parser.set_defaults(func=generate_types_command)


def generate_types_command(args):
    """Execute types generation command."""
    try:
        generator = TypesGenerator()
        generator.generate(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


class TypesGenerator(GeneratorBase):
    """Generator for PDDL types."""
    
    def generate(self, args):
        """Generate types based on command line arguments."""
        # Load domain description (could be file or text)
        domain_desc = self._load_description(args.desc)
        
        # Load existing types if provided
        existing_types = None
        if args.types_file:
            existing_types = self.load_input_file(args.types_file, "existing types")
        
        # Determine template
        if args.template_file:
            template_content = self.load_input_file(args.template_file, "template")
        else:
            template_name = "formalize_type_hierarchy.txt" if args.hierarchy else "formalize_type.txt"
            template_content = self.template_manager.get_template(template_name, "domain")
        
        # Load LLM
        llm = self.load_llm()
        
        # Load DomainBuilder
        from l2p import DomainBuilder, SyntaxValidator
        
        domain_builder = DomainBuilder()
        
        # Configure syntax validator
        syntax_validator = None
        if not args.no_validation:
            syntax_validator = SyntaxValidator()
            syntax_validator.error_types = ["validate_format_types"]
            if args.hierarchy:
                syntax_validator.error_types.append("validate_cyclic_types")
        
        print(f"Generating {'hierarchical' if args.hierarchy else 'flat'} types...")
        
        # Call appropriate method
        if args.hierarchy:
            result = domain_builder.formalize_type_hierarchy(
                model=llm,
                domain_desc=domain_desc,
                prompt_template=template_content,
                types=existing_types,
                syntax_validator=syntax_validator,
                max_retries=args.max_retries
            )
            types_result, llm_output, validation_info = result
        else:
            result = domain_builder.formalize_types(
                model=llm,
                domain_desc=domain_desc,
                prompt_template=template_content,
                types=existing_types,
                syntax_validator=syntax_validator,
                max_retries=args.max_retries
            )
            types_result, llm_output, validation_info = result
        
        # Check validation results
        if validation_info and not validation_info[0]:
            print(f"⚠ Validation warning: {validation_info[1]}")
        
        # Format output
        if args.format == "pddl":
            from l2p.utils.pddl_format import format_types_to_string
            output_content = format_types_to_string(types_result)
        else:
            output_content = self.format_for_output(types_result, args.format)
        
        # Save output
        self.save_output(output_content, args.output, args.format)
        
        # Show token usage
        input_tokens, output_tokens = llm.get_tokens()
        if input_tokens or output_tokens:
            print(f"Token usage: {input_tokens} input, {output_tokens} output")
        
    def _load_description(self, desc_input: str) -> str:
        """Load domain description from file or text.
        
        Args:
            desc_input: Either description text or path to file.
            
        Returns:
            Domain description string.
        """
        # Check if it's a file path
        desc_path = Path(desc_input)
        if desc_path.exists() and desc_path.is_file():
            try:
                return desc_path.read_text().strip()
            except Exception as e:
                raise CLIError(
                    f"Failed to read description file: {e}",
                    ["Check file permissions and encoding", "Ensure file contains text"]
                )
        
        # Otherwise treat as text
        return desc_input