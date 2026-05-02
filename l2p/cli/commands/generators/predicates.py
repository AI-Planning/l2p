"""
Predicates generator for L2P CLI.

Generates PDDL predicates from domain descriptions.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..generate import GeneratorBase
from ...utils.config import get_config_manager, CLIError
from ...utils.templates import get_template_manager
from ...utils.errors import handle_error


def add_subparser(subparsers):
    """Add predicates generator subparser."""
    parser = subparsers.add_parser(
        "predicates",
        help="Generate PDDL predicates",
        description="Generate PDDL predicates from domain description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate predicates with types
  l2p generate predicates --desc "blocksworld" --types-file types.json
  
  # With existing predicates for refinement
  l2p generate predicates --desc "domain" --predicates-file existing_preds.json
  
  # With constants
  l2p generate predicates --desc "domain" --constants-file constants.json
  
  # Output in different formats
  l2p generate predicates --desc "domain" --format json --output predicates.json
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
        "--types-file",
        type=str,
        help="Types file (JSON/YAML) for predicate typing"
    )
    
    parser.add_argument(
        "--constants-file",
        type=str,
        help="Constants file (JSON/YAML) for predicate constants"
    )
    
    parser.add_argument(
        "--predicates-file",
        type=str,
        help="Existing predicates file (JSON/YAML) for refinement"
    )
    
    parser.add_argument(
        "--functions-file",
        type=str,
        help="Functions file (JSON/YAML) for context"
    )
    
    parser.add_argument(
        "--template-file",
        type=str,
        help="Custom template file (default: formalize_predicates.txt)"
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
    
    parser.set_defaults(func=generate_predicates_command)


def generate_predicates_command(args):
    """Execute predicates generation command."""
    try:
        generator = PredicatesGenerator()
        generator.generate(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


class PredicatesGenerator(GeneratorBase):
    """Generator for PDDL predicates."""
    
    def generate(self, args):
        """Generate predicates based on command line arguments."""
        # Load domain description
        domain_desc = self._load_description(args.desc)
        
        # Load supporting files
        types = self._load_optional_file(args.types_file, "types")
        constants = self._load_optional_file(args.constants_file, "constants")
        existing_predicates = self._load_optional_file(args.predicates_file, "existing predicates")
        functions = self._load_optional_file(args.functions_file, "functions")
        
        # Determine template
        if args.template_file:
            template_content = self.load_input_file(args.template_file, "template")
        else:
            template_content = self.template_manager.get_template("formalize_predicates.txt", "domain")
        
        # Load LLM
        llm = self.load_llm()
        
        # Load DomainBuilder
        from l2p import DomainBuilder, SyntaxValidator, Predicate
        
        domain_builder = DomainBuilder()
        
        # Convert existing predicates to Predicate objects if needed
        predicate_objects = None
        if existing_predicates:
            if isinstance(existing_predicates, list):
                predicate_objects = []
                for pred in existing_predicates:
                    if isinstance(pred, dict) and 'name' in pred:
                        predicate_objects.append(Predicate(pred['name'], pred.get('params', [])))
                    else:
                        # Try to parse string representation
                        predicate_objects.append(Predicate.from_string(str(pred)))
        
        # Configure syntax validator
        syntax_validator = None
        if not args.no_validation:
            syntax_validator = SyntaxValidator()
            syntax_validator.error_types = [
                "validate_header",
                "validate_duplicate_headers",
                "validate_unsupported_keywords",
                "validate_types_predicates",
                "validate_format_predicates",
                "validate_duplicate_predicates"
            ]
        
        print("Generating predicates...")
        
        # Call generation method
        result = domain_builder.formalize_predicates(
            model=llm,
            domain_desc=domain_desc,
            prompt_template=template_content,
            types=types,
            constants=constants,
            predicates=predicate_objects,
            functions=functions,
            syntax_validator=syntax_validator,
            max_retries=args.max_retries
        )
        
        predicates_result, llm_output, validation_info = result
        
        # Check validation results
        if validation_info and not validation_info[0]:
            print(f"⚠ Validation warning: {validation_info[1]}")
        
        # Format output
        # if args.format == "pddl":
        #     from l2p.utils.pddl_format import format_predicates_to_string
        #     output_content = format_predicates_to_string(predicates_result)
        else:
            # Convert Predicate objects to serializable format
            serializable_predicates = []
            for pred in predicates_result:
                if hasattr(pred, 'to_dict'):
                    serializable_predicates.append(pred.to_dict())
                elif isinstance(pred, dict):
                    serializable_predicates.append(pred)
                else:
                    serializable_predicates.append({"raw": str(pred)})
            
            output_content = self.format_for_output(serializable_predicates, args.format)
        
        # Save output
        self.save_output(output_content, args.output, args.format)
        
        # Show token usage
        input_tokens, output_tokens = llm.get_tokens()
        if input_tokens or output_tokens:
            print(f"Token usage: {input_tokens} input, {output_tokens} output")
        
    def _load_description(self, desc_input: str) -> str:
        """Load domain description from file or text."""
        desc_path = Path(desc_input)
        if desc_path.exists() and desc_path.is_file():
            try:
                return desc_path.read_text().strip()
            except Exception as e:
                raise CLIError(
                    f"Failed to read description file: {e}",
                    ["Check file permissions and encoding"]
                )
        return desc_input
    
    def _load_optional_file(self, file_path: Optional[str], description: str) -> Any:
        """Load optional file if provided."""
        if not file_path:
            return None
        return self.load_input_file(file_path, description)