"""
Task generator for L2P CLI.

Generates PDDL problem/task specifications.
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
    """Add task generator subparser."""
    parser = subparsers.add_parser(
        "task",
        help="Generate PDDL problem/task",
        description="Generate PDDL problem specification from description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate problem with domain file
  l2p generate task --desc "stack blocks A and B" --domain-file domain.pddl
  
  # With types and predicates
  l2p generate task --desc "problem" --types-file types.json --predicates-file predicates.json
  
  # Output to file
  l2p generate task --desc "problem" --domain-file domain.pddl --output problem.pddl
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--desc",
        type=str,
        required=True,
        help="Problem description (text or path to file)"
    )
    
    # Domain specification (either domain file or components)
    domain_group = parser.add_mutually_exclusive_group()
    domain_group.add_argument(
        "--domain-file",
        type=str,
        help="PDDL domain file"
    )
    domain_group.add_argument(
        "--domain-name",
        type=str,
        default="generated-domain",
        help="Domain name (used with component files)"
    )
    
    # Component files (alternative to domain file)
    parser.add_argument(
        "--types-file",
        type=str,
        help="Types file (JSON/YAML)"
    )
    
    parser.add_argument(
        "--predicates-file",
        type=str,
        help="Predicates file (JSON/YAML)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output PDDL file (default: stdout)"
    )
    
    parser.add_argument(
        "--problem-name",
        type=str,
        help="Problem name (default: derived from description)"
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
    
    parser.set_defaults(func=generate_task_command)


def generate_task_command(args):
    """Execute task generation command."""
    try:
        generator = TaskGenerator()
        generator.generate(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


class TaskGenerator(GeneratorBase):
    """Generator for PDDL tasks/problems."""
    
    def generate(self, args):
        """Generate task based on command line arguments."""
        # Load problem description
        problem_desc = self._load_description(args.desc)
        
        # Load domain components
        domain_name = args.domain_name
        types = None
        predicates = None
        
        if args.domain_file:
            # Parse domain file to extract components
            domain_name, types, predicates = self._parse_domain_file(args.domain_file)
        else:
            # Load component files
            types = self._load_optional_file(args.types_file, "types")
            predicates = self._load_optional_file(args.predicates_file, "predicates")
        
        # Determine template
        template_content = self.template_manager.get_template("formalize_task.txt", "task")
        
        # Load LLM
        llm = self.load_llm()
        
        # Load TaskBuilder
        from l2p import TaskBuilder, SyntaxValidator
        
        task_builder = TaskBuilder()
        
        # Configure syntax validator
        syntax_validator = None
        if not args.no_validation:
            syntax_validator = SyntaxValidator()
            syntax_validator.error_types = [
                "validate_header",
                "validate_duplicate_headers",
                "validate_unsupported_keywords"
            ]
        
        print("Generating problem specification...")
        
        # Call generation method
        result = task_builder.formalize_task(
            model=llm,
            problem_desc=problem_desc,
            prompt_template=template_content,
            syntax_validator=syntax_validator,
            max_retries=args.max_retries
        )
        
        objects, initial, goal, llm_output, validation_info = result
        
        # Check validation results
        if validation_info and not validation_info[0]:
            print(f"⚠ Validation warning: {validation_info[1]}")
        
        # Determine problem name
        problem_name = args.problem_name or self._extract_problem_name(problem_desc)
        
        # Generate complete PDDL problem
        problem_pddl = task_builder.generate_task(
            domain_name=domain_name,
            problem_name=problem_name,
            objects=objects,
            initial=initial,
            goal=goal
        )
        
        # Save or output problem
        self.save_output(problem_pddl, args.output, "pddl")
        
        # Show token usage
        input_tokens, output_tokens = llm.get_tokens()
        if input_tokens or output_tokens:
            print(f"Token usage: {input_tokens} input, {output_tokens} output")
        
    def _parse_domain_file(self, domain_file: str) -> tuple:
        """Parse PDDL domain file to extract domain name, types, and predicates.
        
        Returns:
            Tuple of (domain_name, types, predicates)
        """
        try:
            with open(domain_file, 'r') as f:
                content = f.read()
            
            # Simple parsing - extract domain name
            import re
            
            # Extract domain name
            domain_match = re.search(r'\(domain\s+([^)\s]+)\)', content)
            domain_name = domain_match.group(1) if domain_match else "unknown-domain"
            
            # Note: Full parsing would require PDDL parser
            # For now, return minimal information
            print(f"  Using domain: {domain_name} from {domain_file}")
            print("  Note: Full domain parsing not implemented - using basic information")
            
            return domain_name, None, None
            
        except Exception as e:
            raise CLIError(
                f"Failed to parse domain file: {e}",
                [
                    "Check file format and syntax",
                    "Ensure it's a valid PDDL domain file",
                    "Try using component files instead: --types-file and --predicates-file"
                ]
            )
    
    def _extract_problem_name(self, problem_desc: str) -> str:
        """Extract problem name from description."""
        # Simple heuristic: use first few words
        words = problem_desc.split()[:3]
        name = "-".join(words).lower()
        # Remove non-alphanumeric characters
        import re
        name = re.sub(r'[^a-z0-9-]', '', name)
        return name or "generated-problem"
    
    def _load_description(self, desc_input: str) -> str:
        """Load description from file or text."""
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