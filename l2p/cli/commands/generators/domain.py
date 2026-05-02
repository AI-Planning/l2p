"""
Domain generator for L2P CLI.

Generates complete PDDL domain using pipeline approach.
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..generate import GeneratorBase
from ...utils.config import get_config_manager, CLIError
from ...utils.templates import get_template_manager
from ...utils.errors import handle_error


def add_subparser(subparsers):
    """Add domain generator subparser."""
    parser = subparsers.add_parser(
        "domain",
        help="Generate complete PDDL domain",
        description="Generate complete PDDL domain using pipeline approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate complete domain with pipeline
  l2p generate domain --desc "blocksworld domain" --pipeline
  
  # Generate domain with specific actions
  l2p generate domain --desc "blocksworld" --actions "pick-up, put-down, stack, unstack"
  
  # Save intermediate files
  l2p generate domain --desc "domain" --save-intermediate --output-dir ./domain_components
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--desc",
        type=str,
        required=True,
        help="Domain description (text or path to file)"
    )
    
    # Pipeline options
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use automatic pipeline to generate all components"
    )
    
    parser.add_argument(
        "--actions",
        type=str,
        help="Comma-separated list of action names to generate"
    )
    
    parser.add_argument(
        "--requirements",
        type=str,
        default=":strips,:typing",
        help="PDDL requirements (default: :strips,:typing)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output PDDL file (default: stdout)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for intermediate files (used with --save-intermediate)"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate component files"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for LLM (default: 3)"
    )
    
    parser.set_defaults(func=generate_domain_command)


def generate_domain_command(args):
    """Execute domain generation command."""
    try:
        generator = DomainGenerator()
        generator.generate(args)
    except Exception as e:
        handle_error(e)
        sys.exit(1)


class DomainGenerator(GeneratorBase):
    """Generator for complete PDDL domains."""
    
    def generate(self, args):
        """Generate complete domain based on command line arguments."""
        # Load domain description
        domain_desc = self._load_description(args.desc)
        
        if not args.pipeline and not args.actions:
            raise CLIError(
                "Must specify either --pipeline or --actions",
                [
                    "Use --pipeline for automatic component generation",
                    "Or --actions to specify which actions to generate"
                ]
            )
        
        # Create output directory if needed
        output_dir = None
        if args.output_dir or args.save_intermediate:
            output_dir = Path(args.output_dir if args.output_dir else ".")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating domain from description...")
        print(f"Description: {domain_desc[:100]}..." if len(domain_desc) > 100 else f"Description: {domain_desc}")
        
        # Step 1: Generate types
        print("\n" + "="*60)
        print("Step 1: Generating types...")
        types = self._generate_types(domain_desc, output_dir, args.max_retries)
        
        # Step 2: Generate constants (optional)
        print("\n" + "="*60)
        print("Step 2: Generating constants...")
        constants = self._generate_constants(domain_desc, types, output_dir, args.max_retries)
        
        # Step 3: Generate predicates
        print("\n" + "="*60)
        print("Step 3: Generating predicates...")
        predicates = self._generate_predicates(domain_desc, types, constants, output_dir, args.max_retries)
        
        # Step 4: Generate actions
        print("\n" + "="*60)
        print("Step 4: Generating actions...")
        
        action_names = []
        if args.actions:
            action_names = [name.strip() for name in args.actions.split(",")]
        else:
            # Extract action names from description using NL action extraction
            action_names = self._extract_action_names(domain_desc, types, args.max_retries)
        
        actions = self._generate_actions(
            domain_desc, action_names, types, constants, predicates, 
            output_dir, args.max_retries
        )
        
        # Step 5: Generate complete domain
        print("\n" + "="*60)
        print("Step 5: Generating complete PDDL domain...")
        
        domain_pddl = self._generate_domain_pddl(
            domain_desc=domain_desc,
            requirements=args.requirements,
            types=types,
            constants=constants,
            predicates=predicates,
            actions=actions
        )
        
        # Save or output domain
        self.save_output(domain_pddl, args.output, "pddl")
        
        print("\n" + "="*60)
        print("✅ Domain generation complete!")
        
        if output_dir:
            print(f"Intermediate files saved to: {output_dir.resolve()}")
        
    def _generate_types(self, domain_desc: str, output_dir: Optional[Path], max_retries: int) -> Any:
        """Generate types from domain description."""
        from l2p import DomainBuilder
        
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        
        template_content = self.template_manager.get_template("formalize_type_hierarchy.txt", "domain")
        
        print("  Using hierarchical type generation...")
        result = domain_builder.formalize_type_hierarchy(
            model=llm,
            domain_desc=domain_desc,
            prompt_template=template_content,
            max_retries=max_retries
        )
        
        types_result, llm_output, validation_info = result
        
        if validation_info and not validation_info[0]:
            print(f"  ⚠ Validation warning: {validation_info[1]}")
        
        # Save intermediate file if requested
        if output_dir:
            types_file = output_dir / "types.json"
            with open(types_file, 'w') as f:
                json.dump(types_result, f, indent=2)
            print(f"  Types saved to: {types_file}")
        
        return types_result
    
    def _generate_constants(self, domain_desc: str, types: Any, output_dir: Optional[Path], max_retries: int) -> Any:
        """Generate constants from domain description."""
        from l2p import DomainBuilder
        
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        
        template_content = self.template_manager.get_template("formalize_constants.txt", "domain")
        
        print("  Generating constants...")
        result = domain_builder.formalize_constants(
            model=llm,
            domain_desc=domain_desc,
            prompt_template=template_content,
            types=types,
            max_retries=max_retries
        )
        
        constants_result, llm_output, validation_info = result
        
        if validation_info and not validation_info[0]:
            print(f"  ⚠ Validation warning: {validation_info[1]}")
        
        # Save intermediate file if requested
        if output_dir and constants_result:
            constants_file = output_dir / "constants.json"
            with open(constants_file, 'w') as f:
                json.dump(constants_result, f, indent=2)
            print(f"  Constants saved to: {constants_file}")
        
        return constants_result
    
    def _generate_predicates(self, domain_desc: str, types: Any, constants: Any, 
                           output_dir: Optional[Path], max_retries: int) -> Any:
        """Generate predicates from domain description."""
        from l2p import DomainBuilder
        
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        
        template_content = self.template_manager.get_template("formalize_predicates.txt", "domain")
        
        print("  Generating predicates...")
        result = domain_builder.formalize_predicates(
            model=llm,
            domain_desc=domain_desc,
            prompt_template=template_content,
            types=types,
            constants=constants,
            max_retries=max_retries
        )
        
        predicates_result, llm_output, validation_info = result
        
        if validation_info and not validation_info[0]:
            print(f"  ⚠ Validation warning: {validation_info[1]}")
        
        # Save intermediate file if requested
        if output_dir:
            predicates_file = output_dir / "predicates.json"
            # Convert to serializable format
            serializable_predicates = []
            for pred in predicates_result:
                if hasattr(pred, 'to_dict'):
                    serializable_predicates.append(pred.to_dict())
                elif isinstance(pred, dict):
                    serializable_predicates.append(pred)
                else:
                    serializable_predicates.append({"raw": str(pred)})
            
            with open(predicates_file, 'w') as f:
                json.dump(serializable_predicates, f, indent=2)
            print(f"  Predicates saved to: {predicates_file}")
        
        return predicates_result
    
    def _extract_action_names(self, domain_desc: str, types: Any, max_retries: int) -> List[str]:
        """Extract action names from domain description."""
        from l2p import DomainBuilder
        
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        
        template_content = self.template_manager.get_template("extract_nl_actions.txt", "domain")
        
        print("  Extracting action names from description...")
        result = domain_builder.extract_nl_actions(
            model=llm,
            domain_desc=domain_desc,
            prompt_template=template_content,
            types=types,
            max_retries=max_retries
        )
        
        nl_actions, llm_output = result
        
        if not nl_actions:
            print("  ⚠ No actions extracted, using default action names")
            return ["action1", "action2", "action3"]
        
        action_names = list(nl_actions.keys())
        print(f"  Extracted {len(action_names)} actions: {', '.join(action_names)}")
        return action_names
    
    def _generate_actions(self, domain_desc: str, action_names: List[str], types: Any, 
                        constants: Any, predicates: Any, output_dir: Optional[Path], 
                        max_retries: int) -> List[Any]:
        """Generate actions from domain description."""
        from l2p import DomainBuilder
        
        llm = self.load_llm()
        domain_builder = DomainBuilder()
        
        actions = []
        
        for i, action_name in enumerate(action_names, 1):
            print(f"  [{i}/{len(action_names)}] Generating action: {action_name}")
            
            template_content = self.template_manager.get_template("formalize_pddl_action.txt", "domain")
            
            result = domain_builder.formalize_pddl_action(
                model=llm,
                domain_desc=domain_desc,
                prompt_template=template_content,
                action_name=action_name,
                action_desc=f"{action_name} action for the domain",
                types=types,
                constants=constants,
                predicates=predicates,
                max_retries=max_retries
            )
            
            action_result, new_predicates, llm_output, validation_info = result
            
            if validation_info and not validation_info[0]:
                print(f"    ⚠ Validation warning: {validation_info[1]}")
            
            actions.append(action_result)
            
            # Save intermediate file if requested
            if output_dir:
                action_file = output_dir / f"action_{action_name}.json"
                with open(action_file, 'w') as f:
                    json.dump(action_result, f, indent=2)
        
        return actions
    
    def _generate_domain_pddl(self, domain_desc: str, requirements: str, types: Any, 
                            constants: Any, predicates: Any, actions: List[Any]) -> str:
        """Generate complete PDDL domain string."""
        from l2p import DomainBuilder
        
        domain_builder = DomainBuilder()
        
        # Parse requirements string
        requirements_list = [req.strip() for req in requirements.split(",") if req.strip()]
        
        # Generate domain
        domain_pddl = domain_builder.generate_domain(
            domain_name=self._extract_domain_name(domain_desc),
            requirements=requirements_list,
            types=types,
            constants=constants,
            predicates=predicates,
            actions=actions
        )
        
        return domain_pddl
    
    def _extract_domain_name(self, domain_desc: str) -> str:
        """Extract domain name from description."""
        # Simple heuristic: use first few words
        words = domain_desc.split()[:3]
        name = "-".join(words).lower()
        # Remove non-alphanumeric characters
        import re
        name = re.sub(r'[^a-z0-9-]', '', name)
        return name or "generated-domain"
    
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