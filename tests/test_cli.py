"""
Unit tests for L2P CLI.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from l2p.cli.commands.generators.types import TypesGenerator
from tests.mock_llm import MockLLM


class TestCLITypesGenerator(unittest.TestCase):
    """Test types generator."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.yaml"
        
        # Create a minimal config
        config = {
            "model": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "config_path": "l2p/llm/utils/llm.yaml",
                "api_key": "test-key",
            },
            "generation": {
                "default_format": "pddl",
                "max_retries": 1,
                "syntax_validation": False,
            },
            "templates": {
                "default_path": "l2p/templates/",
                "custom_path": None,
            }
        }
        
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Mock LLM
        self.mock_llm = MockLLM()
        self.mock_llm.output = """
### TYPES
```
{
    "block": "block that can be stacked",
    "table": "table that blocks sit on"
}
```
"""
        # Add missing method
        self.mock_llm.get_tokens = lambda: (0, 0)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('l2p.DomainBuilder')
    @patch('l2p.SyntaxValidator')
    def test_generate_types_basic(self, mock_syntax_validator_cls, mock_domain_builder_cls):
        """Test basic types generation with mocked LLM."""
        # Mock DomainBuilder instance
        mock_builder = MagicMock()
        mock_domain_builder_cls.return_value = mock_builder
        
        # Mock SyntaxValidator instance
        mock_syntax_validator = MagicMock()
        mock_syntax_validator_cls.return_value = mock_syntax_validator
        
        # Set up the formalize_types return value
        mock_builder.formalize_types.return_value = (
            {"block": "block that can be stacked", "table": "table that blocks sit on"},
            "LLM output",
            (True, "Validation passed")
        )
        
        # Create generator
        generator = TypesGenerator()
        
        # Mock load_llm to return our mock LLM
        with patch.object(generator, 'load_llm', return_value=self.mock_llm):
            # Mock template manager
            with patch.object(generator.template_manager, 'get_template') as mock_get_template:
                mock_get_template.return_value = "Template with {domain_desc} and {types}"
                
                # Mock config manager to avoid file reading
                with patch('l2p.cli.commands.generators.types.get_config_manager') as mock_get_cm:
                    mock_cm = MagicMock()
                    mock_cm.get_model_config.return_value = {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "config_path": "l2p/llm/utils/llm.yaml",
                        "api_key": "test-key",
                    }
                    mock_get_cm.return_value = mock_cm
                    
                    # Run generator with args
                    from argparse import Namespace
                    args = Namespace(
                        desc="blocksworld domain",
                        hierarchy=False,
                        types_file=None,
                        template_file=None,
                        format="pddl",
                        output=None,
                        max_retries=1,
                        no_validation=True
                    )
                    
                    # Capture stdout
                    import io
                    from contextlib import redirect_stdout
                    f = io.StringIO()
                    with redirect_stdout(f):
                        generator.generate(args)
                    
                    output = f.getvalue()
                    
                    # Verify
                    self.assertIn("block", output)
                    self.assertIn("table", output)
                    mock_builder.formalize_types.assert_called_once()


if __name__ == '__main__':
    unittest.main()