"""
Configuration manager for L2P CLI.

Handles loading, saving, and validating configuration files.
Manages environment variables and provides model configuration.
"""

import os
import sys
import json
import copy
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import shutil

from .errors import CLIError
from ...llm.base import resolve_config_path


class ConfigManager:
    """Manages L2P CLI configuration."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        "model": {
            "backend": "unified",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "config_path": "l2p/llm/utils/llm.yaml",
            "api_key": "${OPENAI_API_KEY}",
        },
        "generation": {
            "default_format": "pddl",
            "max_retries": 3,
            "syntax_validation": True,
        },
        "templates": {
            "default_path": "l2p/templates/",
            "custom_path": None,
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager.
        
        Args:
            config_path: Optional path to config file. If None, uses default.
        """
        self.config_dir = Path.home() / ".l2p"
        self.config_file = self.config_dir / "config.yaml"
        
        if config_path:
            self.config_file = Path(config_path).expanduser().resolve()
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default.
        
        Returns:
            Configuration dictionary.
            
        Raises:
            CLIError: If config file exists but is invalid.
        """
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if config file exists
        if not self.config_file.exists():
            return self.DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                return self.DEFAULT_CONFIG.copy()
            
            # Merge with defaults to ensure all sections exist
            merged = copy.deepcopy(self.DEFAULT_CONFIG)
            self._deep_update(merged, config)
            return merged
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise CLIError(
                f"Failed to parse configuration file: {self.config_file}\n"
                f"Error: {e}\n\n"
                "Troubleshooting:\n"
                "• Check YAML syntax in config file\n"
                "• Delete config file to regenerate: rm {self.config_file}\n"
                "• Run 'l2p init' to create new configuration"
            )
    
    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """Save configuration to file.
        
        Args:
            config: Configuration to save. If None, saves current config.
        """
        if config is not None:
            self.config = config
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise CLIError(
                f"Failed to save configuration: {e}\n\n"
                "Troubleshooting:\n"
                "• Check write permissions for {self.config_file}\n"
                "• Ensure enough disk space"
            )
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration with environment variables resolved.
        
        Returns:
            Model configuration dictionary with resolved API keys.
        """
        model_config = self.config.get("model", {}).copy() # retrieve model configuration file
        api_key = model_config.get("api_key", "") # retrieve API key entry

        # check if env variable
        if api_key.endswith("_API_KEY"):
            env_key = os.getenv(api_key)
            # if env key is empty
            if not env_key:
                print(
                    f"[WARNING] Could not find environment variable: {api_key}. Setting API key to None.\n"
                    f"Env variable must strictly end with `_API_KEY`. Example: export {{MY_API_KEY}}=\"your-key\""
                )
                model_config["api_key"] = ""
            else:
                model_config["api_key"] = env_key
        
        return model_config
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration."""
        return self.config.get("generation", {}).copy()
    
    def get_templates_config(self) -> Dict[str, Any]:
        """Get templates configuration."""
        return self.config.get("templates", {}).copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates.
        """
        self._deep_update(self.config, updates)
        self.save_config()
    
    def validate_model_config(self) -> Tuple[bool, str]:
        """Validate current model configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            model_config = self.get_model_config()
            
            # Check required fields
            required = ["provider", "model", "config_path"]
            for field in required:
                if not model_config.get(field):
                    return False, f"Missing required field: {field}"
            
            # Check config path exists
            config_path = model_config.get("config_path")
            if config_path:
                try:
                    resolve_config_path(config_path)
                except FileNotFoundError as e:
                    return False, str(e)
            
            # Check backend matches config_path for known pairs
            backend = model_config.get("backend", "")
            if config_path.endswith("openaiSDK.yaml") and backend != "openai":
                return False, f"Backend is '{backend}' but config_path '{config_path}' suggests 'openai'"
            if config_path.endswith("llm.yaml") and backend != "unified":
                return False, f"Backend is '{backend}' but config_path '{config_path}' suggests 'unified'"
            
            return True, "Configuration valid"
            
        except CLIError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Recursively update target dictionary with source values."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def get_config_path(self) -> Path:
        """Get path to configuration file."""
        return self.config_file
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save_config()


class CLIError(Exception):
    """Custom exception for CLI errors with troubleshooting tips."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


# Convenience function for getting config manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get or create config manager instance.
    
    Args:
        config_path: Optional path to config file.
        
    Returns:
        ConfigManager instance.
    """
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager