"""
Error handling utilities for L2P CLI.

Provides custom exception classes and error handling functions
with embedded troubleshooting tips.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Optional


class CLIError(Exception):
    """Base exception for CLI errors with troubleshooting tips."""

    def __init__(self, message: str, tips: Optional[list] = None):
        super().__init__(message)
        self.message = message
        self.tips = tips or []

    def __str__(self):
        error_msg = f"ERROR: {self.message}"
        if self.tips:
            error_msg += "\n\nTroubleshooting:"
            for i, tip in enumerate(self.tips, 1):
                error_msg += f"\n• {tip}"
        return error_msg


class ConfigError(CLIError):
    """Configuration-related errors."""

    pass


class ModelError(CLIError):
    """Model loading and connection errors."""

    pass


class GenerationError(CLIError):
    """PDDL generation errors."""

    pass


class ValidationError(CLIError):
    """PDDL validation errors."""

    pass


class TemplateError(CLIError):
    """Template loading errors."""

    pass


def handle_error(error: Exception):
    """Handle an exception and display appropriate error message.

    Args:
        error: The exception to handle.
    """
    # if it is already a CLIError just print it
    if isinstance(error, CLIError):
        print(str(error), file=sys.stderr)
        return

    # for other exceptions create a generic error with troubleshooting
    error_type = type(error).__name__
    error_msg = str(error) or "Unknown error"

    # get traceback for debugging
    tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
    tb_text = "".join(tb_lines[-3:])  # Last 3 frames

    # determine error category and specific troubleshooting tips
    tips = []

    if isinstance(error, ImportError):
        tips = [
            "Ensure all required packages are installed: `pip install llm tiktoken rich`",
            "If using a virtual environment, activate it first",
            "Check Python version (requires >= 3.10)",
            "Try reinstalling L2P: `pip install -e .[cli]`",
        ]
    elif isinstance(error, FileNotFoundError):
        tips = [
            "Check if the file or directory exists",
            "Verify file permissions",
            "Use absolute paths for better reliability",
            "Ensure you're in the correct working directory",
        ]
    elif isinstance(error, PermissionError):
        tips = [
            "Check file and directory permissions",
            "Try running with appropriate permissions",
            "Ensure you have write access to the target directory",
        ]
    elif isinstance(error, ConnectionError) or "connection" in error_msg.lower():
        tips = [
            "Check your internet connection",
            "Verify API keys are correct and have not expired",
            "Check if the LLM service is available",
            "Try increasing timeout or retry count",
            "Test with `l2p models test` to verify connection",
        ]
    elif isinstance(error, ValueError) and (
        "model" in error_msg.lower() or "provider" in error_msg.lower()
    ):
        tips = [
            "Run `l2p models list` to see available models",
            "Check model configuration with 'l2p config show'",
            "Ensure the model exists in your provider's configuration",
            "Run `l2p init` to reconfigure model settings",
        ]
    else:
        # generic troubleshooting tips
        tips = [
            "Check command syntax with `l2p <command> --help`",
            "Ensure L2P is properly installed: `pip install -e .[cli]`",
            "Check configuration with `l2p config show`",
            "Run with --verbose flag for more details",
            f"See error details: {tb_text.strip()}",
        ]

    # display error
    cli_error = CLIError(f"{error_type}: {error_msg}", tips)

    print(str(cli_error), file=sys.stderr)

    # suggest documentation for complex errors
    if not isinstance(error, (ImportError, FileNotFoundError, PermissionError)):
        print(
            "\nFor more help, visit: https://github.com/AI-Planning/l2p",
            file=sys.stderr,
        )


def check_required_env_var(var_name: str) -> str:
    """Check if a required environment variable is set."""
    value = os.environ.get(var_name)
    if not value:
        raise ConfigError(
            f"Required environment variable {var_name} is not set.",
            [
                f'Set the environment variable: export {var_name}="your-value"',
                "Or add it to your shell configuration file (.bashrc, .zshrc, etc.)",
                "For temporary testing, set it before running: {var_name}=value l2p ...",
            ],
        )
    return value


def check_file_exists(file_path: str, description: str = "file") -> Path:
    """Check if a file exists and return Path object."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{description.capitalize()} not found: {file_path}",
            [
                f"Check the path: {file_path}",
                "Use absolute path for better reliability",
                f"Ensure the {description} exists and is readable",
            ],
        )
    return path
