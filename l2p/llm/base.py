"""
This file contains code for calling LLMs and saving raw model outputs.
Currently, this builder class contains the generic method to run a wide range of LLMs.
"""

import logging, functools
from abc import ABC, abstractmethod
from typing import Any
import yaml

LOG: logging.Logger = logging.getLogger(__name__)


def load_yaml(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def require_llm(func):
    """
    Decorator to check if an LLM instance is provided and catch errors.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # check if LLM instance is provided
        model = kwargs.get("model", None)
        if model is None:
            # attempt tp get model from positional argument
            for arg in args:
                if isinstance(arg, BaseLLM):
                    model = arg
                    break

        if not model:
            raise ValueError("An LLM instance must be provided to use this method.")

        # try-catch to ensure LLM is even used properly
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(
                f"An error occurred in {func.__name__}: {e}.\n You must provide an LLM engine OR proper configuration to use L2P. Refer to https://github.com/AI-Planning/l2p."
            )
            raise

    return wrapper


class BaseLLM(ABC):
    def __init__(self, model: str, api_key: str | None = None) -> None:

        if not self.valid_models():
            raise ValueError(
                f"No valid models exist in '{self.provider}' provider. Please check if your provider is spelt correctly, or exists in your llm.yaml configuration."
            )

        if model not in self.valid_models():
            raise ValueError(
                f"'{model}' is not a valid model for '{self.provider}' provider. "
                f"Valid models are: {', '.join(self.valid_models())}."
            )
        self.model: str = model
        self.api_key: str | None = api_key

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM
        Returns:
            str: The response from the LLM
        """
        pass

    def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
        """
        Abstract method to query an LLM with a given prompt and system prompt and return the response.

        Args:
            system_prompt (str): The system prompt to send to the LLM
            prompt (str): The prompt to send to the LLM
        Returns:
            str: The response from the LLM
        """
        return self.query(system_prompt + "\n" + prompt)

    def valid_models(self) -> list[str]:
        """
        List of valid model parameters, e.g., 'gpt4o-mini' for GPT
        """
        return []
